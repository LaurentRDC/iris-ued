# -*- coding: utf-8 -*-
"""
Raw dataset classes
===================

The following classes are defined herein:

.. autosummary::
    :toctree: classes/

    AbstractRawDataset
    McGillRawDataset
    MerlinRawDataset
    FSURawDataset

Subclassing RawDatasetBase
==========================
"""
from abc import abstractmethod
from collections import OrderedDict
from functools import partial

import numpy as np

from npstreams import average, pmap
from skued import ialign

from .meta import ExperimentalParameter, MetaRawDataset


class AbstractRawDataset(metaclass = MetaRawDataset):
    """
    Abstract base class for ultrafast electron diffraction data set. 
    RawDatasetBase allows for enforced metadata types and values, 
    as well as a standard interface. For example, AbstractRawDataset
    implements the context manager interface.

    Minimally, the following method must be implemented in subclasses:

        * raw_data

    It is suggested to also implement the following magic methods:

        * __init__ 
        * __exit__
    
    The call signature should remain the same for all overwritten methods.
    """

    # List of valid metadata below
    # Using the ExperimentalParameter allows for automatic registering
    # of the parameters as valid.
    # These attributes can be accessed using the usual property access
    date            = ExperimentalParameter('date',            str,   default = '')
    energy          = ExperimentalParameter('energy',          float, default = 90)       # keV
    pump_wavelength = ExperimentalParameter('pump_wavelength', int,   default = 800)      # nanometers
    fluence         = ExperimentalParameter('fluence',         float, default = 0)        # mj / cm**2
    time_zero_shift = ExperimentalParameter('time_zero_shift', float, default = 0)        # picoseconds
    temperature     = ExperimentalParameter('temperature',     float, default = 293)      # Kelvins
    exposure        = ExperimentalParameter('exposure',        float, default = 1)        # seconds
    resolution      = ExperimentalParameter('resolution',      tuple, default = (2048, 2048))
    time_points     = ExperimentalParameter('time_points',     tuple, default = tuple())  # picoseconds
    scans           = ExperimentalParameter('scans',           tuple, default = (1,))
    camera_length   = ExperimentalParameter('camera_length',   float, default = 0.23)     # meters
    pixel_width     = ExperimentalParameter('pixel_width',     float, default = 14e-6)    # meters
    notes           = ExperimentalParameter('notes',           str,   default = '')

    def __init__(self, source = None, metadata = dict()):
        """
        Parameters
        ----------
        source : object
            Data source, for example a directory or external file.
        metadata : dict or None, optional
            Metadata and experimental parameters. Dictionary keys that are
            not valid metadata, they are ignored. Metadata can also be
            set directly later.

        Raises
        ------
        TypeError : if an item from the metadata has an unexpected type.
        """
        self.source = source
        if metadata:
            self.update_metadata(metadata)
    
    def __repr__(self):
        string = '< RawDataset object. '
        for k, v in self.metadata.items():
            string.join('\n {key}: {value} '.format(key = k, value = v))
        string.join(' >')
        return string
    
    def __enter__(self):
        """ Return `self` upon entering the runtime context. """
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Raise any exception triggered within the runtime context. """
        # Perform cleanup operations here
        pass
    
    def update_metadata(self, metadata):
        """
        Update metadata from a dictionary. Only appropriate keys are used; irrelevant keys are ignored.

        Parameters
        ----------
        metadata : dictionary
            See ``AbstractRawDataset.valid_metadata`` for valid keys.
        """
        for k, v in metadata.items():
            if k in self.valid_metadata:
                setattr(self, k, v)

    @property
    def metadata(self):
        """ Experimental parameters and dataset metadata as a dictionary. """
        meta = {k:getattr(self, k) for k in self.valid_metadata} 
        # Ordered dictionary by keys is easiest to inspect
        return OrderedDict(sorted(meta.items(), 
                           key = lambda t: t[0]))

    def iterscan(self, scan, **kwargs):
        """
        Generator function of images as part of a scan, in 
        time-delay order.

        Parameters
        ----------
        scan : int
            Scan from which to yield the data.
        kwargs
            Keyword-arguments are passed to ``raw_data`` method.
        
        Yields
        ------
        data : `~numpy.ndarray`, ndim 2
        """
        if scan not in set(self.scans):
            raise ValueError('There is no scan {} in available scans'.format(scan))
        
        for timedelay in self.time_points:
            yield self.raw_data(timedelay = timedelay, scan = scan, **kwargs)
    
    @abstractmethod
    def raw_data(self, timedelay, scan = 1, bgr = True, **kwargs):
        """
        Returns an array of the image at a timedelay and scan.
        
        Parameters
        ----------
        timdelay : float
            Acquisition time-delay.
        scan : int, optional
            Scan number. Default is 1.
        bgr : bool, optional
            If True (default), laser background is removed before being returned.
        
        Returns
        -------
        arr : `~numpy.ndarray`, ndim 2
        
        Raises
        ------
        IOError : Filename is not associated with an image/does not exist.
        """ 
        pass
    
    def reduced(self, exclude_scans = tuple(), align = True, processes = 1, dtype = np.float):
        """
        Generator of reduced dataset.

        Parameters
        ----------
        exclude_scans : iterable, optional
            Iterable of ints. These scans will be skipped when reducing the dataset.
        align : bool, optional
            If True (default), raw images will be aligned on a per-scan basis.
        processes : int or None, optional
            Number of Processes to spawn for processing. 
        dtype : numpy.dtype or None, optional
            Patterns will be cast to ``dtype``. If None (default), ``dtype`` will be set to the same
            data-type as the first pattern in ``patterns``.

        Yields
        ------
        pattern : `~numpy.ndarray`, ndim 2
        """
        valid_scans = list(sorted(set(self.scans) - set(exclude_scans)))

        yield from pmap(_raw_combine, self.time_points,
                        args = (self, valid_scans, align), 
                        processes = processes,
                        ntotal = len(self.time_points))

# For multiprocessing, the function to be mapped must be 
# global, hence defined outside of the AbstractRawDataset class
# TODO: include dtype in _raw_combine
def _raw_combine(raw, valid_scans, align, timedelay):
    images = map(partial(raw.raw_data, timedelay), valid_scans)
    if align:
        images = ialign(images)
    return average(images)
