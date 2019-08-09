# -*- coding: utf-8 -*-
"""
Raw dataset classes
===================
"""
from abc import abstractmethod
from collections import OrderedDict
from contextlib import AbstractContextManager
from functools import wraps
from pathlib import Path

import numpy as np

from npstreams import average, itercopy, peek, pmap
from skued import ialign

from .meta import ExperimentalParameter, MetaRawDataset


def open_raw(path):
    """
    Open a raw data item, guessing the AbstractRawDataset instance that 
    should be used based on available plug-ins.

    This function can also be used as a context manager::

        with open_raw('.') as dset:
            ...
    
    Parameters
    ----------
    path : path-like
        Path to the file/folder containing the raw data.
    
    Returns
    -------
    raw : AbstractRawDataset instance
        The raw dataset. If no format could be guessed, an RuntimeError is raised.
    
    Raises
    ------
    RuntimeError : if the data format could not be guessed.
    """
    if isinstance(path, Path):
        path = str(path)

    # For easier debugging, data formats are checked in deterministic order
    for dataformat in sorted(AbstractRawDataset.implementations, key=str):
        try:
            return dataformat(path)
        except:
            pass

    raise RuntimeError(
        "No data format could be guessed for item located at: \n {}".format(path)
    )


class AbstractRawDataset(AbstractContextManager, metaclass=MetaRawDataset):
    """
    Abstract base class for ultrafast electron diffraction data set. 
    AbstractRawDataset allows for enforced metadata types and values, 
    as well as a standard interface. For example, AbstractRawDataset
    implements the context manager interface.

    Minimally, the following method must be implemented in subclasses:

        * raw_data

    It is suggested to also implement the following magic method:

        * __init__
        * __exit__
    
    For better results or performance during reduction, the following methods
    can be specialized:

        * reduced
    
    A list of concrete implementations of AbstractRawDatasets is available in 
    the ``implementations`` class attribute. Subclasses are automatically added.
    
    The call signature must remain the same for all overwritten methods.
    """

    # List of valid metadata below
    # Using the ExperimentalParameter allows for automatic registering
    # of the parameters as valid.
    # These attributes can be accessed using the usual property access
    date = ExperimentalParameter("date", str, default="")
    energy = ExperimentalParameter("energy", float, default=90)  # keV
    pump_wavelength = ExperimentalParameter(
        "pump_wavelength", int, default=800
    )  # nanometers
    fluence = ExperimentalParameter("fluence", float, default=0)  # mj / cm**2
    time_zero_shift = ExperimentalParameter(
        "time_zero_shift", float, default=0
    )  # picoseconds
    temperature = ExperimentalParameter("temperature", float, default=293)  # Kelvins
    exposure = ExperimentalParameter("exposure", float, default=1)  # seconds
    resolution = ExperimentalParameter("resolution", tuple, default=(2048, 2048))
    time_points = ExperimentalParameter(
        "time_points", tuple, default=tuple()
    )  # picoseconds
    scans = ExperimentalParameter("scans", tuple, default=(1,))
    camera_length = ExperimentalParameter(
        "camera_length", float, default=0.23
    )  # meters
    pixel_width = ExperimentalParameter("pixel_width", float, default=14e-6)  # meters
    notes = ExperimentalParameter("notes", str, default="")

    def __init__(self, source=None, metadata=None):
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

    def __exit__(self, *exc):
        pass

    def __repr__(self):
        rep = "< {} object with following metadata: ".format(type(self).__name__)
        for key, val in self.metadata.items():
            rep += "\n    {key}: {val}".format(key=key, val=val)

        return rep + " >"

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
        meta = {k: getattr(self, k) for k in self.valid_metadata}
        # Ordered dictionary by keys is easiest to inspect
        return OrderedDict(sorted(meta.items(), key=lambda t: t[0]))

    def iterscan(self, scan, **kwargs):
        """
        Generator function of diffraction patterns as part of a scan, in 
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

        See Also
        --------
        itertime : generator of diffraction patterns for a single time-delay, in scan order
        """
        if scan not in set(self.scans):
            raise ValueError("There is no scan {} in available scans".format(scan))

        for timedelay in self.time_points:
            yield self.raw_data(timedelay=timedelay, scan=scan, **kwargs)

    def itertime(self, timedelay, exclude_scans=None, **kwargs):
        """
        Generator function of diffraction patterns of the same time-delay, in 
        scan order.

        Parameters
        ----------
            timedelay : float
            Scan from which to yield the data.
        exclude_scans : iterable or None, optional
            These scans will be skipped.
        kwargs
            Keyword-arguments are passed to ``raw_data`` method.
        
        Yields
        ------
        data : `~numpy.ndarray`, ndim 2

        See Also
        --------
        iterscan : generator of diffraction patterns for a single scan, in time-delay order
        """
        if not exclude_scans:
            exclude_scans = set([])

        if timedelay not in set(self.time_points):
            raise ValueError("There is no scan {} in available scans".format(scan))

        valid_scans = sorted(set(self.scans) - set(exclude_scans))
        for scan in valid_scans:
            yield self.raw_data(timedelay=timedelay, scan=scan, **kwargs)

    @abstractmethod
    def raw_data(self, timedelay, scan=1, **kwargs):
        """
        Returns an array of the image at a timedelay and scan.
        
        Parameters
        ----------
        timdelay : float
            Acquisition time-delay.
        scan : int, optional
            Scan number. Default is 1.
        kwargs
            Keyword-arguments are ignored.
        
        Returns
        -------
        arr : `~numpy.ndarray`, ndim 2
        
        Raises
        ------
        ValueError : if ``timedelay`` or ``scan`` are invalid / out of bounds.
        IOError : Filename is not associated with an image/does not exist.
        """
        pass

    def reduced(
        self,
        exclude_scans=None,
        align=True,
        normalize=True,
        mask=None,
        processes=1,
        dtype=np.float,
    ):
        """
        Generator of reduced dataset. The reduced diffraction patterns are generated in order of time-delay.

        This particular implementation normalizes diffracted intensity of pictures acquired at the same time-delay
        while rejecting masked pixels.

        Parameters
        ----------
        exclude_scans : iterable or None, optional
            These scans will be skipped when reducing the dataset.
        align : bool, optional
            If True (default), raw diffraction patterns will be aligned using the masked normalized
            cross-correlation approach. See `skued.align` for more information.
        normalize : bool, optional
            If True (default), equivalent diffraction pictures (e.g. same time-delay, different scans) 
            are normalized to the same diffracted intensity.
        mask : array-like of bool or None, optional
            If not None, pixels where ``mask = True`` are ignored for certain operations (e.g. alignment).
        processes : int or None, optional
            Number of Processes to spawn for processing. 
        dtype : numpy.dtype or None, optional
            Reduced patterns will be cast to ``dtype``.

        Yields
        ------
        pattern : `~numpy.ndarray`, ndim 2
        """
        kwargs = {
            "raw": self,
            "exclude_scans": exclude_scans,
            "align": align,
            "normalize": normalize,
            "invalid_mask": mask,
            "dtype": dtype,
        }

        combined = pmap(
            _raw_combine,
            self.time_points,
            kwargs=kwargs,
            processes=processes,
            ntotal=len(self.time_points),
        )

        # Each image at the same time-delay are aligned to each other. This means that
        # the reference image is different for each time-delay. We align the reduced images
        # to each other as well.
        # Note : the fast = False fixes issue #11, where single crystal images were not successfully aligned.
        if align:
            yield from ialign(combined, mask=mask, fast=False)
        else:
            yield from combined


# For multiprocessing, the function to be mapped must be
# global, hence defined outside of the class method
def _raw_combine(timedelay, raw, exclude_scans, normalize, align, invalid_mask, dtype):

    images = raw.itertime(timedelay, exclude_scans=exclude_scans)

    if align:
        # Note : the fast = False fixes issue #11, where single crystal images were not successfully aligned.
        images = ialign(images, mask=invalid_mask, fast=False)

    # Set up normalization
    if normalize:
        valid = np.logical_not(invalid_mask)
        images, images2 = itercopy(images, copies=2)

        # Compute the total intensity of first image
        # This will be the reference point
        first2, images2 = peek(images2)
        initial_weight = np.sum(first2[valid])
        weights = (initial_weight / np.sum(image[valid]) for image in images2)
    else:
        weights = None

    return average(images, weights=weights).astype(dtype)


def check_raw_bounds(method):
    """
    Decorator that automatically checks out-of-bounds errors while
    querying raw diffraction data.

    In case of out-of-bounds queries, a ``ValueError`` is raised.

    ``method`` is expected to be a reimplemented :meth:`AbstractRawDataset.raw_data` method.
    See :meth:`AbstractRawDataset.raw_data` for the expected call signature.
    """
    # Note that we use the expected call signature from AbstractRawDataset.raw_data
    @wraps(method)
    def checked_method(self, timedelay, scan, *args, **kwargs):
        timedelay = float(timedelay)
        scan = int(scan)

        valid_scan = scan in self.scans
        valid_timedelay = timedelay in self.time_points

        if (not valid_scan) or (not valid_timedelay):
            raise ValueError(
                "Requested time-delay {t} and scan {s} are invalid or out-of-bounds".format(
                    t=timedelay, s=scan
                )
            )

        return method(self, timedelay, scan, *args, **kwargs)

    return checked_method
