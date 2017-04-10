"""
@author: Laurent P. Rene de Cotret
"""

import glob
import h5py
import numpy as n
import os
from os.path import join
import re
from warnings import warn

from .dualtree import baseline, dualtree_max_level
from .optimizations import cached_property
from .io import RESOLUTION, read, cast_to_16_bits
from .utils import angular_average, scattering_length

class ExperimentalParameter(object):
    """ Descriptor to experimental parameters. """
    def __init__(self, name, output, default = None):
        """ 
        Parameters
        ----------
        name : str
        output : callable
            Callable to format output.
            e.g. numpy.array, tuple, float, ...
        default : object or None, optional
            Default value returned. If None, exception is raised.
        """
        self.name = name
        self.output = output
        self.default = default
    
    def __get__(self, instance, cls):
        try:
            return self.output(instance.experimental_parameters_group.attrs[self.name])
        except:
            # Some parameters might have a default value, others not.
            if self.default is not None:
                return self.default
    
    def __set__(self, instance, value):
        instance.experimental_parameters_group.attrs[self.name] = value
    
    def __delete__(self, instance):
        del instance.experimental_parameters_group.attrs[self.name]

class DiffractionDataset(h5py.File):
    """
    Abstraction of an HDF5 file to represent diffraction datasets.
    """

    _processed_group_name = '/processed'
    _exp_params_group_name = '/'
    _pumpoff_pictures_group_name = '/pumpoff'
    
    # Experimental parameters as descriptors
    nscans = ExperimentalParameter('nscans', tuple)
    time_points = ExperimentalParameter('time_points', tuple)
    acquisition_date = ExperimentalParameter('acquisition_date', str)
    fluence = ExperimentalParameter('fluence', float)
    current = ExperimentalParameter('current', float)
    exposure = ExperimentalParameter('exposure', float)
    energy = ExperimentalParameter('energy', float)
    resolution = ExperimentalParameter('resolution', tuple, default = (2048, 2048))
    center = ExperimentalParameter('center', tuple)
    beamblock_rect = ExperimentalParameter('beamblock_rect', tuple)
    sample_type = ExperimentalParameter('sample_type', str)
    time_zero_shift = ExperimentalParameter('time_zero_shift', float, default = 0.0)

    def __init__(self, name, mode = 'r', **kwargs):
        """
        Keyword arguments are passed to the h5py.File constructor.

        Parameters
        ----------
        name : str
        mode : str, {'w', 'r' (default), 'r+', 'a', 'w-'}
        """
        super().__init__(name = name, mode = mode, **kwargs)
    
    def __repr__(self):
        return '< DiffractionDataset object. \
                  Sample type: {}, \n \
                  Acquisition date : {}, \n \
                  fluence {} mj/cm**2 >'.format(self.sample_type,self.acquisition_date, self.fluence)
    
    @property
    def corrected_time_points(self):
        """ Time points corrected for time-zero shift. """
        return tuple(n.array(self.time_points) - self.time_zero_shift)
    
    def shift_time_zero(self, shift):
        """
        Shift time-zero uniformly across time-points.

        Parameters
        ----------
        shift : float
            Shift [ps].
        """
        self.time_zero_shift = shift
            
    def averaged_data(self, timedelay, out = None):
        """
        Returns data at a specific time-delay.

        Parameters
        ----------
        timdelay : float or None
            Timedelay [ps]. If None, the entire block is returned.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.
        
        Returns
        -------
        arr : ndarray or None
            Time-delay data. If out is provided, None is returned.
        
        Raises
        ------
        ValueError
            If timedelay does not exist.
        """
        dataset = self.processed_measurements_group['intensity']

        if timedelay is None:
            if out is None:
                out = n.empty_like(dataset)
            dataset.read_direct(out)

        else:
            # time_index cannot be cast to int() if n.argwhere returns an empty array
            # catch the corresponding TypeError
            try:
                time_index = int(n.argwhere(n.array(self.corrected_time_points) == float(timedelay)))
            except TypeError:
                raise ValueError('Time-delay {} does not exist.'.format(timedelay))

            if out is None:
                out = n.empty(self.resolution)
            dataset.read_direct(out, source_sel = n.s_[:,:, time_index], dest_sel = n.s_[:,:])

        return out

    def averaged_error(self, timedelay, out = None):
        """ 
        Returns error in measurement.

        Parameters
        ----------
        timdelay : float or None
            Timedelay [ps]. If None, the entire block is returned.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.
        
        Returns
        -------
        arr : ndarray or None
            Time-delay error. If out is provided, None is returned.

        Raises
        ------
        ValueError
            If timedelay does not exist.
        """
        dataset = self.processed_measurements_group['error']

        if timedelay is None:
            if out is None:
                out = n.empty_like(dataset)
            dataset.read_direct(out)

        else:
            # time_index cannot be cast to int() if n.argwhere returns an empty array
            # catch the corresponding TypeError
            try:
                time_index = int(n.argwhere(n.array(self.corrected_time_points) == float(timedelay)))
            except TypeError:
                raise ValueError('Time-delay {} does not exist.'.format(timedelay))

            if out is None:
                out = n.empty(self.resolution)
            dataset.read_direct(out, source_sel = n.s_[:,:, time_index], dest_sel = n.s_[:,:])
        
        return out
    
    def time_series(self, rect, out = None):
        """
        Integrated intensity over time inside bounds.

        Parameters
        ----------
        rect : 4-tuple of ints
            Bounds of the region in px.
        out : ndarray or None, optional
            1-D ndarray in which to store the results. The shape
            should be compatible with (len(time_points),)
        
        Returns
        -------
        out : ndarray, ndim 1
        """
        x1, x2, y1, y2 = rect
        data = self.processed_measurements_group['intensity'][y1:y2, x1:x2, :]  # Numpy axes are transposed
        return n.sum(data, axis = (0,1), out = out)

    def pumpoff_data(self, scan, out = None):
        """
        Returns a pumpoff picture from a specific scan.

        Parameters
        ----------
        scan : int or None
            If None, the entire block (i.e. for all scans) is returned.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.

        Returns
        -------
        arr : ndarray or None
            Pump-off picture. If out is provided, None is returned.
        """
        #Scans start at 1, ndarray indices start at 0
        dataset = self.pumpoff_pictures_group['pumpoff_pictures']
        if scan:
            if out is None:
                out = n.empty(self.resolution)
            dataset.read_direct(out, source_sel = n.s_[:,:,scan - 1], dest_sel = n.s_[:,:])
        else:
            if out is None:
                out = n.empty_like(dataset)
            dataset.read_direct(out)
        
        return out
       
    @property
    def background_pumpon(self):
        return n.array(self.processed_measurements_group['background_pumpon'])
    
    @property
    def background_pumpoff(self):
        return n.array(self.processed_measurements_group['background_pumpoff'])
    
    @property
    def experimental_parameters_group(self):
        return self.require_group(name = self._exp_params_group_name)
    
    @property
    def processed_measurements_group(self):
        return self.require_group(name = self._processed_group_name)
    
    @property
    def pumpoff_pictures_group(self):
        return self.require_group(name = self._pumpoff_pictures_group_name)
    
    @cached_property
    def compression_params(self):
        """ Compression options in the form of a dictionary """
        dataset = self.processed_measurements_group['intensity']
        ckwargs = dict()
        ckwargs['compression'] = dataset.compression
        ckwargs['fletcher32'] = dataset.fletcher32
        ckwargs['shuffle'] = dataset.shuffle
        ckwargs['chunks'] = True if dataset.chunks else False
        if dataset.compression_opts: #could be None
            ckwargs.update(dataset.compression_opts)
        return ckwargs

class PowderDiffractionDataset(DiffractionDataset):
    """ 
    Abstraction of HDF5 files for powder diffraction datasets.
    """

    first_stage = ExperimentalParameter(name = 'powder_wavelet_baseline_first_stage', output = str)
    wavelet = ExperimentalParameter(name = 'powder_baseline_wavelet', output = str)
    level = ExperimentalParameter(name = 'powder_baseline_transform_level', output = int)
    baseline_removed = ExperimentalParameter(name = 'powder_baseline_removed', output = bool, default = False)

    @property
    def powder_group(self):
        return self.require_group('/powder')
    
    @property
    def scattering_length(self):
        return n.array(self.powder_group['scattering_length'])

    def powder_data(self, timedelay, bgr = False, out = None):
        """
        Returns the angular average data from scan-averaged diffraction patterns.

        Parameters
        ----------
        timdelay : float or None
            Time-delay [ps]. If None, the entire block is returned.
        bgr : bool
            If True, background is removed.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.
        
        Returns
        -------
        I : ndarray, shape (N,)
            Diffracted intensity [counts]
        """
        dataset = self.powder_group['intensity']

        if timedelay is None:
            if out is None:
                out = n.empty_like(dataset)
            dataset.read_direct(out)

        else:
            time_index = n.argwhere(n.array(self.corrected_time_points) == float(timedelay))
            if out is None:
                out = n.empty_like(self.scattering_length)
            dataset.read_direct(out, source_sel = n.s_[time_index,:], dest_sel = n.s_[:])

        if bgr:
            out -= self.baseline(timedelay)
        return out     
    
    def powder_error(self, timedelay, out = None):
        """
        Returns the angular average error from scan-averaged diffraction patterns.

        Parameters
        ----------
        timdelay : float or None
            Time-delay [ps]. If None, the entire block is returned.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.
        
        Returns
        -------
        out : ndarray, shape (N,)
            Error in diffracted intensity [counts].
        """
        dataset = self.powder_group['error']

        if timedelay is None:
            if out is None:
                out = n.empty_like(dataset)
            dataset.read_direct(out)
        
        else:
            time_index = n.argwhere(n.array(self.corrected_time_points) == float(timedelay))
            if out is None:
                out = n.empty_like(self.scattering_length)
            dataset.read_direct(out, source_sel = n.s_[time_index,:], dest_sel = n.s_[:])  
        return out

    def baseline(self, timedelay, out = None):
        """ 
        Returns the baseline data. 

        Parameters
        ----------
        timdelay : float or None
            Time-delay [ps]. If None, the entire block is returned.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.
        
        Returns
        -------
        out : ndarray
            If a baseline hasn't been computed yet, the returned
            array is an array of zeros.
        """
        if not self.baseline_removed:
            return n.zeros_like(self.scattering_length)

        dataset = self.powder_group['baseline']

        if timedelay is None:
            if out is None:
                out = n.empty_like(dataset)
            dataset.read_direct(out)
        
        else:
            time_index = n.argwhere(n.array(self.corrected_time_points) == float(timedelay))
            if out is None:
                out = n.empty_like(self.scattering_length)
            dataset.read_direct(out, source_sel = n.s_[time_index,:], dest_sel = n.s_[:]) 
        
        return out
    
    def powder_time_series(self, smin, smax, bgr = False, out = None):
        """
        Integrated intensity in a scattering angle range, over time.
        Diffracted intensity is integrated in the closed interval [smin, smax]

        Parameters
        ----------
        smin : float
            Lower scattering angle bound [rad/A]
        smax : float
            Higher scattering angle bound [rad/A]. 
        bgr : bool, optional
            If True, background is removed. Default is False.
        out : ndarray or None, optional
            1-D ndarray in which to store the results. The shape
            should be compatible with (len(time_points),)
        
        Returns
        -------
        out : ndarray, shape (N,)
            Integrated diffracted intensity over time.
        """
        # TODO: handle out parameter more efficiently?
        # Python slices are semi-open by design, therefore i_max + 1 is used
        # so that the integration interval is closed.
        i_min, i_max = n.argmin(n.abs(smin - self.scattering_length)), n.argmin(n.abs(smax - self.scattering_length))
        trace = n.array(self.powder_group['intensity'][:, i_min:i_max + 1])
        if bgr :
            trace -= n.array(self.powder_group['baseline'][:, i_min:i_max + 1])
        
        if out is not None:
            return n.sum(axis = 1, out = out)
        return n.squeeze(n.sum(trace, axis = 1))
    
    def compute_baseline(self, first_stage, wavelet, max_iter = 100, level = 'max'):
        """
        Compute and save the baseline computed from the dualtree package.

        Parameters
        ----------
        first_stage : str, optional
            Wavelet to use for the first stage. See dualtree.ALL_FIRST_STAGE for a list of suitable arguments
        wavelet : str, optional
            Wavelet to use in stages > 1. Must be appropriate for the dual-tree complex wavelet transform.
            See dualtree.ALL_COMPLEX_WAV for possible
        max_iter : int, optional

        level : int or 'max', optional
        """
        background = baseline(array = self.powder_data(timedelay = None, bgr = False), max_iter = max_iter, 
                              level = level, first_stage = first_stage, wavelet = wavelet, background_regions = tuple(),
                              mask = None, axis = 1)
        
        if not self.baseline_removed:
            self.powder_group.create_dataset(name = 'baseline', data = background, **self.compression_params)
        else:
            self.powder_group['baseline'].write_direct(background)
        
        # Record parameters
        if level == 'max':
            level = dualtree_max_level(data = self.scattering_length, first_stage = first_stage, wavelet = wavelet)
        self.level = level
        self.first_stage = first_stage
        self.wavelet = wavelet
        self.baseline_removed = True
    
    def _compute_angular_averages(self):
        """ Compute the angular averages. This method is 
        only called by RawDataset.process once"""

        # Because it is difficult to know the angular averaged data's shape in advance, 
        # we calculate it first and store it next
        results = list()
        for timedelay in self.time_points:
            results.append( angular_average(self.averaged_data(timedelay), center = self.center, 
                                            beamblock_rect = self.beamblock_rect, 
                                            error = self.averaged_error(timedelay)) )
        
        # Concatenate arrays for intensity and error
        rintensity = n.stack([I for _, I, _ in results], axis = 0)
        rerror =  n.stack([e for _, _, e in results], axis = 0)
        
        dataset = self.powder_group.require_dataset(name = 'intensity', shape = rintensity.shape, dtype = rintensity.dtype)
        dataset[:,:] = rintensity

        dataset = self.powder_group.require_dataset(name = 'error', shape = rerror.shape, dtype = rerror.dtype)
        dataset[:,:] = rerror
        
        # TODO: variable pixel_width and camera distance in the future
        px_radius = results[0][0]
        s_length = scattering_length(px_radius, energy = self.energy)
        self.powder_group.create_dataset(name = 'scattering_length', data = s_length, dtype = n.float)
        self.baseline_removed = False