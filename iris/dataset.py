"""
@author: Laurent P. Rene de Cotret
"""

import glob
import h5py
import numpy as n
import os
from os.path import join
import re

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

class AnalysisParameter(object):
    """ Descriptor to analysis parameters. """
    def __init__(self, name, output, group_name):
        """ 
        Parameters
        ----------
        name : str
        output : callable
            Callable to format output.
            e.g. numpy.array, tuple, float, ...
        group_name : str
            Path to the hdf5 group in which the parameter is 
            stored (as an attribute).
        """
        self.name = name
        self.output = output
        self.group_name = group_name
    
    def __get__(self, instance, cls):
        try:
            return self.output(instance[self.group_name].attrs[self.name])
        except KeyError:
            return None
    
    def __set__(self, instance, value):
        instance[self.group_name].attrs[self.name] = value
    
    def __delete__(self, instance):
        try:
            del instance[self.group_name].attrs[self.name]
        except:
            pass

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
        Parameters
        ----------
        name : str or unicode
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
            
    def averaged_data(self, timedelay, tcor = False, out = None):
        """
        Returns data at a specific time-delay.

        Parameters
        ----------
        timdelay : float
            Timedelay [ps]
        tcor : bool, optional
            If True, the time-zero shift corrected time-points are used
            instead of experimental time-points.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.
        
        Returns
        -------
        arr : ndarray or None
            Time-delay data. If out is provided, None is returned.
        """
        time_index = n.argwhere(n.array(self.time_points) == float(timedelay))
        dataset = self.processed_measurements_group['intensity']
        if out:
            return dataset.read_direct(array = out, source_sel = n.s_[:,:, time_index], dest_sel = n.s_[:,:])
        return n.array(dataset[:,:,time_index])
    
    def averaged_data_block(self):
        """
        Array containing all time-delay (averaged) data.

        Returns
        -------
        out : ndarray, ndim 3
        """
        return n.array(self.processed_measurements_group['intensity'])
    
    def averaged_error(self, timedelay, tcor = False, out = None):
        """ 
        Returns error in measurement.

        Parameters
        ----------
        timdelay : float
            Timedelay [ps]
        tcor : bool, optional
            If True, the time-zero shift corrected time-points are used
            instead of experimental time-points.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.
        
        Returns
        -------
        arr : ndarray or None
            Time-delay error. If out is provided, None is returned.
        """
        time_index = n.argwhere(n.array(self.time_points) == float(timedelay))
        dataset = self.processed_measurements_group['error']
        if out:
            return dataset.read_direct(array = out, source_sel = n.s_[:,:, time_index], dest_sel = n.s_[:,:])
        return n.array(dataset[:,:,time_index])
    
    def time_series(self, rect):
        """
        Integrated intensity over time inside bounds.

        Parameters
        ----------
        rect : 4-tuple of ints
            Bounds of the region in px.
        
        Returns
        -------
        out : ndarray, ndim 1
        """
        x1, x2, y1, y2 = rect

        data = n.array(self.processed_measurements_group['intensity'][y1:y2, x1:x2, :])  # Numpy axes are transposed
        return n.sum(n.sum(data, axis = 0), axis = 0)

    def pumpoff_data(self, scan, out = None):
        """
        Returns a pumpoff picture from a specific scan.

        Parameters
        ----------
        scan : int

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
        if out:
            return dataset.read_direct(array = out, source_sel = n.s_[:,:,scan - 1], dest_sel = n.s_[:,:])
        return n.array(dataset[:,:,scan - 1])
       
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
    _powder_group_name = '/powder'

    # Analysis parameters concerning the wavelet used
    # The dual-tree complex wavelet transform also uses 
    # first stage wavelet
    first_stage = AnalysisParameter(name = 'first_stage', output = str, group_name = _powder_group_name)
    wavelet = AnalysisParameter(name = 'wavelet', output = str, group_name = _powder_group_name)
    level = AnalysisParameter(name = 'level', output = int, group_name = _powder_group_name)
    baseline_removed = AnalysisParameter(name = 'baseline_removed', output = bool, group_name = _powder_group_name)

    @property
    def powder_group(self):
        return self.require_group(name = self._powder_group_name)
    
    @property
    def scattering_length(self):
        return n.array(self.powder_group['scattering_length'])

    def powder_data(self, timedelay, tcor = False, bgr = False, out = None):
        """
        Returns the angular average data from scan-averaged diffraction patterns.

        Parameters
        ----------
        timdelay : float
            Time-delay [ps].
        tcor : bool, optional
            If True, the time-zero shift corrected time-points are used
            instead of experimental time-points.
        bgr : bool
            If True, background is removed.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.
        
        Returns
        -------
        I : ndarray, shape (N,)
            Diffracted intensity [counts]
        
        See also
        --------
        powder_data_block
            All time-delay powder diffraction data into 2D arrays.
        """
        # I'm not sure how to handle out parameter if bgr is True
        # out has no effect if bgr = True
        time_index = n.argwhere(n.array(self.time_points) == float(timedelay))
        dataset = self.powder_group['intensity']
        if not bgr and out:
            return dataset.read_direct(array = out, source_sel = n.s_[time_index,:], dest_sel = n.s_[:])
        elif bgr:
            return n.array(dataset[time_index, :]) - self.baseline(timedelay)
        else:
            return n.array(dataset[time_index, :])
    
    def powder_error(self, timedelay, tcor = False, out = None):
        """
        Returns the angular average error from scan-averaged diffraction patterns.

        Parameters
        ----------
        timdelay : float
            Time-delay [ps].
        tcor : bool, optional
            If True, the time-zero shift corrected time-points are used
            instead of experimental time-points.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.
        
        Returns
        -------
        out : ndarray, shape (N,)
            Error in diffracted intensity [counts].
        
        See also
        --------
        powder_data_block
            All time-delay powder diffraction data into 2D arrays.
        """
        # I'm not sure how to handle out parameter if bgr is True
        # out has no effect if bgr = True
        time_index = n.argwhere(n.array(self.time_points) == float(timedelay))
        dataset = self.powder_group['error']
        if not bgr and out:
            return dataset.read_direct(array = out, source_sel = n.s_[time_index,:], dest_sel = n.s_[:])
        elif bgr:
            return n.array(dataset[time_index, :]) - self.baseline(timedelay)
        else:
            return n.array(dataset[time_index, :])

    def baseline(self, timedelay, tcor = False, out = None):
        """ 
        Returns the baseline data 

        Parameters
        ----------
        timdelay : float
            Time-delay [ps].
        tcor : bool, optional
            If True, the time-zero shift corrected time-points are used
            instead of experimental time-points.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.
        
        Returns
        -------
        out : ndarray
        """
        if not self.baseline_removed:
            return n.zeros_like(self.scattering_length)

        # I'm not sure how to handle out parameter if bgr is True
        # out has no effect if bgr = True
        time_index = n.argwhere(n.array(self.time_points) == float(timedelay))
        dataset = self.powder_group['baseline']
        if out:
            return dataset.read_direct(array = out, source_sel = n.s_[time_index,:], dest_sel = n.s_[:])
        else:
            return n.array(dataset[time_index, :])
    
    def powder_time_series(self, smin, smax, bgr = False):
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
        
        Returns
        -------
        out : ndarray, shape (N,)
            Integrated diffracted intensity over time.
        """
        # Python slices are semi-open by design, therefore i_max + 1 is used
        # so that the integration interval is closed.
        i_min, i_max = n.argmin(n.abs(smin - self.scattering_length)), n.argmin(n.abs(smax - self.scattering_length))
        trace = n.array(self.powder_group['intensity'][:, i_min:i_max + 1])
        if bgr :
            trace -= n.array(self.powder_group['baseline'][:, i_min:i_max + 1])

        return n.squeeze(n.sum(trace, axis = 1))
        
    def powder_data_block(self, bgr = False):
        """ 
        Return powder diffraction data for all time delays as a single block. 

        Parameters
        ----------
        bgr : bool
            If True, background is removed.

        Returns
        -------
        I : ndarray, shapes (N, M)
            Diffracted intensity
        """
        data = n.array(self.powder_group['intensity'])
        if bgr:
            data -= n.array(self.powder_group['baseline'])
        return data
    
    def powder_error_block(self):
        """
        Return powder diffraction data for all time delays as a single block. 

        Returns
        -------
        e : ndarray, shapes (N, M)
            Error in diffracted intensity
        """
        return n.array(self.powder_group['error'])
    
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
        background = baseline(array = self.powder_data_block(bgr = False), max_iter = max_iter, level = level, 
                              first_stage = first_stage, wavelet = wavelet, background_regions = tuple(),
                              mask = None, axis = 1)
        
        if not self.baseline_removed:
            self.powder_group.create_dataset(name = 'baseline', data = background, **self.compression_params)
        else:
            self.powder_group['baseline'][:] = background
        
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