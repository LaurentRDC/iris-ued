"""
@author: Laurent P. Rene de Cotret
"""

import glob
import h5py
import numpy as n
import os
from os.path import join
import re
import dualtree

from .optimizations import cached_property
from .io import RESOLUTION, read, cast_to_16_bits
from .utils import angular_average

__author__ = 'Laurent P. René de Cotret'
__version__ = '2.0'

class ExperimentalParameter(object):
    """ Descriptor to experimental parameters. """
    def __init__(self, name, output):
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
    
    def __get__(self, instance, cls):
        return self.output(instance.experimental_parameters_group.attrs[self.name])
    
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

    Attributes
    ----------
    experimental_parameters_group

    processed_measurements_group
    """

    _processed_group_name = '/processed'
    _exp_params_group_name = '/'
    _pumpoff_pictures_group_name = '/pumpoff'

    _exp_parameter_names = ('nscans', 'time_points', 'acquisition_date', 'fluence', 
                            'current', 'exposure', 'energy', 'resolution', 'center',
                            'sample_type', 'beamblock_rect')
    
    # Experimental parameters as descriptors
    nscans = ExperimentalParameter('nscans', tuple)
    time_points = ExperimentalParameter('time_points', tuple)
    acquisition_date = ExperimentalParameter('acquisition_date', str)
    fluence = ExperimentalParameter('fluence', float)
    current = ExperimentalParameter('current', float)
    exposure = ExperimentalParameter('exposure', float)
    energy = ExperimentalParameter('energy', float)
    resolution = ExperimentalParameter('resolution', tuple)
    center = ExperimentalParameter('center', tuple)
    beamblock_rect = ExperimentalParameter('beamblock_rect', tuple)
    sample_type = ExperimentalParameter('sample_type', str)

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
                  fluence {} mj/cm2 >'.format(self.sample_type,self.acquisition_date, self.fluence)
        
    def averaged_data(self, timedelay, out = None):
        """
        Returns data at a specific time-delay.

        Parameters
        ----------
        timdelay : float
            Timedelay [ps]
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.
        
        Returns
        -------
        arr : ndarray or None
            Time-delay data. If out is provided, None is returned.
        """
        # TODO: return error somehow
        timedelay = str(float(timedelay))
        dataset = self.processed_measurements_group[timedelay]['intensity']
        if out:
            return dataset.read_direct(array = out, source_sel = n.s_[:,:], dest_sel = n.s_[:,:])
        return n.array(dataset)

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

class PowderDiffractionDataset(DiffractionDataset):
    """ 
    Attributes
    ----------
    """
    _powder_group_name = '/powder'

    # Analysis parameters concerning the wavelet used
    # The dual-tree complex wavelet transform also uses 
    # first stage wavelet
    first_stage = AnalysisParameter(name = 'first_stage', output = str, group_name = _powder_group_name)
    wavelet = AnalysisParameter(name = 'wavelet', output = str, group_name = _powder_group_name)
    level = AnalysisParameter(name = 'level', output = int, group_name = _powder_group_name)
    baseline_removed = AnalysisParameter(name = 'baseline_removed', output = bool, group_name = _powder_group_name)

    @cached_property
    def scattering_length(self):
        return n.array(self.powder_group['scattering_length'])

    def powder_data(self, timedelay, bgr = False):
        """
        Returns the radial average data from scan-averaged diffraction patterns.

        Parameters
        ----------
        timdelay : float
            Time-delay [ps].
        bgr : bool
            If True, background is removed.
        
        Returns
        -------
        s, I, e : ndarrays, shapes (N,)
            1D arrays for the scattering length [1/Angs], 
            diffracted intensity [counts] and error [counts].
        
        See also
        --------
        powder_data_block
            All time-delay powder diffraction data into 2D arrays.
        """
        timedelay = str(float(timedelay))
        gp = self.powder_group[timedelay]
        if bgr:
            return (self.scattering_length,
                    n.array(gp['intensity']) - self.baseline(timedelay),
                    n.array(gp['error']))
        return self.scattering_length, n.array(gp['intensity']), n.array(gp['error'])
    
    def powder_dynamics(self, s1, s2, bgr = False):
        """ 
        Returns the time-dynamics of a region in the powder data.

        Parameters
        ----------
        s1, s2 : floats
            Scattering length bounds of the region of interest.
        bgr : bool
            If True, background is removed.
        
        Returns
        -------
        t, dyn : ndarrays, shape (N,)
            Time points and integral of diffracted intensity in range (s1, s2) over time.
        """
        s1, s2 = min([s1,s2]), max([s1,s2])
        s_length = n.array(self.powder_group['scattering_length'])

        # Indices of the scattering length bounds
        # Include the upper bound
        # This will also take care of the case where
        # s1 == s2, because then i_max = i_min + 1
        i_min, i_max = n.argmin(n.abs(s_length - s1)), n.argmin(n.abs(s_length - s2)) + 1
        
        dynamics = n.empty(shape = (i_max - i_min, len(self.time_points)), dtype = n.float)
        for index, timedelay in enumerate(self.time_points):
            dynamics[:, index] = self.powder_data(timedelay, bgr)[1][i_min:i_max]
        
        # Integral over scattering length range
        ds = (s2 - s1)/(i_max - i_min)
        return n.array(self.time_points), dynamics.sum(axis = 0)*ds
        
    def powder_data_block(self, bgr = False):
        """ 
        Return powder diffraction data for all time delays as a single block. 

        Parameters
        ----------
        bgr : bool
            If True, background is removed.

        Returns
        -------
        s, I, e : ndarrays, shapes (N, M)
        """
        data_block = n.empty(shape = (len(self.time_points), self.scattering_length.size), dtype = n.float)
        error_block = n.empty_like(data_block)

        for row, timedelay in enumerate(self.time_points):
            _, data_block[row, :], error_block[row, :] = self.powder_data(timedelay, bgr = bgr)
        
        return self.scattering_length, data_block, error_block
    
    def baseline(self, timedelay):
        """ 
        Returns the baseline data 

        Parameters
        ----------
        timdelay : float
            Time-delay [ps].
        
        Returns
        -------
        out : ndarray
        """
        try:
            return n.array(self.powder_group[str(float(timedelay))]['baseline'])
        except KeyError:
            return n.zeros_like(self.scattering_length)
    
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
        for timedelay in self.time_points:
            _, intensity, _ = self.powder_data(timedelay)
            self.powder_group[timedelay]['baseline'] = baseline(array = intensity, max_iter = max_iter, 
                                                                level = level, first_stage = first_stage,
                                                                wavelet = wavelet, background_regions = tuple(),
                                                                mask = None)
        # Record parameters
        if level == 'max':
            level = dualtree.dualtree_max_level(data = self.scattering_length, 
                                                first_stage = first_stage, 
                                                wavelet = wavelet)
        self.level = level
        self.first_stage = first_stage
        self.wavelet = wavelet
        self.baseline_removed = True

    @property
    def powder_group(self):
        return self.require_group(name = self._powder_group_name)
    
    def _compute_angular_averages(self, **ckwargs):
        """
        Compute the angular averages. This method is only called by RawDataset.process
        """
        for timedelay in self.time_points:
            s_length, intensity, error = angular_average(n.array(self.averaged_data(timedelay)), 
                                                            center = self.center, beamblock_rect = self.beamblock_rect)
            gp = self.powder_group.create_group(name = str(timedelay))
            for name, value in zip(('intensity', 'error'), (intensity, error)):
                gp.create_dataset(name = name, data = value, dtype = n.float, **ckwargs)
        
        self.powder_group.create_dataset(name = 'scattering_length', data = s_length, dtype = n.float, **ckwargs)
        self.baseline_removed = False