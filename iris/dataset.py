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
from .utils import angular_average, scattering_length

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
    """

    _processed_group_name = '/processed'
    _exp_params_group_name = '/'
    _pumpoff_pictures_group_name = '/pumpoff'

    experimental_parameter_names = ('nscans', 'time_points', 'acquisition_date', 'fluence', 
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
                  fluence {} mj/cm**2 >'.format(self.sample_type,self.acquisition_date, self.fluence)
            
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
        timedelay = str(float(timedelay))
        dataset = self.processed_measurements_group[timedelay]['intensity']
        if out:
            return dataset.read_direct(array = out, source_sel = n.s_[:,:], dest_sel = n.s_[:,:])
        return n.array(dataset)
    
    def averaged_error(self, timedelay, out = None):
        """ 
        Returns error in measurement.

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
            Time-delay error. If out is provided, None is returned.
        """
        timedelay = str(float(timedelay))
        dataset = self.processed_measurements_group[timedelay]['error']
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
    
    @cached_property
    def compression_params(self):
        """ Compression options in the form of a dictionary """
        dataset = self.processed_measurements_group[str(float(self.time_points[0]))]['intensity']
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
    analysis_parameter_names = ('first_stage', 'wavelet', 'level', 'baseline_removed')
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

    def powder_data(self, timedelay, bgr = False, out = None):
        """
        Returns the angular average data from scan-averaged diffraction patterns.

        Parameters
        ----------
        timdelay : float
            Time-delay [ps].
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
        timedelay = str(float(timedelay))
        dataset = self.powder_group[timedelay]['intensity']
        if not bgr and out:
            return dataset.read_direct(array = out)
        elif bgr:
            return n.array(dataset) - self.baseline(timedelay)
        else:
            return n.array(dataset)
    
    def powder_error(self, timedelay, out = None):
        """
        Returns the angular average error from scan-averaged diffraction patterns.

        Parameters
        ----------
        timdelay : float
            Time-delay [ps].
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
        timedelay = str(float(timedelay))
        dataset = self.powder_group[timedelay]['error']
        if out:
            return dataset.read_direct(array = out)
        return n.array(dataset)

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
        
    def powder_data_block(self, bgr = False):
        """ 
        Return powder diffraction data for all time delays as a single block. 

        Parameters
        ----------
        bgr : bool
            If True, background is removed.

        Returns
        -------
        s : ndarray, shape (N,)
        I, e : ndarrays, shapes (N, M)
        """
        data_block = n.empty(shape = (len(self.time_points), self.scattering_length.size), dtype = n.float)
        error_block = n.empty_like(data_block)

        for row, timedelay in enumerate(self.time_points):
            data_block[row, :] = self.powder_data(timedelay, bgr = bgr)
            error_block[row, :] = self.powder_error(timedelay)
        
        return self.scattering_length, data_block, error_block
    
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
        ckwargs = self.compression_params
        for timedelay in self.time_points:
            background = dualtree.baseline(array = self.powder_data(timedelay), max_iter = max_iter, 
                                           level = level, first_stage = first_stage,
                                           wavelet = wavelet, background_regions = tuple(),
                                           mask = None)

            if not self.baseline_removed:
                self.powder_group[str(float(timedelay))].create_dataset(name = 'baseline', 
                                                                        data = background, 
                                                                        **ckwargs)
            else:
                self.powder_group[str(float(timedelay))]['baseline'][:] = background
        
        # Record parameters
        if level == 'max':
            level = dualtree.dualtree_max_level(data = self.scattering_length, 
                                                first_stage = first_stage, 
                                                wavelet = wavelet)
        self.level = level
        self.first_stage = first_stage
        self.wavelet = wavelet
        self.baseline_removed = True
    
    def _compute_angular_averages(self):
        """ Compute the angular averages. This method is 
        only called by RawDataset.process """
        ckwargs = self.compression_params
        for timedelay in self.time_points:
            px_radius, intensity, error = angular_average(self.averaged_data(timedelay), 
                                                            center = self.center, beamblock_rect = self.beamblock_rect, 
                                                            error = self.averaged_error(timedelay))
            gp = self.powder_group.create_group(name = str(timedelay))

            # Error in the powder pattern = image_data / sqrt(nscans) * sqrt(# of pixels at this radius)
            # angular_average() doesn't know about nscans, so we must include it here
            gp.create_dataset(name = 'intensity', data = intensity, dtype = n.float, **ckwargs)
            gp.create_dataset(name = 'error', data = error/n.sqrt(len(self.nscans)), dtype = n.float, **ckwargs)
        
        # TODO: variable pixel_width and camera distance in the future
        s_length = scattering_length(px_radius, energy = self.energy)
        self.powder_group.create_dataset(name = 'scattering_length', data = s_length, dtype = n.float, **ckwargs)
        self.baseline_removed = False