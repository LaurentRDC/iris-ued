"""
@author: Laurent P. Rene de Cotret
"""

import glob
import h5py
import numpy as n
import os
from os.path import join
import re

from .io import RESOLUTION, read, cast_to_16_bits
from .utils import angular_average

# Info
__author__ = 'Laurent P. René de Cotret'
__version__ = '2.0 unreleased'

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
                            'sample_type')
    
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
        return '< DiffractionDataset object. Acquisition date : {}, fluence {} mj/cm2 >'.format(self.acquisition_date, self.fluence)
        
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
        scan -= 1   #Scans start at 1, ndarrays start at 0
        dataset = self.pumpoff_pictures_group['pumpoff_pictures']
        if out:
            return dataset.read_direct(array = out, source_sel = n.s_[:,:,scan], dest_sel = n.s_[:,:])
        return n.array(dataset[:,:,scan])
       
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
    """ """
    _powder_group_name = '/powder'

    def powder_data(self, timedelay):
        """
        Returns the radial average data from scan-averaged diffraction patterns.

        Parameters
        ----------
        timdelay : float
            Timedelay [ps]
        
        Returns
        -------
        s, I, e : ndarrays
            1D arrays for the scattering length, diffracted intensity and error
        """
        timedelay = str(float(timedelay))
        gp = self.powder_group[timedelay]
        return gp['scattering length'], gp['intensity'], gp['error']

    @property
    def powder_group(self):
        return self.require_group(self._powder_group_name)