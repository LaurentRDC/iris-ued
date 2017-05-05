# -*- coding: utf-8 -*-
"""
@author: Laurent P. René de Cotret
"""
import glob
import numpy as n
from os.path import join, isfile, isdir
from os import listdir 
import re
from functools import partial
from skimage.io import imread
import sys
from datetime import datetime as dt
from warnings import warn, catch_warnings
from skued import pmap

from . import cached_property

class ExperimentalParameter(object):
    """ Descriptor to experimental parameters for raw diffraction datasets. """
    def __init__(self, name, output, default):
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
        """
        Reads an experimental parameter from the DiffractionDataset's
        experimental parameter file.
        
        Parameters
        ----------
        key : str
            Name of the parameter
        """
        with open(instance._exp_params_filename, 'r') as exp_params:
            for line in exp_params:
                if line.startswith(self.name): 
                    value = line.split('=')[-1]
                    break
            return self.default
        
        value = value.replace(' ','')
        value = value.replace('s','')                   # For exposure values with units
        value = value.strip('\n')
        try:
            return self.output(value)
        except: # Might be 'BLANK', can't cast
            return self.output(self.default)
    
    def __set__(self, instance, value):
        raise AttributeError('Attribute {} is read-only.'.format(self.name))
    
    def __delete__(self, instance):
        pass

class RawDataset(object):
    """
    Wrapper around raw dataset as produced by UEDbeta.
    
    Attributes
    ----------
    directory : str or path
    
    nscans : list of ints
        Container of the available scans.
    acquisition_date : str
    
    time_points_str : list of str
        Time-points of the dataset as strings. As recorded in the TIFF filenames.
    
    time_points : list of floats
    
    pumpon_background : ndarray
    
    pumpoff_background : ndarray
    
    image_list : list of str
    
    Methods
    -------
    raw_data
        Retrieve a raw image from a specific scan and time-delay.
    """

    resolution = (2048, 2048)
    fluence = ExperimentalParameter('Fluence', float, 0)
    current = ExperimentalParameter('Current', float, 0)
    exposure = ExperimentalParameter('Exposure', float, 0)
    energy = ExperimentalParameter('Energy', float, 90)

    def __init__(self, directory):
        if isdir(directory):
            self.raw_directory = directory
        else:
            raise ValueError('The path {} is not a directory'.format(directory))
    
    @cached_property
    def _exp_params_filename(self):
        return join(self.raw_directory, 'tagfile.txt')
    
    @cached_property
    def nscans(self):
        """ List of integer scans. """
        scans = [re.search('[n][s][c][a][n][.](\d+)', f).group() for f in self.image_list if 'nscan' in f]
        return list(set([int(string.strip('nscan.')) for string in scans])) # Remove duplicates by using a set
    
    @cached_property
    def acquisition_date(self):
        """ Returns the acquisition date from the folder name as a string of the form: '2016.01.06.15.35' """
        try:
            return re.search('(\d+[.])+', self.raw_directory).group()[:-1]      #Last [:-1] removes a '.' at the end
        except(AttributeError):     #directory name does not match time pattern
            return '0.0.0.0.0'
    
    @cached_property
    def time_points(self):
        return tuple(map(float, self.time_points_str))
    
    @cached_property
    def time_points_str(self):
        """ Returns a list of sorted string times. """
        # Get time points. Strip away '+' as they are superfluous.
        time_data = [re.search('[+-]\d+[.]\d+', f).group() for f in self.image_list if 'timedelay' in f]
        time_list =  list(set(time_data))     #Conversion to set then back to list to remove repeated values
        time_list.sort(key = float)
        return time_list

    @property
    def image_list(self):
        """ All images in the raw folder. """
        # Image list can't be a cached property since it's a generator.
        return (f for f in listdir(self.raw_directory) 
                  if isfile(join(self.raw_directory, f)) and f.endswith(('.tif', '.tiff')))
    
    @property
    def pumpon_background(self):
        backgrounds = tuple(map(imread, glob.iglob(join(self.raw_directory, 'background.*.pumpon.tif'))))
        return sum(backgrounds)/len(backgrounds)
    
    @property
    def pumpoff_background(self):
        backgrounds = tuple(map(imread, glob.iglob(join(self.raw_directory, 'background.*.pumpoff.tif'))))
        return sum(backgrounds)/len(backgrounds)
        
    def raw_data(self, timedelay, scan):
        """
        Returns an array of the raw TIFF.
        
        Parameters
        ----------
        timedelay : numerical
            Time-delay in picoseconds.
        scan : int, > 0
            Scan number. 
        
        Returns
        -------
        arr : ndarray, shape (N,M), dtype uint16
        
        Raises
        ------
        ImageNotFoundError
            Filename is not associated with a TIFF/does not exist.
        """ 
        #Template filename looks like:
        #    'data.timedelay.+1.00.nscan.04.pumpon.tif'
        sign = '' if float(timedelay) < 0 else '+'
        str_time = sign + '{0:.2f}'.format(float(timedelay))
        filename = 'data.timedelay.' + str_time + '.nscan.' + str(int(scan)).zfill(2) + '.pumpon.tif'
        return imread(join(self.raw_directory, filename))
    
    def timedelay_filenames(self, timedelay):
        """ Generator of filenames of raw data for a specific time-delay """
        #Template filename looks like:
        #    'data.timedelay.+1.00.nscan.04.pumpon.tif'
        sign = '' if float(timedelay) < 0 else '+'
        str_time = sign + '{0:.2f}'.format(float(timedelay))
        return glob.glob(join(self.raw_directory, 'data.timedelay.' + str_time + '.nscan.*.pumpon.tif'))