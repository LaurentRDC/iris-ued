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

from .optimizations import cached_property

def parse_tagfile(path):
    """ Parse a tagfile.txt from a raw dataset into a dictionary of values """
    metadata = dict()
    with open(path) as f:
        for line in f:
            key, value = re.sub('\s+', '', line).split('=')
            try:
                value = float(value.strip('s'))    # exposure values have units
            except:
                value = None    # value might be 'BLANK'
            metadata[key.lower()] = value
    
    return metadata

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

    def __init__(self, directory):
        if isdir(directory):
            self.raw_directory = directory
        else:
            raise ValueError('The path {} is not a directory'.format(directory))
        
        self.metadata = parse_tagfile(join(directory, 'tagfile.txt'))
    
    @property
    def fluence(self):
        return self.metadata['fluence'] or 0
    
    @property
    def current(self):
        return self.metadata['current'] or 0
    
    @property
    def exposure(self):
        return self.metadata['exposure'] or 0
    
    @property
    def energy(self):
        return self.metadata['energy'] or 90
    
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
    
    @cached_property
    def pumpon_background(self):
        backgrounds = tuple(map(imread, glob.iglob(join(self.raw_directory, 'background.*.pumpon.tif'))))
        return sum(backgrounds)/len(backgrounds)
    
    @cached_property
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
    
    def timedelay_filenames(self, timedelay, exclude_scans = list()):
        """ 
        Returns filenames of raw data for a specific time-delay.

        Parameters
        ----------
        timedelay : str or float

        exclude_scans : iterable of ints, optional
        
        Returns
        -------
        filenames : iterable of str
        """
        #Template filename looks like:
        #    'data.timedelay.+1.00.nscan.04.pumpon.tif'
        valid_scans = set(self.nscans) - set(exclude_scans)
        filenames = list()
        for scan in valid_scans:
            sign = '' if float(timedelay) < 0 else '+'
            str_time = sign + '{0:.2f}'.format(float(timedelay))
            filenames.append(join(self.raw_directory, 'data.timedelay.' + str_time + '.nscan.' + str(int(scan)).zfill(2) + '.pumpon.tif'))
        return filenames