# -*- coding: utf-8 -*-
"""
Raw dataset classes
-------------------

The following classes are defined herein:

.. autosummary::
    :toctree: classes/

    RawDataset
    McGillRawDataset

"""
import glob
import re
from abc import ABCMeta, abstractmethod
from os import listdir
from os.path import isdir, isfile, join

import numpy as np
from cached_property import cached_property
from skimage.io import imread

from npstreams import imean, last

from .dataset import VALID_DATASET_METADATA

def uint_subtract_safe(arr1, arr2):
    """ Subtract two unsigned arrays without rolling over """
    result = np.subtract(arr1, arr2)
    result[np.greater(arr2, arr1)] = 0
    return result

class RawDatasetBase(metaclass = ABCMeta):
    """ 
    Base class for raw dataset objects in iris.
    """

    @property
    def metadata(self):
        metadata = dict()
        for attr, val in VALID_DATASET_METADATA.items():
            metadata[attr] = getattr(self, attr, val.default)
        return metadata

    @property
    def pumpon_background(self): 
        return np.zeros(self.resolution, dtype = np.uint16)

    @property
    def pumpoff_background(self): 
        return np.zeros(self.resolution, dtype = np.uint16)

    @abstractmethod
    def raw_data_filename(timedelay, scan = 1, **kwargs): pass

    def raw_data(self, timedelay, scan = 1, bgr = True, **kwargs): 
        """
        Returns an array of the image at a timedelay and scan.
        
        Parameters
        ----------
        timedelay : float
            Time-delay in picoseconds.
        scan : int, optional
            Scan number. 
        bgr : bool, optional
            If True, pump-on background is removed from the pattern
            before being returned.
        
        Returns
        -------
        arr : ndarray, shape (N,M)
        
        Raises
        ------
        ImageNotFoundError
            Filename is not associated with an image/does not exist.
        """ 
        im = imread(self.raw_data_filename(timedelay, scan, **kwargs))
        if bgr:
            return uint_subtract_safe(im, self.pumpon_background)
        return im

    def timedelay_filenames(self, timedelay, exclude_scans = list(), **kwargs): 
        """ 
        Returns filenames of raw data at a specific time-delay, for all valid scans.

        Parameters
        ----------
        timedelay : float

        exclude_scans : iterable of ints, optional
            Scans to exclude. Scans start counting at one, not zero.
        
        Returns
        -------
        filenames : iterable of str
        """
        valid_scans = set(self.nscans) - set(exclude_scans)
        return [self.raw_data_filename(timedelay, scan) for scan in valid_scans]

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

class McGillRawDataset(RawDatasetBase):
    """ Wrapper around raw dataset as produced by McGill's UEDbeta. """

    def __init__(self, directory):
        if isdir(directory):
            self.raw_directory = directory
        else:
            raise ValueError('The path {} is not a directory'.format(directory))
        
        self._from_tagfile = parse_tagfile(join(directory, 'tagfile.txt'))
        self.fluence = self._from_tagfile.get('fluence', 0)
        self.resolution = (2048, 2048)
        self.current = self._from_tagfile.get('current', 0)
        self.exposure = self._from_tagfile.get('exposure', 0)
        self.energy = self._from_tagfile.get('energy', 90)
        
        try:
            self.acquisition_date = re.search('(\d+[.])+', self.raw_directory).group()[:-1]      #Last [:-1] removes a '.' at the end
        except(AttributeError):     #directory name does not match time pattern
            self.acquisition_date = '0.0.0.0.0'
    
    @cached_property
    def nscans(self): 
        """ List of integer scans. """
        scans = [re.search('[n][s][c][a][n][.](\d+)', f).group() for f in self._image_list if 'nscan' in f]
        return list(set([int(string.strip('nscan.')) for string in scans])) # Remove duplicates by using a set
    
    @cached_property
    def time_points(self):
        # Get time points. Strip away '+' as they are superfluous.
        time_data = [re.search('[+-]\d+[.]\d+', f).group() for f in self._image_list if 'timedelay' in f]
        time_list =  list(set(time_data))     #Conversion to set then back to list to remove repeated values
        time_list.sort(key = float)
        return tuple(map(float, time_list))

    @property
    def _image_list(self):
        """ All images in the raw folder. """
        return (f for f in listdir(self.raw_directory) 
                  if isfile(join(self.raw_directory, f)) and f.endswith(('.tif', '.tiff')))
    
    @cached_property
    def pumpon_background(self):
        backgrounds = map(imread, glob.iglob(join(self.raw_directory, 'background.*.pumpon.tif')))
        return last(imean(backgrounds))
    
    @cached_property
    def pumpoff_background(self):
        backgrounds = map(imread, glob.iglob(join(self.raw_directory, 'background.*.pumpoff.tif')))
        return last(imean(backgrounds))

    def raw_data_filename(self, timedelay, scan = 1):
        #Template filename looks like:
        #    'data.timedelay.+1.00.nscan.04.pumpon.tif'
        sign = '' if float(timedelay) < 0 else '+'
        str_time = sign + '{0:.2f}'.format(float(timedelay))
        filename = 'data.timedelay.' + str_time + '.nscan.' + str(int(scan)).zfill(2) + '.pumpon.tif'
        return join(self.raw_directory, filename)
