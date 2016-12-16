# -*- coding: utf-8 -*-
"""
@author: Laurent P. René de Cotret
"""

from datetime import datetime
import glob
import numpy as n
from os.path import join, isfile, isdir
from os import listdir 
import re
import sys

from . import cached_property
from .io import read, save, RESOLUTION, ImageNotFoundError, cast_to_16_bits
from .dataset import DiffractionDataset
from .utils import shift, find_center, average_tiff

# Info
__author__ = 'Laurent P. René de Cotret'
__version__ = '2.0 unreleased'
TEST_PATH = 'C:\\test_data\\2016.10.18.11.10.VO2_vb_16.2mJ'

def log(message, file):
    """
    Writes a time-stamped message into a log file. Also print to the interpreter
    for debugging purposes.
    """
    now = datetime.now().strftime('[%Y-%m-%d  %H:%M:%S]')
    time_stamped = '{0} {1}'.format(now, str(message))
    print(time_stamped)
    print(time_stamped, file = file)

class RawDataset(object):
    """
    Wrapper around raw dataset as produced by UEDbeta.
    
    Attributes
    ----------
    directory : str or path
    
    nscans : int
    
    acquisition_date : str
    
    time_points_str : list of str
        Time-points of the dataset as strings. As recorded in the TIFF filenames.
    
    time_points : list of floats
    
    processed : bool
    
    pumpon_background : ndarray
    
    pumpoff_background : ndarray
    
    image_list : list of str
    
    Methods
    -------
    raw_image
    
    process
    """
    def __init__(self, directory):
        if isdir(directory):
            self.raw_directory = directory
        else:
            raise ValueError('The path {} is not a directory'.format(directory))
    
    @cached_property
    def _exp_params_filename(self):
        return join(self.raw_directory, 'tagfile.txt')
    
    def _read_experimental_parameter(self, key):
        """
        Reads an experimental parameter from the DiffractionDataset's
        experimental parameter file.
        
        Parameters
        ----------
        key : str
            Name of the parameter
        """
        with open(self._exp_params_filename, 'r') as exp_params:
            for line in exp_params:
                if line.startswith(key): 
                    value = line.split('=')[-1]
                    break
        
        value = value.replace(' ','')
        value = value.replace('s','')                   # For exposure values with units
        if key == 'Acquisition date': 
            return value.strip('\n')
        else:
            try:
                return float(value)
            except: #Value might be an invalid number. E.g. 'BLANK'
                return 0.0

    @cached_property
    def resolution(self):
        return RESOLUTION
        
    @cached_property
    def fluence(self):
        return self._read_experimental_parameter('Fluence')
    
    @cached_property
    def current(self):
        return self._read_experimental_parameter('Current')
    
    @cached_property
    def exposure(self):
        return self._read_experimental_parameter('Exposure')
    
    @cached_property
    def energy(self):
        return self._read_experimental_parameter('Energy')
        
    @cached_property    
    def acquisition_date(self):
        return self._read_experimental_parameter('Acquisition date')
    
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
        return tuple(float(t) for t in self.time_points_str)
    
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
        backgrounds = (read(filename) for filename in glob.glob(join(self.raw_directory, 'background.*.pumpon.tif')))
        return sum(backgrounds)/len(backgrounds)
    
    @property
    def pumpoff_background(self):
        backgrounds = (read(filename) for filename in glob.glob(join(self.raw_directory, 'background.*.pumpoff.tif')))
        return sum(backgrounds)/len(backgrounds)
        
    def raw_image(self, timedelay, scan):
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
        arr : ndarray, shape (N,M)
        
        Raises
        ------
        ImageNotFoundError
            Filename is not associated with a TIFF/does not exist.
        
        Notes
        -----
        Template filename looks like:
            'data.timedelay.+1.00.nscan.04.pumpon.tif'
        """
        sign = '' if float(timedelay) < 0 else '+'
        str_time = sign + '{0:.2f}'.format(float(timedelay))
        filename = 'data.timedelay.' + str_time + '.nscan.' + str(int(scan)).zfill(2) + '.pumpon.tif'
        
        return read(join(self.raw_directory, filename)).astype(n.float)
    
    def process(self, filename, center, radius, beamblock_rect, compression = 'lzf', sample_type = 'powder', callback = print):
        """
        Processes raw data into something useable by iris.
        
        Parameters
        ----------
        filename : str {*.hdf5}
            Filename for the DiffractionDataset object
        center : 2-tuple

        beamblock_rect : 4-tuple

        compression : str, optional

        sample_type : str {'powder', 'single_crystal'}, optional

        callback : callable or None, optional
            Callable with one argument executed at the end of each time-delay processing.
            Argument will be the progress as an integer between 0 and 100.
        
        Returns
        -------
        path
        """
        if callback is None:
            callback = lambda x: None
        
        # Prepare compression kwargs
        ckwargs = dict()
        if compression:
            ckwargs = {'compression' : compression, 'chunks' : True, 'shuffle' : True, 'fletcher32' : True}
        
        with DiffractionDataset(name = filename, mode = 'w') as processed:

            # Copy experimental parameters
            processed.nscans = self.nscans
            processed.time_points = self.time_points
            processed.acquisition_date = self.acquisition_date
            processed.fluence = self.fluence
            processed.current = self.current
            processed.exposure = self.exposure
            processed.energy = self.energy
            processed.resolution = self.resolution
            processed.center = center
            processed.sample_type = sample_type

            # Copy pumpoff pictures
            # Subtract background from all pumpoff pictures
            pumpoff_image_list = glob.glob(join(self.raw_directory, 'data.nscan.*.pumpoff.tif'))
            pumpoff_cube = n.empty(shape = self.resolution + (len(self.nscans),), dtype = n.uint16)
            for index, image_filename in enumerate(pumpoff_image_list):
                scan_str = re.search('[.]\d+[.]', image_filename.split('\\')[-1]).group()
                scan = int(scan_str.replace('.',''))
                pumpoff_cube[:, :, scan - 1] = cast_to_16_bits(read(image_filename))
            processed.pumpoff_pictures_group.create_dataset(name = 'pumpoff_pictures', data = pumpoff_cube, dtype = n.uint16, **ckwargs)

            # Average background images
            # If background images are not found, save empty backgrounds
            try:
                pumpon_background = average_tiff(self.raw_directory, 'background.*.pumpon.tif', background = None)
            except ImageNotFoundError:
                pumpon_background = n.zeros(shape = self.resolution, dtype = n.uint16)
            processed.processed_measurements_group.create_dataset(name = 'background_pumpon', data = pumpon_background, dtype = n.uint16, **ckwargs)

            try:
                pumpoff_background = average_tiff(self.raw_directory, 'background.*.pumpoff.tif', background = None)
            except ImageNotFoundError:
                pumpoff_background = n.zeros(shape = self.resolution, dtype = n.uint16)
            processed.processed_measurements_group.create_dataset(name = 'background_pumpoff', data = pumpoff_background, dtype = n.uint16, **ckwargs)

            # Create beamblock mask right now
            # Evaluates to TRUE on the beamblock
            x1,x2,y1,y2 = beamblock_rect
            beamblock_mask = n.zeros(shape = processed.resolution, dtype = n.bool)
            beamblock_mask[y1:y2, x1:x2] = True
            
            # Pre-allocation
            cube = n.empty(shape = self.resolution + (len(self.nscans),), dtype = n.float)
            int_intensities = n.zeros(shape = (len(self.nscans),), dtype = n.float)
            averaged = n.empty(shape = self.resolution, dtype = n.float)

            def estimator(*args, **kwargs):
                return 2*n.nanstd(*args, **kwargs)
            
            for i, timedelay in enumerate(self.time_points):

                # Concatenate time-delay in data cube
                # Last axis is the scan number
                # Before concatenation, shift around for center
                # Invalid pixels (such as border pixels, and beamblock pixels) are NaN
                for index, scan in enumerate(self.nscans):
                    
                    try:    # Deal with missing pictures
                        image = self.raw_image(timedelay, scan) - pumpon_background
                        image[beamblock_mask] = n.nan       # Don't take into account pixels under the beamblock
                        corr_i, corr_j = n.array(center) - find_center(image, guess_center = center, radius = radius)
                    except ImageNotFoundError:
                        image = n.empty(shape = self.resolution)
                        image[:] = n.nan
                        corr_i, corr_j = 0, 0
                    cube[:,:,index] = shift(image, round(corr_i), round(corr_j), fill = n.nan)

                # Perform statistical test for outliers using estimator function
                # Pixels deemed outliers have their value replaced by NaN so that
                # they will be ignored by nanmean
                mean = n.expand_dims(n.nanmean(cube, axis = -1, dtype = n.float), axis = 2)
                estimation = n.expand_dims(estimator(cube, axis = -1, dtype = n.float), axis = 2)
                cube[n.abs(cube - mean) > estimation] = n.nan

                # Normalize data cube intensity
                # Integrated intensities are computed for each "picture" (each slice in axes (0, 1))
                # Then, the data cube is normalized such that each slice has the same integrated intensity
                n.nansum(n.nansum(cube, axis = 0, dtype = n.float), axis = 0, dtype = n.float, out = int_intensities)
                int_intensities /= n.mean(int_intensities, dtype = n.float)
                int_intensities.reshape((1,1,len(self.nscans)))
                cube /= int_intensities   

                # Store average along axis 2
                # Averaged data is not uint16 anymore
                n.nanmean(cube, axis = -1, dtype = n.float, out = averaged)
                averaged[n.isnan(averaged)] = 0
                processed.processed_measurements_group.create_dataset(name = str(timedelay), data = averaged, shape = self.resolution, dtype = n.float)

                # Update callback
                callback(round(100*i / len(self.time_points)))

            return filename