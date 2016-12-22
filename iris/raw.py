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
from warnings import warn

from . import cached_property
from .io import read, save, RESOLUTION, ImageNotFoundError, cast_to_16_bits
from .dataset import DiffractionDataset, PowderDiffractionDataset
from .utils import shift, find_center, average_tiff, angular_average

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
        arr : ndarray, shape (N,M)
        
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
        
        return read(join(self.raw_directory, filename)).astype(n.float)
    
    def process(self, filename, center, radius, beamblock_rect, compression = 'lzf', 
                sample_type = 'powder', callback = print, window_size = 10):
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
        window_size : int, optional
            Number of pixels the center is allowed to vary.
        
        Returns
        -------
        path
        """
        if callback is None:
            callback = lambda x: None

        # truncate the sides of images along axes (0, 1) due to the
        # window size of the center finder. Since the center can move by
        # as much as 'window_size' pixels in both all directions, we can
        # save space and avoid a lot of NaNs.
        reduced_resolution = tuple(n.array(self.resolution) - 2*window_size)
        def reduced(arr):
            return arr[window_size:-window_size, window_size:-window_size]
        
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
            processed.resolution = reduced_resolution  # Will be modified later due to center finder
            processed.center = center
            processed.beamblock_rect = beamblock_rect
            processed.sample_type = sample_type

            # Copy pumpoff pictures
            # Subtract background from all pumpoff pictures
            pumpoff_image_list = glob.glob(join(self.raw_directory, 'data.nscan.*.pumpoff.tif'))
            pumpoff_cube = n.empty(shape = reduced_resolution + (len(self.nscans),), dtype = n.uint16)
            for index, image_filename in enumerate(pumpoff_image_list):
                scan_str = re.search('[.]\d+[.]', image_filename.split('\\')[-1]).group()
                scan = int(scan_str.replace('.',''))
                pumpoff_cube[:, :, scan - 1] = reduced(cast_to_16_bits(read(image_filename)))
            processed.pumpoff_pictures_group.create_dataset(name = 'pumpoff_pictures', data = pumpoff_cube, dtype = n.uint16, **ckwargs)

            # Average background images
            # If background images are not found, save empty backgrounds
            try:
                pumpon_background = average_tiff(self.raw_directory, 'background.*.pumpon.tif', background = None)
            except ImageNotFoundError:
                pumpon_background = n.zeros(shape = self.resolution, dtype = n.uint16)
            processed.processed_measurements_group.create_dataset(name = 'background_pumpon', data = reduced(pumpon_background), dtype = n.uint16, **ckwargs)

            try:
                pumpoff_background = average_tiff(self.raw_directory, 'background.*.pumpoff.tif', background = None)
            except ImageNotFoundError:
                pumpoff_background = n.zeros(shape = self.resolution, dtype = n.uint16)
            processed.processed_measurements_group.create_dataset(name = 'background_pumpoff', data = reduced(pumpoff_background), dtype = n.uint16, **ckwargs)

            # Create beamblock mask right now
            # Evaluates to TRUE on the beamblock
            # Only valid for images at the FULL RESOLUTION
            x1,x2,y1,y2 = beamblock_rect
            beamblock_mask = n.zeros(shape = self.resolution, dtype = n.bool)
            beamblock_mask[y1:y2, x1:x2] = True
            
            # truncate the sides of 'cube' along axes (0, 1) due to the
            # window size of the center finder. Since the center can move by
            # as much as 'window_size' pixels in both all directions, we can
            # save space and avoid a lot of NaNs.
            cube = n.empty(shape = reduced_resolution + (len(self.nscans),), dtype = n.float)
            averaged = n.empty(shape = reduced_resolution, dtype = n.float)
            deviation = n.empty_like(cube, dtype = n.float)

            # TODO: find best estimator. 3 std or 5 std?
            def estimator(*args, **kwargs):
                # Set 'estimator' to be 0 where all NaNs.
                # Any comparison later will remove those pixels.
                est = 3*n.nanstd(*args, **kwargs)
                est[n.isnan(est)] = 0
                return est
            
            for i, timedelay in enumerate(self.time_points):

                # Concatenate time-delay in data cube
                # Last axis is the scan number
                # Before concatenation, shift around for center
                # Invalid pixels (such as border pixels, and beamblock pixels) are NaN
                missing_pictures = 0
                slice_index = 0
                for scan in self.nscans:
                    # Deal with missing pictures
                    try:
                        image = self.raw_data(timedelay, scan) - pumpon_background
                    except ImageNotFoundError:
                        warn('Image at time-delay {} and scan {} was not found.'.format(timedelay, scan))
                        missing_pictures += 1
                    
                    corr_i, corr_j = n.array(center) - find_center(image, guess_center = center, radius = radius)
                    #print('Center correction by {} pixels.'.format(n.sqrt(corr_i**2 + corr_j**2)))
                    cube[:,:,slice_index] = reduced(shift(image, round(corr_i), round(corr_j), fill = n.nan))
                    slice_index += 1
                
                # cube possibly has some empty slices due to missing pictures
                # Compress cube along the -1 axis
                if missing_pictures > 0:
                    cube = cube[:,:, 0:-missing_pictures]
                    deviation = n.empty_like(cube, dtype = n.float)
                # All pixels under the beamblock, after shifting the images around
                # These pixels will not contribute to the integrated intensity later
                cube[reduced(beamblock_mask), :] = 0

                # Perform statistical test for outliers using estimator function
                # Pixels deemed outliers have their value replaced by NaN so that
                # they will be ignored by nanmean
                mean = n.nanmean(cube, axis = -1, dtype = n.float, keepdims = True)
                estimation = estimator(cube, axis = -1, dtype = n.float, keepdims = True)
                n.abs(cube - mean, out = deviation) # Contains NaNs if missing pictures
                deviation[n.isnan(deviation)] = n.nanmax(estimation)
                cube[deviation > estimation] = n.nan

                # Normalize data cube intensity
                # Integrated intensities are computed for each "picture" (each slice in axes (0, 1))
                # Then, the data cube is normalized such that each slice has the same integrated intensity
                # int_intensities might contain NaNs due to missing pictures
                int_intensities = n.nansum(n.nansum(cube, axis = 0, dtype = n.float, keepdims = True), axis = 1, dtype = n.float, keepdims = True)
                int_intensities /= n.nanmean(int_intensities, dtype = n.float)
                int_intensities[int_intensities == 0] = 1       # nansum returns 0 if all NaNs, e.g. for missing pictures.
                cube /= int_intensities   

                # Store average along axis 2
                # Averaged data is not uint16 anymore
                n.nanmean(cube, axis = -1, dtype = n.float, out = averaged)
                gp = processed.processed_measurements_group.create_group(name = str(timedelay))
                gp.create_dataset(name = 'intensity', data = averaged, shape = reduced_resolution, dtype = n.float)
                # TODO: include error. Can we approximate the error as intensity/sqrt(nscans) ? Otherwise we
                #       need to store an entire array for error, per timedelay... Doubles the size of dataset.

                callback(round(100*i / len(self.time_points)))

                # If there were some missing pictures, arrays will have been resized. Return to original size
                if missing_pictures > 0:
                    cube = n.empty(shape = reduced_resolution + (len(self.nscans),), dtype = n.float)
                    deviation = n.empty_like(cube, dtype = n.float)
            
        # Extra step for powder data: angular average
        # We already have the center + beamblock info
        # scattering length is the same for all time-delays 
        # if the center and beamblock_rect don't change.
        if sample_type == 'powder':
            with PowderDiffractionDataset(name = filename, mode = 'r+') as processed:
                processed._compute_angular_averages(**ckwargs)

        return filename