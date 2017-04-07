# -*- coding: utf-8 -*-
"""
@author: Laurent P. René de Cotret
"""
import glob
import numpy as n
from os.path import join, isfile, isdir
from os import listdir 
import re
from skimage.feature import register_translation
import sys
from datetime import datetime as dt
from warnings import warn, catch_warnings

from . import cached_property
from .io import read, save, RESOLUTION, ImageNotFoundError, cast_to_16_bits
from .dataset import DiffractionDataset, PowderDiffractionDataset
from .subroutines import diff_avg, diff_align, powder_align, shift_image
from .utils import find_center, average_tiff, angular_average

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
    
    process
    """

    resolution = RESOLUTION
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
        backgrounds = tuple(read(filename) for filename in glob.glob(join(self.raw_directory, 'background.*.pumpon.tif')))
        return sum(backgrounds)/len(backgrounds)
    
    @property
    def pumpoff_background(self):
        backgrounds = tuple(read(filename) for filename in glob.glob(join(self.raw_directory, 'background.*.pumpoff.tif')))
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
        
        return read(join(self.raw_directory, filename))
    
    def process(self, filename, center, radius, beamblock_rect, compression = 'lzf', sample_type = 'powder', 
                callback = None, cc = True, window_size = 10, ring_width = 5, mad = True):
        """
        Processes raw data into something useable by iris.
        
        Parameters
        ----------
        filename : str {*.hdf5}
            Filename for the DiffractionDataset object
        center : 2-tuple

        radius : float

        beamblock_rect : 4-tuple

        compression : str, optional

        sample_type : str {'powder', 'single_crystal'}, optional

        callback : callable or None, optional
            Callable with one argument executed at the end of each time-delay processing.
            Argument will be the progress as an integer between 0 and 100.
        cc : bool, optional
            Center correction flag. If True, images are shifted before
            processing to account for electron beam drift.
        window_size : int, optional
            Number of pixels the center is allowed to vary.
        ring_width : int, optional
            Width of the ring over which the intensity integral is calculated.
        mad : bool, optional
            If True (default), the distributions of pixel intensities across scans are included based on a median absolute difference (MAD)
            approach. Set to False for faster performance.
        
        Returns
        -------
        path
        """
        # Preliminary check. If energy is 0kV, then the scattering length calculation will
        # fail at the end of processing, crashing iris.
        if self.energy == 0:
            raise AttributeError('Energy is 0 kV')

        if callback is None:
            callback = lambda x: None
        
        # Prepare compression kwargs
        ckwargs = dict()
        if compression:
            ckwargs = {'compression' : compression, 'chunks' : True, 'shuffle' : True, 'fletcher32' : True}
        
        start_time = dt.now()
        with DiffractionDataset(name = filename, mode = 'w') as processed:

            # Copy experimental parameters
            # Center and beamblock_rect will be modified
            # because of reduced resolution later
            processed.nscans = self.nscans
            processed.time_points = self.time_points
            processed.acquisition_date = self.acquisition_date
            processed.fluence = self.fluence
            processed.current = self.current
            processed.exposure = self.exposure
            processed.energy = self.energy
            processed.resolution = self.resolution
            processed.sample_type = sample_type
            processed.center = center
            processed.beamblock_rect = beamblock_rect
            processed.time_zero_shift = 0.0

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

        # Prepare container for the large array of averaged pictures
        shape = self.resolution + (len(self.time_points),)
        with DiffractionDataset(name = filename, mode = 'r+') as processed:
            gp = processed.processed_measurements_group
            gp.create_dataset(name = 'intensity', shape = shape, dtype = n.float32, **ckwargs)
            gp.create_dataset(name = 'error', shape = shape, dtype = n.float32, **ckwargs)

        # Get reference image for aligning all single crystal images
        ref_im = self.raw_data(self.time_points[0], self.nscans[0]) - pumpoff_background

        # TODO: parallelize this loop
        #       The only reason it is not right now is that
        #       each branch of the loop uses ~ 4-6GBs of RAM for
        #       a 30 scans dataset
        for i, timedelay in enumerate(self.time_points):

            images = list()
            for scan in self.nscans:
                try:
                    images.append( self.raw_data(timedelay, scan) - pumpon_background )
                except ImageNotFoundError:
                    warn('Image at time-delay {} and scan {} was not found.'.format(timedelay, scan))
            
            # Center correction is built as a subroutine
            #
            if cc and sample_type == 'single_crystal':
                images = diff_align(images, reference = ref_im)
            elif cc and sample_type == 'powder':
                images = powder_align(images, guess_center = center, radius = radius, 
                                      window_size = window_size, ring_width = ring_width)
            
            # Creation of the image 'cube' with appropriate mask
            # Create beamblock mask right now
            # Evaluates to TRUE on the beamblock
            x1,x2,y1,y2 = beamblock_rect
            cube = n.dstack(images)
            cube[y1:y2, x1:x2, :] = n.nan

            # Average appropriately using subroutine
            averaged, error = diff_avg(cube, mad = mad, mad_dist = 3)

            with DiffractionDataset(name = filename, mode = 'r+') as processed:
                processed.processed_measurements_group['intensity'].write_direct(n.nan_to_num(averaged), source_sel = n.s_[:,:], dest_sel = n.s_[:,:,i])
                processed.processed_measurements_group['error'].write_direct(n.nan_to_num(error), source_sel = n.s_[:,:], dest_sel = n.s_[:,:,i])
            
            callback(round(100*i / len(self.time_points)))

        # Extra step for powder data: angular average
        # We already have the center + beamblock info
        # scattering length is the same for all time-delays 
        # since the center and beamblock_rect don't change.
        if sample_type == 'powder':
            with PowderDiffractionDataset(name = filename, mode = 'r+') as processed:
                processed._compute_angular_averages()

        callback(100)
        print('Processing has taken {}'.format(str(dt.now() - start_time)))
        return filename