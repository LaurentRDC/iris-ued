# -*- coding: utf-8 -*-
"""
@author: Laurent P. René de Cotret
"""
import sys
from os.path import join, isfile, isdir
from os import listdir, mkdir
from shutil import copy2, rmtree
import glob
from datetime import datetime
import re
import numpy as n
from iris.io import read, save, RESOLUTION, ImageNotFoundError

# Info
__author__ = 'Laurent P. René de Cotret'
__version__ = '1.8 released 2016-06-20'

# Hard-coded parameters
SUBSTRATE_FILENAME = 'subs.tif'

def log(message, file):
    """
    Writes a time-stamped message into a log file. Also print to the interpreter
    for debugging purposes.
    """
    now = datetime.now().strftime('[%Y-%m-%d  %H:%M:%S]')
    time_stamped = '{0} {1}'.format(now, str(message))
    print(time_stamped)
    print(time_stamped, file = file)
    
def average_tiff(directory, filename_template, background = None, normalize_to_intensity = None):
    """
    Averages images matching a filename template within the dataset directory.
    
    Parameters
    ----------
    directory : str
        Absolute path to the directory
    filename_template : string
        Examples of filename templates: 'background.*.pumpon.tif', '*.tif', etc.
    background : array-like, optional
        Background to subtract from the average.
    normalize_to_intensity : float or None, optional
        If not None, images will be normalized to this value before averaging takes place
        
    Returns
    -------
    out : ndarray
    
    Raises
    ------
    ImageNotFoundError
        If filename_template does not match any file in the directory
    """ 
    #Format background correctly
    if background is not None:
        background = background.astype(n.float)
    else:
        background = n.zeros(shape = RESOLUTION, dtype = n.float)
    
    #Get file list
    image_list = glob.glob(join(directory, filename_template))
    if not image_list:      #List is empty
        raise ImageNotFoundError('filename_template does not match any file in the dataset directory')
    
    image = n.zeros(shape = RESOLUTION, dtype = n.float)
    for filename in image_list:
        new_image = read(filename)
        
        # Optional normalization
        if normalize_to_intensity is not None:
            new_image *= normalize_to_intensity/new_image.sum()
        
        image += new_image
        
    # average - background
    return image/len(image_list) - background

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
    
    preprocess
    """
    def __init__(self, directory):
        if isdir(directory):
            self.raw_directory = directory
        else:
            raise ValueError('The path {} is not a directory'.format(directory))
    
    @property
    def nscans(self):
        """ List of integer scans. """
        scans = [re.search('[n][s][c][a][n][.](\d+)', f).group() for f in self.image_list if 'nscan' in f]
        return list(set([int(string.strip('nscan.')) for string in scans])) # Remove duplicates by using a set
    
    @property
    def acquisition_date(self):
        """ Returns the acquisition date from the folder name as a string of the form: '2016.01.06.15.35' """
        try:
            return re.search('(\d+[.])+', self.raw_directory).group()[:-1]      #Last [:-1] removes a '.' at the end
        except(AttributeError):     #directory name does not match time pattern
            return '0.0.0.0.0'
    
    @property
    def time_points(self):
        return [float(t) for t in self.time_points_str]
    
    @property
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
        return [f for f in listdir(self.raw_directory) 
                if isfile(join(self.raw_directory, f)) 
                and f.endswith('tif')] 
    
    @property
    def processed(self):
        return isdir(join(self.raw_directory, 'processed'))
    
    @property
    def pumpon_background(self):
        backgrounds = [read(filename) for filename in glob.glob(join(self.raw_directory, 'background.*.pumpon.tif'))]
        return sum([background for background in backgrounds])/len(backgrounds)
    
    @property
    def pumpoff_background(self):
        backgrounds = [read(filename) for filename in glob.glob(join(self.raw_directory, 'background.*.pumpoff.tif'))]
        return sum([background for background in backgrounds])/len(backgrounds)
        
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
        sign = '-' if timedelay < 0 else '+'
        str_time = sign + '{0:.2f}'.format(float(timedelay))
        filename = 'data.timedelay.' + str_time + '.nscan.' + str(int(scan)).zfill(2) + '.pumpon.tif'
        
        return read(join(self.raw_directory, filename)).astype(n.float)
    
    def preprocess(self, overwrite = False):
        """
        Preprocesses raw data into something useable by iris.
        
        Parameters
        ----------
        directory : str
            Data directory.
        overwrite : bool, optional
            If True, any existing processed data will be deleted and reprocessed.
            Default is False. NOT IMPLEMENTED.
        
        Returns
        -------
        processed_directory : path
        """
        # If the processed folder already exists, abort.
        processed_directory = join(self.raw_directory, 'processed')
        
        if isdir(processed_directory) and not overwrite:
            print('Processed directory already exists but overwriting is disabled. Abort preprocessing.')
            return
        elif isdir(processed_directory) and overwrite:
            print('Processed directory already exists but overwriting is enabled.')
            rmtree(processed_directory)
            
        mkdir(processed_directory)
        
        #Create log file and begin preprocessing
        log_filename = join(processed_directory, 'log.txt')
        with open(log_filename, 'x') as log_file:
            try:
                # Try preprocessing
                self._preprocess_routine(processed_directory, log_file, overwrite)
            except Exception:
                # Record errors and log them
                error, value, traceback = sys.exc_info()
                log('Preprocessing aborted due to an error: {}'.format(value.strerror), file = log_file)
            finally: 
                # If someone is monitoring the interpreter, execute this bit of code
                # no matter if an error was raised.
                print('Preprocessing done. A log is available at {}'.format(log_filename))
        
        return processed_directory
    
    def _preprocess_routine(self, processed_directory, log_file, overwrite = False):
        """
        Preprocessing function. Highly dependent on the formatting of the directory.
        
        Parameters
        ----------
        directory : str
            Data directory.
        processed_directory : str
            Directory of the processed images. Usually '\...\processed'
        log_file : File object
            Log file in which the log is kept.
        overwrite : bool, optional
            If True, any existing processed data will be deleted and reprocessed.
            Default is False. NOT IMPLEMENTED.
        """
        
        # Header of the logging file
        log('Preprocessing script', file = log_file)
        log('Siwick Research Group', file = log_file)
        log('Script author: {0}'.format(__author__), file = log_file)
        log('Script version: {0}'.format(__version__), file = log_file)
        log('---------------------', file = log_file)
        
        #Get time points
        log('Processing directory {0}'.format(self.raw_directory), file = log_file)
        log('Time points found:', file = log_file)
        times = self.time_points_str
        log(str(times), file = log_file)
        log('---------------------', file = log_file)
        
        # Average background images
        # If background images are not found, save empty backgrounds
        try:
            pumpon_background = average_tiff(self.raw_directory, 'background.*.pumpon.tif', background = None)
        except ImageNotFoundError:
            pumpon_background = n.zeros(shape = RESOLUTION, dtype = n.float)
        
        try:
            pumpoff_background = average_tiff(self.raw_directory, 'background.*.pumpoff.tif', background = None)
        except ImageNotFoundError:
            pumpoff_background = n.zeros(shape = RESOLUTION, dtype = n.float)
        
        save(pumpon_background, join(processed_directory, 'background_average_pumpon.tif'))
        save(pumpoff_background, join(processed_directory, 'background_average_pumpoff.tif'))
        log('Background pictures averaged.', file = log_file)
        log('---------------------', file = log_file)
        
        # Copy pumpoff pictures in a separate folder for debugging
        # Subtract background from all pumpoff pictures
        pumpoff_directory = join(processed_directory, 'pumpoff pictures')
        mkdir(pumpoff_directory)
        log('Pumpoff pictures folder created.', file = log_file)
        pumpoff_image_list = glob.glob(join(self.raw_directory, 'data.nscan.*.pumpoff.tif'))
        if not pumpoff_image_list:      #List is empty
            log('Pumpoff pictures missing.', file = log_file)
        i = 1
        for filename in pumpoff_image_list:
            image = read(filename) - pumpoff_background
            save(image, join(pumpoff_directory, 'data_nscan_{0}_pumpoff.tif'.format(str(i))))
            i += 1
        log('Pumpoff pictures background-subtracted and moved.', file = log_file)
        log('---------------------', file = log_file)
        
        # Process substrate picture if it exists
        # Due to the way substrate picture are acquired, the shape might not be the
        # same as the resolution.
        if SUBSTRATE_FILENAME in listdir(self.raw_directory):
            log('Substrate picture found.', file = log_file) 
            subs = read(join(self.raw_directory, SUBSTRATE_FILENAME)) - pumpoff_background
        else:
            log('Substrate picture not found.', file = log_file)
            subs = n.zeros(shape = RESOLUTION, dtype = n.float)
        save(subs, join(processed_directory, 'substrate.tif'))
        log('---------------------', file = log_file)
        
        # Average time delay images
        # Substract background and substrate (if it exists)
        # In the output filename, use str(float(time)) to get rid of extra '+' and trailing zeros
        # e.g.: str(float(time)) = '5.0', not '+005.000'
        
        # Get overall intensity at each time delay from the first scan
        # Each time delay of subsequent scans will be normalized to the corresponding
        # to the overall intensity of the associated time-delay of the first scan
        for time in self.time_points_str:
            # Get integrated intensity at this time delay for the first scan
            first_scan_picture = 'data.timedelay.' + time + '.nscan.01.pumpon.tif'
            integrated_intensity_from_first_scan = read(join(self.raw_directory, first_scan_picture)).sum()
            
            template = 'data.timedelay.' + time + '.nscan.*.pumpon.tif'
            output_filename = ('data_timedelay_' + str(float(time)) + '_average_pumpon.tif')
            average = average_tiff(self.raw_directory, template, background = pumpon_background, 
                                   normalize_to_intensity = integrated_intensity_from_first_scan)
            save(average - subs, join(processed_directory, output_filename)) 
            log('Averaged timepoint {0}'.format(str(time)), file = log_file)
        log('---------------------', file = log_file)
        
        # Copy tagfile.txt to record experimental_parameters
        # Open experimental parameters file in appending mode 'a' and record the acquisition date
        exp_params_filename = join(processed_directory, 'experimental_parameters.txt')
        try:
            copy2(join(self.raw_directory, 'tagfile.txt'), exp_params_filename)
            log('tagfile.txt copied to experimental_parameters.txt', file = log_file)
        except:
            log('tagfile.txt could not be copied to experimental_parameters.txt', file = log_file)
        
        with open(exp_params_filename, 'a') as params_file:
            print('Acquisition date = {0}'.format(self.acquisition_date), file = params_file)
        log('Acquisition date added to experimental parameters file', file = log_file)
        log('---------------------', file = log_file)
        
        # That's all folks
        log('Preprocessing ended with no issues.', file = log_file)


if __name__ == '__main__':
    d = RawDataset('K:\\test_folder')
    d.preprocess()