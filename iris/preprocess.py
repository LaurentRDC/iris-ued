# -*- coding: utf-8 -*-
"""
Preprocess of Ultrafast Electron Diffraction data
-------------------------------------------------

This script averages pictures together, subtracting background images when
possible, in order to reduce the size of datasets. Anything more involved than 
this (e.g. brightness equalization) should be done in processing.

Preprocessed data is put in a folder called 'processed'. Information is logged 
into a .txt file as well.Experimental parameters (from the original tagfile.txt) 
are recorded into a file called experimental_parameters.txt.

Arguments
----------
directory : str, optional
    The directory of the data directory. If left empty, the script will act as
    if it was placed inside the directory itself.

Examples
--------
Local mode in Windows cmd.exe or Powershell:
    C:/Users>
    C:/Users> cd 'K:/2016.03.01.16.57.VO2_1500mW'
    K:/2016.03.01.16.57.VO2_1500mW>
    K:/2016.03.01.16.57.VO2_1500mW> python preprocess.py
    [2016-04-02  13:35:42] Preprocessing begun.
    [2016-04-02  13:35:42] Script author: Laurent P. René de Cotret
    [2016-04-02  13:35:42] Script version: 2016-04-02
    [2016-04-02  13:35:42] Processing directory K:/2016.03.01.16.57.VO2_1500mW
    ...

Remote mode in Windows cmd.exe or Powershell:
    C:/Users>
    C:/Users> python preprocess.py 'K:/2016.03.01.16.57.VO2_1500mW'
    [2016-04-02  13:35:42] Preprocessing begun.
    [2016-04-02  13:35:42] Script author: Laurent P. René de Cotret
    [2016-04-02  13:35:42] Script version: 2016-04-02
    [2016-04-02  13:35:42] Processing directory K:/2016.03.01.16.57.VO2_1500mW
    ...
"""

import sys
from os.path import abspath, join, isfile, isdir, dirname
from os import listdir, mkdir
from shutil import copy2, rmtree
from glob import glob
from datetime import datetime
from time import sleep
import re
import numpy as n
from iris.io import read, save, RESOLUTION

# Info
__author__ = 'Siwick Lab'
__version__ = '1.7 released 2016-06-07'

# Hard-coded parameters
SUBSTRATE_FILENAME = 'subs.tif'

    
def acquisition_date(directory):
    """ Returns the acquisition date from the folder name as a string of the form: '2016.01.06.15.35' """
    try:
        return re.search('(\d+[.])+', directory).group()[:-1]      #Last [:-1] removes a '.' at the end
    except(AttributeError):     #directory name does not match time pattern
        return '0.0.0.0.0'


def log(message, file):
    """
    Writes a time-stamped message into a log file. Also print to the interpreter
    for debugging purposes.
    """
    now = datetime.now().strftime('[%Y-%m-%d  %H:%M:%S]')
    time_stamped = '{0} {1}'.format(now, str(message))
    print(time_stamped)
    print(time_stamped, file = file)


def time_points(directory):
    """ 
    Returns a list of sorted string times.
    
    Parameters
    ----------
    directory : str
        Absolute path to the data directory.
    
    Returns
    -------
    time_list : list of str
    """
    #Get TIFF images
    image_list = [f for f in listdir(directory) 
            if isfile(join(directory, f)) 
            and f.startswith('data.timedelay.') 
            and f.endswith('pumpon.tif')]
    # Get time points. Strip away '+' as they are superfluous.
    time_data = [re.search('[+-]\d+[.]\d+', f).group() for f in image_list]
    time_list =  list(set(time_data))     #Conversion to set then back to list to remove repeated values
    
    time_list.sort(key = lambda x: float(x))
    return time_list


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
    IOError
        If filename_template does not match any file in the directory
    """ 
    #Format background correctly
    if background is not None:
        background = background.astype(n.float)
    else:
        background = n.zeros(shape = RESOLUTION, dtype = n.float)
    
    #Get file list
    image_list = glob(join(directory, filename_template))
    if not image_list:      #List is empty
        raise IOError('filename_template does not match any file in the dataset directory')
    
    image = n.zeros(shape = RESOLUTION, dtype = n.float)
    for filename in image_list:
        new_image = read(filename)
        
        # Optional normalization
        if normalize_to_intensity is not None:
            new_image *= normalize_to_intensity/new_image.sum()
        
        image += new_image
        
    # average - background
    return image/len(image_list) - background    

def preprocess(directory, overwrite = False):
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
    # TODO: make overwriting possible
    processed_directory = join(directory, 'processed')
    
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
            _preprocess_routine(directory, processed_directory, log_file, overwrite)
        except Exception:
            # Record errors and log them
            error, value, traceback = sys.exc_info()
            log('Preprocessing aborted due to an error: {}'.format(value.strerror), file = log_file)
        finally: 
            # If someone is monitoring the interpreter, execute this bit of code
            # no matter if an error was raised.
            print('Preprocessing done. A log is available at {0}'.format(str(log_filename)))
    
    return processed_directory

def _preprocess_routine(directory, processed_directory, log_file, overwrite = False):
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
    log('Processing directory {0}'.format(directory), file = log_file)
    log('Time points found:', file = log_file)
    times = time_points(directory)
    log(str(times), file = log_file)
    log('---------------------', file = log_file)
    
    # Average background images
    # If background images are not found, save empty backgrounds
    try:
        pumpon_background = average_tiff(directory, 'background.*.pumpon.tif', background = None)
    except IOError:
        pumpon_background = n.zeros(shape = RESOLUTION, dtype = n.float)
    
    try:
        pumpoff_background = average_tiff(directory, 'background.*.pumpoff.tif', background = None)
    except IOError:
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
    pumpoff_image_list = glob(join(directory, 'data.nscan.*.pumpoff.tif'))
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
    if SUBSTRATE_FILENAME in listdir(directory):
        log('Substrate picture found.', file = log_file) 
        subs = read(join(directory, SUBSTRATE_FILENAME)) - pumpoff_background
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
    for time in time_points(directory):
        # Get integrated intensity at this time delay for the first scan
        first_scan_picture = 'data.timedelay.' + time + '.nscan.01.pumpon.tif'
        integrated_intensity_from_first_scan = read(join(directory, first_scan_picture)).sum()
        
        template = 'data.timedelay.' + time + '.nscan.*.pumpon.tif'
        output_filename = ('data_timedelay_' + str(float(time)) + '_average_pumpon.tif')
        average = average_tiff(directory, template, background = pumpon_background, 
                               normalize_to_intensity = integrated_intensity_from_first_scan)
        save(average - subs, join(processed_directory, output_filename)) 
        log('Averaged timepoint {0}'.format(str(time)), file = log_file)
    log('---------------------', file = log_file)
    
    # Copy tagfile.txt to record experimental_parameters
    # Open experimental parameters file in appending mode 'a' and record the acquisition date
    exp_params_filename = join(processed_directory, 'experimental_parameters.txt')
    try:
        copy2(join(directory, 'tagfile.txt'), exp_params_filename)
        log('tagfile.txt copied to experimental_parameters.txt', file = log_file)
    except:
        log('tagfile.txt could not be copied to experimental_parameters.txt', file = log_file)
    
    with open(exp_params_filename, 'a') as params_file:
        print('Acquisition date = {0}'.format(acquisition_date(directory)), file = params_file)
    log('Acquisition date added to experimental parameters file', file = log_file)
    log('---------------------', file = log_file)
    
    # That's all folks
    log('Preprocessing ended with no issues.', file = log_file)