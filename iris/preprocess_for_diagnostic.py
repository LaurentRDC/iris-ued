# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 13:04:23 2016

@author: SiwickWS1
"""

# -*- coding: utf-8 -*-
"""
Advanced Preprocess of Ultrafast Electron Diffraction data
-------------------------------------------------

Functions
---------
preprocess_for_diagnostic
    Preprocess a raw data folder for pic by pic diagnostics.
    
@author : Laurent P. René de Cotret and Mark Stern
"""

import pdb
import sys
import traceback
import pyqtgraph as pg
from os.path import join, isdir
from os import mkdir, listdir
from .io import read
from .preprocess import log
from .dataset import radial_average

# Info
__author__ = 'Siwick Lab'
__version__ = '0.1'
__all__ = ['preprocess_for_diagnostic']

# Hard-coded parameters
SUBSTRATE_FILENAME = 'subs.tif'
DEFAULT_DIRECTORY = 'C:\\Users\\SiwickWS1\\Desktop\\2016.06.15.21.31.VO2_11mJ'

def read_radav_tool_params(file_path):
        """
        Reads a file to determine the positions for the center
        finder and mask.
        
        Parameters
        ----------
        key : str
            Name of the parameter
        """
        
        with open(file_path, 'r') as config_file:
            for line in config_file:
                if line.startswith('MASK_POSITION'):
                    mask_params = eval(line.split('=')[-1])
                elif line.startswith('CENTER_FINDER_POSITION'):    
                    center_finder_pos = eval(line.split('=')[-1])                 
        return center_finder_pos, mask_params



def preprocess_for_diagnostic(directory = DEFAULT_DIRECTORY, overwrite = False):
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
    processed_directory = join(directory, 'processed_for_diagnostic')
    if not isdir(processed_directory):
        mkdir(processed_directory)
    
    #Create log file and begin preprocessing
    log_filename = join(processed_directory, 'log.txt')
    with open(log_filename, 'w+') as log_file:
        try:
            # Try preprocessing
            _preprocess_routine(directory, processed_directory, log_file, overwrite)
        except Exception:
            # Record errors and log them
            error, value, traceback_exc = sys.exc_info()
            traceback.print_tb(traceback_exc)
            log('Preprocessing aborted due to an error: {}'.format(value), file = log_file)
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
    for file_name in listdir(directory):
        if file_name[-4:] == '.tif' and file_name.startswith('data.timedelay'):
            radial_output = _radially_average(directory,file_name)
            save_as_txt(radial_output, join(processed_directory, file_name[:-4] + '.txt'))
            print(file_name)
   
def _radially_average(directory, file_name):
    image = read(join(directory, file_name), normalize = True)
    background = read(join(directory,'processed','background_average_pumpon.tif'), normalize = True)
    center, beamblock_mask = read_radav_tool_params(join(directory,'processed','radav_tools.txt'))
    return radial_average(image-background, center, beamblock_mask, return_error = False)

def save_as_txt(radial_output, output_file_path):
    with open(output_file_path,'w') as out_file:
        for x, y in zip(radial_output[0], radial_output[1]):
            out_file.write(str(x) + ', ' + str(y) + '\n')          

if __name__ == '__main__':
    preprocess_for_diagnostic()