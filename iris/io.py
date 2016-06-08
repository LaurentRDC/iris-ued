# -*- coding: utf-8 -*-
"""
TIFF handling and other I/O operations.

Functions
---------
read
    Read a TIFF image into a NumPy array.

save
    Save a NumPy array to a TIFF.

resize
    Resize an array to a certain resolution.

cast_to_16_bits
    Normalize a NumPy array to a 16-bit array. Useful when
    dealing with CCD camera images-as-array.

Global variables
----------------
RESOLUTION
    CCD camera resolution.

@author: Laurent P. René de Cotret
"""

import numpy as n
from iris.tifffile import imread, imsave

__all__ = ['read', 'save', 'resize', 'cast_to_16_bits', 'RESOLUTION']

RESOLUTION = (2048, 2048)
_HOT_PIXEL_THRESHOLD = 25000
_HOT_PIXEL_VALUE = 0

def cast_to_16_bits(array):
    """ 
    Returns an array in int16 format. Array values below 0 are cast to 0,
    and array values above (2**16) - 1 are cast to (2**16) - 1.
    
    Parameters
    ----------
    array : array-like
        Array to be cast into 16 bits.
    
    Returns
    -------
    out : ndarray, dtype numpy.int16
    """
    array = n.asarray(array)
    array[ array < 0] = 0
    array[ array > (2**16) - 1] = (2**16) - 1
    return n.around(array, decimals = 0).astype(n.int16)

def resize(array, resolution = RESOLUTION):
    """
    Resizes an array to a certain resolution by zero-padding or cropping.
    
    Parameters
    ----------
    array : ndarray
    resolution : 2-tuple
    """
    if array.shape == resolution:
        return array
    elif array.size > resolution[0]*resolution[1]:
        base = n.zeros(shape = resolution, dtype = n.float)
        base[:resolution[0],:resolution[1]] = array
    else:
        base = n.zeros(shape = resolution, dtype = n.float)
        base[:array.shape[0],:array.shape[1]] = array
    return base
    
def read(filename, return_mask = False):
    """ 
    Returns a ndarray from an image filename. 
    
    Parameters
    ----------
    filename : str
        Absolute path to the file.
    return_mask : bool, optional
        If True, returns a mask the same size as image that evaluates to False
        on pixels that are invalid due to being above a certain threshold 
        (hot pixels). Default is False.
    
    Returns
    -------
    out : ndarray, dtype numpy.float
        Numpy array from the image.
    mask : ndarray, dtype numpy.bool
        Numpy array of the valid pixels. Only returned if return_mask is True
    """
    image = imread(filename).astype(n.float)
    if image.shape != RESOLUTION:
        image = resize(image, RESOLUTION)
    image[image < 0] = 0
    
    # Deal with saturated pixels
    mask = image < _HOT_PIXEL_THRESHOLD   # Mask evaluated to False on hot pixels
    if not return_mask:
        # Set hot pixels to _HOT_PIXEL_VALUE
        image[n.logical_not(mask)] = _HOT_PIXEL_VALUE
        return image
    elif return_mask:
        return image, mask

    
def save(array, filename):
    """ 
    Saves a ndarray to an image filename. 
    
    Parameters
    ----------
    array : ndarray
        Image as an array of any type. The array will be cast to 16 bits.
    filename : str
        Absolute path to the file.
    
    See also
    --------        
    cast_to_16_bits
        For casting rules.
    """
    imsave(filename, cast_to_16_bits(array))