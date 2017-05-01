"""
Array manipulation subroutines

@author: Laurent P. Rene de Cotret
"""
from functools import partial
import numpy as n
from skimage.feature import register_translation
from warnings import catch_warnings, simplefilter

from .optimizations import pmap
from .utils import find_center

def diff_avg(arr, weights = None):
    """
    Average diffraction pictures from the same time-delay together. It is assumed that the 
    pictures have been aligned already.

    Parameters
    ----------
    arr : ndarray, dtype uint16
        Array to be averaged.
    weights : ndarray or None, optional
        Array representing how much an image should be 'worth'. E.g.: a weight below 1 means that
        a picture is not bright enough, and therefore it should count more in the averaging.
        If None (default), total picture intensity is used to weight each picture.
    
    Returns
    -------
    avg : ndarray, ndim 2, dtype float
        'Average' of arr.
    err : ndarray, ndim 2, dtype float
        Standard error in the mean.
    """
    arr = n.array(arr, copy = False, dtype = n.uint16)

    # Handle weights of images
    # The sum of weights should be equal to 1 per picture
    if weights is None:
        weights = n.sum(arr, axis = (0, 1)).astype(n.float)
    weights *= arr.shape[2] / n.sum(weights)    # Normalize weights

    # Final averaging
    # Error in the mean is only approximate, but much faster.
    # For a true measure of error, see scipy.stats.sem (masked standard error in mean)
    avg = n.average(arr, axis = 2, weights = weights) 
    err = n.std(arr, axis = 2) / n.sqrt(arr.shape[2])
    return avg, err

non = lambda s: s if s < 0 else None
mom = lambda s: max(0, s)
def shift_image(arr, shift):
    """ 
    Shift an image on at a 1-pixel resolution.

    Parameters
    ----------
    arr : ndarray
    shift : ndarray

    Returns
    -------
    out : ndarray
        Invalid pixels are set to NaN
    """
    x, y = tuple(shift)
    x, y = int(x), int(y)

    shifted = n.full_like(arr, n.nan)
    shifted[mom(y):non(y), mom(x):non(x)] = arr[mom(-y):non(-y), mom(-x):non(-x)]
    return shifted

def powder_align(images, guess_center, radius, window_size = 10, ring_width = 5):
    """
    Align powder diffraction images together, making use of the azimuthal symmetry
    of the patterns to speed up computations.

    Parameters
    ----------
    images : iterable
        Iterable of ndarrays
    guess_center : array_like, shape (2,)
        Initial guess for a center
    radius : int    

    window_size : int, optional

    ring_width : int, optional
        
    
    Returns
    -------
    aligned : tuple of ndarrays, ndim 2
        Aligned images
    """
    images = iter(images)
    center = n.array(guess_center, dtype = n.float, copy = False)
    
    aligned = list()
    for image in images:
        shift = center - find_center(image, guess_center = center, radius = radius, 
                                     window_size = window_size, ring_width = ring_width)
        aligned.append(shift_image(image, -shift))
    return aligned

def diff_align(images, reference = None, upsample_factor = 1, processes = None):
    """
    Align diffraction images to each other. Optimized for single-crystal images for now.

    Parameters
    ----------
    images : iterable
        Iterable of ndarrays
    reference : ndarray or None, optional
        If not None, this is the reference image to which all images will be aligned. Otherwise,
        images will be aligned to the first element of the iterable 'images'.
    upsample_factor : int, optional
        Images will be aligned to within 1 / upsample_factor of a pixel. 
        For example upsample_factor == 20 means the images will be aligned within 1/20th of a pixel. 
    processes : int or None, optional
        Number of processors to use for this task. If None, all CPUs are used.
    
    Returns
    -------
    aligned : tuple of ndarrays, ndim 2
        Aligned images
    """
    images = iter(images)

    # TODO: crop images to a subimage, that is used for alignment
    aligned = list()
    if reference is None:
        reference = next(images)
        aligned = [reference]
    
    return aligned + list(pmap(_align_im, images, kwargs = {'reference': reference, 
                                                            'upsample_factor': upsample_factor}, 
                                                  processes = processes))

def _align_im(image, reference, upsample_factor):
    """ Align a single image to a reference. """
    shifts, *_ = register_translation(reference, image, upsample_factor = upsample_factor, space = 'real')
    return shift_image(image, -shifts)