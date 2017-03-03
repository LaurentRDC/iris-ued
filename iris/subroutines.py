"""
Array manipulation subroutines

@author: Laurent P. Rene de Cotret
"""
import numpy as n
from scipy.stats.mstats import sem  #Standard error in mean
from scipy.ndimage import fourier_shift
from skimage.feature import register_translation

from .optimizations import pmap
from .utils import find_center

try:
    from numpy.fft_intel import fft2, ifft2, fftshift
except ImportError:
    from numpy.fft import fft2, ifft2, fftshift

def diff_avg(arr, weights = None, mad = True, mad_dist = 3):
    """
    Average diffraction pictures from the same time-delay together. Median-abolute-difference (MAD)
    filtering can also be used to clean up the data.

    It is assumed that the pictures have been aligned already.

    Parameters
    ----------
    arr : ndarray or MaskedArray
        Array to be averaged.
    weights : ndarray or None, optional
        Array representing how over-estimated each image is. If None (default),
        total picture intensity is used to weight each picture.
    mad : bool, optional
        If True (default), the distributions of pixel intensities across scans are included based on a median absolute difference (MAD)
        approach. Set to False for faster performance.
    mad_dist : float, optional
        The number of median-absolute-differences allowable inside the pixel intensity distribution.
        Setting this number lower will 'filter' out more pixels.
    
    Returns
    -------
    avg : ndarray, ndim 2
        'Average' of arr.
    err : ndarray, ndim 2
        Standard error in the mean.
    """
    # Making sure it is a masked array
    arr = n.ma.masked_array(arr)

    if mad:
        # Mask outliers according to the median-absolute-difference criterion
        # Consistency constant of 1.4826 due to underlying normal distribution
        # http://eurekastatistics.com/using-the-median-absolute-deviation-to-find-outliers/
        absdiff = n.ma.abs(arr - n.ma.median(arr, axis = 2, keepdims = True))
        MAD = 1.4826*n.ma.median(absdiff, axis = 2, keepdims = True)     # out = mad bug with keepdims = True
        arr[absdiff > mad_dist*MAD] = n.ma.masked
    
    if weights is None:
        integrated = n.ma.sum(n.ma.sum(arr, axis = 0), axis = 0)
        weights = n.ma.mean(integrated) / integrated
    
    avg = n.ma.average(arr, axis = 2, weights = weights)
    err = sem(arr, axis = 2)
    return avg, err

def shift_image(arr, shift):
    """
    Shift array with subpixel accuracy.

    Parameters
    ----------
    arr : ndarray
        Image to be shifted
    shift : ndarray

    Returns
    -------
    shifted : ndarray
    """
    return n.real(ifft2(fourier_shift(fft2(arr), shift = shift)))

def powder_align(images, guess_center, radius, window_size = 10, ring_width = 10):
    """
    Align powder diffraction images together, making use of the azimuthal symmetry
    of the patterns to speed up computations.

    Parameters
    ----------
    image : ndarray, ndim 2

    guess_center : 2-tuple

    radius : int

    window_size : int, optional

    ring_width : int, optional
    """
    images = iter(images)
    center = n.array(guess_center, dtype = n.float)
    
    # TODO: parallelize?
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
    images = iter(images)   # Making sure we have an iterator here

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