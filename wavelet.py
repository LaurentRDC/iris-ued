# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:57:31 2016

@author: Laurent P. René de Cotret

References
----------
[1] An Iterative Algorithm for Background Removal in Spectroscopy by 
    Wavelet Transforms. Galloway, Le Ru and Etchegoin
"""
import numpy as n
import pywt
from scipy import interpolate

# Boundary extension mode
extension_mode = 'constant'

def dyadic_upsampling(array):
    """
    Upsamples a 1D signal or 2D array by a factor of 2 using linear interpolation.
    
    Parameters
    ----------
    array : ndarray
        Array to be upsampled. Can be either a 1D signal or 2D array.

    Returns
    -------
    out : ndarray
        Upsampled array with same dimensionality as input.
    
    Raises
    ------
    ValueError
        If mode argument is not recognized.
    """
    if array.ndim == 1:
        return _dyadic_upsampling_1d(array)
    
    # Create array support
    # These are integer coordinates for the array
    x_support = n.arange(0, array.shape[0], step = 1)
    y_support = n.arange(0, array.shape[1], step = 1)
    interpolator = interpolate.RectBivariateSpline( x = x_support, y = y_support, z = array)
    
    # Create interpolated support
    interp_x_support = n.arange(0, array.shape[0], step = 1/2)
    interp_y_support = n.arange(0, array.shape[1], step = 1/2)
    
    return interpolator(x = interp_x_support, y = interp_y_support, grid = True)

def _dyadic_upsampling_1d(signal):
    """
    Upsamples a signal by a factor of 2 using cubic spline interpolation.
    
    Parameters
    ----------
    signal : ndarray, ndim 1
        Array to be upsampled.

    Returns
    -------
    out : ndarray, ndim 1
        Upsampled array
    """
    if signal.ndim == 2:
        return dyadic_upsampling(signal)

    signal = n.asarray(signal)
    support = n.arange(0, signal.shape[0], dtype = n.int)
    
    # Support for interpolated values
    interpolated_support = n.arange(0, signal.shape[0], step = 1/2)
    return n.interp(interpolated_support, support, signal)
    
def dyadic_downsampling(array):
    """
    Downsamples a 1D signal or 2D array by a factor of 2.
    
    Parameters
    ----------
    signal : ndarray
        Array to be downsampled.
    
    Returns
    -------
    out : ndarray, ndim 1
        Downsampled array
    """
    array = n.asarray(array)
    if array.ndim == 1:
        return array[::2]
    elif array.ndim == 2:
        return array[::2, ::2]
    else:
        raise ValueError('Dyadic downsampling supported for 1D or 2D arrays.')

def approx_rec(array, level, wavelet, array_mask = []):
    """
    Uses the multilevel discrete wavelet transform to decompose a signal or an image, 
    and reconstruct it using approximate coefficients only.
    
    Parameters
    ----------
    array : ndarray, ndim 1 or ndim 2
        Array to be decomposed.
    level : int
        Decomposition level. A higher level will result in a coarser approximation of
        the input array.
    wavelet : str or Wavelet object
        Can be any argument accepted by PyWavelet.Wavelet, e.g. 'db10'
            
    Returns
    -------
    reconstructed : ndarray
        Approximated reconstruction of the input array.
    """
    dim = array.ndim
    
    if dim == 1:
        dec_func = pywt.wavedec
        rec_func = pywt.waverec
    elif dim == 2:
        dec_func = pywt.wavedec2
        rec_func = pywt.waverec2

    original_array = n.array(array)         # Copy
    if isinstance(wavelet, str):
        wavelet = pywt.Wavelet(wavelet)
        
    # Check maximum decomposition level and interpolate if needed
    # Upsample the signal via interpolation as long as we cannot reach the desired
    # decomposition level
    upsampling_factor = 1
    while pywt.dwt_max_level(data_len = min(array.shape), filter_len = wavelet.dec_len) < level:
        array = dyadic_upsampling(array)
        upsampling_factor *= 2
        
    # By now, we are sure that the decomposition level will be supported.
    # Decompose the signal using the multilevel wavelet transform
    coeffs = dec_func(data = array, wavelet = wavelet, level = level, mode = extension_mode)
    app_coeffs, det_coeffs = coeffs[0], coeffs[1:]
    
    # Replace detail coefficients by 0; keep the right length so that the
    # reconstructed signal has the same length as the (possibly upsampled) signal
    # The structure of coefficients depends on the dimensionality
    zeroed = list()
    if dim == 1:
        for detail in det_coeffs:
            zeroed.append( n.zeros_like(detail) )
    elif dim == 2:
        for detail_tuples in det_coeffs:
            cHn, cVn, cDn = detail_tuples
            zeroed.append( (n.zeros_like(cHn), n.zeros_like(cVn), n.zeros_like(cDn)) )  
        
    # Reconstruct signal
    reconstructed = rec_func([app_coeffs] + zeroed, wavelet = wavelet, mode = extension_mode)
    
    # Downsample
    while upsampling_factor > 1:
        reconstructed = dyadic_downsampling(reconstructed)
        upsampling_factor /= 2
        
    # Adjust size of reconstructed signal so that it is the same size as input
    # TODO: clean up by removing special cases in terms of dimensionality?
    if reconstructed.size == original_array.size:
        return reconstructed
        
    elif original_array.size < reconstructed.size:
        if dim == 1:
            return reconstructed[:len(original_array)]
        elif dim == 2:
            return reconstructed[:original_array.shape[0], :original_array.shape[1]]
        
    elif original_array.size > reconstructed.size:
        extended_reconstructed = n.zeros_like(original_array, dtype = original_array.dtype)        
        if dim == 1:
            extended_reconstructed[:len(reconstructed)] = reconstructed
        elif dim == 2:
            extended_reconstructed[:reconstructed.shape[0], :reconstructed.shape[1]] = reconstructed
        return extended_reconstructed
        
def apply_mask(array, mask):
    """
    Interpolates the values of an array over the regions bounded by a rectangular mask.
    
    Parameters
    ----------
    array : ndarray, ndim 2
        Array
    mask : list of ints
        2D mask as rectangle. Syntax is [x1, x2, y1, y2]. Values of array[x1:x2, y1:y2]
        are interpolated from the edges of the mask.
    """
    x1, x2, y1, y2 = mask
    
    # support
    x_support, y_support = n.meshgrid(n.arange(0, array.shape[0]), n.arange())
    
    raise NotImplemented
    

def baseline(array, max_iter, level, wavelet = 'db10', background_regions = [], conv_tol = 1e-3):
    """
    Iterative method of baseline determination from [1].
    
    Parameters
    ----------
    array : ndarray
        Data with background. Can be either 1D signal or 2D array.
    max_iter : int
        Number of iterations to perform.
    level : int
        Decomposition level. A higher level will result in a coarser approximation of
        the input signal (read: a lower frequency baseline).
    wavelet : PyWavelet.Wavelet object or str, optional
        Wavelet with which to perform the algorithm. See PyWavelet documentation
        for available values. Default is 'db10'.
    background_regions : list, optional
        Indices of the array values that are known to be purely background. Depending
        on the dimensions of array, the format is different:
        
        ``array.ndim == 1``
          background_regions is a list of ints (indices) or slices
          E.g. >>> background_regions = [0, 7, 122, slice(534, 1000)]
          
        ``array.ndim == 2``
          background_regions is a list of tuples of ints (indices) or tuples of slices
          E.g. >>> background_regions = [(14, 19), (42, 99), (slice(59, 82), slice(81,23))]
          
    conv_tol : float, optional
        Convergence tolerance. If the sum square difference of the background
        between two steps is smaller than this tolerance, the algorithm stops.
    
    Returns
    -------
    baseline : ndarray
        Baseline of the input array.
    """

    # Initial starting point
    previous_background = n.zeros_like(array, dtype = array.dtype)
    background = n.array(array)
    
    for i in range(max_iter):
        
        background = approx_rec(array = background, level = level, wavelet = wavelet)
        
        # Check convergence
        if n.sum(n.abs(previous_background - background)**2) < conv_tol:
            break
        
        # Modify the background so it cannot be more than the signal itself
        # Set the background to be equal to the signal in the places where it
        # is more than the signal
        background_over_signal = (array - background) < 0.0
        background[background_over_signal] = array[background_over_signal]
        
        # Make sure the background values are equal to the signal values in the
        # background regions
        for index in background_regions:
            background[index] = array[index]
        
        # Save computation to check for convergence
        previous_background = n.array(background)
    
    return background

if __name__ == '__main__':
    from PIL import Image
    test_image = n.array(Image.open('graphite_test.tif'))
    bg = baseline(array = test_image, max_iter = 10, level = 5, wavelet = 'db10')