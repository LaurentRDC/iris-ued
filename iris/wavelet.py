# -*- coding: utf-8 -*-
"""
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
        If input array is neither 1D nor 2D.
    """
    if array.ndim == 1:
        signal = n.asarray(array)
        support = n.arange(0, signal.shape[0], dtype = n.int)
        
        # Support for interpolated values
        interpolated_support = n.arange(0, signal.shape[0], step = 1/2)
        return n.interp(interpolated_support, support, signal)
    
    elif array.ndim == 2:
        # Create array support
        # These are integer coordinates for the array
        x_support = n.arange(0, array.shape[0], step = 1)
        y_support = n.arange(0, array.shape[1], step = 1)
        interpolator = interpolate.RectBivariateSpline( x = x_support, y = y_support, z = array)
        
        # Create interpolated support
        interp_x_support = n.arange(0, array.shape[0], step = 1/2)
        interp_y_support = n.arange(0, array.shape[1], step = 1/2)
        
        return interpolator(x = interp_x_support, y = interp_y_support, grid = True)
    
    else:
        raise ValueError('Input array must be 1D or 2D.')
    
def dyadic_downsampling(array):
    """
    Downsamples a 1D signal or 2D array by a factor of 2.
    
    Parameters
    ----------
    signal : ndarray
        Array to be downsampled.
    
    Returns
    -------
    out : ndarray
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
    # Choose deconstruction and reconstruction functions based on dimensionality
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
    # For 2D array, check the condition with shortest dimension min(array.shape). This is how
    # it is done in PyWavelet.wavedec2.
    upsampling_factor = 1
    while pywt.dwt_max_level(data_len = min(array.shape), filter_len = wavelet.dec_len) < level:
        array = dyadic_upsampling(array)
        upsampling_factor *= 2
        
    # By now, we are sure that the decomposition level will be supported.
    # Decompose the signal using the multilevel discrete wavelet transform
    coeffs = dec_func(data = array, wavelet = wavelet, level = level, mode = extension_mode)
    app_coeffs, det_coeffs = coeffs[0], coeffs[1:]
    
    # Replace detail coefficients by 0; keep the correct length so that the
    # reconstructed signal has the same size as the (possibly upsampled) signal
    # The structure of coefficients depends on the dimensionality
    zeroed = list()
    if dim == 1:
        for detail in det_coeffs:
            zeroed.append( n.zeros_like(detail) )
    elif dim == 2:
        for detail_tuples in det_coeffs:
            cHn, cVn, cDn = detail_tuples       # See PyWavelet.wavedec2 documentation for coefficients structure.
            zeroed.append( (n.zeros_like(cHn), n.zeros_like(cVn), n.zeros_like(cDn)) )  
        
    # Reconstruct signal
    reconstructed = rec_func([app_coeffs] + zeroed, wavelet = wavelet, mode = extension_mode)
    
    # Downsample
    while upsampling_factor > 1:
        reconstructed = dyadic_downsampling(reconstructed)
        upsampling_factor /= 2
        
    # Adjust size of reconstructed signal so that it is the same size as input
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


def baseline(array, max_iter, level, wavelet = 'db10', background_regions = [], mask = None):
    """
    Iterative method of baseline determination modified from [1]. This function handles
    both 1D radial patterns and 2D single-crystal diffraction images.
    
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
         
         Default is empty list.
    
    mask : ndarray, dtype bool, optional
        Mask array that evaluates to True for pixels that are valid. Only available
        for 2D arrays (i.e. images). Useful to determine which pixels are masked
        by a beam block.
    
    Returns
    -------
    baseline : ndarray
        Baseline of the input array.
    
    Raises
    ------
    NotImplemented
        If a 2D array is provided with a non-trivial mask.
    
    Notes
    -----
    While the algorithm presented in [1] modifies the input signal to never be over
    the background, this function does the opposite: the background is forced to
    not exceed the input. This yields identical results to [1].
    """
    if (array.ndim == 2) and (mask is not None):
        raise NotImplemented
    
    # In case a mask is not provided, all data points are valid
    # Masks are also not available for 1D signals
    if (mask is None) or array.ndim == 1:
        mask = n.ones_like(array, dtype = n.bool)

    # Initial starting point
    background = n.array(array)
    
    for i in range(max_iter):
        
        # TODO: check convergence after each iteration?
        background = approx_rec(array = background, level = level, wavelet = wavelet)
        
        # Modify the background so it cannot be more than the signal itself
        # Set the background to be equal to the signal in the places where it
        # is more than the signal
        background_over_signal = (array - background)*mask < 0.0    # Background where mask is false is invalid
        background[background_over_signal] = array[background_over_signal]
        
        # Make sure the background values are equal to the signal values in the
        # background regions
        # This reduces the influence of the array peaks from the baseline
        for index in background_regions:
            background[index] = array[index]
    
    # The background should be identically 0 where the data points are invalid
    # A boolean ndarray is 0 where False, and 1 where True
    # Therefore, simple multiplication does what we want
    return background*mask
    
    
def denoise(array, level, wavelet = 'db20'):
    """
    Denoise an array using the wavelet transform.
    """
    #TODO: this
    pass

    
if __name__ == '__main__':
    from iris import dataset
    import matplotlib.pyplot as plt
    directory = 'K:\\2012.11.09.19.05.VO2.270uJ.50Hz.70nm'
    d = dataset.PowderDiffractionDataset(directory)
    p = d.pattern(0.0)
    bg = baseline(p.data, max_iter = 200, level = 10)
    plt.plot(p.data, 'b', bg, 'r')