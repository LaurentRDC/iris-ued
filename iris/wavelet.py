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
from warnings import warn

__all__ = ['approx_rec', 'baseline', 'denoise']

# Boundary extension mode
EXTENSION_MODE = 'constant'

def approx_rec(array, level, wavelet, array_mask = []):
    """
    Approximate reconstruction of a signal/image. Uses the multilevel discrete wavelet 
    transform to decompose a signal or an image, and reconstruct it using approximate 
    coefficients only.
    
    Parameters
    ----------
    array : ndarray
        Array to be decomposed. Currently, only 1D and 2D arrays are supported.
        nD support is on the way.
    level : int or None
        Decomposition level. A higher level will result in a coarser approximation of
        the input array. If the level is higher than the maximum possible decomposition level,
        the maximum level is used.
        If None, the maximum possible decomposition level is used.
    wavelet : str or Wavelet object
        Can be any argument accepted by PyWavelet.Wavelet, e.g. 'db10'
            
    Returns
    -------
    reconstructed : ndarray
        Approximated reconstruction of the input array.
    
    Raises
    ------    
    NotImplementedError
        If input array has dimension > 2 
    """
    original_array = n.array(array)         # Copy
    
    # Choose deconstruction and reconstruction functions based on dimensionality
    dim = array.ndim 
    if dim == 1:
        dec_func, rec_func = pywt.wavedec, pywt.waverec
    elif dim == 2:
        dec_func, rec_func = pywt.wavedec2, pywt.waverec2
    elif dim > 2:
        raise NotImplementedError
        dec_func, rec_func = pywt.wavedecn, pywt.waverecn

    # Build Wavelet object
    if isinstance(wavelet, str):
        wavelet = pywt.Wavelet(wavelet)
        
    # Check maximum decomposition level
    # For 2D array, check the condition with shortest dimension min(array.shape). This is how
    # it is done in PyWavelet.wavedec2.
    max_level = pywt.dwt_max_level(data_len = min(array.shape), filter_len = wavelet.dec_len)
    if level is None:
        level = max_level
    elif max_level < level:
        warn('Decomposition level {} higher than maximum {}. Maximum is used.'.format(level, max_level))
        level = max_level
        
    # By now, we are sure that the decomposition level will be supported.
    # Decompose the signal using the multilevel discrete wavelet transform
    coeffs = dec_func(data = array, wavelet = wavelet, level = level, mode = EXTENSION_MODE)
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
    elif dim > 2:
        pass
        #TODO: set the detail coefficients to zero
        
    # Reconstruct signal
    reconstructed = rec_func([app_coeffs] + zeroed, wavelet = wavelet, mode = EXTENSION_MODE)
    
    # Adjust size of reconstructed signal so that it is the same size as input
    if reconstructed.size == original_array.size:
        return reconstructed
        
    elif original_array.size < reconstructed.size:
        if dim == 1:
            return reconstructed[:original_array.shape[0]]
        elif dim == 2:
            return reconstructed[:original_array.shape[0], :original_array.shape[1]]
        elif dim > 2:
            pass
        
    elif original_array.size > reconstructed.size:
        extended_reconstructed = n.zeros_like(original_array, dtype = original_array.dtype)        
        if dim == 1:
            extended_reconstructed[:reconstructed.shape[0]] = reconstructed
        elif dim == 2:
            extended_reconstructed[:reconstructed.shape[0], :reconstructed.shape[1]] = reconstructed
        elif dim > 2:
            pass
        return extended_reconstructed


def baseline(array, max_iter, level = None, wavelet = 'sym6', background_regions = [], mask = None):
    """
    Iterative method of baseline determination modified from [1]. This function handles
    both 1D curves and 2D images.
    
    Parameters
    ----------
    array : ndarray, shape (M,N)
        Data with background. Can be either 1D signal or 2D array.
    max_iter : int
        Number of iterations to perform.
    level : int or None, optional
        Decomposition level. A higher level will result in a coarser approximation of
        the input signal (read: a lower frequency baseline). If None (default), the maximum level
        possible is used.
    wavelet : PyWavelet.Wavelet object or str, optional
        Wavelet with which to perform the algorithm. See PyWavelet documentation
        for available values. Default is 'sym6'.
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
    baseline : ndarray, shape (M,N)
        Baseline of the input array.
    
    Raises
    ------
    ValueError
        If input array is neither 1D nor 2D. Inherited behavior from approx_rec.
    NotImplementedError
        If a 2D array is provided with a non-trivial mask.
    """
    if (array.ndim == 2) and (mask is not None):
        raise NotImplementedError
        
    # In case a mask is not provided, all data points are valid
    # Masks are also not available for 1D signals
    if (mask is None) or array.ndim == 1:
        mask = n.ones_like(array, dtype = n.bool)

    # Initial starting point
    if max_iter == 0:
        return array*mask
    signal = n.array(array)
    
    for i in range(max_iter):
        
        # Make sure the background values are equal to the original signal values in the
        # background regions
        for index in background_regions:
            signal[index] = array[index]
        
        # Wavelet reconstruction using approximation coefficients
        background = approx_rec(array = signal, level = level, wavelet = wavelet)
        
        # Modify the signal so it cannot be more than the background
        # This reduces the influence of the peaks in the wavelet decomposition
        signal_over_background = (signal - background)*mask > 0.0    # Background where mask is false is invalid
        signal[signal_over_background] = background[signal_over_background]
    
    # The background should be identically 0 where the data points are invalid
    # A boolean ndarray is 0 where False, and 1 where True
    # Therefore, simple multiplication does what we want        
    return background*mask
    
    
def denoise(array, wavelet = 'db5'):
    """
    Denoise an array using the wavelet transform.
    
    Parameters
    ----------
    array : ndarray, shape (M,N)
        Data with background. Can be either 1D signal or 2D array.
    wavelet : PyWavelet.Wavelet object or str, optional
        Wavelet with which to perform the algorithm. See PyWavelet documentation
        for available values. Default is 'db5'.
    
    Returns
    -------
    out : ndarray, shape (M,N)
    """
    return approx_rec(array = array, level = 1, wavelet = wavelet)
    
if __name__ == '__main__':
    from iris import dataset
    import matplotlib.pyplot as plt
    directory = 'K:\\2012.11.09.19.05.VO2.270uJ.50Hz.70nm'
    d = dataset.PowderDiffractionDataset(directory)  
    signal = d.pattern(0.0).data
    bg1 = baseline(signal, 1000, wavelet = 'sym6')
    plt.plot(signal, 'b', bg1, 'r')