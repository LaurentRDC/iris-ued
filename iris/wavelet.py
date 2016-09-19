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

def approx_rec(array, level, wavelet, mask = None):
    """
    Approximate reconstruction of a signal/image. Uses the multilevel discrete wavelet 
    transform to decompose a signal or an image, and reconstruct it using approximate 
    coefficients only.
    
    Parameters
    ----------
    array : array-like
        Array to be decomposed. Currently, only 1D and 2D arrays are supported.
        nD support is on the way.
    level : int or 'max' or None (deprecated)
        Decomposition level. A higher level will result in a coarser approximation of
        the input array. If the level is higher than the maximum possible decomposition level,
        the maximum level is used.
        If None, the maximum possible decomposition level is used.
    wavelet : str or Wavelet object
        Can be any argument accepted by PyWavelet.Wavelet, e.g. 'db10'
    mask : ndarray
        Same shape as array. Must evaluate to True where data is invalid.
            
    Returns
    -------
    reconstructed : ndarray
        Approximated reconstruction of the input array.
    
    Raises
    ------    
    NotImplementedError
        If input array has dimension > 2 
    """
    array = n.asarray(array, dtype = n.float)
    original_array = n.copy(array)
    
    # Choose deconstruction and reconstruction functions based on dimensionality
    # TODO: dim > 2
    dim = array.ndim
    if dim > 2:
        raise NotImplementedError
    func_dict = {1: (pywt.wavedec, pywt.waverec), 2: (pywt.wavedec2, pywt.waverec2)}
    dec_func, rec_func = func_dict[dim]

    # Build Wavelet object
    if isinstance(wavelet, str):
        wavelet = pywt.Wavelet(wavelet)
        
    # Check maximum decomposition level
    # For 2D array, check the condition with shortest dimension min(array.shape). This is how
    # it is done in PyWavelet.wavedec2.
    max_level = pywt.dwt_max_level(data_len = min(array.shape), filter_len = wavelet.dec_len)
    if level is None or level is 'max':
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
        
    elif original_array.size > reconstructed.size:
        extended_reconstructed = n.zeros_like(original_array, dtype = original_array.dtype)        
        if dim == 1:
            extended_reconstructed[:reconstructed.shape[0]] = reconstructed
        elif dim == 2:
            extended_reconstructed[:reconstructed.shape[0], :reconstructed.shape[1]] = reconstructed
        return extended_reconstructed


def baseline(array, max_iter, level = 'max', wavelet = 'sym6', background_regions = [], mask = None):
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
        Mask array that evaluates to True for pixels that are invalid. Useful to determine which pixels are masked
        by a beam block.
    
    Returns
    -------
    baseline : ndarray, shape (M,N)
        Baseline of the input array.
    
    Raises
    ------
    NotImplementedError
        If input array is neither 1D nor 2D. Inherited behavior from approx_rec.
    """
    array = n.asarray(array, dtype = n.float)
    if mask is None:
        mask = n.zeros_like(array, dtype = n.bool)
    
    signal = n.copy(array)
    background = n.zeros_like(array, dtype = n.float)
    for i in range(max_iter):
        
        # Make sure the background values are equal to the original signal values in the
        # background regions
        for index in background_regions:
            signal[index] = array[index]
        
        # Wavelet reconstruction using approximation coefficients
        background = approx_rec(array = signal, level = level, wavelet = wavelet, mask = mask)
        
        # Modify the signal so it cannot be more than the background
        # This reduces the influence of the peaks in the wavelet decomposition
        signal[signal > background] = background[signal > background]
    
    # The background should be identically 0 where the data points are invalid
    background[mask] = 0  
    return background
    
def denoise(array, level = 1, wavelet = 'db5', mask = None):
    """
    Denoise an array using the wavelet transform.
    
    Parameters
    ----------
    array : ndarray, shape (M,N)
        Data with background. Can be either 1D signal or 2D array.
    level : int, optional
        Decomposition level. Higher level means that lower frequency noise is removed. Default is 1
    wavelet : PyWavelet.Wavelet object or str, optional
        Wavelet with which to perform the algorithm. See PyWavelet documentation
        for available values. Default is 'db5'.
    
    Returns
    -------
    out : ndarray, shape (M,N)
    """
    if mask is None:
        mask = n.zeros_like(array, dtype = n.bool)

    return approx_rec(array = array, level = level, wavelet = wavelet, mask = mask)

def enhance(array, level = 1, wavelet = 'db5', mask = None):
    """
    Enhance an array by denoising and removing background.

    Parameters
    ----------
    array : ndarray, shape (M,N)
        Data with background. Can be either 1D signal or 2D array.
    level : int, optional
        Decomposition level. Higher level means that lower frequency noise is removed. Default is 1
    wavelet : PyWavelet.Wavelet object or str, optional
        Wavelet with which to perform the algorithm. See PyWavelet documentation
        for available values. Default is 'db5'.
    mask : ndarray, dtype bool, optional
        Evaluates to True on array values that are invalid.

    Returns
    -------
    enhanced : ndarray, shape (M,N)
    """
    if mask is None:
        mask = n.zeros_like(array, dtype = n.bool)
    
    return denoise(array, level = level, wavelet = wavelet, mask = mask) - baseline(array, max_iter = 50, level = None, wavelet = wavelet, mask = mask)
    
if __name__ == '__main__':
    from iris import dataset
    import matplotlib.pyplot as plt
    directory = 'K:\\2012.11.09.19.05.VO2.270uJ.50Hz.70nm'
    d = dataset.PowderDiffractionDataset(directory)  
    signal = d.pattern(0.0).data
    bg1 = baseline(signal, 1000, wavelet = 'sym6')
    plt.plot(signal, 'b', bg1, 'r')