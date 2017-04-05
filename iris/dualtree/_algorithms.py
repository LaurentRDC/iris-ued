"""
Algorithms based on the dual-tree complex wavelet transform.

Author : Laurent P. Rene de Cotret
"""
from ._dtcwt import approx_rec, DEFAULT_FIRST_STAGE, DEFAULT_CMP_WAV
import numpy as n

__all__ = ['baseline', 'denoise']

def baseline(array, max_iter, level = 'max', first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV, 
             background_regions = [], mask = None, axis = -1):
    """
    Iterative method of baseline determination based on the dual-tree complex wavelet transform. Modified from [1]
    
    Parameters
    ----------
    array : ndarray, shape (M,N)
        Data with background. Can be either 1D signal or 2D array.
    max_iter : int
        Number of iterations to perform.
    level : int or 'max', optional
        Decomposition level. A higher level will result in a coarser approximation of
        the input signal (read: a lower frequency baseline). If 'max' (default), the maximum level
        possible is used.
    first_stage : str, optional
        Wavelet to use for the first stage. See dualtree.ALL_FIRST_STAGE for a list of suitable arguments
    wavelet : str, optional
        Wavelet to use in stages > 1. Must be appropriate for the dual-tree complex wavelet transform.
        See dualtree.ALL_COMPLEX_WAV for possible
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
        Mask array that evaluates to True for pixels that are invalid. 
    axis : int, optional
        Axis over which to compute the wavelet transform. Default is -1
    
    Returns
    -------
    baseline : ndarray, shape (M,N)
        Baseline of the input array.
    
    Raises
    ------
    NotImplementedError
        If input is a 2D array
    ValueError
        If input array is neither 1D nor 2D.
        
    References
    ----------
    [1] Galloway et al. 'An Iterative Algorithm for Background Removal in Spectroscopy by Wavelet Transforms', 
        Applied Spectroscopy pp. 1370 - 1376, September 2009.
    """   
    array = n.asarray(array, dtype = n.float)

    # Since dualtree() only works on even-length signals, we might have to extend.
    # See numpy.pad docs for a formatting of the padding tuple constructed below
    original_shape = array.shape
    if original_shape[axis] % 2 == 1:
        padding = [(0,0) for dim in original_shape]     # e.g. 2D : padding = [ (0,0), (0,0) ]
        padding[axis] = (0, 1)
        array = n.pad(array, tuple(padding), mode = 'constant')

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
        background[:] = approx_rec(array = signal, level = level, first_stage = first_stage, 
                                   wavelet = wavelet, axis = axis)
        
        # Modify the signal so it cannot be more than the background
        # This reduces the influence of the peaks in the wavelet decomposition
        signal[signal > background] = background[signal > background]
    
    # The background should be identically 0 where the data points are invalid
    background[mask] = 0 

    # Readjust size for odd input signals
    return n.resize(background, new_shape = original_shape)

def denoise(array, level = 1, first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV, mask = None):
    """
    Denoise an array using the dual-tree complex wavelet transform.
    
    Parameters
    ----------
    array : ndarray, ndim 1
        Data with background. 2D array support is in the works.
    level : int, optional
        Decomposition level. Higher level means that lower frequency noise is removed. Default is 1
    first_stage : str, optional
        Wavelet to use for the first stage. See dualtree.ALL_FIRST_STAGE for a list of suitable arguments
    wavelet : str, optional
        Wavelet to use in stages > 1. Must be appropriate for the dual-tree complex wavelet transform.
        See dualtree.ALL_COMPLEX_WAV for possible
    
    Returns
    -------
    denoised : ndarray, shape (M,N)

    Raises
    ------
    NotImplementedError
        If input array has dimension 2
    ValueError
        If input array has dimension > 2
    """
    #TODO: automatically determine 'level' by iterating; if the array hasn't
    #      changed much at a certain level, go a level further?
    array = n.asarray(array)
    if array.ndim == 2:
        raise NotImplementedError('2D array support is planned but not implemented.')
    elif array.ndim > 2:
        raise ValueError('{}D array denoising is not supported.'.format(array.ndim))
    
    if mask is None:
        mask = n.zeros_like(array, dtype = n.bool)
    
    return approx_rec(array = array, level = level, first_stage = first_stage, wavelet = wavelet, mask = mask)