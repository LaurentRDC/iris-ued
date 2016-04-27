# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:57:31 2016

@author: Laurent P. René de Cotret

References
----------
[1] A wavelet transform algorithm for peak detection and application to powder 
    x-ray diffraction data. Gregoire, Dale, and van Dover

[2] An Iterative Algorithm for Background Removal in Spectroscopy by 
    Wavelet Transforms. Galloway, Le Ru and Etchegoin
"""
import numpy as n
import pywt
from numpy import convolve
    
def _zero_pad_to_length(array, length):
    """
    Extends a 1D array to a specific length
    """
    array = n.asarray(array)
    if len(array) >= length:
        return array
    else:
        new_array = n.zeros( shape = (length,), dtype = array.dtype )
        new_array[:len(array)] = array
        return new_array

def downsample(array, factor = 2, mode = 'discard'):
    """
    Downsample (by averaging) a vector by an integer factor.
    
    Parameters
    ----------
    array : array-like, ndim 1
        Array to be downsampled
    factor : int, optional
        Downsampling factor. Default is 2
    mode : str {'discard' (default), 'average'}, optional
        If 'average', the array values are averaged to yield the correct downsampling
        shape. If 'discard', numbers from array are discarded completely.
    
    Returns
    -------
    out : ndarray
    """
    # Extend arrays not divisible by the factor via zero padding
    # Recursively extend the array until it works...
    if (len(array) % factor):
        new_array = n.zeros( shape = (len(array) + 1,) )
        new_array[:len(array)] = array
        return downsample(new_array, factor, mode)
        
    array = n.asarray(array)
    if mode == 'discard':
        return array[::factor]
        
    elif mode == 'average':
        array.shape = (int(len(array)/factor), factor)
        return n.mean(array, axis = 1)
    
    else:
        raise ValueError('mode argument not recognized')

def upsample(array, factor = 2, length = None):
    """
    Upsamples an array by zero-padding.
    
    Parameters
    ----------
    array : array-like, ndim 1
        Array to be upsampled
    factor : int, optional
        Upsampling factor. Default is 2
    length : int, optional
        If not None (default), the output array is set to be this length by zero-padding.
        
    Returns
    -------
    out : ndarray
    """
    array = n.asarray(array)
    output = n.zeros( shape = (factor*len(array),), dtype = array.dtype)
    output[::factor] = array
    
    if length is None:
        return output
    else:
        if len(output) > length:    # Cut the array
            return output[:length]
        else:                       # Zero pad the array
            zero_padded = n.zeros(shape = (length, ) )
            zero_padded[:len(output)] = output
            return zero_padded
    
def dwt(signal, level, wavelet):
    """
    Computes the multi-level discrete wavelet transform.
    
    Parameters
    ----------
    signal : ndarray, ndim 1
        Discrete signal to be approximated.
    level : int
        Level of decomposition. A higher level means that the approximation will be coarser.
    wavelet : str or PyWavelet.Wavelet object
        Wavelet. See PyWavelet documentation for details.
    
    Returns
    -------
    coeff_list : list of ndarrays
        Signal coefficients. The order is given so that coeff_list = [c_A_n, c_D_n, ..., c_D_1]
    """
    if isinstance(wavelet, str):
        wavelet = pywt.Wavelet(wavelet)
    
    # deconstruction coefficients
    lpk, hpk, _, _ = wavelet.filter_bank
    assert len(lpk) == len(hpk)
    
    # Loop preparation
    result = list()
    signal_temp = n.copy(signal)
    
    for i in range(level):
        # Filter signals
        low_pass_signal = convolve(signal_temp, lpk, mode = 'same')
        high_pass_signal = convolve(signal_temp, hpk, mode = 'same')

        # Downsample both output by half. See Nyquist theorem
        # Downsampling using average mode results in better reconstruction
        low_pass_downsampled = downsample(low_pass_signal, factor = 2, mode = 'average')
        high_pass_downsampled = downsample(high_pass_signal, factor = 2, mode = 'average')
        
        # Force downsampled signals to have the same length
        if len(low_pass_downsampled) != len(high_pass_downsampled):
            length = max(len(low_pass_downsampled), len(high_pass_downsampled))
            low_pass_downsampled = _zero_pad_to_length(low_pass_downsampled, length)
            high_pass_downsampled = _zero_pad_to_length(high_pass_downsampled, length)
        
        # Store the high pass signal (also known as detail coefficients)
        # Further decompose the low-pass (or approximate) signal
        result.append( high_pass_downsampled )
        signal_temp = n.copy(low_pass_downsampled)
    
    result.append( low_pass_downsampled )
    result.reverse()        #For compatibility with MATLAB Wavelet toolbox and PyWavelet
    return result

def idwt(coefficients, wavelet, approximate = False):
    """
    Computes the multi-level inverse discrete wavelet transform.
    
    Parameters
    ----------
    coeff_list : list
        Discrete signal to be approximated.
    wavelet : str or PyWavelet.Wavelet object
        Wavelet. See PyWavelet documentation for details.
    approximate : bool, optional
        If True, only approximate coefficients of the maximum level are used
        in the reconstruction. Default is False.
    """
    # shortcut for using wavelets
    if isinstance(wavelet, str):
        wavelet = pywt.Wavelet(wavelet)
        
    # Reconstruction coefficient    
    _, _, lpk, hpk = wavelet.filter_bank
    assert len(lpk) == len(hpk)
    
    reconstructed, detail_coeffs = coefficients[0], coefficients[1:]
    for detail_coeff in detail_coeffs:
        
        # Due to downsampling and upsampling, reconstructed arrays might change length.
        # Zero pad arrays to make them the same length
        if len(reconstructed) < len(detail_coeff):
            reconstructed = _zero_pad_to_length(reconstructed, len(detail_coeff))
        elif len(reconstructed) > len(detail_coeff):
            detail_coeff = _zero_pad_to_length(detail_coeff, len(reconstructed))
        assert len(reconstructed) == len(detail_coeff)
        
        # If approximate reconstruction, set the detail coefficients to 0
        if approximate:
            detail_coeff = n.zeros( shape = (len(detail_coeff),) )
            
        # Upsample by 2
        low_pass_upsampled = upsample(reconstructed, factor = 2)
        high_pass_upsampled = upsample(detail_coeff, factor = 2)
        
        # Convolve with reconstruction coefficient
        low_pass_convolved = convolve(low_pass_upsampled, lpk, mode = 'same')
        high_pass_convolved = convolve(high_pass_upsampled, hpk, mode = 'same')
        
        # Prepare next iteration
        reconstructed = low_pass_convolved + high_pass_convolved
    
    return reconstructed

def approx_rec(signal, level, wavelet):
    """
    Performs the multi-level discrete wavelet transform and reconstructs the signal
    using only approximate coefficients of the appropriate level
    
    Parameters
    ----------
    signal : ndarray, ndim 1
        Discrete signal to be approximated.
    level : int
        Level of decomposition. A higher level means that the approximation will be coarser.
    wavelet : str or PyWavelet.Wavelet object
        Wavelet. See PyWavelet documentation for details.
    """
    coeffs = dwt(signal, level, wavelet)
    reconstructed = idwt(coeffs, wavelet, approximate = True)
    
    return _zero_pad_to_length(reconstructed, len(signal))
    
def baseline(signal, niter = 10, level = None, wavelet = 'db10', background_regions = []):
    """
    Iterative method of baseline determination from [2].
    
    Parameters
    ----------
    signal : ndarray, ndim 1
        Signal with background.
    niter : int
        Number of iterations to perform.
    wavelet : PyWavelet.Wavelet object or str, optional
        Wavelet with which to perform the algorithm. See PyWavelet documentation
        for available values. Default is 'db10'.
    background_regions : list of ints, optional
        Indices of the signal values that are purely background.
    """
    # Initial starting point
    background = n.array(signal)
    
    # TODO: make sure background and signal always have the same shape
    # Idea: make approx_rec output be the same shape as input
    for i in range(niter):
        background = approx_rec(signal = background, level = level, wavelet = wavelet)
        
        # Modify the background so it cannot be more than the signal itself
        # Set the background to be equal to the signal in the places where it
        # is more than the signal
        background_over_signal = (signal - background) < 0.0
        background[background_over_signal] = signal[background_over_signal]
        
        # Make sure the background values are equal to the signal values in the
        # background regions
        for index in background_regions:
            background[index] = signal[index]
    
    return background
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    base = n.arange(0,10.5, 0.01)
    signal = n.sin(base)# + n.random.normal(loc = 0, scale = 0.1, size = base.shape)
    bg = dwt(signal, level = 3, wavelet = 'db10')
    rec = idwt(bg, wavelet = 'db10')
    
    #pyrec = pywt.waverec(bg, 'db10')
    
    plt.plot(signal)
    plt.plot(rec)
    #approx_reconstruction(signal, level = 2, wavelet = 'db10')