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
from numpy import convolve
import pywt


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
    array = n.asarray(array)
    if mode == 'discard':
        return array[::factor]
        
    elif mode == 'average':
        if (len(array) % factor):
            print('Array shape not a multiple of factor. Discard mode is used.')
            return downsample(array, factor, mode = 'discard')
        array.shape = (len(array)/factor, factor)
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
    
def multi_level_dwt(signal, level, wavelet):
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
    """
    if isinstance(wavelet, str):
        wavelet = pywt.Wavelet(wavelet)
    
    # deconstruction coefficients
    lpk, hpk, _, _ = wavelet.filter_bank 
    
    # Loop preparation
    result = [[]]*(level+1)
    signal_temp = signal[:]
    
    for i in range(level):
        # Filter signals
        low_pass_signal = convolve(signal_temp, lpk, mode = 'same')
        high_pass_signal = convolve(signal_temp, hpk, mode = 'same')

        # Downsample both output by half. See Nyquist theorem
        low_pass_downsampled = downsample(low_pass_signal, factor = 2, mode = 'discard')
        high_pass_downsampled = downsample(high_pass_signal, factor = 2, mode = 'discard')
        
        # Store the high pass signal (also known as detail coefficients)
        # Further decompose the low-pass (or approximate) signal
        result[level-i] = high_pass_downsampled
        signal_temp = low_pass_downsampled[:]
    
    result[0] = low_pass_downsampled
    return result

def multi_level_idwt(coefficients, level, wavelet):
    """
    Computes the multi-level inverse discrete wavelet transform.
    
    Parameters
    ----------
    coefficients : list
        Discrete signal to be approximated.
    level : int
        Level of decomposition. A higher level means that the approximation will be coarser.
    wavelet : str or PyWavelet.Wavelet object
        Wavelet. See PyWavelet documentation for details.
    """
    if isinstance(wavelet, str):
        wavelet = pywt.Wavelet(wavelet)
        
    # Reconstruction coefficient    
    _, _, lpk, hpk = wavelet.filter_bank
    
    # Reconstruct level by level
    iter_coeff = coefficients[:]
    for i in range(level):
        low_pass_coeff = iter_coeff[0]
        high_pass_coeff = iter_coeff[1]
        
        # Verify new length
        if len(low_pass_coeff) > len(high_pass_coeff):
            length = 2*len(high_pass_coeff)
        else:
            length = 2*len(low_pass_coeff)

        # Upsampling by 2
        low_pass_upsampled = upsample(low_pass_coeff, factor = 2, length = length)
        high_pass_upsampled = upsample(high_pass_coeff, factor = 2, length = length)

        # Convolve with reconstruction coefficient
        lpc = convolve(low_pass_upsampled, lpk, mode = 'same')
        hpc = convolve(high_pass_upsampled, hpk, mode = 'same')

        # Add both signal reconstruction for this level
        signal = lpc + hpc
        
        if len(iter_coeff) > 2:
            iter_coeff = [signal] + iter_coeff[2:]
        else:
            iter_coeff = [signal]

    return iter_coeff[0]

def approx_reconstruction(signal, level, wavelet):
    """
    Computes the multi-level discrete wavelet transform of a signal, and reconstructs
    the signal using only the approximate (low-pass) coefficients.
    
    Parameters
    ----------
    signal : ndarray, ndim 1
        Discrete signal to be approximated.
    level : int
        Level of decomposition. A higher level means that the approximation will be coarser.
    wavelet : str or PyWavelet.Wavelet object
        Wavelet. See PyWavelet documentation for details.
    """
    coefficients = multi_level_dwt(signal = signal, level = level, wavelet = wavelet)
    
    # Separate the approximate coefficients from the detail coefficients
    approx_coefficients = coefficients[0]
    detail_coefficients = coefficients[1:]
    
    # Set detail coefficients to zero
    detail_coefficients = n.zeros_like(n.array(detail_coefficients))
    print(detail_coefficients)
    
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
    
    for i in range(niter):
        approximation = approx_reconstruction(background, level = level, wavelet = wavelet)
        
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
    base = n.arange(0,10, 0.1)
    signal = n.sin(base)# + n.random.normal(loc = 0, scale = 0.1, size = base.shape)
    bg = multi_level_dwt(signal, level = 1, wavelet = 'db10')
    rec = multi_level_idwt(bg, level = 1, wavelet = 'db10')
    plt.plot(signal)
    plt.plot(rec)
    approx_reconstruction(signal, level = 2, wavelet = 'db10')