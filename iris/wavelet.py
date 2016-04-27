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

def dyadic_upsampling(signal):
    """
    Upsamples a signal by a factor of 2 using cubic spline interpolation.
    
    Parameters
    ----------
    signal : ndarray, ndim 1
    
    Returns
    -------
    out : ndarray, ndim 1
        Upsampled array
    
    See also
    --------
    scipy.interpolate
        Interpolation packakge
    """
    signal = n.asarray(signal)
    support = n.arange(0, len(signal), dtype = n.int)
    
    # Support for interpolated values
    interpolated_support = n.arange(0, len(signal), step = 1/2)
    
    # Interpolation
    tck = interpolate.splrep(support, signal, s = 0)
    return interpolate.splev(interpolated_support, tck, der = 0)

def dyadic_downsampling(signal):
    """
    Downsamples a signal by a factor of 2.
    
    Parameters
    ----------
    signal : ndarray, ndim 1
    
    Returns
    -------
    out : ndarray, ndim 1
        Downsampled array
    """
    signal = n.asarray(signal)
    return signal[::2]

def approx_rec(signal, level, wavelet):
    """
    Uses the multilevel discrete wavelet transform to decompose a signal, and
    reconstruct it using approximate coefficients only.
    
    Parameters
    ----------
    signal : ndarray, shape (N,)
        Signal to be decomposed.
    level : int
        Decomposition level. A higher level will result in a coarser approximation of
        the input signal.
    wavelet : str or Wavelet object
        Can be any argument accepted by PyWavelet.Wavelet, e.g. 'db10'
    
    Returns
    -------
    reconstructed : ndarray, shape (N,)
        Approximated reconstruction of the input signal.
    """
    original_signal = n.array(signal)
    if isinstance(wavelet, str):
        wavelet = pywt.Wavelet(wavelet)
    
    # Check maximum decomposition level and interpolate if needed
    # Upsample the signal via interpolation as long as we cannot reach the desired
    # decomposition level
    upsampling_factor = 1
    while pywt.dwt_max_level(data_len = len(signal), filter_len = wavelet.dec_len) < level:
        signal = dyadic_upsampling(signal)
        upsampling_factor *= 2
    
    # By now, we are sure that the decomposition level will be supported.
    # Decompose the signal using the multilevel wavelet transform
    coeffs = pywt.wavedec(data = signal, wavelet = wavelet, level = level)
    app_coeffs, det_coeffs = coeffs[0], coeffs[1:]
    
    # Replace detail coefficients by 0; keep the right length so that the
    # reconstructed signal has the same length as the (possibly upsampled) signal
    zeroed = list()
    for detail in det_coeffs:
        zeroed.append( n.zeros(shape = (len(detail),) ) )
    
    # Reconstruct signal
    reconstructed = pywt.waverec([app_coeffs] + zeroed, wavelet = wavelet)
    
    # Downsample
    while upsampling_factor > 1:
        reconstructed = dyadic_downsampling(reconstructed)
        upsampling_factor /= 2
    
    #Adjust size of reconstructed signal so that it is the same length as input
    if original_signal.size < reconstructed.size:
        return reconstructed[:len(original_signal)]
    elif original_signal.size > reconstructed.size:
        extended_reconstructed = n.zeros_like(original_signal, dtype = original_signal.dtype)
        extended_reconstructed[:len(reconstructed)] = reconstructed
        return extended_reconstructed
    else:
        return reconstructed
    
def baseline(signal, niter = 10, level = 5, wavelet = 'db10', background_regions = []):
    """
    Iterative method of baseline determination from [1].
    
    Parameters
    ----------
    signal : ndarray, shape (N,)
        Signal with background.
    niter : int
        Number of iterations to perform.
    level : int
        Decomposition level. A higher level will result in a coarser approximation of
        the input signal (read: a lower frequency baseline).
    wavelet : PyWavelet.Wavelet object or str, optional
        Wavelet with which to perform the algorithm. See PyWavelet documentation
        for available values. Default is 'db10'.
    background_regions : list of ints, optional
        Indices of the signal values that are purely background.
    
    Returns
    -------
    baseline : ndarray, shape (N,)
        Baseline of the input signal.
    """
    # Initial starting point
    background = n.array(signal)
    
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