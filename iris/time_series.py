"""
Time-series analysis.

@author: Laurent P. René de Cotret
"""

import numpy as n

try:
    from numpy.fft_intel import fft, ifft, rfft, irfft, fftshift
except ImportError:
    from scipy.fftpack import fft, ifft, rfft, irfft, fftshift

def chirpz(signal, A = 1, W = None, M = None):
    """
    Chirp Z-transform.

    Parameters
    ----------
    signal : ndarray

    A : float or complex, optional

    W : float or complex, optional 

    M : int or None, optional

    Returns
    -------
    out : ndarray
    """
    if M is None:
        M = signal.size
    if W is None:
        W = n.exp(2j*n.pi/M)

    A, W = n.complex(A), n.complex(W)
    signal = n.asarray(signal, dtype = n.complex)

    # To speed things up, determine the smallest padding to get the
    # signal size to a power of two
    padded = int(2**n.ceil(n.log2(M + signal.size - 1)))

    integers = n.arange(signal.size)
    y = n.power(A, -integers) * n.power(W, 0.5 * integers**2) * signal
    Y = fft(y, n = padded)

    v = 


def autocorr1d(arr, axis = -1):
    """
    Autocorrelation via FFTs

    Parameters
    ----------
    arr : ndarray
    
    axis : int, optional
        Axis along which to compute autocorrelation. Default
        is last axis.
    
    Returns
    -------
    out : ndarray
        autocorrelation, same shape as input.
    """
    # Bring autocorrelation axis to front
    # since arr is real, no need to complex conjugate
    arr = n.swapaxes(arr, axis, 0)

    f = rfft(arr, axis = 0)
    f *= rfft(arr[::-1], axis = 0)
    out = irfft(f, axis = 0)

    out[:] = fftshift(out, axes = (0,))
    return n.swapaxes(out, 0, axis)

