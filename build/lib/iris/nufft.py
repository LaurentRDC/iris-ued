""" 
Non-uniform Fast Fourier Transform for Time-series Analysis
"""

import numpy as np

def nufftfreq(n, d = 1.0):
    """ 
    Returns the frequency grid on which to compute the NUFFT
    API is same as numpy.fft.fftfreq.

    Parameters
    ----------
    n : int
        Window length
    d : float, optional
        Sample spacing (inverse of sampling rate).
    
    Returns
    -------
    out : ndarray
        Array of length n containing sample frequencies.
    """
    return d*np.arange( -(n/2), n - (n/2))

def nufft(x, y):
    """
    Non-uniform Fast Fourier Transform of type 1

    Parameters
    ----------
    x, y : ndarrays, shape (N,)

    d : float, optional
        Sample spacing
    """
    k = nufftfreq(len(x), d = np.diff(x).mean())
    return k, (1/len(x)) * np.dot(y, np.exp(-1j * k * x[:,None]))