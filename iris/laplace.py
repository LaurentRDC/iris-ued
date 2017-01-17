
import numpy as n
from scipy.fftpack import fftfreq, fft, fftshift, ifftshift

def heaviside(x):
    """ 
    Heaviside step function

    Parameters
    ----------
    x : iterable or numerical
    """
    return 1 * (n.array(x) >= 0)

def laplace(x, y, sigma = 0):
    """
    Laplace transform of y = f(x).

    Parameters
    ----------
    x, y : ndarrays, shapes (N,)
        Input y = f(x).
    sigma : float
        Real-part of the Laplace transform variable s.
        Default is 0, which makes the Laplace transform equivalent to the
        Fourier transform.

    Returns
    -------
    out : ndarray, shape (M,N) , dtype complex

    Raises
    ------
    ValueError
        If x and y don't have the same shape.
    """
    # Reshape arrays to vectorize fft for multiple sigmas
    # If input sigma is numerical, change to (1,1) array
    # Sigma rows along rows, x and y along colums
    if isinstance(sigma, (int, float)):
        sigma = (sigma,)
    sigma = n.asarray(sigma, dtype = n.float)[:, None] 
    x, y = x[None, :], y[None, :]

    # We include a factor of 2 pi to sigma because fft uses exp(2 pi 1j k t)
    return fftshift(fft(heaviside(x) * y * n.exp(-2*n.pi*sigma*x), axis = 1), axes = 1)