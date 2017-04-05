from math import sqrt, log  # Faster than numpy on non-arrays
import numpy as n
from scipy.optimize import curve_fit
from scipy.special import erf
import sys

def cdf(x, amplitude, std, center = 0, offset = 0):
    """ Cumulative distribution function for the normal distribution """
    return amplitude * 0.5 * (1 + erf( (x - center)/(std * sqrt(2)))) + offset

def knife_edge(x, y, fit_parameters = dict()):
    """
    Measure a beam size FWHM from a knife edge measurement, assuming a beam with a gaussian profile.

    Parameters
    ----------
    x : ndarray, shape (N,)
        Position of the knife edge
    y : ndarray, shape (N,)
        Beam property at corresponding knife edge position, e.g. power, intensity.
    fit_parameters : dict, optional
        The fit parameters can be saved if a dictionary is provided. Then, the fit function
        can be recovered as cdf(x, **fit_parameters)
    
    Returns
    -------
    out : float
        FWHM of the beam. The FWHM is related to a Gaussian standard deviation by the relation:
        fwhm = 2 sqrt(2 log 2) std
    """
    # TODO: handle knife edge measurements made in increasing x but decreasing y
    x, y = n.asfarray(x, dtype = n.float), n.asfarray(y, n.float)
    x -= x.min()
    y -= y.min()
    
    (amplitude, std, center, offset), *_ = curve_fit(cdf, xdata = x, ydata = y, 
                                                     p0 = [y.max(), x.max()/2, n.median(x), y.min()],
                                                     bounds = ([0, 0, 0, -2*y.min()], 
                                                               [2*y.max(), 2*x.max(), x.max(), y.max()]))
    
    # Modify the provided dictionary
    fit_parameters.update( {'amplitude': amplitude, 'std': std, 'center':center, 'offset': offset} )
    
    # FWHM of the normal distribution is related to the std by a factor
    return 2 * sqrt(2 * log(2)) * std