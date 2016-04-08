# -*- coding: utf-8 -*-

#Basics
import numpy as n
import scipy.optimize as opt

# -----------------------------------------------------------------------------
#           HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def lorentzian(x, xc = 0, width_l = 0.1):
    """ Returns a lorentzian with maximal height of 1 (not area of 1)."""
    core = ((width_l/2)**2)/( (x-xc)**2 + (width_l/2)**2 )
    return core
    
def biexp(x, a, b, c, d, center, const):
    """ Returns a biexponential of the form a*n.exp(-b*(x-e)) + c*n.exp(-d*(x-e)) + f"""
    return a*n.exp(-b*(x-center)) + c*n.exp(-d*(x-center)) + const

def biexponential(x, amplitude1, amplitude2, decay1, decay2, offset1, offset2, floor):
    """
    Returns a biexponential function evaluated over an array.
    
    Notes
    -----
    In case of fitting, there are 7 free parameters, and thus at least 7 points
    must be provided.
    """
    exp1 = amplitude1*n.exp(decay1*(x - offset1))
    exp2 = amplitude2*n.exp(decay2*(x - offset2))
    return exp1 + exp2 + floor

def bilor(x, center, amp1, amp2, width1, width2, const):
    """ Returns a Bilorentzian functions. """
    return amp1*lorentzian(x, center, width1) + amp2*lorentzian(x, center, width2) + const
        
# -----------------------------------------------------------------------------
#           RADIAL CURVE CLASS
# -----------------------------------------------------------------------------

class Curve(object):
    """
    This class represents any radially averaged diffraction pattern or fit.
    
    Attributes
    ----------
    xdata : ndarray, shape (N,)
        Abscissa values of the curve.
    ydata : ndarray, shape (N,)
        Ordinate values of the curve.
    name : str
        String identifier.
    color : str
        Plotting color.
    
    Methods
    -------
    cutoff
        Cuts off a part of the curve.
    
    inelastic_background
        Fits a biexponential inelastic background to the curve
    """
    def __init__(self, xdata, ydata, name = '', color = 'b'):
        
        self.xdata = n.asarray(xdata, dtype = n.float)
        self.ydata = n.asarray(ydata, dtype = n.float)
        self.name = name
        
        #Plotting attributes
        self.color = color
    
    def __sub__(self, pattern):
        #Interpolate values so that substraction makes sense
        return Curve(self.xdata, self.ydata - n.interp(self.xdata, pattern.xdata, pattern.ydata), name = self.name, color = self.color)
    
    def __add__(self, pattern):
        #Interpolate values so that addition makes sense
        return Curve(self.xdata, self.ydata + n.interp(self.xdata, pattern.xdata, pattern.ydata), name = self.name, color = self.color)
    
    def __copy__(self):
        return Curve(self.xdata, self.ydata, self.name, self.color)
    
    def plot(self):
        """ Diagnostic tool. """
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.title(self.name)
        plt.plot(self.xdata, self.ydata, self.color)

    def cutoff(self, cutoff):
        """ Cuts off a part of the pattern"""
        if isinstance(cutoff, list):
            cutoff = cutoff[0]
        
        cutoff_index = n.argmin(n.abs(self.xdata - cutoff))
        self.xdata = self.xdata[cutoff_index::] 
        self.ydata = self.ydata[cutoff_index::]

    def inelastic_background(self, xpoints = list()):
        """
        Inelastic scattering background fit.
        
        Parameters
        ----------
        points : array-like
            x-values of the points to fit to.
        fit : str {'biexp', 'bilor'}
            Function to use as fit. Default is biexponential.
        """        
        # Preliminaries
        assert len(xpoints) >= 7    # The biexponential implementation has 7 free parameters
        xpoints = n.array(xpoints, dtype = n.float) 
        
        #Create initial guesses
        # amp1, amp2, decay1, decay2, offset1, offset2, floor
        # TODO: find better guesses?
        guesses = (self.ydata.max(), self.ydata.max(), 0, 0, 0, 0, 0)
        
        # Value bounds
        # amp1, amp2, decay1, decay2, offset1, offset2, floor
        min_bounds = n.array( [ 0.0, 0.0, -n.inf,-n.inf,-n.inf,-n.inf, 0.0 ] )
        max_bounds = n.array( [n.inf,n.inf,n.inf, n.inf, n.inf, n.inf, self.ydata.max()] )
        bounds = (min_bounds, max_bounds)
        
        # Interpolate the values of the patterns at the x points
        # Fit with guesses if optimization does not converge
        ypoints = n.interp(xpoints, self.xdata, self.ydata)
        optimal_parameters, parameters_covariance = opt.curve_fit(biexponential, xpoints, ypoints, p0 = guesses, bounds = bounds) 
    
        # Create inelastic background function 
        amp1, amp2, dec1, dec2, off1, off2, floor = optimal_parameters
        new_fit = biexponential(self.xdata, amp1, amp2, dec1, dec2, off1, off2, floor)
        return Curve(self.xdata, new_fit, 'IBG {0}'.format(self.name), 'red')