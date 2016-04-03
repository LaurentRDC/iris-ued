# -*- coding: utf-8 -*-

#Basics
import numpy as n
from numpy import pi
import scipy.optimize as opt

# -----------------------------------------------------------------------------
#           HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def lorentzian(x, xc = 0, width_l = 0.1):
    """ Returns a lorentzian with maximal height of 1 (not area of 1)."""
    core = ((width_l/2)**2)/( (x-xc)**2 + (width_l/2)**2 )
    return core
    
def biexp(x, a, b, c, d, e, f):
    """ Returns a biexponential of the form a*exp(-b*x) + c*exp(-d*x) + e"""
    return a*n.exp(-b*(x-f)) + c*n.exp(-d*(x-f)) + e

def bilor(x, center, amp1, amp2, width1, width2, const):
    """ Returns a Bilorentzian functions. """
    return amp1*lorentzian(x, center, width1) + amp2*lorentzian(x, center, width2) + const

def generate_circle(xc, yc, radius):
    """
    Generates scatter value for a cicle centered at [xc,yc] of radius 'radius'.
    """
    xvals = xc + radius*n.cos(n.linspace(0, 2*pi, 500))
    yvals = yc + radius*n.sin(n.linspace(0, 2*pi, 500))
    
    circle = zip(xvals.tolist(), yvals.tolist())
    circle.append( (xc, yc) )
    return circle
        
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
    def __init__(self, xdata = n.zeros(shape = (100,)), ydata = n.zeros(shape = (100,)), name = '', color = 'b'):
        
        self.xdata = n.asarray(xdata)
        self.ydata = n.asarray(ydata)
        self.name = name
        
        #Plotting attributes
        self.color = color
    
    def __sub__(self, pattern):
        """ Definition of the subtraction operator. """ 
        #Interpolate values so that substraction makes sense
        return Curve(self.xdata, self.ydata - n.interp(self.xdata, pattern.xdata, pattern.ydata), name = self.name, color = self.color)
    
    def __copy__(self):
        return Curve(self.xdata, self.ydata, self.name, self.color)

    def cutoff(self, cutoff = [0,0]):
        """ Cuts off a part of the pattern"""
        cutoff_index = n.argmin(n.abs(self.xdata - cutoff[0]))
        self.xdata = self.xdata[cutoff_index::] 
        self.ydata = self.ydata[cutoff_index::]

    def inelastic_background(self, points = list(), fit = 'biexp'):
        """
        Inelastic scattering background fit.
        
        Parameters
        ----------
        points : list of tuples
            List of tuples of the form (x,y). y-values are ignored, and the 
            interpolated y-values at the provided x-values is used.
        fit : str {'biexp', 'bilor'}
            Function to use as fit. Default is biexponential.
        """
        #Preliminaries
        function = bilor if fit == 'bilor' else biexp
        
        #Create x arrays for the points 
        points = n.array(points, dtype = n.float) 
        x = points[:,0]
        
        #Create guess 
        guesses = {'biexp': (self.ydata.max()/2, 1/50.0, self.ydata.max()/2, 1/150.0, self.ydata.min(), self.xdata.min()), 
                   'bilor':  (self.xdata.min(), self.ydata.max()/1.5, self.ydata.max()/2.0, 50.0, 150.0, self.ydata.min())}
        
        #Interpolate the values of the patterns at the x points
        y = n.interp(x, self.xdata, self.ydata)
        
        #Fit with guesses if optimization does not converge
        try:
            optimal_parameters, parameters_covariance = opt.curve_fit(function, x, y, p0 = guesses[fit]) 
        except(RuntimeError):
            print('Runtime error')
            optimal_parameters = guesses[fit]
    
        #Create inelastic background function 
        a,b,c,d,e,f = optimal_parameters
        new_fit = function(self.xdata, a, b, c, d, e, f)
        
        return Curve(self.xdata, new_fit, 'IBG {0}'.format(self.name), 'red')