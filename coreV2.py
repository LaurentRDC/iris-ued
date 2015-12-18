# -*- coding: utf-8 -*-

import sys
import os.path
import numpy as n
import scipy.optimize as opt

#plotting backends
from matplotlib.backends import qt_compat
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.backends.backend_qt4agg as qt4agg
from matplotlib.figure import Figure
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE

#GUI backends
if use_pyside:
    from PySide import QtGui, QtCore
else:
    from PyQt4 import QtGui, QtCore
# -----------------------------------------------------------------------------
#           HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def Gaussian(x, xc = 0, width_g = 0.1):
    """ Returns a Gaussian with maximal height of 1 (not area of 1)."""
    exponent = (-(x-xc)**2)/((2*width_g)**2)
    return n.exp(exponent)

def Lorentzian(x, xc = 0, width_l = 0.1):
    """ Returns a lorentzian with maximal height of 1 (not area of 1)."""
    core = ((width_l/2)**2)/( (x-xc)**2 + (width_l/2)**2 )
    return core
    
def pseudoVoigt(x, height, xc, width_g, width_l, constant = 0):
    """ Returns a pseudo Voigt profile centered at xc with weighting factor 1/2. """
    return height*(0.5*Gaussian(x, xc, width_g) + 0.5*Lorentzian(x, xc, width_l)) + constant

def biexp(x, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0):
    """ Returns a biexponential of the form a*exp(-b*x) + c*exp(-d*x) + e"""
    return a*n.exp(-b*(x-f)) + c*n.exp(-d*(x-f)) + e

def bilor(x, center, amp1, amp2, width1, width2, const):
    """ Returns a Bilorentzian functions. """
    return amp1*Lorentzian(x, center, width1) + amp2*Lorentzian(x, center, width2) + const

def generateCircle(xc, yc, radius):
    """
    Generates scatter value for a cicle centered at [xc,yc] of radius 'radius'.
    """
    xvals = xc + radius*n.cos(n.linspace(0,2*n.pi,100))
    yvals = yc + radius*n.sin(n.linspace(0,2*n.pi,100))
    return xvals, yvals

# -----------------------------------------------------------------------------
#               STATE CLASS
# -----------------------------------------------------------------------------
class State(object):
    
    def __init__(self, previous_state = None, next_state = None, application = None, name = ''):
        
        self.previous_state = previous_state
        self.next_state = next_state
        self.application = application
        self.execute_method = executeNothing
        self.plotting_method = plotDefault
        self.instructions = 'test'
        self.data = list()
        self.name = name
        self.on_click = nothingHappens    #CLicking behavior on the Image Viewer (by default, nothingHappens(self, self.application.image_viewer))
        self.others = dict()    #Storing data that is relevant for only one state
    
    def __repr__(self):
        return self.name
    
    def plot(self, axes, **kwargs):
        axes.cla()
        self.plotting_method(self, axes, **kwargs)
        self.application.image_viewer.draw()
    
    def loadData(self, data):
        pass
    
    def setInstructions(self, message):
        assert isinstance(message, str)
        self.instructions = message
    
    def click(self, event):
        self.on_click(self, self.application.image_viewer)
        
# -----------------------------------------------------------------------------
#           DATA CLASS
# -----------------------------------------------------------------------------
      
class Image(object):
    
    def __init__(self, array, source = None, name = ''):

        self.image = array.astype(n.float)
        self.source = source
        self.name = name
    
    def load(self, filename):
        from PIL.Image import open
        self.image = n.array(open(filename), dtype = n.float)
    
    def plot(self, axes, **kwargs):
        #Handle empty images
        if self.image is None:
            self.image = n.zeros(shape = (1024,1024), dtype = n.float)
        #Plot
        axes.imshow(self.image, vmin = self.image.min(), vmax = self.image.max(), label = self.name)
    
    def radialAverage(self, center = [562,549]):
        """
        This function returns a radially-averaged pattern computed from a TIFF image.
        
        Parameters
        ----------
        image : list of ndarrays, shape(N,N)
            List of images that have the same shape and share the same center.
        center : array-like, shape (2,)
            [x,y] coordinates of the center (in pixels)
        beamblock_rectangle : list, shape (2,)
            Two corners of the rectangle, in the form [ [x0,y0], [x1,y1] ]  
        Returns
        -------
        [[radius1, pattern1, name1], [radius2, pattern2, name2], ... ], : list of ndarrays, shapes (M,), and an ID string
        """
        assert self.image != None
        
        #Get shape
        im_shape = self.image.shape
        #Preliminaries
        xc, yc = center     #Center coordinates
        x = n.linspace(0, im_shape[0], im_shape[0])
        y = n.linspace(0, im_shape[1], im_shape[1])
        
        #Create meshgrid and compute radial positions of the data
        X, Y = n.meshgrid(x,y)
        R = n.around(n.sqrt( (X - xc)**2 + (Y - yc)**2 ), decimals = 0)
            
        #Flatten arrays
        intensity = self.image.flatten()
        radius = R.flatten()
        
        #Sort by increasing radius
        intensity = intensity[n.argsort(radius)]
        radius = n.around(radius, decimals = 0)
        
        #Average intensity values for equal radii
        unique_radii = n.unique(radius)
        accumulation = n.zeros_like(unique_radii)
        bincount =  n.ones_like(unique_radii)
        
        #loop over image
        for (i,j), value in n.ndenumerate(self.image):
          
            #Ignore top half image (where the beamblock is)
            if i < center[0]:
                continue
    
            r = R[i,j]
            #bin
            ind = n.where(unique_radii==r)
            #increment
            accumulation[ind] += value
            bincount[ind] += 1
    
        return RadialCurve(xdata = unique_radii, ydata = n.divide(accumulation, bincount), source = self, name = self.name + ' radial average')
    
    def fCenter(self, xg, yg, rg, scalefactor = 20):
        """
        Finds the center of a diffraction pattern based on an initial guess of the center.
        
        Parameters
        ----------
        xg, yg, rg : ints
            Guesses for the (x,y) position of the center, and the radius
        im : ndarray, shape (N,N)
            ndarray of a TIFF image
        
        Returns
        -------
        optimized center and peak position
        
        See also
        --------
        Scipy.optimize.fmin - Minimize a function using the downhill simplex algorithm
        """
        
    
        #find maximum intensity
        xgscaled, ygscaled, rgscaled = n.array([xg,yg,rg])/scalefactor
        c1 = lambda x: self.circ(x[0],x[1],x[2])
        xcenter, ycenter, rcenter = n.array(\
            opt.minimize(c1,[xgscaled,ygscaled,rgscaled],\
            method = 'Nelder-Mead').x)*scalefactor
        rcenter = rg    
        return xcenter, ycenter, rcenter
    
    def circ(self, xg, yg, rg, scalefactor = 20):
    
        """
        Sums the intensity over a circle of given radius and center position
        on an image.
        
        Parameters
        ----------
        xg, yg, rg : ints
            The (x,y) position of the center, and the radius
        im : ndarray, shape (N,N)
            ndarray of a TIFF image
        
        Returns
        -------
        Total intensity at pixels on the given circle. 
        
        """
         #image size
        s = self.image.shape[0]
        xgscaled, ygscaled, rgscaled = n.array([xg,yg,rg])*scalefactor
        xMat, yMat = n.meshgrid(n.linspace(1, s, s),n.linspace(1, s, s))
        # find coords on circle and sum intensity
        
        residual = (xMat-xgscaled)**2+(yMat-ygscaled)**2-rgscaled**2
        xvals, yvals = n.where(((residual < 10) & (yMat > 550)))
        ftemp = n.mean(self.image[xvals, yvals])
        
        return 1/ftemp

# -----------------------------------------------------------------------------
#               1D DATA CLASS
# -----------------------------------------------------------------------------

class RadialCurve(object):
    """
    This class represents any radially averaged diffraction pattern or fit.
    """
    def __init__(self, xdata, ydata, source = None, name = '', color = 'b'):
        
        self.xdata = xdata
        self.ydata = ydata
        self.source = source
        self.name = name
        
        #Plotting attributes
        self.color = color
    
    def plot(self, axes, **kwargs):
        """ Plots the pattern in the axes specified """
        axes.plot(self.xdata, self.ydata, color = self.color, label = self.name, **kwargs)
       
        #Plot parameters
        axes.set_xlim(self.xdata.min(), self.xdata.max())  #Set xlim and ylim on the first pattern args[0].
        axes.set_ylim(self.ydata.min(), self.ydata.max())
        axes.set_aspect('auto')
        axes.set_title('Diffraction pattern')
        axes.set_xlabel('radius (px)')
        axes.set_ylabel('Intensity')
        axes.legend( loc = 'upper right', numpoints = 1)
    
    def cutoff(self, cutoff = [0,0]):
        """ Cuts off a part of the pattern"""
        cutoff_index = n.argmin(n.abs(self.xdata - cutoff[0]))
        self.xdata = self.xdata[cutoff_index::]
        self.ydata = self.ydata[cutoff_index::]
        
    def __sub__(self, pattern):
        """ Definition of the subtraction operator. """ 
        #Interpolate values so that substraction makes sense
        self.ydata -= n.interp(self.xdata, pattern.xdata, pattern.ydata)
    
    def inelasticBGfit(self, points = list(), fit = 'biexp'):
        """
        Inelastic scattering background substraction.
        
        Parameters
        ----------
        patterns : list of lists of the form [xdata, ydata, name]
        
        points : list of lists of the form [x,y]
        
        fit : string
            Function to use as fit. Allowed values are 'biexp' and 'bilor'
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
            print 'Runtime error'
            optimal_parameters = guesses[fit]
    
        #Create inelastic background function 
        a,b,c,d,e,f = optimal_parameters
        new_fit = function(self.xdata, a, b, c, d, e, f) 
        fit = [self.xdata, new_fit, 'IBG ' + self.name]
        
        return RadialCurve(self.xdata, new_fit, 'Inelastic Background fit on ' + self.name)

# -----------------------------------------------------------------------------
#           SPECIFIC PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------

def plotDefault(state, axes, **kwargs):
    """ For handling lists of objects """
    if isinstance(state.data, list):
        for item in state.data:
            item.plot(axes, **kwargs)
    else:
        state.data.plot(axes, **kwargs)

def plotGuessCenter(state, axes, **kwargs):
    """ """
    #Start by plotting data
    plotDefault(state, axes, **kwargs)
    
    #Overlay other informations
    if state.others['guess center'] != None:
        xc, yc = state.others['guess center']           #Center coordinates
        axes.scatter(xc, yc, color = 'red')
        axes.set_xlim(0, state.data.image.shape[0])
        axes.set_ylim(state.data.image.shape[1],0)
    
    if state.others['guess radius'] != None:
        xc, yc = state.others['guess center']
        xvals, yvals = generateCircle(xc, yc, state.others['guess radius'])
        axes.scatter(xvals, yvals, color = 'red')
        #Restrict view to the plotted circle (to better evaluate the fit)
        axes.set_xlim(xvals.min() - 10, xvals.max() + 10)
        axes.set_ylim(yvals.max() + 10, yvals.min() - 10)
    
    #Draw changes
    state.application.image_viewer.draw()

def plotComputedCenter(state, axes, **kwargs):
    """ """
    #Start by plotting data
    plotDefault(state, axes, **kwargs)
    
    #Overlay other informations
    if state.others['center'] != None:
        xc, yc = state.others['center']           #Center coordinates
        axes.scatter(xc, yc, color = 'green')
        axes.set_xlim(0, state.data.image.shape[0])
        axes.set_ylim(state.data.image.shape[1],0)
    
    if state.others['radius'] != None:
        xc, yc = state.others['center']
        xvals, yvals = generateCircle(xc, yc, state.others['radius'])
        axes.scatter(xvals, yvals, color = 'green')
        #Restrict view to the plotted circle (to better evaluate the fit)
        axes.set_xlim(xvals.min() - 10, xvals.max() + 10)
        axes.set_ylim(yvals.max() + 10, yvals.min() - 10)
        
    #Draw changes
    state.application.image_viewer.draw()

# -----------------------------------------------------------------------------
#           EXECUTE METHODS
# These methods are used to pass onto the next state, sometimes involving computations
# -----------------------------------------------------------------------------

def executeNothing(state):
    pass

def computeCenter(state):
    """ Only valid for the data_loaded_state """
    #Compute center
    guess_center = state.others['guess center'] 
    guess_radius = state.others['guess radius']
    if guess_center == None or guess_radius == None:
        pass
    else:
        xg, yg = guess_center
        xc, yc, rc = state.data.fCenter(xg, yg, guess_radius)
        
    #Go to next state
    state.application.center_found_state.others = {'center': [xc, yc], 'radius': rc, 'substrate image': state.others['substrate image']}
    state.application.center_found_state.data = state.data
    state.application.current_state = state.application.center_found_state      #Includes plotting new data

def radiallyAverage(state):
    
    center = state.others['center']
    assert center != None
    
    #Compute radial average
    rav = state.data.radialAverage(center)
    substrate_rav = state.others['substrate image'].radialAverage(center)
    
    #Set up next state
    state.application.current_state.next_state.data = [rav, substrate_rav]
    state.application.current_state = state.application.current_state.next_state         #Includes ploting new data
# -----------------------------------------------------------------------------
#           IMAGE VIEWER CLICK METHOD
# -----------------------------------------------------------------------------

def nothingHappens(state, image_viewer):
    pass

def returnLastClickPosition(state, image_viewer):
    return image_viewer.last_click_position

def guessCenterOrRadius(state, image_viewer):
    """ Only valid for the data_loaded_state state. """
    if state.others['guess center'] == None:
        state.others['guess center'] = image_viewer.last_click_position
        state.plot(state.application.image_viewer.axes)
    elif state.others['guess radius'] == None:
        ring_position = n.asarray(image_viewer.last_click_position)
        state.others['guess radius'] = n.linalg.norm(state.others['guess center'] - ring_position)
        state.plot(state.application.image_viewer.axes)
            
def cutoff(state, image_viewer):
    state.others['cutoff'] = image_viewer.last_click_position
    image_viewer.axes.axvline(state.others['cutoff'][0],ymax = image_viewer.axes.get_ylim()[1])
    image_viewer.draw()