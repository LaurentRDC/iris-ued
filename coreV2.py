# -*- coding: utf-8 -*-

import sys
import os.path
import numpy as n
from PIL.Image import open
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

# -----------------------------------------------------------------------------
#               STATE CLASS
# -----------------------------------------------------------------------------
class State(object):
    
    def __init__(self, previous_state = None, next_state = None, application = None, execute_button = None):
        
        self.previous_state = previous_state
        self.next_state = next_state
        self.application = application
        self.execute_button = None
        self.execute_method = None
        self.instructions = 'test'
        self.data = list()
        self.on_click = nothingHappens    #CLicking behavior on the Image Viewer (by default, nothingHappens(self, self.application.image_viewer))
        self.layout = None
        self.others = dict()    #Storing data that is relevant for only one state
        
        self.hangleGui()
    
    def loadData(self, data):
        pass
    
    def plot(self, axes, **kwargs):
        """ For handling lists of objects """
        if isinstance(self.data, list):
            for item in self.data:
                item.plot(axes, **kwargs)
        else:
            self.data.plot(axes, **kwargs)
    
    def setInstructions(self, message):
        assert isinstance(message, str)
        self.instructions = message
    
    def click(self, event):
        self.on_click(self, self.application.image_viewer)
    
    def hangleGui(self):
        
        #Layout
        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(QtGui.QLabel(self.instructions))
        if self.execute_button != None:
            self.layout.addWidget(self.execute_button)
        
        #Signals
        if self.execute_button != None and self.execute_method != None:
            self.execute_button.clicked.connect(self.execute_method)
        
# -----------------------------------------------------------------------------
#           DATA CLASS
# -----------------------------------------------------------------------------
      
class Image(object):
    
    def __init__(self, array, source = ''):

        self.image = array.astype(n.float)
        self.source = source
    
    def load(self, filename):
        from PIL.Image import open
        self.image = n.array(open(filename), dtype = n.float)
    
    def plot(self, axes, **kwargs):
        #Handle empty images
        if self.image is None:
            self.image = n.zeros(shape = (1024,1024), dtype = n.float)
        #Plot
        axes.imshow(self.image, vmin = self.image.min(), vmax = self.image.max(), label = self.source)
    
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
    
        return RadialCurve(unique_radii, n.divide(accumulation, bincount), self.source + ' radial average')

# -----------------------------------------------------------------------------
#               PATTERN CLASS
# -----------------------------------------------------------------------------

class RadialCurve(object):
    """
    This class represents any radially averaged diffraction pattern or fit.
    """
    def __init__(self, xdata, ydata, source = '', color = 'b'):
        
        self.xdata = xdata
        self.ydata = ydata
        self.source = source
        
        #Plotting attributes
        self.color = color
    
    def plot(self, axes, **kwargs):
        """ Plots the pattern in the axes specified """
        axes.plot(self.xdata, self.ydata, color = self.color, label = self.name, **kwargs)
    
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
        fit = [self.xdata, new_fit, 'IBG ' + self.source]
        
        return RadialCurve(self.xdata, new_fit, 'Inelastic Background fit on ' + self.source)

# -----------------------------------------------------------------------------
#           SPECIFIC STATE RESET METHODS
# -----------------------------------------------------------------------------

def reset(state):
    pass

# -----------------------------------------------------------------------------
#           IMAGE VIEWER CLICK METHOD
# -----------------------------------------------------------------------------

def nothingHappens(state, image_viewer):
    pass

def returnLastClickPosition(state, image_viewer):
    return image_viewer.last_click_position

def guessCenterOrRadius(state, image_viewer):
    """ Only valid for the data_loaded_state state. """
    if state.other['guess center'] == None:
        state.other['guess center'] = image_viewer.last_click_position
        #TODO: update image_viewer with center
    elif state.other['guess radius'] == None:
        ring_position = n.asarray(image_viewer.last_click_position)
        state.other['guess radius'] = n.linalg.norm(state.other['guess center'] - ring_position)
        #TODO: update image_viewer with guess center + ring
    
def cutoff(state, image_viewer):
    state.other['cutoff'] = image_viewer.last_click_position
    image_viewer.axes.axvline(state.other['cutoff'][0],ymax = image_viewer.axes.get_ylim()[1])
    image_viewer.draw()