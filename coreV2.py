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
    points = n.vstack( (xvals, yvals) ).T
    
    #Extend with the center point
    center = n.array([xc, yc])
    return n.vstack( (points, center) )
    
# -----------------------------------------------------------------------------
#           WORKING CLASS
# -----------------------------------------------------------------------------

class WorkThread(QtCore.QThread):
    """
    Object taking care of computations
    """
    def __init__(self, function, *args, **kwargs):
        QtCore.QThread.__init__(self)
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.result = None
    
    def __del__(self):
        self.wait()
    
    def run(self):
        """ Compute and emit a 'done' signal."""
        self.emit(QtCore.SIGNAL('Display Loading'), '\n Computing...')
        self.result = self.function(*self.args, **self.kwargs)
        self.emit(QtCore.SIGNAL('Remove Loading'), '\n Done.')
        self.emit(QtCore.SIGNAL('Computation done'), self.result)

# -----------------------------------------------------------------------------
#               DATA HANDLER CLASS
# -----------------------------------------------------------------------------    
    
class DataHandler(object):
    
    def __init__(self, application):
        
        self._data = None
        self.application = application
        self.image_viewer = self.application.image_viewer
        self.on_click = None
        self.execute_function = None
        
        #Important  attributes for batch processing
        self._diffraction_center = None
        self.__diffraction_radius = None
        self._cutoff = None
        self.background_fit_parameters = None
    
    # Property makes sure that modified data is concurrently plotted
    @property
    def data(self):
        return self._data
    
    @property
    def _diffraction_radius(self):
        return self.__diffraction_radius
    
    @property
    def diffraction_center(self):
        return self._diffraction_center
    
    @property
    def cutoff(self):
        return self._cutoff
        
    @data.setter
    def data(self, value):
        self._data = value
        self.plot()
    
    @_diffraction_radius.setter
    def _diffraction_radius(self, value):
        self.__diffraction_radius = value
        
        #Generate a circle and add replace previous scatterpoints
        if value is not None:
            xc, yc = self.diffraction_center
            old_data = self.data
            self.data = Image(old_data.image, scatterpoints = generateCircle(xc, yc, value), source = old_data)
    
    @diffraction_center.setter
    def diffraction_center(self, value):
        self._diffraction_center = value
        
        #Add center to the scatterpoints
        if value is not None:
            old_data = self.data
            self.data = Image(old_data.image, scatterpoints = value, source = old_data)
    
    @cutoff.setter
    def cutoff(self, value):
        self._cutoff = value
        
        #Set new cutoff
        if value is not None:
            old_data = self.data
            self.data = RadialCurve(old_data.xdata, old_data.ydata, vert_lines = value, source = old_data)
    
    def plot(self, **kwargs):
        """ Handles plotting Data objects or lists of Data objects. """
        
        self.image_viewer.axes.cla()            #Clears axes
        
        if isinstance(self.data, list):         #Plot an item at a time
            for item in self.data:
                item.plot(self.image_viewer.axes, **kwargs)
        else:
            self.data.plot(self.image_viewer.axes, **kwargs)            #Plot the only item
        
        #Update drawing
        self.image_viewer.draw()
    
    def onClick(self):
        """ Operates according to whatever function has been passed. """
        if self.on_click != None:
            self.on_click(self)
    
    def executeFunction(self):
        """ """
        if self.execute_function != None:
            self.execute_function(self)
        
    def revert(self):
        """ """
        
        if isinstance(self.data, list):
            reverted = list()
            for item in self.data:
                reverted.append(item.source)
            self.data = reverted
        else:
            self.data = self.data.source
        
        #Specific hacks
        if self.diffraction_center is not None and self._diffraction_radius is not None:
            self.diffraction_center = None
            

def guessCenter(dataHandler):
    
    #Check cases
    point = n.asarray(dataHandler.image_viewer.last_click_position)
    if dataHandler.diffraction_center is None and dataHandler._diffraction_radius is None:      #If center and radius have not been set
        dataHandler.data.scattercolor = 'red'
        dataHandler.diffraction_center = point
    elif dataHandler.diffraction_center is not None and dataHandler._diffraction_radius is None:    #If center is set but not the radius
        dataHandler.data.scattercolor = 'red'
        dataHandler._diffraction_radius = n.linalg.norm(n.asarray(dataHandler.diffraction_center) - point)
    elif dataHandler.diffraction_center is not None and dataHandler._diffraction_radius is not None:    #If both center and radius have been set, overwrite all
        dataHandler.data.scattercolor = 'red'
        dataHandler.diffraction_center = point
        dataHandler._diffraction_radius = None

def computeCenter(dataHandler):
    """ Only valid for the data_loaded_state """
    #Compute center
    guess_center = dataHandler.diffraction_center
    guess_radius = dataHandler._diffraction_radius
    
    if guess_center is None or guess_radius is None:
        return
    else:
        xc, yc, rc = dataHandler.data.fCenter(guess_center[0], guess_center[1], guess_radius)
    
    #Assignment
    dataHandler.data.scattercolor = 'green'
    dataHandler.diffraction_center = [xc,yc]
    dataHandler._diffraction_radius = rc

def radiallyAverage(dataHandler):
    """ """
    center = dataHandler.diffraction_center #Shorthand
    assert isinstance(dataHandler.data, Image) and center is not None #Check validity of radial average
    dataHandler.data = dataHandler.data.radialAverage(center) #Compute radial average

def cutoffRadialCurve(dataHandler):
    """ """
    cutoff = dataHandler.cutoff
    #Do nothing if cutoff is not set or the data is not a RadialCurve
    if cutoff is None or not isinstance(dataHandler.data, RadialCurve): 
        return
    
    dataHandler.data = dataHandler.data.cutoff(cutoff)

def cutoffClick(dataHandler):
    """ Only valid for radial_averaged_state. """
    dataHandler.cutoff = dataHandler.image_viewer.last_click_position
    
        
# -----------------------------------------------------------------------------
#           DATA CLASS
# -----------------------------------------------------------------------------
        
class Data(object):
    """
    Abstract data container.
    
    Attributes
    ----------
    source - Data object
        Data object from which the current data object is derived. This attribute should be used for intra-state calculations (e.g. the source of a cutoff radial curve is the uncut radial curve), which makes
        it easier to revert operations.
    
    name - string
        String identifier for debugging
    """
    
    def __init__(self, source = None, name = ''):
        
        self.source = source
        self.name = name
    
    def __repr__(self):
        return self.name    
    
class Image(Data):
    
    def __init__(self, array, scatterpoints = None, scattercolor = 'red', source = None, name = ''):
        
        Data.__init__(self, source, name)
        self.image = array.astype(n.float)
        self._scatterpoints = scatterpoints  #in format [ [x1,y1], [x2,y2], ...]  ]
        self.scattercolor = scattercolor
    
    @property
    def scatterpoints(self):
        return self._scatterpoints
    
    @scatterpoints.setter
    def scatterpoints(self, value):
        """ Makes sure scatterpoints is a NumPy array. """
        if isinstance(value, list):
            self._scatterpoints = n.asarray(value)
    
    def load(self, filename):
        from PIL.Image import open
        self.image = n.array(open(filename), dtype = n.float)
    
    def plot(self, axes, **kwargs):
        #Handle empty images
        if self.image is None:
            self.image = n.zeros(shape = (1024,1024), dtype = n.float)
            
        #Plot image
        axes.imshow(self.image, vmin = self.image.min(), vmax = self.image.max(), cmap = 'jet', label = self.name)
        
        #plot scatterpoints
        if self.scatterpoints is not None and n.size(self.scatterpoints) >= 2 :    #If array is not empty
            if n.size(self.scatterpoints) == 2:     #Indexing hack
                xp, yp = self.scatterpoints
            else:
                xp, yp = self.scatterpoints[:,0], self.scatterpoints[:,1]
            axes.scatter(xp, yp, color = self.scattercolor)
        
            #Restrict to the outline of the circle if circle there is (that is, not only a center, or len(scatterpoints) > 1)
            if n.size(self.scatterpoints) == 2:
                axes.set_xlim(0, self.image.shape[0])
                axes.set_ylim(self.image.shape[1],0)
            if n.size(self.scatterpoints) > 2:
                axes.set_xlim(xp.min() - 10, xp.max() + 10)
                axes.set_ylim(yp.max() + 10, yp.min() - 10)

    
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
        assert self.image is not None
        
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

class RadialCurve(Data):
    """
    This class represents any radially averaged diffraction pattern or fit.
    """
    def __init__(self, xdata, ydata, vert_lines = list(), source = None, name = '', color = 'b'):
        
        Data.__init__(self, source, name)
        self.xdata = xdata
        self.ydata = ydata
        self.vert_lines = vert_lines
        
        #Plotting attributes
        self.color = color
    
    def plot(self, axes, **kwargs):
        """ Plots the pattern in the axes specified """
        axes.plot(self.xdata, self.ydata, '.', color = self.color, label = self.name, **kwargs)
        
        #Plot vertical lines if there are any
        if len(self.vert_lines) > 0:
            if len(self.vert_lines) == 2:
                axes.axvline(self.vert_lines[0],ymax = axes.get_ylim()[1], color = 'black')
            else:
                for point in self.vert_lines:
                    axes.axvline(point[0],ymax = axes.get_ylim()[1], color = 'black')
       
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
        return RadialCurve(self.xdata[cutoff_index::], self.ydata[cutoff_index::], source = self, name = 'Cutoff ' + self.name, color = self.color)
        
    def __sub__(self, pattern):
        """ Definition of the subtraction operator. """ 
        #Interpolate values so that substraction makes sense
        return RadialCurve(self.xdata, self.ydata - n.interp(self.xdata, pattern.xdata, pattern.ydata), source = self, name = self.name, color = self.color)
    
    def inelasticBGFit(self, points = list(), fit = 'biexp'):
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
        
        return RadialCurve(self.xdata, new_fit, source = None, name = 'Inelastic Background fit on ' + self.name, color = 'red')

# -----------------------------------------------------------------------------
#           EXECUTE METHODS
# These methods are used to pass onto the next state, sometimes involving computations
# -----------------------------------------------------------------------------

def inelasticBackgroundFit(state):
    #Identify if appropriate state
    try:
        guesses = state.others['guesses']
    except:
        return
    #Do nothing if not enough guesses
    if len(guesses) < 6:
        return
    
    #Actual fitting
    if isinstance(state.data, list):
        result = state.data
        for item in state.data:
            result.append(state.data.inelasticBGFit(guesses, 'biexp'))
    else:
        result = [state.data]
        result.append(state.data.inelasticBGFit(guesses, 'biexp'))

    #Set up next state with a list of data + fit
    state.application.current_state.next_state.data = result
    state.application.current_state = state.application.current_state.next_state
    
    
# -----------------------------------------------------------------------------
#           IMAGE VIEWER CLICK METHOD
# -----------------------------------------------------------------------------

            

def baselineGuesses(state, image_viewer):
    """ Only valid for data_baseline_state. """
    state.has_been_modified = True
    state.others['guesses'].append(image_viewer.last_click_position)
    image_viewer.axes.axvline(state.others['guesses'][-1][0],ymax = image_viewer.axes.get_ylim()[1], color = 'black')
    image_viewer.draw()
