# -*- coding: utf-8 -*-
#Basics
from __future__ import division
import numpy as n
import scipy.optimize as opt

import os.path
import PIL.Image
import glob
import re
from tqdm import tqdm
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
#           I/O FUNCTIONS
# -----------------------------------------------------------------------------

def diffractionFileList(folder_path = 'C:\\' ):
    """
    returns a list of filenames corresponding to diffraction pictures
    """
    return
    
# -----------------------------------------------------------------------------
#           RADIAL CURVE CLASS
# -----------------------------------------------------------------------------

class RadialCurve(object):
    """
    This class represents any radially averaged diffraction pattern or fit.
    """
    def __init__(self, xdata, ydata, name = '', color = 'b'):
        
        self.xdata = xdata
        self.ydata = ydata
        self.name = name
        #Plotting attributes
        self.color = color
    
    def plot(self, axes, **kwargs):
        """ Plots the pattern in the axes specified """
        axes.plot(self.xdata, self.ydata, '.-', color = self.color, label = self.name, **kwargs)
       
        #Plot parameters
        axes.set_xlim(self.xdata.min(), self.xdata.max())  #Set xlim and ylim on the first pattern args[0].
        axes.set_ylim(self.ydata.min(), self.ydata.max())
        axes.set_aspect('auto')
        axes.set_title('Diffraction pattern')
        axes.set_xlabel('radius (px)')
        axes.set_ylabel('Intensity')
        axes.legend( loc = 'upper right', numpoints = 1)
    
    def __sub__(self, pattern):
        """ Definition of the subtraction operator. """ 
        #Interpolate values so that substraction makes sense
        return RadialCurve(self.xdata, self.ydata - n.interp(self.xdata, pattern.xdata, pattern.ydata), name = self.name, color = self.color)

    def cutoff(self, cutoff = [0,0]):
        """ Cuts off a part of the pattern"""
        cutoff_index = n.argmin(n.abs(self.xdata - cutoff[0]))
        return RadialCurve(self.xdata[cutoff_index::], self.ydata[cutoff_index::], name = 'Cutoff ' + self.name, color = self.color)

    def inelasticBG(self, points = list(), fit = 'biexp'):
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
        
        return RadialCurve(self.xdata, new_fit, 'IBG ' + self.name, 'red')

# -----------------------------------------------------------------------------
#           FIND CENTER OF DIFFRACTION PATTERN
# -----------------------------------------------------------------------------

def fCenter(xg, yg, rg, im, scalefactor = 20):
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
    c1 = lambda x: circ(x[0],x[1],x[2],im)
    xcenter, ycenter, rcenter = n.array(\
        opt.minimize(c1,[xgscaled,ygscaled,rgscaled],\
        method = 'Nelder-Mead').x)*scalefactor
    rcenter = rg    
    return xcenter, ycenter, rcenter

def circ(xg, yg, rg, im, scalefactor = 20):

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
    s = im.shape[0]
    xgscaled, ygscaled, rgscaled = n.array([xg,yg,rg])*scalefactor
    xMat, yMat = n.meshgrid(n.linspace(1, s, s),n.linspace(1, s, s))
    # find coords on circle and sum intensity
    
    residual = (xMat-xgscaled)**2+(yMat-ygscaled)**2-rgscaled**2
    xvals, yvals = n.where(((residual < 10) & (yMat > 550)))
    ftemp = n.mean(im[xvals, yvals])
    
    return 1/ftemp

# -----------------------------------------------------------------------------
#               RADIAL AVERAGING
# -----------------------------------------------------------------------------

def radialAverage(image, name, center = [562,549]):
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
    
    #Get shape
    im_shape = image.shape
    #Preliminaries
    xc, yc = center     #Center coordinates
    x = n.linspace(0, im_shape[0], im_shape[0])
    y = n.linspace(0, im_shape[1], im_shape[1])
    
    #Create meshgrid and compute radial positions of the data
    X, Y = n.meshgrid(x,y)
    R = n.around(n.sqrt( (X - xc)**2 + (Y - yc)**2 ), decimals = 0)
    
    #Flatten arrays
    intensity = image.flatten()
    radius = R.flatten()
    
    #Sort by increasing radius
    intensity = intensity[n.argsort(radius)]
    radius = n.around(radius, decimals = 0)
    
    #Average intensity values for equal radii
    unique_radii = n.unique(radius)
    accumulation = n.zeros_like(unique_radii)
    bincount =  n.ones_like(unique_radii)
    
    #loop over image
    for (i,j), value in n.ndenumerate(image):
      
        #Ignore top half image (where the beamblock is)
        if i < center[0]:
            continue

        r = R[i,j]
        #bin
        ind = n.where(unique_radii==r)
        #increment
        accumulation[ind] += value
        bincount[ind] += 1
        
        #Return normalized radial average
    return RadialCurve(unique_radii, n.divide(accumulation,bincount), name + ' radial average')

# -----------------------------------------------------------------------------
#           BATCH FILE PROCESSING
# -----------------------------------------------------------------------------

class DiffractionDataset(object):
    
    def __init__(self, directory, resolution = (2048, 2048)):
        
        self.directory = directory
        self.resolution = resolution
        self.pumpon_background = self.averageImages('background.*.pumpon.tif')
        self.pumpoff_background = self.averageImages('background.*.pumpoff.tif')
        self.time_points = self.getTimePoints()
        
    
    def averageImages(self, filename_template, background = None):
        """
        Averages images matching a filename template within the dataset directory.
        
        Parameters
        ----------
        filename_templates : string
            Examples of filename templates: 'background.*.pumpon.tif', '*.jpeg', etc.
        
        See also
        --------
        Glob.glob
        """ 
        #Preliminaries               
        if background is not None:
            background = background.astype(n.float)
            
        image_list = glob.glob(os.path.join(self.directory, filename_template))
        
        image = n.zeros(shape = self.resolution, dtype = n.float)
        for filename in tqdm(image_list):
            new_image = n.array(PIL.Image.open(filename), dtype = n.float)
            if background is not None:
                new_image -= background
            image += new_image
            
        #Average    
        return image/len(image_list)
    
    def getTimePoints(self):
        """ """
        #Get TIFF images
        image_list = [f for f in os.listdir(self.directory) 
                if os.path.isfile(os.path.join(self.directory, f)) 
                and f.startswith('data.timedelay.') 
                and f.endswith('pumpon.tif')]
        
        #get time points
        #TODO: check if sorting is necessary
        time_data = [float( re.search('[+-]\d+[.]\d+', f).group() ) for f in image_list]
        return list(set(time_data))     #Conversion to set then back to list to remove repeated values
    
    def dataAverage(self, time_point, pump = 'pumpon'):
        """ 
        Returns a UNIX-like file template, to be used in conjunction with self.averageImages.
        
        Parameters
        ----------
        time_point : string or float
            string in the form of +150.00, -10.00, etc. If a float is provided, it will be converted to a string of the correct format.
        """
        
        #preliminaries
        if isinstance(time_point, float):
            print 'Time point entered as float'
            sign_prefix = '+' if time_point >= 0.0 else '-'
            time_point = sign_prefix + str(time_point) + '0'
            print 'New time point format: {0}'.format(time_point)
        
        assert time_point.endswith('.00')
        
        glob_template = 'data.timedelay.' + time_point + '.nscan.*.' + pump + '.tif'
        background = self.pumpon_background if pump == 'pumpon' else self.pumpoff_background
        
        return self.averageImages(glob_template, background)
        
        
        
        