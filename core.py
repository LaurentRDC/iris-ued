# -*- coding: utf-8 -*-
#Basics
from __future__ import division
import numpy as n
import scipy.optimize as opt

#Batch processing libraries
import os.path
import h5py
import tifffile as t
import glob
import re
from tqdm import tqdm
import datetime

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
#           RADIAL CURVE CLASS
# -----------------------------------------------------------------------------

class RadialCurve(object):
    """
    This class represents any radially averaged diffraction pattern or fit.
    """
    def __init__(self, xdata = n.zeros(shape = (100,)), ydata = n.zeros(shape = (100,)), name = '', color = 'b'):
        
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

def circ(xg, yg, rg, im, scalefactor = 5):

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
    
    return 1.0/ftemp

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
    
    #radii beyond r_max don't fit a full circle within the image
    edge_values = n.array([R[0,:], R[-1,:], R[:,0], R[:,-1]])
    r_max = n.min(n.array(edge_values))  #Maximal valid radius
    
    #Average intensity values for equal radii
    unique_radii = n.unique(radius)
    loc = n.argmin(n.abs(unique_radii - r_max))
    unique_radii = unique_radii[:loc]       #Remove radii that don't fit in the image
    accumulation = n.zeros_like(unique_radii)
    bincount =  n.ones_like(unique_radii)
    
    #loop over image
    for (i,j), value in n.ndenumerate(image):
        #TODO: replace the below condition with something that does not
        #      ignore so much data
        #Ignore top half image (where the beamblock is)
        if j < center[1]:
            continue

        r = R[i,j]
        #bin
        ind = n.where(unique_radii==r)
        accumulation[ind] += value
        bincount[ind] += 1
        
    #Return normalized radial average
    return RadialCurve(unique_radii, n.divide(accumulation,bincount), name)

# -----------------------------------------------------------------------------
#           BATCH FILE PROCESSING
# -----------------------------------------------------------------------------

class DiffractionDataset(object):
    
    def __init__(self, directory, resolution = (2048, 2048)):
        
        self.directory = directory
        self.resolution = resolution
        self.substrate = self.getSubstrateImage()
        self.pumpon_background = self.averageTiffFiles('background.*.pumpon.tif')
        self.time_points = self.getTimePoints()
        self.acquisition_date = self.getAcquisitionDate()
        
    def getAcquisitionDate(self):
        """ Returns a string containing the date of acquisition. """
        path = os.path.join(self.directory, 'background.1.pumpon.tif')
        try:
            acquisition_date = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%B %d, %Y")
        except(WindowsError):       #Can't find any file
            acquisition_date = ''
        return acquisition_date
        
    def getSubstrateImage(self):
        """ Finds and stores a substrate image, and returns None if criterias are not matched. """
        
        substrate_filenames = [os.path.join(self.directory, possible_filename) for possible_filename in ['subs.tif', 'substrate.tif']]
        for possible_substrate in substrate_filenames:
            if possible_substrate in os.listdir(self.directory):
                print 'Substrate image found'
                return t.imread(possible_substrate).astype(n.float)
        return n.zeros(shape = self.resolution, dtype = n.float)         #If file not found
    
    def averageTiffFiles(self, filename_template, background = None):
        """
        Averages images matching a filename template within the dataset directory.
        
        Parameters
        ----------
        filename_templates : string
            Examples of filename templates: 'background.*.pumpon.tif', '*.tif', etc.
        
        See also
        --------
        Glob.glob
        """ 
        #Format background correctly
        if background is not None:
            if background.dtype != n.float:
                background = background.astype(n.float)
        
        #Get file list
        image_list = glob.glob(os.path.join(self.directory, filename_template))
        
        image = n.zeros(shape = self.resolution, dtype = n.float)
        for filename in tqdm(image_list, nested = True):
            new_image = t.imread(filename).astype(n.float)
            if background is not None:
                new_image -= background
            image += new_image
            
        #Average    
        return image/float(len(image_list))

    def dataAverage(self, time_point, export = False):
        """         
        Parameters
        ----------
        time_point : string or numerical
            string in the form of +150.00, -10.00, etc. If a float or int is provided, it will be converted to a string of the correct format.
        pump : string
            Determines whether to average 'pumpon' data or 'pumpoff' data
        """
        
        time_point = self.timeToString(time_point, units = False)
        glob_template = 'data.timedelay.' + time_point + '.nscan.*.pumpon.tif'
        
        #Average calculation
        average = self.averageTiffFiles(glob_template, self.pumpon_background)
        
        if export:
            output_directory = os.path.join(self.directory, 'processed')
            if not os.path.isdir(output_directory):             #Find out if output directory exists and create if necessary
                os.makedirs(output_directory)                
            save_path = os.path.join(output_directory, 'data.timedelay.' + time_point + '.average.pumpon.tif')
            
            #Format array to be uint16 by removing data outside the bounds
            average[average < 0] = 0
            average[average >= 65536] = 65536 - 1
            #TODO: add datetime parameter that mirrors the source data
            t.imsave(save_path, average.astype(n.uint16))
        
        return average
        
    def batchAverage(self, check_for_averages = False):
        """ 
        Averages all relevant TIFF images and saves them in self.directory\processed. Saved images can be radially-averaged later.
        
        Parameters
        ----------
        check_for_duplicates : bool
            Stops if the average has already been computed (i.e. if the folder 'processed' exists)        
        """
        if check_for_averages:
            if os.path.isdir( os.path.join(self.directory, 'processed') ):      #Data has been processed already
                return None
                
        #Average images        
        for time in tqdm(self.time_points):
            self.dataAverage(time, export = True)
    
    def getTimePoints(self):
        """ """
        #Get TIFF images
        image_list = [f for f in os.listdir(self.directory) 
                if os.path.isfile(os.path.join(self.directory, f)) 
                and f.startswith('data.timedelay.') 
                and f.endswith('pumpon.tif')]
        
        #get time points
        time_data = [float( re.search('[+-]\d+[.]\d+', f).group() ) for f in image_list]
        return list(set(time_data))     #Conversion to set then back to list to remove repeated values
    
    @staticmethod
    def timeToString(time, units = False):
        """ 
        Converts input time to string notation of the form: '+150.00', '-10.00', etc. 
        If units = True, returns units as well: '+150.00ns'
        """
        if isinstance(time, str):
            return time
        if isinstance(time, int):
            time = float(time)
        if isinstance(time, float):
            sign_prefix = '+' if time >= 0.0 else '-'
            str_time = sign_prefix + str(abs(time)) + '0'
        
        assert str_time.endswith('.00')
        if units:
            return str_time + 'ns'
        else:
            return str_time
        
    def processImage(self, filename, center, cutoff, inelasticBGCurve):
        """
        Returns a processed radial curve associated with a radial diffraction pattern at a certain time point.
        
        Parameters
        ----------
        time : string or numerical
            Either a string formatted as {'+150.00', '-10.00'} or equivalent float or integer. See self.dataAverage
        TBD
        """
        image = t.imread(filename).astype(n.float)
        
        if self.substrate is not None:                           #substract substrate if it is provided
            assert image.shape == self.substrate.shape
            self.substrate = self.substrate.astype(n.float)
            image -= self.substrate
            
        curve = radialAverage(image, 'Radial average', center)     #Radially average
        curve = curve.cutoff(cutoff)                        #cutoff curve

        if inelasticBGCurve is not None:                    #substract inelastic scattering background if it is provided
            assert isinstance(inelasticBGCurve, RadialCurve) 
            return curve - inelasticBGCurve
        else:
            return curve
        
    def batchProcess(self, center = [0,0], cutoff = [0,0], inelasticBGCurve = None):
        """
        Returns a list of RadialCurve objects (one for every time point)
        """
        
        results = list()
        for time in tqdm(self.time_points):
            filename = os.path.join(self.directory, 'processed', 'data.timedelay.' + self.timeToString(time) + '.average.pumpon.tif')
            curve = self.processImage(filename, center, cutoff, inelasticBGCurve)
            results.append( (self.timeToString(time), curve) )
        
        self.export(results)          
            
    def export(self, results):
        """ """
        
        #save filename including acquisition time
        save_filename = os.path.join(self.directory, 'processed', self.acquisition_date + '.radial.averages.hdf5')
        #assert isinstance(save_filename, str) and save_filename.endswith('.hdf5')
        
        f = h5py.File(save_filename, 'w', libver = 'latest')
        f.attrs['date'] = self.acquisition_date
        
        #Iteratively create a group for each timepoint
        for item in results:
            timepoint, curve = item
            timepoint = self.timeToString(timepoint, units = True)      #Checking that the timepoint string formatting is consistent
            
            #Create group and attribute
            group = f.create_group(timepoint)
            group.attrs['timepoint'] = timepoint
            
            #Add some data to the file
            group.create_dataset(name = 'Radav ' + timepoint, data = n.vstack((curve.xdata, curve.ydata)) )
        
        f.close()

# -----------------------------------------------------------------------------
#           TESTING FUNCTION FOR PLOTTING DYNAMIC DATA 
# -----------------------------------------------------------------------------

def plotTimeResolved(filename):    
    
    f = h5py.File(filename, 'r')
    times = f.keys()
    times.sort()
    datasets = [(f[time]['Radav ' + time], time) for time in times]
    
    #Plotting
    import matplotlib.pyplot as plt
    for dataset in datasets:
        data, time = dataset
        plt.plot(data[0], data[1], label = str(time))
        plt.legend()