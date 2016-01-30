# -*- coding: utf-8 -*-
#Basics
from __future__ import division
import numpy as n
from numpy import pi
import scipy.optimize as opt
import matplotlib.pyplot as plt #For testing only

#Batch processing libraries
import os.path
import h5py
import tifffile as t
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

def generateCircle(xc, yc, radius):
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

class RadialCurve(object):
    """
    This class represents any radially averaged diffraction pattern or fit.
    """
    def __init__(self, xdata = n.zeros(shape = (100,)), ydata = n.zeros(shape = (100,)), name = '', color = 'b'):
        
        self.xdata = n.asarray(xdata)
        self.ydata = n.asarray(ydata)
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
     optimized center and radius
     """ 
     #find maximum intensity 
     xgscaled, ygscaled, rgscaled = [qty/scalefactor for qty in [xg,yg,rg]] 
     s = im.shape[0] 
     xMat, yMat = n.meshgrid(n.linspace(1, s, s),n.linspace(1, s, s)) 
     
     def circ(params): 
         """ Sums the intensity over a circle of given radius and center position on an image. """

         xg, yg, rg = params
         xgscaled, ygscaled, rgscaled = [qty*scalefactor for qty in [xg,yg,rg]]     
         
         # find coords on circle and sum intensity 
         residual = (xMat-xgscaled)**2+(yMat-ygscaled)**2-rgscaled**2 
         xvals, yvals = n.where(((residual < 25) & (yMat > s/2.0))) 
         ftemp = n.mean(im[xvals, yvals]) 
          
         return 1.0/ftemp
         
     xcenter, ycenter, rcenter = n.array(opt.minimize(circ,[xgscaled,ygscaled,rgscaled],method = 'Nelder-Mead').x)*scalefactor 
     rcenter = rg     
     return xcenter, ycenter, rcenter  
    
# -----------------------------------------------------------------------------
#               RADIAL AVERAGING
# -----------------------------------------------------------------------------

def radialAverage(image, name, center, mask_rect = None):
    """
    This function returns a radially-averaged pattern computed from a TIFF image.
    
    Parameters
    ----------
    image : ndarray
        image data from the diffractometer.
    center : array-like, shape (2,)
        [x,y] coordinates of the center (in pixels)
    name : str
        String identifier for the output RadialCurve
    mask_rect : Tuple, shape (4,)
        Tuple containing x- and y-bounds (in pixels) for the beamblock mask
        mast_rect = (x1, x2, y1, y2)
        
    Returns
    -------
    RadialCurve object
    """

    #preliminaries
    image = image.astype(n.float)
    xc, yc = center     #Center coordinates
    
    #Create meshgrid and compute radial positions of the data
    X, Y = n.meshgrid(n.arange(0,image.shape[0],1), n.arange(0, image.shape[1],1))
    R = n.around(n.sqrt( (X - xc)**2 + (Y - yc)**2 ), decimals = 0)         #Round radius down/up to the nearest pixel
    
    #radii beyond r_max don't fit a full circle within the image
    image_edge_values = n.array([R[0,:], R[-1,:], R[:,0], R[:,-1]])
    r_max = n.min(n.array(image_edge_values))           #Maximal radius that fits completely in the image
    
    # Replace all values in R corresponding to beamblock or other irrelevant
    # data by -1: this way, it will never count in any calculation
    R[R > r_max] = -1
    if mask_rect is None:
        R[:xc, :] = -1      #All poins above center of the image are disregarded (because of beamblock)
    else:
        x1, x2, y1, y2 = mask_rect
        R[x1:x2, y1:y2] = -1
    
    #Average intensity values for equal radii
    radial_position = n.unique(R)
    radial_intensity = n.empty_like(radial_position) 
    
    #Radial average for realz
    for index, radius in n.ndenumerate(radial_position):
        radial_intensity[index] = n.mean(image[R == radius])
        
    #Return normalized radial average
    return RadialCurve(radial_position, radial_intensity, name)

# -----------------------------------------------------------------------------
#           BATCH FILE PROCESSING
# -----------------------------------------------------------------------------

class DiffractionDataset(object):
    
    def __init__(self, directory, resolution = (2048, 2048)):
        
        self.directory = directory
        self.resolution = resolution
        self.pumpon_background = self.averageTiffFiles('background.*.pumpon.tif')
        self.pumpoff_background = self.averageTiffFiles('background.*.pumpoff.tif')
        self.substrate = self.getSubstrateImage()
        self.time_points = self.getTimePoints()
        self.acquisition_date = self.getAcquisitionDate()
        self.exposure = None
        self.fluence = None
        
    def getAcquisitionDate(self):
        """ Returns the acquisition date from the folder name as a string of the form: '2016.01.06.15.35' """
        try:
            return re.search('(\d+[.])+', self.directory).group()[:-1]      #Last [:-1] removes a '.' at the end
        except(AttributeError):     #directory name does not match time pattern
            return '0.0.0.0.0'
    
    def getFluence(self):
        pass
    
    def getExposure(self):
        pass
        
    def getSubstrateImage(self):
        """ Finds and stores a substrate image, and returns an array of zeros if criterias are not matched. """
        substrate_filename = 'subs.tif'
        subs = n.zeros(shape = self.resolution, dtype = n.float)
        if substrate_filename in os.listdir(self.directory):
            print 'Substrate image found'
            absolute_path = os.path.join(self.directory, substrate_filename)
            subs = t.imread(absolute_path).astype(n.float)
            
            subs = subs - self.pumpoff_background
            subs[subs < 0] = 0
        
        return subs
    
    @staticmethod
    def castTo16Bits(array):
        array[ array < 0] = 0
        array[ array > (2**16) - 1] = (2**16) - 1
        return array.astype(n.uint16)
    
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
            background = background.astype(n.float)
        
        #Get file list
        image_list = glob.glob(os.path.join(self.directory, filename_template))
        
        if not image_list:      #List is empty
            raise ValueError('filename_template does not match any file in the dataset directory')
        
        image = n.zeros(shape = self.resolution, dtype = n.float)
        for filename in tqdm(image_list, nested = True):
            new_image = t.imread(filename).astype(n.float)
            image += new_image
            
        average = image/float(len(image_list))
            
        if background is not None:
            average -= background
        
        return average
            
    def dataAverage(self, time_point, export = False, substract_substrate = False):
        """         
        Parameters
        ----------
        time_point : string or numerical
            string in the form of +150.00, -10.00, etc. If a float or int is provided, it will be converted to a string of the correct format.
        """
        
        time_point = self.timeToString(time_point, units = False)
        glob_template = 'data.timedelay.' + time_point + '.nscan.*.pumpon.tif'
        
        #Average calculation
        average = self.averageTiffFiles(glob_template, self.pumpon_background)
        
        if substract_substrate:
            average = average - self.substrate
        
        if export:
            output_directory = os.path.join(self.directory, 'processed')
            if not os.path.isdir(output_directory):             #Find out if output directory exists and create if necessary
                os.makedirs(output_directory)                
            save_path = os.path.join(output_directory, 'data.timedelay.' + time_point + '.average.pumpon.tif')
            
            t.imsave(save_path, self.castTo16Bits(average))
        
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
        
        if units:
            return str_time + 'ps'
        else:
            return str_time
        
    def processImage(self, filename, center, cutoff, inelasticBGCurve, mask_rect):
        """
        Returns a processed radial curve associated with a radial diffraction pattern at a certain time point.
        
        Parameters
        ----------
        time : string or numerical
            Either a string formatted as {'+150.00', '-10.00'} or equivalent float or integer. See self.dataAverage
        TBD
        """
        image = t.imread(filename).astype(n.float)            
        curve = radialAverage(image, 'Radial average', center, mask_rect)     #Radially average
        curve = curve.cutoff(cutoff)                        #cutoff curve

        if inelasticBGCurve is not None:                    #substract inelastic scattering background if it is provided
            return curve - inelasticBGCurve
        else:
            return curve
        
    def batchProcess(self, center = [0,0], cutoff = [0,0], inelasticBGCurve = None, mask_rect = None):
        """
        Returns a list of RadialCurve objects (one for every time point)
        """
        
        results = list()
        for time in tqdm(self.time_points):
            filename = os.path.join(self.directory, 'processed', 'data.timedelay.' + self.timeToString(time) + '.average.pumpon.tif')
            curve = self.processImage(filename, center, cutoff, inelasticBGCurve, mask_rect)
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
    for dataset in datasets:
        data, time = dataset
        plt.plot(data[0], data[1], label = str(time))
        plt.legend()

if __name__ == '__main__':
    
    #Testing
    directory = 'K:\\2016.01.28.18.21.VO2_17mJ\\processed\\2016.01.28.18.21.radial.averages.hdf5'
    plotTimeResolved(directory)
    #d = DiffractionDataset(directory)
    #d.batchProcess(center = [937.4, 998.7], cutoff = [0,0], inelasticBGCurve = None, mask_rect = [926, 1049, 0, 1091])
