# -*- coding: utf-8 -*-
#Basics
import numpy as n
import scipy.optimize as opt

# -----------------------------------------------------------------------------
#           HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def biexp(x, a = 0, b = 0, c = 0, d = 0, e = 0):
    """ Returns a biexponential of the form a*exp(-b*x) + c*exp(-d*x) + e"""
    return a*n.exp(-b*x) + c*n.exp(-d*x) + e

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

# -----------------------------------------------------------------------------
#           I/O FUNCTIONS
# -----------------------------------------------------------------------------

def diffractionFileList(folder_path = 'C:\\' ):
    """
    returns a list of filenames corresponding to diffraction pictures
    """
    return
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

def radialAverage(images = list(), names = list(), center = [562,549]):
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
    im_shape = images[0].shape
    #Preliminaries
    xc, yc = center     #Center coordinates
    x = n.linspace(0, im_shape[0], im_shape[0])
    y = n.linspace(0, im_shape[1], im_shape[1])
    
    #Create meshgrid and compute radial positions of the data
    X, Y = n.meshgrid(x,y)
    R = n.around(n.sqrt( (X - xc)**2 + (Y - yc)**2 ), decimals = 0)
    
    results = list()
    for image, name in zip(images, names):
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
        results.append([unique_radii, n.divide(accumulation,bincount), name + ' radial average'])

    return results

def cutoff(patterns = list(), cutoff = [0,0]):
    """ Cuts off the radial patterns. """
    
    first_pattern = patterns[0]
    xdata, ydata, name = first_pattern
    
    #Find index of cutoff[0]
    cutoff_index = n.argmin(n.abs(xdata - cutoff[0]))
    
    #cut
    cut_patterns = list() 
    for pattern in patterns:
        xdata, ydata, name = pattern
        cut_patterns.append( [xdata[cutoff_index::], ydata[cutoff_index::], name] )

    return cut_patterns
# -----------------------------------------------------------------------------
#           INELASTIC SCATTERING BACKGROUND SUBSTRACTION
# -----------------------------------------------------------------------------
def prototypeInelasticBGSubstract(xdata, ydata, points = list(), chunk_size = 5):
    """ 
    Following Vance's inelastic background substraction method. We assume that the data has been corrected for diffuse scattering by substrate 
    (e.g. silicon nitride substrate for VO2 samples)
    
    In order to determine the shape of the background, we use a list of points selected by the user as 'diffraction feature'. These diffraction features are fit with 
    a pseudo-Voigt + constant. Concatenating this constant for multiple diffraction features, we can get a 'stair-steps' description of the background. We then smooth this
    data to get a nice background.
    
    Parameters
    ----------
    xdata, ydata : ndarrays, shape (N,)
    
    points : list of array-like
    """
    
    #Determine data chunks based on user-input points
    xfeatures = n.asarray(points)[:,0]      #Only consider x-position of the diffraction feature
    xchunks, ychunks= list(), list()
    for feature in xfeatures:
        #Find where in xdata is the feature
        ind = n.argmin(n.abs(xdata - feature))
        chunk_ind = n.arange(ind - chunk_size, ind + chunk_size + 1)    #Add 1 to stop parameter because chunk = [start, stop)
        xchunks.append(xdata[chunk_ind])
        ychunks.append(ydata[chunk_ind])
    
    #Fit a pseudo-Voigt + constant for each xchunk and save constant
    voigt_parameters = list()
    constants = list()
    for xchunk, ychunk in zip(xchunks, ychunks):
        temp_ychunk = ychunk - ychunk.min()         #Trick to get a better fit: remove most of the offset
        parameter_guesses = [temp_ychunk.max(), (xchunk.max()-xchunk.min())/2 + xchunk.min(), 0.1, 0.1, 0]
        opt_parameters = opt.curve_fit(pseudoVoigt, xchunk, ychunk, p0 = parameter_guesses)[0]
        voigt_parameters.append(opt_parameters)
        constants.append(opt_parameters[-1]*n.ones_like(xchunk) + ychunk.min())    # constant is the last parameter in the definition of pseudoVoigt
    
    #Extend constants to x-values outside xchunks
    constant_background = n.asarray(constants).flatten()
    x_background = n.asarray(xchunks).flatten()
    background = n.interp(xdata, x_background, constant_background)
    
    #Create diagnostics Voigt profiles
    profiles = list()
    for params in voigt_parameters:
        profiles.append( pseudoVoigt(xdata, params[0], params[1], params[2], params[3], params[4]) )
    
    #TODO: smooth background data
    return background, profiles
    
def inelasticBGSubstract(patterns, points = list()):
    """
    
    """
    #Create x arrays for the points 
    points = n.array(points) 
    x = points[:,0]
    
    biexponentials = patterns
    for pattern in patterns:
        xdata, ydata, name = pattern
        
        #Create guess 
        guesses = ydata.max(), 1/50, ydata.max()/2, 1/150, ydata.min() 
        
        #Interpolate the values of the patterns at the x points
        y = n.interp(x, xdata, ydata)
        
        #Fit with guesses 
        optimal_parameters, parameters_covariance = opt.curve_fit(biexp, x, y, p0 = guesses, maxfev = 20000) 
        
        #Create inelastic background function 
        #a,b,c,d,e = optimal_parameters 
        biexponentials.append([xdata, biexp(xdata, *optimal_parameters), 'Biexp ' + name]) 
        
    return biexponentials
