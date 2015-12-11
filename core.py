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

def radialAverage(image, center = [512,512]):
    """
    This function returns a radially-averaged pattern computed from a TIFF image.
    
    Parameters
    ----------
    image : ndarray, shape(N,N)
    
    center : array-like, shape (2,)
        [x,y] coordinates of the center (in pixels)
    
    Returns
    -------
    [radius, pattern] : list of ndarrays, shapes (M,)
    """
    
    #Preliminaries
    xc, yc = center     #Center coordinates
    x = n.linspace(0, image.shape[0], image.shape[0])
    y = n.linspace(image.shape[1], 0, image.shape[1])
    
    #Create meshgrid and compute radial positions of the data
    X, Y = n.meshgrid(x,y)
    R = n.around(n.sqrt( (X - xc)**2 + (Y - yc)**2 ), decimals = 0)
    
    #Flatten arrays
    intensity = image.flatten()
    radius = R.flatten()
    
    #Sort by increasing radius
    intensity = intensity[n.argsort(radius)]
    radius = n.around(n.sort(radius), decimals = 0)
    
    #Average intensity values for equal radii
    unique_radii, inverse = n.unique(radius, return_inverse = True)
    radial_average = n.zeros_like(unique_radii)
    bincount =  n.zeros_like(unique_radii)
    
    #loop over image
    for (i,j), value in n.ndenumerate(image):
        r = R[i,j]
        #bin
        ind = n.where(unique_radii==r)[0][0]
        #increment
        radial_average[ind] += value
        bincount[ind] += 1
        
    #Return normalized radial average
    return [unique_radii, n.divide(radial_average,bincount)]

# -----------------------------------------------------------------------------
#           INELASTIC SCATTERING BACKGROUND SUBSTRACTION
# -----------------------------------------------------------------------------
def prototypeIBS(xdata, ydata, points = list(), chunk_size = 20):
    """ 
    Following Vance's inelastic background substraction method. We assume that the data has been corrected for diffuse scattering by substrate 
    (e.g. silicon nitride substrate for VO2 samples)
    
    In order to determine the shape of the background, we use a list of points selected by the user as 'diffraction feature'. These diffraction features are fit with 
    a pseudo-Voigt + constant. Concatenating this constant for multiple diffraction features, we can get a 'stair-steps' description of the background. We then smooth this
    data to get a nice background.
    """
    
    #Determine data chunks based on user-input points
    xfeatures = n.asarray(points)[:,0]      #Only consider x-position of the diffraction feature
    xchunks, ychunks,chunk_indices = list(), list(), list()
    for feature in xfeatures:
        #Find where in xdata is the feature
        ind = n.argmin(n.abs(xdata - feature))
        chunk_ind = n.arange(ind - chunk_size, ind + chunk_size + 1)    #Add 1 to stop parameter because chunk = [start, stop)
        xchunks.append(xdata[chunk_ind])
        ychunks.append(ydata[chunk_ind])
        chunk_indices.append(chunk_ind)
    
    #Fit a pseudo-Voigt + constant for each xchunk and save constant
    constants = list()
    for xchunk, ychunk in zip(xchunks, ychunks):
        parameter_guesses = [ychunk.max()-ychunk.min(), (xchunk.max()-xchunk.min())/2, 1, 1, ychunk.min()]
        opt_parameters = opt.curve_fit(pseudoVoigt, xchunk, ychunk, p0 = parameter_guesses)[0]        
        constants.append(opt_parameters[-1])    # constant is the last parameter in the definition of pseudoVoigt
    
    #Extend constants to x-values outside xchunks
    constant_background = n.zeros_like(xdata) #preallocation
    for constant, indices in zip(constants,chunk_indices):
        constant_background[indices] = constant
    
    return constant_background
    
def inelasticBGSubstract(xdata, ydata, points = list()):
    """
    Returns the radial diffraction pattern with the inelastic scattering background removed.
    
    Parameters
    ----------
    xdata, ydata : ndarrays, shape (N,)
    
    
    """
    
    #Create guess
    guesses = ydata.max()/2, 1/50, ydata.max()/2, 1/150, ydata.min()
    
    #Create x and y arrays for the points
    points = n.array(points)
    x, y = points[:,0], points[:,1]
    
    #Fit with guesses
    optimal_parameters, parameters_covariance = opt.curve_fit(biexp, x, y, p0 = guesses)
    
    #Create inelastic background function
    a,b,c,d,e = optimal_parameters
    background_ordinate = biexp(xdata, a, b, c, d, e)

    return [xdata, ydata - background_ordinate, 'Radial diffraction pattern']