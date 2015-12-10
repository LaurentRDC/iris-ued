# -*- coding: utf-8 -*-
#Basics
import numpy as n
import scipy.optimize as opt

#const scale factor


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
    print xgscaled, ygscaled, rgscaled
    xMat, yMat = n.meshgrid(n.linspace(1, s, s),n.linspace(1, s, s))
    # find coords on circle and sum intensity
    
    residual = (xMat-xgscaled)**2+(yMat-ygscaled)**2-rgscaled**2
    xvals, yvals = n.where(((residual < 10) & (yMat > 550)))
    ftemp = n.mean(im[xvals, yvals])
    
    return 1/ftemp

# -----------------------------------------------------------------------------
#               RADIAL AVERAGING
# -----------------------------------------------------------------------------

def radialAverage(image, center = [0,0]):
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
    R = n.sqrt( (X - xc)**2 + (Y - yc)**2 )
    
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
    for xindex in xrange(0,image.shape[0]):
        for yindex in xrange(0,image.shape[1]):
            r = n.around(n.sqrt((xindex - xc)**2+(yindex-yc)**2))
            #bin
            ind = n.where(unique_radii==r)
            #increment
            radial_average[ind] += image[xindex,yindex]
            bincount[ind] += 1
            
    for rindex in xrange(0,radial_average.shape[0]):
        #normalize
        radial_average = radial_average/bincount
        
    return [unique_radii, radial_average]

# -----------------------------------------------------------------------------
#           INELASTIC SCATTERING BACKGROUND SUBSTRACTION
# -----------------------------------------------------------------------------

def biexp(x, a, b, c, d, e):
    """ Returns a biexponential of the form a*exp(-b*x) + c*exp(-d*x)+e """
    return a*n.exp(-b*x) + c*n.exp(-d*x) + e

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