# -*- coding: utf-8 -*-
#Basics
import numpy as n
import scipy.optimize as opt

# -----------------------------------------------------------------------------
#           FIND CENTER OF DIFFRACTION PATTERN
# -----------------------------------------------------------------------------

def fCenter(xg, yg, rg, im):
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
    c = lambda x: circ(x[0],x[1],x[2],im)

    return opt.minimize(c,[xg,yg,rg]).x

def circ(xg,yg,rg,im):
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
    
    xMat, yMat = n.meshgrid(n.linspace(1, s, s),n.linspace(1, s, s))
    # find coords on circle and sum intensity
    xvals, yvals = n.where(((n.around(n.sqrt((xMat-xg)**2+(yMat-yg)**2))-n.around(rg)) < .1) & (yMat > 550))
    ftemp = n.sum(im[xvals, yvals])
    
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
    radius = n.around(n.sort(radius), decimals = 1)
    
    #Average intensity values for equal radii
    unique_radii, inverse = n.unique(radius, return_inverse = True)
    radial_average = n.zeros_like(unique_radii)
    
    for index, value in enumerate(unique_radii):
        relevant_intensity = intensity[n.where(inverse == value)]   #Find intensity that correspond to the radius 'value'
        if relevant_intensity.size == 0:
            radial_average[index] = 0
        else:
            radial_average[index] = n.mean(relevant_intensity)          #Average intensity
    
    return unique_radii, radial_average

# -----------------------------------------------------------------------------
#           INELASTIC SCATTERING BACKGROUND SUBSTRACTION
# -----------------------------------------------------------------------------