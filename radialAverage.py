# -*- coding: utf-8 -*-

import numpy as n
from PIL.Image import open 

filename = 'C:\Users\Laurent\Dropbox\Powder\VO2\NicVO2\NicVO2_2.tif'
im = n.array(open(filename))

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
    [s, pattern] : list of ndarrays, shapes (M,)
    """
    
    #Preliminaries
    x = n.asarray(range(image.shape[0]))
    y = n.asarray(range(image.shape[1]))
    
    #Create meshgrid and compute radial positions of the data
    X, Y = n.meshgrid(x,y)
    R = n.sqrt( (X - center[0])**2 + (Y - center[1])**2 )
    
    #Flatten arrays
    intensity = image.flatten()
    radius = R.flatten()
    
    #Sort by increasing radius
    intensity = intensity[n.argsort(radius)]
    radius = n.sort(radius)
    
    #Average intensity values for equal radii
    
    
    return