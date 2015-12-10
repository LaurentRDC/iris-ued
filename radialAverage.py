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
    radius = n.around(n.sort(radius), decimals = 2)
    
    #Average intensity values for equal radii
    unique_radii, inverse = n.unique(radius, return_inverse = True)
    radial_average = n.zeros_like(unique_radii)
    
    for index, value in enumerate(unique_radii):
        relevant_intensity = intensity[n.where(inverse == value)]   #Find intensity that correspond to the radius 'value'
        radial_average[index] = n.mean(relevant_intensity)          #Average intensity
    
    return unique_radii, radial_average