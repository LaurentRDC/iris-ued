# -*- coding: utf-8 -*-

import numpy as n
from PIL.Image import open 

filename = 'C:\Users\Laurent\Dropbox\Powder\VO2\NicVO2\NicVO2_2.tif

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
    return