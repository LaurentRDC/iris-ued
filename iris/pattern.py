# -*- coding: utf-8 -*-

#Basics
import matplotlib.pyplot as plt
import numpy as n
from scipy.ndimage.filters import maximum_filter
import scipy.optimize as opt
import wavelet

class Pattern(object):
    """
    This class represents single-crystal diffraction patterns.
    
    Attributes
    ----------
    data : ndarray, shape (M,N)
        Image data as an array.
    name : str
        Description.
    
    Methods
    -------
    inelastic_background
        Fits a biexponential inelastic scattering background to the data
    """
    def __init__(self, image, name = ''):
        """
        Parameters
        ----------
        image : array-like, shape (M,N)
            Image data
        """
        self.data = n.asarray(image, dtype = n.float)
        self.name = name
    
    def __sub__(self, pattern):
        return Pattern(data = self.data - pattern.data, name = self.name)
    
    def __add__(self, pattern):
        return Pattern(data = self.data + pattern.data, name = self.name)
    
    def __copy__(self):
        return Pattern(data = self.data, name = self.name)
    
    def plot(self):
        from matplotlib.pyplot import imshow
        imshow(self.data)
    
    def inelastic_background(self, points = [], level = 10):
        """
        Master method for inelastic background determination via wavelet decomposition.
        
        Parameters
        ---------
        points : list, optional
            List of (i,j) at which the pattern is entirely background.
        level : int, optional
            Wavelet decomposition level. A higher level implies a coarser approximation 
            to the baseline.
        
        Returns
        -------
        background : Pattern object
            Background pattern.
        """
        background = wavelet.baseline(array = self.data,
                                             max_iter = 50,
                                             level = level,
                                             wavelet = 'db10',
                                             background_regions = points)
        
        return Pattern(data = background)