# -*- coding: utf-8 -*-

from __future__ import division
import numpy as n

def gaussian2D(x, y, xc, yc, sigma_x, sigma_y):
    """ Returns a Gaussian with maximal height of 1 (not area of 1)."""
    exponent = ( ((x-xc)**2)/(2*sigma_x**2) + ((y-yc)**2)/(2*sigma_y**2) )
    return n.exp(-exponent)

def gaussianFluenceFunction(incident_pulse_energy, FWHM_x, FWHM_y, sample_size = [250,250]):
    """ 
    Calculates fluence given a 2D guaussian FWHM values (x and y) in microns and an incident pulse energy in uJ.
    
    Parameters
    ----------
    incident_pulse_energy : float
        Incident pump beam energy in uJ
    
    """
    global gaussian
    
    #Setup
    step = 0.5
    maxRange = 500                               # Max square dimensions of the sample
    xRange = n.arange(-maxRange, maxRange, step) # Grid range x, microns
    yRange = n.arange(-maxRange, maxRange, step) # Grid range y, microns
    
    # From FWHM to standard deviation: http://mathworld.wolfram.com/GaussianFunction.html
    wx = FWHM_x/2.35    
    wy = FWHM_y/2.35
    
    # Calculate 2D Gaussian (max intensity of 1)
    xx, yy = n.meshgrid(xRange, yRange, indexing = 'ij')  
    gaussian = gaussian2D(xx, yy, 0, 0, wx, wy)    
    energy_profile = gaussian*incident_pulse_energy
    
    #Find which indices of the Gaussian correspond to a spot on the sample
    xlim = (sample_size[0]/2)
    ylim = (sample_size[1]/2)
    x_cond = n.logical_and( xx >= -xlim, xx <= xlim)
    y_cond = n.logical_and( yy >= -ylim, yy <= ylim)
    energy_profile[ n.logical_and(x_cond, y_cond) ] = 0    #Bitwise &(and) is used 
    
    #Integrate over sample
    dx, dy = step, step
    energy_on_sample = n.sum(energy_profile)*dx*dy  # in microjoules
    sample_area = sample_size[0]*sample_size[1]     # in microns
    
    #Change units: 
    # mj = 1000*uj; cm = 10 000*microns
    # uj/microns^2 -> 1000/(10 000)**2
    units_factor = 1000.0/(10000**2)
    
    return units_factor*(energy_on_sample/sample_area) #in mJ/cm**2
    
if __name__ == '__main __':
    
    gaussianFluenceFunction()