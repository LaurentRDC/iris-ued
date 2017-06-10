"""
A collection of functions useful to the operation of Iris.
"""
import glob
from itertools import product
import numpy as n
from os.path import join
from functools import partial
from skimage.io import imread

def scattering_length(radius, energy, pixel_width = 14e-6, camera_distance = 0.2235):
    """
    Returns the scattering length s = G/4pi for an array of radius data in pixels.
    
    Parameters
    ----------
    radius : array-like, shape (N,)
        Radius from center of diffraction pattern [px]
    energy : numerical
        Electron energy [kV]
    pixel_width : numerical
        CCD pixel width [m]
    camera_distance : float, optional
        Sample-to-CCD distance [m]

    Returns
    -------
    out : ndarray, shape (N,)
        Scattering length
        
    Notes
    -----
    Default values for pixel width and camera distance correspond to experimental
    values for the Siwick diffractometer as of April 2016.
    """
    m = 9.109*10**(-31)     #in kg
    e = 1.602*10**(-19)     #in C
    c = 299792458*(10**10)  #in m/s
    h = 6.63*10**(-34)      #in J*s
    V = energy * 1000       #in eV

    e_wavelength_angs = 1e10*n.sqrt((h**2*c**2)/(e*V*(2*m*c**2+e*V)))

    radius = n.array(radius) * pixel_width
    diffraction_half_angle = n.arctan(radius/camera_distance)/2
    return n.sin(diffraction_half_angle)/e_wavelength_angs

def gaussian2D(x, y, xc, yc, sigma_x, sigma_y):
    """ 
    Returns a Gaussian with integrated area of 1.
    
    Parameters
    ----------
    x, y: ndarrays, shape (M,N)
        Points over which to calculate the gaussian distribution
    xc, yc : floats
        Center of the gaussian
    sigma_x, sigma_y : floats
        Standard deviation in specific directions.
    
    Returns
    -------
    gaussian : ndarray, shape (M,N)
    """
    norm = 1.0/(2*n.pi*sigma_x*sigma_y)
    exponent = ( ((x-xc)**2)/(2*sigma_x**2) + ((y-yc)**2)/(2*sigma_y**2) )
    return norm*n.exp(-exponent)

def fluence(incident_pulse_power, laser_reprate, FWHM, sample_size = [250,250]):
    """ 
    Calculates fluence given a 2D guaussian FWHM values (x and y) in microns and an incident pulse energy.
    
    Parameters
    ----------
    incident_pulse_power : float
        Incident pump beam power in mW
    laser_reprate : float
       Laser repetition rate in Hz.
    FWHM: numerical or list
        Laser beam FWHM in microns. 
        If provided as a single numerical (int or float), it is assumed that the laser
        beam is radially symmetric. If provided as a list, it is assumed that 
        FWHM = [FWHM_x, FWHM_y]
    sample_size : numerical or list (optional)
        sample dimensions in microns. Default is 250um by 250um
        If provided as a numerical (int or float), it is assumed that the sample
        is square. If provided as a list, it is assumed that sample_size = [width_x, width_y]
    
    Returns
    -------
    fluence : float
        Fluence in mJ/cm**2.
    """
    #Unit conversion: microns to centimeters
    um_to_cm = 1.0/10000.0
    
    #Distribute FWHM values correctly
    if isinstance(FWHM, (list, tuple)):
        FWHM_x, FWHM_y = FWHM
    else:
        FWHM_x, FWHM_y = float(FWHM), float(FWHM)
    
    #Distribute smaple_size correctly
    if not isinstance(sample_size, list):
        sample_size = [sample_size, sample_size]
    
    #Everything is in either mJ or cm
    FWHM_x = FWHM_x *um_to_cm
    FWHM_y = FWHM_y *um_to_cm
    sample_size = [sample_size[0]*um_to_cm, sample_size[1]*um_to_cm]    
    
    step = 0.5*um_to_cm                                     # Computational step size in cm
    maxRange = 500*um_to_cm                                 # Max square dimensions of the sample
    xRange = n.arange(-maxRange, maxRange + step, step)     # Grid range x, cm
    yRange = n.arange(-maxRange, maxRange + step, step)     # Grid range y, cm
    
    # From FWHM to standard deviation: http://mathworld.wolfram.com/GaussianFunction.html
    wx = FWHM_x/2.35    
    wy = FWHM_y/2.35
    
    # Distribute total beam energy over 2D Gaussian
    incident_pulse_energy = incident_pulse_power/laser_reprate     #energy un mJ
    xx, yy = n.meshgrid(xRange, yRange, indexing = 'xy')  
    energy_profile = incident_pulse_energy * gaussian2D(xx, yy, 0, 0, wx, wy)    
    
    #Find which indices of the Gaussian are not on the sample
    xlim, ylim = sample_size[0]/2, sample_size[1]/2
    not_on_sample_x = n.logical_xor( xx >= -xlim, xx <= xlim)
    not_on_sample_y = n.logical_xor( yy >= -ylim, yy <= ylim)
    energy_profile[ n.logical_or(not_on_sample_x, not_on_sample_y) ] = 0    

    #Integrate over sample
    dx, dy = step, step
    energy_on_sample = n.sum(energy_profile)*dx*dy  # in millijoules
    sample_area = sample_size[0]*sample_size[1]     # in cm^2
    
    return energy_on_sample/sample_area             #in mJ/cm**2