"""
A collection of functions useful to the operation of Iris.
"""
import glob
from itertools import product
import numpy as n
from os.path import join
from scipy.optimize import brute, fmin_slsqp
from .io import RESOLUTION, ImageNotFoundError, read

def electron_wavelength(kV, units = 'meters'):
    """ 
    Returns the relativistic wavelength of an electron.
        
    Parameters
    ----------
    kV : float
        Voltage in kilovolts of the instrument
    units : str, optional
        Acceptable units are 'meters' or 'm', and 'angstroms' or 'A'.
    """
    m = 9.109*10**(-31)     #in kg
    e = 1.602*10**(-19)     #in C
    c = 299792458*(10**10)  #in m/s
    h = 6.63*10**(-34)      #in J*s
    V = kV * 1000
    
    #from: en.wikipedia.org/wiki/Electron_diffraction#Wavelength_of_electrons
    wavelength_meters = n.sqrt((h**2*c**2)/(e*V*(2*m*c**2+e*V)))
    if units == 'meters' or units == 'm':
        return wavelength_meters
    elif units == 'angstroms' or units.lower() == 'a':
        return wavelength_meters*(10**10)
    else:
        raise ValueError('Invalid units')

def scattering_length(radius, energy, pixel_width = 14e-6, camera_distance = 0.2235):
    """
    Returns the scattering length s = G/4pi for an array of radius data in pixels.
    
    Parameters
    ----------
    radius : array-like, shape (N,)
        Radius from center of diffraction pattern, in units of pixels.
    pixel_width : numerical
        CCD pixel width in meters.
    energy : numerical
        Electron energy in keV.
        
    Notes
    -----
    Default values for pixel width and camera distance correspond to experimental
    values for the Siwick diffractometer as of April 2016.
    """
    radius = n.array(radius) * pixel_width
    diffraction_half_angle = n.arctan(radius/camera_distance)/2
    return n.sin(diffraction_half_angle)/electron_wavelength(energy, units = 'angstroms')

def shift(arr, x, y):
    """
    Shift an array to center.

    Parameters
    ----------
    arr : ndarray
    x : int
    y : int

    Returns
    -------
    out : MaskedArray
    """
    shifted = n.empty_like(arr)
    shifted.fill(n.nan)

    if x > 0:
        x_source = slice(0, -x)
        x_cast = slice(x, None)
    elif x < 0:
        x_source = slice(-x, None)
        x_cast = slice(0, x)
    else:
        x_source = slice(0, None)
        x_cast = slice(0, None)

    if y > 0:
        y_source = slice(0, -y)
        y_cast = slice(y, None)
    elif y < 0:
        y_source = slice(-y, None)
        y_cast = slice(0, y)
    else:
        y_source = slice(0, None)
        y_cast = slice(0, None)
    
    shifted[x_cast, y_cast] = arr[x_source, y_source]
    return n.ma.array(shifted, mask = n.isnan(shifted), fill_value = 0.0)

def average_tiff(directory, wildcard, background = None):
    """
    Averages images matching a filename template within the dataset directory.
    
    Parameters
    ----------
    directory : str
        Absolute path to the directory
    wildcard : string
        Filename wildcard, e.g. 'background.*.pumpon.tif', '*.tif', etc.
    background : array-like, optional
        Background to subtract from the average.
        
    Returns
    -------
    out : ndarray
    
    Raises
    ------
    ImageNotFoundError
        If wildcard does not match any file in the directory
    """ 
    #Format background correctly
    if background is not None:
        background = background.astype(n.float)
    else:
        background = n.zeros(shape = RESOLUTION, dtype = n.float)
    
    #Get file list
    image_list = glob.glob(join(directory, wildcard))
    if not image_list:      #List is empty
        raise ImageNotFoundError('wildcard does not match any file in the dataset directory')
    
    image = n.zeros(shape = RESOLUTION, dtype = n.float)
    for filename in image_list:
        image += read(filename)
        
    # average - background
    return image/len(image_list) - background

MASK_CACHE = dict()
def find_center(image, guess_center, radius, window_size = 10, ring_width = 10):
    """
    Find the best guess for diffraction center.

    Parameters
    ----------
    image : ndarray, ndim 2
        Invalid pixels (such as pixels under the beamblock) should be represented by NaN

    center : 2-tuple

    radius : int

    window_size : int, optional

    ring_width : int, optional
    """
    xx, yy = n.meshgrid(n.arange(0, image.shape[0]), n.arange(0, image.shape[1]))
    xc, yc = guess_center
    centers = product(range(xc - window_size, window_size + xc + 1),
                      range(yc - window_size, window_size + yc + 1))
    
    # Reduce image size down to the bounding box that encompasses
    # all possible circles
    extra = window_size + ring_width + radius
    reduced = image[yc - extra:yc + extra, xc - extra:xc + extra]
    xx = xx[yc - extra:yc + extra, xc - extra:xc + extra]
    yy = yy[yc - extra:yc + extra, xc - extra:xc + extra]

    def integrated(c):
        """ Integrate intensity over the ring """
        if c not in MASK_CACHE:
            rr = n.sqrt((xx - c[0])**2 + (yy - c[1])**2)
            MASK_CACHE[c] = n.logical_and(rr >= radius - ring_width, rr <= radius + ring_width)
        return 1/(1+reduced[MASK_CACHE[c]].sum())
    
    # TODO: average centers with the same max intensity
    (best_x, best_y), _ =  max(zip(centers, map(integrated, centers)), key = lambda x: x[-1])
    return best_x, best_y

def angular_average(image, center, beamblock_rect, mask = None):
    """
    This function returns a radially-averaged pattern computed from a TIFF image.
    
    Parameters
    ----------
    image : ndarray
        image data from the diffractometer.
    center : array-like shape (2,)
        [x,y] coordinates of the center (in pixels).
    beamblock_rect : Tuple, shape (4,)
        Tuple containing x- and y-bounds (in pixels) for the beamblock mask
        mast_rect = (x1, x2, y1, y2)
    mask : ndarray or None, optional
        Array of booleans that evaluates to False on pixels that should be discarded.
        If None (default), all pixels are treated as valid (except for beamblock)
        
    Returns
    -------
    radius : ndarray, shape (N,)
    intensity : ndarray, shape (N,)
    error : ndarray, shape (N,)
    
    Raises
    ------
    ValueError
        If 'mask' is an array of a different shape than the image shape.
    """    
    #preliminaries
    if mask is None:
        mask = n.ones_like(image, dtype = n.bool)
    elif mask.shape != image.shape:
        raise ValueError("'mask' array (shape {}) must have the same shape as 'image' (shape {})".format(mask.shape, image.shape))
    
    image = image.astype(n.float)
    mask = mask.astype(n.bool)
    
    xc, yc = center     #Center coordinates
    x1, x2, y1, y2 = beamblock_rect     
    
    #Create meshgrid and compute radial positions of the data
    X, Y = n.meshgrid(n.arange(0, image.shape[0]), n.arange(0, image.shape[1]))
    R = n.sqrt( (X - xc)**2 + (Y - yc)**2 )
    
    #radii beyond r_max don't fit a full circle within the image
    r_max = min((X.max()/2, Y.max()/2))           #Maximal radius that fits completely in the image
    
    r_min = 0 # = min([ n.sqrt((xc - x1)**2 + (yc - y1)**2), n.sqrt((xc - x2)**2 + (yc - y2)**2) ])
    if x1 < xc < x2 and y1 < yc < y2:
        r_min = min([ n.sqrt((xc - x1)**2 + (yc - y1)**2), n.sqrt((xc - x2)**2 + (yc - y2)**2) ])

    # Replace all values in the image corresponding to beamblock or other irrelevant
    # data by 0: this way, it will never count in any calculation because image
    # values are used as weights in numpy.bincount
    # Create a composite mask that uses the pixels mask, beamblock mask, and maximum/minimum
    # radii     
    composite_mask = n.array(mask, dtype = n.bool)      # start from the pixel mask
    composite_mask[R > r_max] = False
    composite_mask[R < r_min] = False
    composite_mask[x1:x2, y1:y2] = False
    image[ n.logical_not(composite_mask) ] = 0    # composite_mask is false where pixels should be disregarded
    
    #Radial average
    px_bin = n.bincount(R.ravel().astype(n.int), weights = image.ravel())
    r_bin = n.bincount(R.ravel().astype(n.int))  
    radial_intensity = px_bin/r_bin
    
    # Compute the error as square-root-N
    radial_intensity_error = n.sqrt(px_bin)/r_bin
    
    # Only return values with radius between r_min and r_max
    radius = n.unique(R.ravel().astype(n.int))      # Round up to integer number of pixel
    r_max_index = n.argmin(n.abs(r_max - radius))
    r_min_index = n.argmin(n.abs(r_min - radius))
    
    return radius[r_min_index + 1:r_max_index], radial_intensity[r_min_index + 1:r_max_index], radial_intensity_error[r_min_index + 1:r_max_index]

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