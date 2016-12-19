"""
A collection of functions useful to the operation of Iris.
"""
import glob
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

def shift(arr, x, y, fill = n.nan):
    """
    Shift an array to center.

    Parameters
    ----------
    x : int
    y : int
    fill : numeric
    """
    canvas = n.empty_like(arr)
    if x > 0:
        canvas[0:x, :] = fill
        canvas[x::, :] = arr[0:-x, :]
    elif x < 0:
        canvas[x::, :] = fill
        canvas[0:x, :] = arr[-x::, :]
    else:
        canvas[:,:] = arr
    
    if y > 0:
        canvas[:, 0:y] = fill
        canvas[:, y::] = arr[:, 0:-y]
    elif y < 0:
        canvas[:, y::] = fill
        canvas[:, 0:y] = arr[:, -y::]
    
    return canvas

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

def find_center(image, guess_center, radius, window_size = 10, ring_width = 5, method = 'opt'):
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

    method : str {'brute', 'opt' (default)}
    """
    # Reduce image size for performance
    xc, yc = guess_center
    reduced = image[xc - window_size - radius - ring_width : xc + window_size + radius + ring_width + 1,
                    yc - window_size - radius - ring_width : yc + window_size + radius + ring_width + 1]

    # All relevant quantities expressed in the reference frame of the 
    # reduced image
    # Meshgrid built so that the guess center is (0,0)
    extent = n.arange(- (window_size + radius + ring_width), window_size + radius + ring_width + 1)
    xx, yy = n.meshgrid(extent, extent)

    def integrated(center):
        """ Integrate intensity over the ring """
        rr = n.sqrt((xx - center[0])**2 + (yy - center[1])**2)
        ring = reduced[n.logical_and(radius - ring_width <= rr, rr <= radius + ring_width)]
        return 1/(1 + n.nansum(ring))
    
    if method == 'brute':
        best_x, best_y = brute(func = integrated,
                               ranges = (slice(-window_size, window_size, 1), slice(-window_size, window_size, 1)))
    elif method == 'opt':
        best_x, best_y = fmin_slsqp(func = integrated, 
                                    x0 = n.array((0,0)),
                                    bounds = [(-window_size, window_size), (-window_size, window_size)],
                                    iprint = 0)
    
    # Re-express best center (best_x, best_y) into coordinates of the full image
    return round(best_x + xc), round(best_y + yc)

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
    
    # #TODO: Find the smallest circle that completely fits inside the mask rectangle
    r_min = 0 # = min([ n.sqrt((xc - x1)**2 + (yc - y1)**2), n.sqrt((xc - x2)**2 + (yc - y2)**2) ])

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