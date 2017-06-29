"""
A collection of functions useful to the operation of Iris.
"""
import glob
from itertools import product
import numpy as n
from numpy.linalg import inv, eig
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

def ellipse_fit(x, y, circle_constraint = False):
    """
    Returns the ellipse parameters that fits data points. The special 
    case of a circle can also be treated.

    Parameters
    ----------
    x, y : ndarrays, shape (N,)
        Data points to be fitted.
    circle_constraint : bool, optional
        If True, a condition is added to fit a circle.
        Default is False.
    
    Returns
    -------
    a, b, c, d, e, f : float
        Conic section parameters for an ellipse.

    Raises
    ------
    RunTimeError
        If a solution cannot be found.

    Notes
    -----
    In the case of an ellipse, b**2 - 4*a*c < 0, while for a circle, b**2 - 4*a*c = 0.
    From the parameters, the center of the ellipse is given by (xc, yc) = (d/2, e/2) 

    References
    ----------
    .. [#] Halir, R.; Flusser, J. Numerically Stable Direct Least Squares Fitting of Ellipses. 
        Proceedings of the 6th International Conference in Central Europe on Computer Graphics
         and Visualization, Plzeň, Czech Republic, 9–13 February 1998; pp. 125–132.
    """
    x, y = n.asarray(x, dtype = n.float), n.asarray(y, dtype = n.float)
    x, y = x.ravel(), y.ravel()

    # Translation of the code in Fig 2: quadratic part of the design matrix
    D1 = n.zeros(shape = (x.size, 3), dtype = n.float)
    D1[:,0] = x**2
    D1[:,1] = x*y
    D1[:,2] = y**2

    if circle_constraint:
        D1[:,1] = 0     # equivalent to forcing b = 0, or no x*y terms.

    # Linear part of the design matrix
    D2 = n.zeros(shape = (x.size, 3), dtype = n.float)
    D2[:,0] = x
    D2[:,1] = y
    D2[:,2] = 1

    # scatter matrix
    S1 = n.dot(D1.conj().transpose(), D1)      #S1
    S2 = n.dot(D1.conj().transpose(), D2)      #S2
    S3 = n.dot(D2.conj().transpose(), D2)      #S3

    C1 = n.array([ [0, 0, 2], [0, -1, 0], [2, 0, 0] ])
    
    # Solve 
    T = n.dot( -inv(S3), S2.conj().transpose() )
    M = S1 + n.dot(S2, T)
    M = n.dot(-inv(C1), M) 

    # Solution is the eigenvector associated with the minimal
    # nonzero eigenvalue of M.
    # To do this, we change the negative eigenvalues to +infinity
    # and find the minimal element of vals
    vals, vecs = eig(M)
    if vals.max() < 0: 
        raise RuntimeError('Unstable system?')

    vals[vals < 0] = n.inf
    min_pos_eig = vals.argmin()
    a1 = vecs[:, min_pos_eig].ravel()
    a2 = n.dot(T, a1) # Second half of coefficients

    # Ellipse coefficients from eq. 1
    # a, b, c, d, e, f = tuple(a1) + tuple(a2)
    return tuple(a1) + tuple(a2)

def ellipse_center(x, y):
    """
    High-level function to find the center of an ellipse.

    Parameters
    ----------
    x, y : ndarrays, shape (N,)
        Data points to be fitted.
    
    Returns
    -------
    xc, yc : float
        Center position
    """
    a, b, c, d, e, f = ellipse_fit(x, y)
    return -d/(2*a), -e/(2*c)

MASK_CACHE = dict()
def find_center(image, guess_center, radius, window_size = 10, ring_width = 10):
    """
    Fit a diffraction center from polycrystalline data.

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
    
    # Reduce image size down to the bounding bx that encompasses
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
        return reduced[MASK_CACHE[c]].sum()
    
    # TODO: average centers with the same max intensity
    (best_x, best_y), _ =  max(zip(centers, map(integrated, centers)), key = lambda x: x[-1])
    return best_x, best_y
