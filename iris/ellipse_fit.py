"""
Author : Laurent P. Rene de Cotret

Ref.:
[1] NUMERICALLY STABLE DIRECT LEAST SQUARES FITTING OF ELLIPSES
"""
from iris.wavelet import denoise, baseline
import matplotlib.pyplot as plt
import numpy as n
from numpy.linalg import inv, eig

import skimage
from skimage.segmentation import random_walker
from skimage.feature import canny

def ring_mask(shape, center, inner_radius, outer_radius):
    """
    Mark array representing a circlet (thick circle).

    Parameters
    ----------
    shape : 2-tuple

    center : 2-tuple

    inner_radius, outer_radius : numerical

    Returns
    -------
    mask : ndarray, dtype bool

    Raises
    ------
    ValueError
        If inner_radis > outer_radius
    """
    if inner_radius > outer_radius:
        raise ValueError('Inner radius {} must be smaller than the outer radius {}.'.format(inner_radius, outer_radius))
    x, y = n.arange(0, shape[0]), n.arange(0, shape[1])
    xx, yy = n.meshgrid(x, y, indexing = 'ij')
    xc, yc = center
    inner = (xx - xc)**2 + (yy - yc)**2 <= inner_radius**2
    outer = (xx - xc)**2 + (yy - yc)**2 <= outer_radius**2
    mask = outer
    mask[inner] = False

    return mask

def diffraction_center(image, mask = None):
    """
    Returns the diffraction center from a diffraction pattern. The mask must highlight
    one diffraction ring.

    Parameters
    ----------
    image : ndarray, ndim 2
        Grayscale image
    mask : ndarray, dtype bool
        Pixels where mask is False will be set to 0 (non-object pixels).

    Returns
    -------
    xc, yc : floats
        Center coordinates.
    """
    image -= baseline(image, max_iter = 10)
    image = skimage.filters.gaussian(image, sigma = 5)
    image -= n.min(image[mask])
    image[image < 0] = 0
    image[n.logical_not(mask)] = 0

    # Make into binary image
    binary = skimage.morphology.closing(image)
    binary = skimage.feature.canny(binary, sigma = 5, mask = mask)
    binary = skimage.morphology.remove_small_objects(binary, connectivity = 2)

    # From image to list of coordinates
    xx, yy = n.meshgrid(n.arange(binary.shape[0]), n.arange(binary.shape[1]), indexing = 'ij')
    x, y = xx[binary.astype(n.bool)].ravel(), yy[binary.astype(n.bool)].ravel()
    
    return ellipse_center(x, y)

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
    a, b, c, d, e, f : numerical
        Conic section parameters for an ellipse.

    Raises
    ------
    RunTimeError
        If a solution cannot be found.

    Notes
    -----
    In the case of an ellipse, b**2 - 4*a*c < 0, while for a circle, b**2 - 4*a*c = 0.
    From the parameters, the center of the ellipse is given by (xc, yc) = (d/2, e/2) 
    """
    x, y = n.asarray(x, dtype = n.float), n.asarray(y, dtype = n.float)
    x, y = x.ravel(), y.ravel()

    # Translation of the code in Fig 2, ref [1]
    # quadratic part of the design matrix
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
        raise RunTimeError('Unstable system?')
    vals[vals < 0] = n.inf
    min_pos_eig = vals.argmin()
    a1 = vecs[:, min_pos_eig].ravel()
    a2 = n.dot(T, a1) # Second half of coefficients

    # Ellipse coefficients from eq. 1
    a, b, c, d, e, f = tuple(a1) + tuple(a2)
    return a, b, c, d, e, f

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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from os.path import join, dirname
    from iris.io import read
    from uediff import diffshow

    image = read(join(dirname(__file__), 'tests\\test_diff_picture.tif'))
    TEST_MASK = ring_mask(image.shape, center = (990, 940), inner_radius = 215, outer_radius = 280)

    print(diffraction_center(image, mask = TEST_MASK))
    #diffshow(diffraction_center(image, mask = TEST_MASK))W