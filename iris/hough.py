# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:20:33 2016

@author: Laurent

This module is concerned with finding the center of polycrystalline diffraction images.

Functions
---------
diffraction_center
    
"""
from dualtree import denoise, baseline
import numpy as n
import matplotlib.pyplot as plt

import skimage
from skimage.transform import hough_circle
from skimage.filters import threshold_adaptive
from skimage.feature import peak_local_max, canny

from scipy.cluster.vq import whiten, kmeans2

def diffraction_center(image, beamblock = None, min_rad = 150, n_centroids = 5):
    """
    Find the diffraction center of an image using the Hough transform and a clustering
    procedure.
    
    Parameters
    ----------
    image : ndarray, ndim 2
        Diffraction pattern
    beamblock : Tuple, shape (4,) or None, optional
        Tuple containing x- and y-bounds (in pixels) for the beamblock mask
        mast_rect = (x1, x2, y1, y2). If None (default), no mask is applied.
    min_rad: int, optional
        Minimum radius to investigate. Default is 150 px.
    n_centroids : int, optional
        Number of centroids to use during clustering of potential centers. 
        Default is 5.
    
    Return
    -------
    center : 2-tuple
        Center indices (i,j)
    
    Notes
    -----
    The Hough transform is very sensitive to noise. Therefore, using a beamblock
    is highly recommended.
    """
    radii = n.arange(min_rad, image.shape[0]/2, step = 100)
    # Find possible centers
    centers = list()
    for min_radius in radii:
        centers += _candidate_centers(image, beamblock, min_rad = min_radius, block_size = 100)
    centers = n.asarray(centers)
    
    # Use k-means algorithm to group centers
    whitened = whiten(centers)
    centroid, label = kmeans2(data = whitened, k = n_centroids)
    
    # Returns the average center of the biggest k-cluster
    center_cluster = n.argmax(n.bincount(label))    # most numerous cluster
    clustered_centers = centers[label == center_cluster, :]
    
    # Definition of center should be transposed
    j,i =  n.mean(clustered_centers, axis = 0)
    return (i,j)

def _candidate_centers(image, beamblock, min_rad = 150, block_size = 100):
    """
    Returns potential centers based on the circle Hough transform. 
    The circles are assumed to be concentric; therefore, centers from circles with same radii are
    averaged together.
    
    Parameters
    ----------
    image : ndarray, ndim 2
        Binary image of a diffraction pattern
    beamblock : Tuple, shape (4,) or None
        Tuple containing x- and y-bounds (in pixels) for the beamblock mask. 
        If None, no mask is applied.
    min_rad : int, optional
        Minimum radius to investigate. Default is 150px.
    block_size : int, optional
        Maximum radius is taken as min_rad + block_size. Do not play with this.
    
    Returns
    -------
    unique_centers : list of ndarrays, shape (2,)
        Rows of this array are potential centers with shape (xc, yc)
    """
    # Set up the binary image
    mask = n.zeros_like(image, dtype = n.bool)
    if beamblock is not None:
        y1, y2, x1, x2 = beamblock 
        mask[x1:x2, y1:y2] = True
    binary_image = _binary_edge(image, mask)
    
    # Determine a range of radii. From:
    # http://stackoverflow.com/questions/32287032/circular-hough-transform-misses-circles
    # we want to iterate over small ranges of radii
    radii = n.arange(min_rad, min_rad + block_size, step = 1)
    accumulator = hough_circle(image = image, radius = radii)
    centers = peak_local_max(image = accumulator, num_peaks = radii.size)

    # At this point, we might have centers for circles with the same radii. We want
    # to average centers of the same radius.
    unique_radii = n.unique(centers[:,0])
    unique_centers = [n.mean(centers[centers[:,0] == r], axis = 0)[1::] for r in unique_radii]
    return unique_centers
        
def binary(image, mask = None):
    """
    Returns a binary image of the rings in a diffraction patterns.
    
    Parameters
    ----------
    image : ndarray, ndim 2
        Grayscale image
    mask : ndarray, dtype bool, optional
        Pixels where mask is True will be set to 0 (non-object pixels).
        Default is trivial mask.
    
    Returns
    -------
    edges : ndarray
        Same shape as input
    """
    if mask is None:
        mask = n.zeros_like(image, dtype = n.bool)
    
    image -= baseline(image, max_iter = 10)
    image = denoise(image, level = 3, wavelet = 'db5')
    image -= n.median(image)
    image[image < 0] = 0
    image[n.logical_not(mask)] = 0

    # Threshold image into foreground and background.
    smoothed = skimage.filters.gaussian(image, sigma = 5)
    binary = image > skimage.filters.threshold_otsu(image[mask])

    return image

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from os.path import join, dirname
    from iris.io import read
    from uediff import diffshow
    from iris.ellipse_fit import ring_mask

    image = read(join(dirname(__file__), 'tests\\test_diff_picture.tif'))
    TEST_MASK = ring_mask(image.shape, center = (990, 940), inner_radius = 215, outer_radius = 280)

    #print(diffraction_center(image, mask = TEST_MASK))
    diffshow(binary(image, mask = TEST_MASK))