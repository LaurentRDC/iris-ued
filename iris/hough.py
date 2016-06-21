# -*- coding: utf-8 -*-
"""
Created on Tue May 31 11:20:33 2016

@author: Laurent

This module is concerned with finding the center of polycrystalline diffraction images.

Functions
---------
diffraction_center
    
"""
from iris.wavelet import denoise
import numpy as n
import matplotlib.pyplot as plt

from skimage.transform import hough_circle
from skimage.filters import threshold_adaptive
from skimage.feature import peak_local_max

# Clustering
from scipy.cluster.vq import whiten, kmeans2

def diffraction_center(image, beamblock = None, n_centroids = 5):
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
    n_centroids : int, optional
        Number of centroids to use during clustering of potential centers. 
        Default is 3.
    
    Returns
    -------
    center : 2-tuple
        Center indices (i,j)
    
    Notes
    -----
    The Hough transform is very sensitive to noise. Therefore, using a beamblock
    is highly recommended.
    """
    # Find possible centers
    centers = _candidate_centers(image, beamblock)
    
    # Use k-means algorithm to group centers
    whitened = whiten(centers)
    centroid, label = kmeans2(data = whitened, k = n_centroids)
    
    # Returns the average center of the biggest k-cluster
    center_cluster = n.argmax(n.bincount(label))    # most numerous cluster
    clustered_centers = centers[label == center_cluster, :]
    
    # Definition of center should be transposed
    j,i =  n.mean(clustered_centers, axis = 0)
    return (i,j)

def _candidate_centers(image, beamblock):
    """
    Returns potential centers based on the circle Hough transform.
    
    Parameters
    ----------
    image : ndarray, ndim 2
        Diffraction pattern
    beamblock : Tuple, shape (4,) or None
        Tuple containing x- and y-bounds (in pixels) for the beamblock mask. 
        If None, no mask is applied.
    
    Returns
    -------
    centers : ndarray, shape (N, 2)
        Rows of this array are potential centers
    """
    # Set up the binary image
    mask = n.zeros_like(image, dtype = n.bool)
    if beamblock is not None:
        y1, y2, x1, x2 = beamblock 
        mask[x1:x2, y1:y2] = True
    binary_image = _binary_edge(image, mask)
    
    # Determine a range of radii. Determined by testing.
    radii = n.arange(150, min(image.shape)/3, step = 50)
    
    centers = list()
    for radius in radii:
        centers.append(_circle_center(binary_image, radius, num_centers = 5))
        
    return n.vstack(tuple(centers))
    
def _circle_center(image, radius, num_centers = 1):
    """
    Finds the diffraction center in an image for a specific radius.
    
    Parameters
    ----------
    image : ndarray, ndim 2
        Binary image
    radius : numerical
        Circle radius. If not an integer, will be rounded to the nearest integer.
    num_centers : int or None, optional
        Number of centers to return. If None, there is no upper limit on the number
        of centers. Default is 1
    
    Returns
    -------
    centers : list of 2-tuples
        Array with each row being the indices of a potential center
    """    
    # Compute Hough transform accumulator
    accumulator = hough_circle(image = image, radius = radius)
    
    # Find the local maximas in the accumulator: these are the centers
    # Since we are only looking for a single radius, we can project the accumulator to 2D
    return peak_local_max(image = accumulator[0,:,:], num_peaks = num_centers).tolist()

def _binary_edge(image, mask):
    """
    Returns a binary image of the ring edges in a diffraction patterns.
    
    Parameters
    ----------
    image : ndarray, ndim 2
        Grayscale image
    mask : ndarray, dtype bool
        Pixels where mask is True will be set to 0 (non-object pixels).
    """
    # The Hough transform is very sensitive to noise
    image = denoise(image, wavelet = 'db5')
    
    #TODO: use morphology techniques to clean up canny output?
    
    array = threshold_adaptive(image, block_size = 101, method = 'gaussian',
                               offset = image.min(), mode = 'constant')
    array[mask] = 0 # Apply mask
    
    return array
    #return canny(array, sigma = 0.5)    # sigma determined by testing

# -----------------------------------------------------------------------------
#           TESTING
# -----------------------------------------------------------------------------
def test_diffraction_center(image, beamblock):
    """
    
    """
    center = diffraction_center(image, beamblock)
    
    fig = plt.figure()
    plt.imshow(image)
    
    circle = plt.Circle(tuple(center), 100, color = 'r')
    fig.gca().add_artist(circle)

def test_binary(image):
    mask = n.zeros_like(image, dtype = n.bool)
    plt.imshow(_binary_edge(image, mask))

if __name__ == '__main__':
    TEST_BEAMBLOCK = (800, 1110, 0, 1100)
    
    from iris import dataset

    directory = 'K:\\2012.11.09.19.05.VO2.270uJ.50Hz.70nm'
    d = dataset.PowderDiffractionDataset(directory)  
    image = d.image(0.0)
    #test_diffraction_center(image, beamblock = TEST_BEAMBLOCK)
    test_diffraction_center(image, TEST_BEAMBLOCK)