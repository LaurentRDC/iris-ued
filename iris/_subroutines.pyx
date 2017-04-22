#!python
#cython: cdivision = True
#cython: boundscheck = False
#cython: nonecheck = False
#cython: wraparound = False
#cython: language_level = 3

cimport numpy as cnp
import numpy as np

cdef extern from "numpy/npy_math.h" nogil:
    bint npy_isnan(double x)

cdef cnp.double_t nanmean(cnp.double_t [:] arr) nogil:
    """
    Returns mean of 1D array, ignoring NAN
    """
    cdef double l = 0   # number of non-NAN elements
    cdef double s = 0   # accumulator
    cdef double item
    for i in range(arr.shape[0]):
        item = arr[i]
        if not npy_isnan(item):
            l += 1
            s += item
    
    return s / l

def _diff_avg(cnp.double_t [:,:,:] arr, 
              cnp.double_t [:] weights):
    """
    Average diffraction pictures from the same time-delay together. Median-abolute-difference (MAD)
    filtering can also be used to clean up the data.

    It is assumed that the pictures have been aligned already.

    Parameters
    ----------
    arr : ndarray
        Array to be averaged.
    weights : ndarray
        Array representing how much an image should be 'worth'. E.g.: a weight below 1 means that
        a picture is not bright enough, and therefore it should count more in the averaging.
    
    Returns
    -------
    avg : ndarray, ndim 2
        'Average' of arr.
    err : ndarray, ndim 2
        Standard error in the mean.
    """
    cdef cnp.double_t [:,:] avg
    cdef cnp.double_t [:] transverse = np.zeros(shape = (arr.shape[2], ))

    with nogil:

        # Iterate over all 'transverse columns':
        for row in range(arr.shape[0]):
            for column in range(arr.shape[1]):

                # Get transverse column affected by weights
                for d in range(arr.shape[2]):
                    transverse[d] = weights[d] * arr[row, column, d]
                avg[row, column] = nanmean(transverse)
                #err[row, column] = cnp.nanstd(transverse)

    return avg
