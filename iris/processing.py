"""
Parallel processing of RawDataset

@author: Laurent P. René de Cotret
"""
import glob
from datetime import datetime as dt
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import cpu_count
from os.path import join

import numpy as np
from scipy.stats import sem
from skimage.io import imread
from skued.image_analysis import align, powder_center, shift_image

from .dataset import DiffractionDataset, PowderDiffractionDataset

def diff_avg(images, valid_mask = None, weights = None):
    """ 
    Streaming average of diffraction images.

    Parameters
    ----------
    images : iterable of ndarrays, ndim 2

    valid_mask : ndarray or None, optional
        Mask that evaluates to True on pixels that are valid, e.g. not on the beamblock.
        If None, all pixels are valid.
    weights : ndarray or None, optional
        Array of weights. see `numpy.average` for further information. If None (default), 
        total picture intensity of valid pixels is used to weight each picture.
    
    Returns
    -------
    avg, err: ndarrays, ndim 2
        Weighted average and standard error in the mean of the 
    
    References
    ----------
    .. D. Knuth, The Art of Computer Programming 3rd Edition, Vol. 2, p. 232
    """
    images = iter(images)

    if valid_mask is None:
        valid_mask = np.s_[:]
    
    if weights is None:
        AUTO_WEIGHTS = True
    else:
        weights = iter(weights)
        AUTO_WEIGHTS = False

    # Streaming variance: https://www.johndcook.com/blog/standard_deviation/
    first = next(images)
    old_M = new_M = np.array(first, copy = True)
    old_S = new_S = np.zeros_like(first, dtype = np.float)

    sum_of_weights = np.sum(first[valid_mask], dtype = np.float) if AUTO_WEIGHTS else next(weights)
    weighted_sum = np.asfarray(first * sum_of_weights)

    # Running calculation
    # `k` represents the number of images consumed so far
    for k, image in enumerate(images, start = 2):

        # streaming weighted average
        weight = np.sum(image[valid_mask], dtype = np.float) if AUTO_WEIGHTS else next(weights)
        sum_of_weights += weight
        weighted_sum += weight * image

        # streaming variance
        # TODO: weighted variance
        _sub = image - old_M
        new_M[:] = old_M + _sub/k
        new_S[:] = old_S + _sub*(image - new_M)
        old_M, old_S = new_M, new_S
    
    avg = weighted_sum/sum_of_weights
    err = np.sqrt(new_S)/(k-1)  # variance = S / k-1, sem = std / sqrt(k)
    return avg, err

def uint_subtract_safe(arr1, arr2):
    """ Subtract two unsigned arrays without rolling over """
    result = np.subtract(arr1, arr2)
    result[np.greater(arr2, arr1)] = 0
    return result

def process(raw, destination, beamblock_rect, processes = None, callback = None):
    """ 
    Parallel processing of RawDataset into a DiffractionDataset.

    Parameters
    ----------
    raw : RawDataset
        Raw dataset instance.
    destination : str
        Path to the destination HDF5.
    beamblock_rect : 4-tuple
    
    processes : int or None, optional
        Number of Processes to spawn for processing. Default is number of available
        CPU cores.
    callback : callable or None, optional
        Callable that takes an int between 0 and 99. This can be used for progress update.
    """
    if callback is None:
        callback = lambda i: None

    # Prepare compression kwargs
    ckwargs = {'compression' : 'lzf', 
               'chunks' : True, 
               'shuffle' : True, 
               'fletcher32' : True}

    start_time = dt.now()
    with DiffractionDataset(name = destination, mode = 'w') as processed:

        processed.sample_type = 'single_crystal'       # By default
        processed.nscans = raw.nscans
        processed.time_points = raw.time_points
        processed.acquisition_date = raw.acquisition_date
        processed.fluence = raw.fluence
        processed.current = raw.current
        processed.exposure = raw.exposure
        processed.energy = raw.energy
        processed.resolution = raw.resolution
        processed.beamblock_rect = beamblock_rect
        processed.time_zero_shift = 0.0

        # Preallocation
        shape = raw.resolution + (len(raw.time_points),)
        gp = processed.processed_measurements_group
        gp.create_dataset(name = 'intensity', shape = shape, dtype = np.float32, **ckwargs)
        gp.create_dataset(name = 'error', shape = shape, dtype = np.float32, **ckwargs)

    # Average background images
    # If background images are not found, save empty backgrounds
    # NOTE: sum of images must be done as float arrays, otherwise the values
    #       can loop back if over 2**16 - 1
    pumpon_filenames = glob.glob(join(raw.raw_directory, 'background.*.pumpon.tif'))
    pumpon_background = sum(map(lambda f: np.asfarray(imread(f)), pumpon_filenames))/len(pumpon_filenames)

    pumpoff_filenames = glob.glob(join(raw.raw_directory, 'background.*.pumpoff.tif'))
    pumpoff_background = sum(map(lambda f: np.asfarray(imread(f)), pumpoff_filenames))/len(pumpoff_filenames)

    with DiffractionDataset(name = destination, mode = 'r+') as processed:
        gp = processed.processed_measurements_group
        gp.create_dataset(name = 'background_pumpon', data = pumpon_background, dtype = np.float32, **ckwargs)
        gp.create_dataset(name = 'background_pumpoff', data = pumpoff_background, dtype = np.float32, **ckwargs)

    # Create a mask of valid pixels (e.g. not on the beam block, not a hot pixel)
    x1,x2,y1,y2 = beamblock_rect
    valid_mask = np.ones(raw.resolution, dtype = np.bool)
    valid_mask[y1:y2, x1:x2] = False

    ref_im = uint_subtract_safe(raw.raw_data(raw.time_points[0], raw.nscans[0]), pumpon_background) # Reference for alignment
    mapkwargs = {'background': pumpon_background, 'ref_im': ref_im, 'valid_mask': valid_mask}

    # an iterator is used so that writing to the HDF5 file can be done in
    # the current process; otherwise, writing to disk can fail.
    # TODO: imap chunksize has been kept at 1 because for 2048x2048 images,
    #       memory usage is abount ~600MB per core. Would it be beneficial to
    #       increase chunksize to two or three?
    # NOTE: It is important the fnames_iterators are sorted by time
    #       therefore, enumerate() gives the right index that goes in the pipeline function
    time_points_processed = 0
    fnames_iterators = map(raw.timedelay_filenames, sorted(raw.time_points))
    with Pool(processes) as pool:
        results = pool.imap_unordered(func = partial(pipeline, **mapkwargs), 
                                      iterable = enumerate(fnames_iterators))
        
        for index, avg, err in results:

            time_points_processed += 1
            with DiffractionDataset(name = destination, mode = 'r+') as processed:
                gp = processed.processed_measurements_group
                gp['intensity'].write_direct(avg, source_sel = np.s_[:,:], dest_sel = np.s_[:,:,index])
                gp['error'].write_direct(err, source_sel = np.s_[:,:], dest_sel = np.s_[:,:,index])
            
            callback(round(100*time_points_processed / len(raw.time_points)))

    print('Processing has taken {}'.format(str(dt.now() - start_time)))
    return destination

def pipeline(values, background, ref_im, valid_mask):
    """
    Processing pipeline for a single time-point.

    Parameters
    ----------
    values : 2-tuple
        Index and filenames of diffraction pictures
    background : ndarray, dtype uint16
        Pump-on diffraction background
    ref_im : ndarray
        Background-subtracted diffraction pattern used as reference for alignment.
    valid_mask : ndarray, dtype bool
        Image mask that evaluates to True on valid pixels.
    
    Returns
    -------
    index : int
        Time-point index.
    avg, err : ndarrays, ndim 2
        Weighted average and standard error on processing.
    """
    # Generator chains helps keep memory usage lower
    # This in turns allows for more cores to be active at the same time
    index, fnames = values
    images = map(imread, fnames)

    images_bs = map(partial(uint_subtract_safe, **{'arr2': background}), images)
    aligned = align(images_bs, reference = ref_im)
    
    avg, err = diff_avg(aligned, valid_mask = valid_mask, weights = None)
    return index, avg, err
