"""
Parallel processing of RawDataset

@author: Laurent P. René de Cotret
"""
import glob
from datetime import datetime as dt
from functools import partial
from itertools import tee
from multiprocessing import Pool
from os.path import join

import numpy as np
from skimage.io import imread

from skued import pmap
from skued.baseline import baseline_dt
from skued.image import (angular_average, ialign, powder_center, shift_image)
from npstreams import iaverage, isem, last

from .dataset import DiffractionDataset, PowderDiffractionDataset
from .utils import scattering_length

def process(raw, destination, beamblock_rect, exclude_scans = list(), 
            processes = 1, callback = None, align = True):
    """ 
    Parallel processing of RawDataset into a DiffractionDataset.

    Parameters
    ----------
    raw : RawDataset
        Raw dataset instance.
    destination : str
        Path to the destination HDF5.
    beamblock_rect : 4-tuple

    exclude_scans : iterable of ints, optional
        Scans to exclude from the processing.
    processes : int or None, optional
        Number of Processes to spawn for processing. Default is number of available
        CPU cores.
    callback : callable or None, optional
        Callable that takes an int between 0 and 99. This can be used for progress update.
    align : bool, optional
        If True (default), raw images will be aligned on a per-scan basis.
    """
    # Create a mask of valid pixels (e.g. not on the beam block, not a hot pixel)
    x1,x2,y1,y2 = beamblock_rect
    valid_mask = np.ones(raw.resolution, dtype = np.bool)
    valid_mask[y1:y2, x1:x2] = False

    new_dataset = DiffractionDataset.from_raw(raw, destination, valid_mask = valid_mask, 
                                              callback = callback, align = align, processes = processes)

    return destination

def perscan(raw, srange, center, mask = None, trange = None, exclude_scans = list(), baseline_kwargs = dict(), callback = None):
    """ 
    Build the scan-by-scan array, for which each row is a time-series of a diffraction
    peak for a single scan. Only powder datasets are supported for now.

    Parameters
    ----------
    raw : RawDataset
        Raw dataset instance.
    srange : 2-tuple of floats
        Diffracted intensity will be integrated between those bounds.
    center : 2-tuple of ints
        Center of the diffraction pattern.
    mask : ndarray or None, optional
        Mask that evaluates to True on invalid pixels
    trange : iterable or None, optional
        If not None, time-points between min(trange) and max(trange) (inclusive) are used.
    exclude_scans : iterable of ints, optional
        Scans to exclude from the processing.
    baseline_kwargs : dict, optional
        Dictionary passed to skued.baseline.baseline_dt. If empty dict (default),
        no baseline is subtracted from each azimuthal average.
    callback : callable or None, optional
        Callable that takes an int between 0 and 99. This can be used for progress update.
    
    Returns
    -------
    scans : iterable of ints
        Scans appearing in the scan-by-scan array
    time_points : iterable of floats
        Time-points appearing in the scan-by-scan array [ps]
    scan_by_scan : ndarray, shape (N, M)
        Scan-by-scan analysis. Each row is a time-series of a scan.
    """
    if callback is None:
        callback = lambda _: None
    
    if trange is None:
        trange = (min(raw.time_points), max(raw.time_points))

    # NOTE: for the rare options 'pumpon only' in UEDbeta, there is no pumpoff_background
    pumpon_filenames = glob.glob(join(raw.raw_directory, 'background.*.pumpon.tif'))
    pumpon_background = sum(map(lambda f: np.asfarray(imread(f)), pumpon_filenames))/len(pumpon_filenames)

    if mask is None:
        mask = np.zeros_like(pumpon_background, dtype = np.bool)
    
    tmin, tmax = min(trange), max(trange)

    scans = set(raw.nscans) - set(exclude_scans)
    time_points = [time for time in raw.time_points if tmin <= times <= tmax]
    total = len(scans) * len(time_points)

    # Determine range between which to integrate
    r, _ = angular_average(pumpon_background, center = center, mask = mask)
    s = scattering_length(r, energy = raw.energy)
    i_min = np.argmin(np.abs(s - min(srange)))
    i_max = np.argmin(np.abs(s - max(srange)))

    scan_by_scan = np.empty(shape = (len(scans), len(time_points)), dtype = np.float)
    im = np.empty_like(pumpon_background, np.uint16)

    progress = 0
    for time_index, time_point in enumerate(sorted(time_points)):

        # Each row is an azimuthally-averaged diffraction pattern for a scan
        patterns = np.empty( shape = (len(scans), len(s)), dtype = np.float)

        for scan_index, scan in enumerate(scans):
            im[:] = raw.raw_data(time_point, scan = scan)
            im[:] = uint_subtract_safe(im, pumpon_background)
            _, I = angular_average(im, center = center, mask = mask)
            patterns[scan_index, :] = I
        
        if baseline_kwargs:
            patterns -= baseline_dt(patterns, axis = 1, **baseline_kwargs)

        scan_by_scan[:, time_index] = np.sum(patterns[:, i_min : i_max + 1], axis = 1)

        callback(int(100*time_index / len(time_points)))

            # TODO: parallel
            # TODO: alignment
        
    return scans, time_points, scan_by_scan
