
import numpy as np
from skimage.io import imread
from os.path import join
from glob import glob
from warnings import warn

def beam_properties(directory, reprate = 1000, exposure = None, energy = None):
    """ 
    Calculates electron beam properties from a ponderomotive-style
    measurement. Properties monitored are average electron count per shot and
    electron count stability.

    Parameters
    ----------
    directory : str or path-like object
        Absolute path to the directory
    reprate : int, optional
        Laser repetition rate [Hz]. Default is 1000.
    exposure : float, optional
        Exposure time [s]. If None (default), the value will be inferred from
        `directory/tagfile.txt`
    energy : float, optional
        Electron energy [keV]. If None (default), the value will be inferred from
        `directory/tagfile.txt`.
    
    Returns
    -------
    properties : dict
        Dictionary with keys 'count' and 'stability'. The value for the 'count' key is
        the average number of electron per shot. The value for the 'stability' key is the
        standard deviation (in percent of the average) in the 'count' value.
    """
    # TODO: more elegant parsing?
    if not exposure:
        with open(join(directory, 'tagfile.txt')) as metadata:
            exposure_line = next(filter(lambda line: line.startswith('Exposure'), metadata)).strip('\n')
    exposure_str = exposure_line.split('=')[-1]
    exposure = float(exposure_str.strip('s'))

    if not energy:
        with open(join(directory, 'tagfile.txt')) as metadata:
            try:
                energy_line = next(filter(lambda line: line.startswith('Energy'), metadata)).strip('\n')
                energy = float(energy_line.split('=')[-1])
            except StopIteration:
                warn('Electron energy not stored in {}. Using 90kV as default.'.format(directory))
                energy = 90

    # Conversion between pixel intensity and number of electrons
    # depends on electron energy
    # Conversion taken from lowECount.m
    conv = 0.1862*energy - 0.5660
    n_pulses = exposure*reprate

    bg_filenames = glob(join(directory, 'background*.tif'))
    background = sum(imread(path).astype(np.float) for path in bg_filenames)/len(bg_filenames)

    images = glob(join(directory, 'data.timedelay.*.pumpon.tif'))
    e_count = list()
    accumulator = np.zeros_like(background, dtype = np.float)
    for fn in images:
        im = imread(fn).astype(np.float) - background
        im[im < 0] = 0
        accumulator += im
        e_count.append(np.sum(im))
    average = accumulator/len(images)

    count = np.sum(average)/conv/n_pulses
    stability = np.std(e_count)/np.mean(e_count) * 100

    return {'count': count, 'stability': stability}