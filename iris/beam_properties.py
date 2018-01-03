
import numpy as np
from skimage.io import imread
from os.path import join
from glob import glob
from warnings import warn

from . import McGillRawDataset

# TODO: beam pointing stability using scikit-image's register_translation?

def beam_properties(directory, reprate = 1000, exposure = None, energy = None, callback = None):
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
    callback : callable or None, optional
        Callable of a single argument, to which the calculation progress will be passed as
        an integer between 0 and 100.
    
    Returns
    -------
    properties : dict
        Dictionary with keys 'count' and 'stability'. The value for the 'count' key is
        the average number of electron per shot. The value for the 'stability' key is the
        standard deviation (in percent of the average) in the 'count' value.
    """
    if callback is None:
        callback = lambda _: None
    
    metadata = McGillRawDataset.parse_tagfile(join(directory, 'tagfile.txt'))
    exposure = metadata['exposure']
    energy = metadata['energy'] or 90

    # Conversion between pixel intensity and number of electrons
    # depends on electron energy
    # Conversion taken from lowECount.m
    conv = 0.1862*energy - 0.5660
    n_pulses = exposure*reprate

    bg_filenames = glob(join(directory, 'background*.tif'))
    background = sum(imread(path).astype(np.float) for path in bg_filenames)/len(bg_filenames)

    images = glob(join(directory, 'data.timedelay.*.pumpon.tif'))
    e_count = list()
    for i, fn in enumerate(images):
        im = (imread(fn).astype(np.float) - background).clip(min = 0)
        e_count.append(np.sum(im)/conv/n_pulses)
        callback(int(100*i/len(images)))

    return {'count': np.mean(e_count), 
            'stability': np.std(e_count)/np.mean(e_count) * 100}