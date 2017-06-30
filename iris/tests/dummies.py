
import os
import tempfile
from ..dataset import DiffractionDataset, PowderDiffractionDataset, SinglePictureDataset
from ..raw import RawDataset
import numpy as np
from numpy.random import random, randint
from skimage.io import imsave
import os.path
import tempfile

def dummy_dataset(cls = DiffractionDataset, **kwargs):
    """ Create a dummy DiffractionDataset, fills it with random data, and returns the filename. 
    All keyword arguments are passed to the DiffractionDataset's experimental parameters. """
    dataset_attributes = {'nscans': tuple(range(0, 5)),
                          'time_points': tuple(range(-5, 10)),
                          'acquisition_date': '0.0.0.0',
                          'fluence': 10.0,
                          'current': 0.0,
                          'exposure': 5,
                          'energy': 90,
                          'resolution': (128, 128),
                          'center': (64,64),
                          'beamblock_rect': (0,0,0,0),
                          'sample_type': 'single_crystal',
                          'time_zero_shift': 0,
                          'notes': ''}
    dataset_attributes.update(kwargs)

    name = os.path.join(tempfile.gettempdir(), 'test_dataset.hdf5')
    with cls(name = name, mode = 'w') as dataset:

        dataset.experimental_parameters_group.attrs.update(dataset_attributes)

        # Fill dataset with random data
        dataset.pumpoff_pictures_group.create_dataset('pumpoff_pictures',
                                                      data = random(size = dataset.resolution + (len(dataset.nscans),)))
        
        dataset.experimental_parameters_group.create_dataset(name = 'valid_mask', 
                                                             data = randint(low = 0, high = 2, size = dataset.resolution), 
                                                             dtype = np.bool)
        
        block_shape = dataset.resolution + (len(dataset.time_points),)
        dataset.processed_measurements_group.create_dataset('intensity', data = random(size = block_shape))
        dataset.processed_measurements_group.create_dataset('error', data = random(size = block_shape))
        dataset.processed_measurements_group.create_dataset('background_pumpon', data = random(size = dataset.resolution))
        dataset.processed_measurements_group.create_dataset('background_pumpoff', data = random(size = dataset.resolution))

    return cls(name = name, mode = 'r+')

def dummy_powder_dataset(**kwargs):
    """ Create a dummy PowderDiffractionDataset, fills it with random data, and returns the filename. 
    All keyword arguments are passed to the PowderDiffractionDataset's experimental parameters. """

    kwargs.update({'sample_type': 'powder'})
    dataset = dummy_dataset(cls = PowderDiffractionDataset, **kwargs)
    dataset.compute_angular_averages()
    return dataset

def dummy_single_picture_dataset(image):
    """ Create a dummy SinglePictureDataset """
    path = os.path.join(tempfile.gettempdir(), 'test.tif')
    imsave(path, image)

    return SinglePictureDataset(path)

def dummy_raw_dataset(**kwargs):
    return RawDataset(os.path.join(os.path.dirname(__file__), 'raw_dataset_test'))