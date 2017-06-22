
from ..dataset import DiffractionDataset, PowderDiffractionDataset, SinglePictureDataset
import numpy as np
from numpy.random import random, randint
from skimage.io import imsave
import os.path
import tempfile
import unittest

np.random.seed(23)

def dummy_dataset(**kwargs):
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
    with DiffractionDataset(name = name, mode = 'w') as dataset:

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

    return name

def dummy_powder_dataset(**kwargs):
    """ Create a dummy PowderDiffractionDataset, fills it with random data, and returns the filename. 
    All keyword arguments are passed to the PowderDiffractionDataset's experimental parameters. """

    kwargs.update({'sample_type': 'powder'})
    filename = dummy_dataset(**kwargs)
    with PowderDiffractionDataset(name = filename, mode = 'r+') as dataset:
        dataset.compute_angular_averages()
        
    return filename

def dummy_single_picture_dataset(image):
    """ Create a dummy SinglePictureDataset """
    path = os.path.join(tempfile.gettempdir(), 'test.tif')
    imsave(path, image)

    return SinglePictureDataset(path)

class TestDiffractionDataset(unittest.TestCase):

    def test_corrected_time_points(self):
        with DiffractionDataset(name = dummy_dataset(), mode = 'r+') as dataset:
            dataset.time_zero_shift = 0
            self.assertSequenceEqual(dataset.corrected_time_points, dataset.time_points)
    
    def test_notes(self):
        """ Test that updating the notes works as intended """
        filename = dummy_dataset(**{'notes': 'test notes'})
        with DiffractionDataset(name = filename, mode = 'r+') as dataset:
            self.assertEqual(dataset.notes, 'test notes')
            dataset.notes = 'different notes'
        
        with DiffractionDataset(name = filename, mode = 'r+') as dataset:
            self.assertEqual(dataset.notes, 'different notes')
    
    def test_averaged_data_retrieval(self):
        """ Test the size of the output from DiffractionDataset.averaged_data 
        and DiffractionDataset.averaged_error """
        with DiffractionDataset(name = dummy_dataset(), mode = 'r+') as dataset:
            full_shape = dataset.resolution + (len(dataset.time_points),)

            full_data = dataset.averaged_data(timedelay = None)
            full_error = dataset.averaged_error(timedelay = None)
            self.assertSequenceEqual(full_data.shape, full_shape)
            self.assertSequenceEqual(full_error.shape, full_shape)

            time_slice = dataset.averaged_data(timedelay = dataset.time_points[0])
            error_slice = dataset.averaged_error(timedelay = dataset.time_points[0])
            self.assertSequenceEqual(time_slice.shape, dataset.resolution)
            self.assertSequenceEqual(error_slice.shape, dataset.resolution)

    def test_averaged_equilibrium(self):
        """ Test DiffractionDataset.averaged_equilibrium() """
        with PowderDiffractionDataset(name = dummy_dataset(), mode = 'r+') as dataset:
            eq = dataset.averaged_equilibrium()
            self.assertSequenceEqual(eq.shape, dataset.resolution)
            
            with self.subTest('No data before time-zero'):
                dataset.shift_time_zero(1 + abs(min(dataset.time_points)))
                eq = dataset.averaged_equilibrium()
                self.assertSequenceEqual(eq.shape, dataset.resolution)
                self.assertTrue(np.allclose(eq, np.zeros_like(eq)))
    
    def test_averaged_data_retrieval_out_parameter(self):
        """ Tests that the out parameter is working properly for  DiffractionDataset.averaged_data 
        and DiffractionDataset.averaged_error """
        with DiffractionDataset(name = dummy_dataset(), mode = 'r+') as dataset:
            full_shape = dataset.resolution + (len(dataset.time_points),)
            container = np.empty(full_shape)

            dataset.averaged_data(timedelay = None, out = container)
            self.assertTrue(np.allclose(container, dataset.averaged_data(None)))

            dataset.averaged_error(timedelay = None, out = container)
            self.assertTrue(np.allclose(container, dataset.averaged_error(None)))
    
    def test_averaged_data_retrieval_nonexisting_timepoint(self):
        """ Tests that trying to access a non-existing time-point raises an appropriate error """
        with DiffractionDataset(name = dummy_dataset(), mode = 'r+') as dataset:

            with self.assertRaises(ValueError):
                time_slice = dataset.averaged_data(timedelay = np.sum(dataset.time_points))
            
            with self.assertRaises(ValueError):
                error_slice = dataset.averaged_error(timedelay = np.sum(dataset.time_points))
    
    def test_time_series(self):
        """ Test that the DiffractionDataset.time_series() method is working """
        with DiffractionDataset(name = dummy_dataset(), mode = 'r+') as dataset:

            with self.subTest('Test out parameter'):
                container = np.zeros((len(dataset.time_points), ))
                x1 = y1 = int(dataset.resolution[0] / 3)
                x2 = y2 = int((2/3)*dataset.resolution[0])
                dataset.time_series((x1, x2, y1, y2), out = container)

class TestPowderDiffractionDataset(unittest.TestCase):

    def test_scattering_length(self):
        """ Test that scattering_length attribute exists """
        with PowderDiffractionDataset(name = dummy_powder_dataset(), mode = 'r+') as dataset:
            self.assertSequenceEqual(dataset.scattering_length.shape, 
                                     dataset.powder_data(dataset.time_points[0]).shape)
    
    def test_baseline_attributes(self):
        """ Test that the attributes related to baseline have correct defaults and are
        set to the correct values after computation """
        with PowderDiffractionDataset(name = dummy_powder_dataset(), mode = 'r+') as dataset:
            self.assertIs(dataset.first_stage, None)
            self.assertIs(dataset.wavelet, None)
            self.assertIs(dataset.level, None)
            self.assertFalse(dataset.baseline_removed)

            dataset.compute_baseline(first_stage = 'sym6', wavelet = 'qshift3', level = 1, mode = 'periodic')
            self.assertEqual(dataset.first_stage, 'sym6')
            self.assertEqual(dataset.wavelet, 'qshift3')
            self.assertEqual(dataset.level, 1)
            self.assertTrue(dataset.baseline_removed)

    def test_powder_data_retrieval(self):
        """ Test the size of the output from PowderDiffractionDataset.powder_data 
        and PowderDiffractionDataset.powder_error """
        with PowderDiffractionDataset(name = dummy_powder_dataset(), mode = 'r+') as dataset:
            full_shape = (len(dataset.time_points), dataset.scattering_length.size)

            full_data = dataset.powder_data(timedelay = None)
            full_error = dataset.powder_error(timedelay = None)
            self.assertSequenceEqual(full_data.shape, full_shape)
            self.assertSequenceEqual(full_error.shape, full_shape)

            time_slice = dataset.powder_data(timedelay = dataset.time_points[0])
            error_slice = dataset.powder_error(timedelay = dataset.time_points[0])
            self.assertSequenceEqual(time_slice.shape, dataset.scattering_length.shape)
            self.assertSequenceEqual(error_slice.shape, dataset.scattering_length.shape)
    
    def test_powder_equilibrium(self):
        """ Test PowderDiffractionDataset.powder_equilibrium() """
        with PowderDiffractionDataset(name = dummy_powder_dataset(), mode = 'r+') as dataset:
            with self.subTest('bgr = False'):
                eq = dataset.powder_equilibrium()
                self.assertSequenceEqual(eq.shape, dataset.scattering_length.shape)

            with self.subTest('bgr = True'):
                dataset.compute_baseline(first_stage = 'sym6', wavelet = 'qshift3', mode = 'constant')
                eq = dataset.powder_equilibrium(bgr = True)
                self.assertSequenceEqual(eq.shape, dataset.scattering_length.shape)
            
            with self.subTest('No data before time-zero'):
                dataset.shift_time_zero(1 + abs(min(dataset.time_points)))
                eq = dataset.powder_equilibrium()
                self.assertSequenceEqual(eq.shape, dataset.scattering_length.shape)
                self.assertTrue(np.allclose(eq, np.zeros_like(eq)))
    
    def test_recomputing_angular_average(self):
        """ Test that recomputing the angular average multiple times will work. This also
        tests resizing all powder data multiple times. """
        with PowderDiffractionDataset(name = dummy_powder_dataset(), mode = 'r+') as dataset:
            dataset.compute_angular_averages(center = (34, 56))
            dataset.compute_baseline(first_stage = 'sym6', wavelet = 'qshift1')
            dataset.compute_angular_averages(center = (45, 45))
            dataset.compute_baseline(first_stage = 'sym5', wavelet = 'qshift2')
    
    def test_baseline(self):
        """ Test the computation of wavelet baselines """
        with PowderDiffractionDataset(name = dummy_powder_dataset(), mode = 'r+') as dataset:
            dataset.compute_baseline(first_stage = 'sym6', wavelet = 'qshift3', mode = 'smooth')
            
            self.assertSequenceEqual(dataset.baseline(timedelay = 0).shape, 
                                     dataset.scattering_length.shape)

class TestSinglePictureDataset(unittest.TestCase):
    
    def setUp(self):
        self.image = np.random.random(size = (128, 128))
        self.dataset = dummy_single_picture_dataset(self.image)
    
    def test_averaged_data(self):
        self.assertTrue(np.allclose(self.image, self.dataset.averaged_data(0)))
    
    def test_averaged_error(self):
        # check that average_error() is an array of zeros
        self.assertFalse(np.any(self.dataset.averaged_error(0)))
    
    def test_valid_mask(self):
        self.assertTrue(np.all(self.dataset.valid_mask))
    

