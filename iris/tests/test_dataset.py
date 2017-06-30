
from .dummies import dummy_dataset, dummy_single_picture_dataset, dummy_powder_dataset
import numpy as np
from numpy.random import random, randint
from skimage.io import imsave
import os.path
import tempfile
import unittest

np.random.seed(23)

class TestDiffractionDataset(unittest.TestCase):
    
    def setUp(self):
        self.dataset = dummy_dataset()
    
    def tearDown(self):
        self.dataset.close()

    def test_corrected_time_points(self):

        self.dataset.time_zero_shift = 0
        self.assertSequenceEqual(self.dataset.corrected_time_points, self.dataset.time_points)
    
    def test_notes(self):
        """ Test that updating the notes works as intended """
        self.dataset.notes = 'test notes'
        self.assertEqual(self.dataset.notes, 'test notes')
        self.dataset.notes = 'different notes'
        self.assertEqual(self.dataset.notes, 'different notes')
    
    def test_averaged_data_retrieval(self):
        """ Test the size of the output from DiffractionDataset.averaged_data 
        and DiffractionDataset.averaged_error """

        full_shape = self.dataset.resolution + (len(self.dataset.time_points),)

        full_data = self.dataset.averaged_data(timedelay = None)
        full_error = self.dataset.averaged_error(timedelay = None)
        self.assertSequenceEqual(full_data.shape, full_shape)
        self.assertSequenceEqual(full_error.shape, full_shape)

        time_slice = self.dataset.averaged_data(timedelay = self.dataset.time_points[0])
        error_slice = self.dataset.averaged_error(timedelay = self.dataset.time_points[0])
        self.assertSequenceEqual(time_slice.shape, self.dataset.resolution)
        self.assertSequenceEqual(error_slice.shape, self.dataset.resolution)

    def test_averaged_data_retrieval_out_parameter(self):
        """ Tests that the out parameter is working properly for  DiffractionDataset.averaged_data 
        and DiffractionDataset.averaged_error """
        full_shape = self.dataset.resolution + (len(self.dataset.time_points),)
        container = np.empty(full_shape)

        self.dataset.averaged_data(timedelay = None, out = container)
        self.assertTrue(np.allclose(container, self.dataset.averaged_data(None)))

        self.dataset.averaged_error(timedelay = None, out = container)
        self.assertTrue(np.allclose(container, self.dataset.averaged_error(None)))
    
    def test_averaged_data_retrieval_nonexisting_timepoint(self):
        """ Tests that trying to access a non-existing time-point raises an appropriate error """

        with self.assertRaises(ValueError):
            time_slice = self.dataset.averaged_data(timedelay = np.sum(self.dataset.time_points))
        
        with self.assertRaises(ValueError):
            error_slice = self.dataset.averaged_error(timedelay = np.sum(self.dataset.time_points))

    def test_averaged_equilibrium(self):
        """ Test DiffractionDataset.averaged_equilibrium() """
        eq = self.dataset.averaged_equilibrium()
        self.assertSequenceEqual(eq.shape, self.dataset.resolution)
        
        with self.subTest('No data before time-zero'):
            self.dataset.shift_time_zero(1 + abs(min(self.dataset.time_points)))
            eq = self.dataset.averaged_equilibrium()
            self.assertSequenceEqual(eq.shape, self.dataset.resolution)
            self.assertTrue(np.allclose(eq, np.zeros_like(eq)))
    
    def test_time_series(self):
        """ Test that the DiffractionDataset.time_series() method is working """

        with self.subTest('Test out parameter'):
            container = np.zeros((len(self.dataset.time_points), ))
            x1 = y1 = int(self.dataset.resolution[0] / 3)
            x2 = y2 = int((2/3)*self.dataset.resolution[0])
            self.dataset.time_series((x1, x2, y1, y2), out = container)

class TestPowderDiffractionDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = dummy_powder_dataset()
    
    def tearDown(self):
        self.dataset.close()

    def test_scattering_length(self):
        """ Test that scattering_length attribute exists """
        self.assertSequenceEqual(self.dataset.scattering_length.shape, 
                                    self.dataset.powder_data(self.dataset.time_points[0]).shape)
    
    def test_baseline_attributes(self):
        """ Test that the attributes related to baseline have correct defaults and are
        set to the correct values after computation """
        self.assertIs(self.dataset.first_stage, None)
        self.assertIs(self.dataset.wavelet, None)
        self.assertIs(self.dataset.level, None)

        self.dataset.compute_baseline(first_stage = 'sym6', wavelet = 'qshift3', level = 1, mode = 'periodic')
        self.assertEqual(self.dataset.first_stage, 'sym6')
        self.assertEqual(self.dataset.wavelet, 'qshift3')
        self.assertEqual(self.dataset.level, 1)

    def test_powder_data_retrieval(self):
        """ Test the size of the output from PowderDiffractionDataset.powder_data 
        and PowderDiffractionDataset.powder_error """
        full_shape = (len(self.dataset.time_points), self.dataset.scattering_length.size)

        full_data = self.dataset.powder_data(timedelay = None)
        full_error = self.dataset.powder_error(timedelay = None)
        self.assertSequenceEqual(full_data.shape, full_shape)
        self.assertSequenceEqual(full_error.shape, full_shape)

        time_slice = self.dataset.powder_data(timedelay = self.dataset.time_points[0])
        error_slice = self.dataset.powder_error(timedelay = self.dataset.time_points[0])
        self.assertSequenceEqual(time_slice.shape, self.dataset.scattering_length.shape)
        self.assertSequenceEqual(error_slice.shape, self.dataset.scattering_length.shape)
    
    def test_powder_equilibrium(self):
        """ Test PowderDiffractionDataset.powder_equilibrium() """
        with self.subTest('bgr = False'):
            eq = self.dataset.powder_equilibrium()
            self.assertSequenceEqual(eq.shape, self.dataset.scattering_length.shape)

        with self.subTest('bgr = True'):
            self.dataset.compute_baseline(first_stage = 'sym6', wavelet = 'qshift3', mode = 'constant')
            eq = self.dataset.powder_equilibrium(bgr = True)
            self.assertSequenceEqual(eq.shape, self.dataset.scattering_length.shape)
        
        with self.subTest('No data before time-zero'):
            self.dataset.shift_time_zero(1 + abs(min(self.dataset.time_points)))
            eq = self.dataset.powder_equilibrium()
            self.assertSequenceEqual(eq.shape, self.dataset.scattering_length.shape)
            self.assertTrue(np.allclose(eq, np.zeros_like(eq)))
    
    def test_recomputing_angular_average(self):
        """ Test that recomputing the angular average multiple times will work. This also
        tests resizing all powder data multiple times. """
        self.dataset.compute_angular_averages(center = (34, 56))
        self.dataset.compute_baseline(first_stage = 'sym6', wavelet = 'qshift1')
        self.dataset.compute_angular_averages(center = (45, 45), normalized = False)
        self.dataset.compute_baseline(first_stage = 'sym5', wavelet = 'qshift2')
        self.dataset.compute_angular_averages(center = (34, 56), angular_bounds = (15.3, 187))
        self.dataset.compute_baseline(first_stage = 'sym6', wavelet = 'qshift1')
    
    def test_baseline(self):
        """ Test the computation of wavelet baselines """
        self.dataset.compute_baseline(first_stage = 'sym6', wavelet = 'qshift3', mode = 'smooth')
        
        self.assertSequenceEqual(self.dataset.powder_baseline(timedelay = 0).shape, 
                                 self.dataset.scattering_length.shape)

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