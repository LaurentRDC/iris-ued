
from .. import DiffractionDataset, PowderDiffractionDataset, McGillRawDataset, VALID_DATASET_METADATA
from contextlib import suppress
import numpy as np
from itertools import repeat
from numpy.random import random
import os.path
import unittest

np.random.seed(23)

class TestDiffractionDatasetCreation(unittest.TestCase):

    def setUp(self):
        self.fname = 'test.hdf5'
    
    def test_from_raw(self):
        raw = McGillRawDataset(os.path.join(os.path.dirname(__file__), 'raw_dataset_test'))

        with self.subTest('align = False'):
            with DiffractionDataset.from_raw(raw, filename = self.fname, align = False, mode = 'w') as dataset:
                self.assertSequenceEqual(dataset.diffraction_group['intensity'].shape, (2048, 2048, 2))
            with suppress(OSError):
                os.remove(self.fname)

        with self.subTest('align = True'):
            with DiffractionDataset.from_raw(raw, filename = self.fname, align = True, mode = 'w') as dataset:
                self.assertSequenceEqual(dataset.diffraction_group['intensity'].shape, (2048, 2048, 2))
            with suppress(OSError):
                os.remove(self.fname)

        with self.subTest('processes = 2'):
            with DiffractionDataset.from_raw(raw, filename = self.fname, align = False, processes = 2, mode = 'w') as dataset:
                self.assertSequenceEqual(dataset.diffraction_group['intensity'].shape, (2048, 2048, 2))
            with suppress(OSError):
                os.remove(self.fname)
    
    def test_from_collection(self):
        """ Test the creation of a DiffractionDataset from a collection of patterns """
        patterns = repeat(random(size = (256, 256)), 10)
        metadata = {'fluence': 10, 'energy': 90}

        with DiffractionDataset.from_collection(patterns, filename = self.fname, 
                                                time_points = list(range(10)), metadata = metadata, mode = 'w') as dataset:
            
            self.assertSequenceEqual(dataset.diffraction_group['intensity'].shape, (256, 256, 10))
            self.assertEqual(dataset.fluence, metadata['fluence'])
            self.assertEqual(dataset.energy, metadata['energy'])
            self.assertSequenceEqual(tuple(dataset.time_points), list(range(10)))
    
    def tearDown(self):
        with suppress(OSError):
            os.remove(self.fname)

class TestDiffractionDataset(unittest.TestCase):

    def setUp(self):
        self.patterns = list(repeat(random(size = (256, 256)), 5))
        self.metadata = {'fluence': 10, 'energy': 90}
        self.dataset = DiffractionDataset.from_collection(self.patterns, 
                                                          filename = 'test.hdf5', 
                                                          time_points = range(5), 
                                                          metadata = self.metadata,
                                                          mode = 'w')
    
    def test_dataset_metadata(self):
        """ Test that the property 'metadata' is working correctly"""
        metadata = self.dataset.metadata
        for required in VALID_DATASET_METADATA:
            self.assertIn(required, metadata)
        self.assertIn('filename', metadata)

    def test_notes(self):
        """ Test that updating the notes works as intended """
        self.dataset.notes = 'test notes'
        self.assertEqual(self.dataset.notes, 'test notes')
        self.dataset.notes = 'different notes'
        self.assertEqual(self.dataset.notes, 'different notes')
    
    def test_resolution(self):
        """ Test that dataset resolution is correct """
        self.assertSequenceEqual(self.patterns[0].shape, self.dataset.resolution)
    
    def test_data(self):
        """ Test that data stored in DiffractionDataset is correct """
        for time, pattern in zip(list(self.dataset.time_points), self.patterns):
            self.assertTrue(np.allclose(self.dataset.diff_data(time), pattern))
    
    def test_diff_eq(self):
        """ test that DiffractionDataset.diff_eq() returns the correct array """
        with self.subTest('No data before time-zero'):
            self.dataset.shift_time_zero(10)
            self.assertTrue(np.allclose(self.dataset.diff_eq(), np.zeros(self.dataset.resolution)))

        with self.subTest('All data before time-zero'):
            self.dataset.shift_time_zero(-20)
            eq = np.mean(np.stack(self.patterns, axis = -1), axis = 2)
            self.assertTrue(np.allclose(self.dataset.diff_eq(), eq))
    
    def test_time_series(self):
        """ Test that the DiffractionDataset.time_series method is working """
        
        r1, r2, c1, c2 = 100, 120, 45, 57
        stack = np.stack(self.patterns, axis = -1)
        ts = np.mean(stack[r1:r2, c1:c2], axis = (0, 1))

        self.assertTrue(np.allclose(self.dataset.time_series([r1,r2,c1,c2]), ts))

    def tearDown(self):
        self.dataset.close()
        del self.dataset
        with suppress(OSError):
            os.remove('test.hdf5')

class TestPowderDiffractionDataset(unittest.TestCase):

    def setUp(self):
        self.patterns = list(repeat(random(size = (128, 128)), 5))
        self.metadata = {'fluence': 10, 'energy': 90}
        diff_dataset = DiffractionDataset.from_collection(self.patterns, 
                                                          filename = 'test.hdf5', 
                                                          time_points = range(5), 
                                                          metadata = self.metadata,
                                                          mode = 'w')
        self.dataset = PowderDiffractionDataset.from_dataset(diff_dataset, center = (64, 64))

    def test_baseline_attributes(self):
        """ Test that the attributes related to baseline have correct defaults and are
        set to the correct values after computation """
        self.assertIs(self.dataset.first_stage, '')
        self.assertIs(self.dataset.wavelet, '')
        self.assertEqual(self.dataset.level, 0)
        self.assertEqual(self.dataset.niter, 0)

        self.dataset.compute_baseline(first_stage = 'sym6', wavelet = 'qshift3', level = 1, mode = 'periodic')
        self.assertEqual(self.dataset.first_stage, 'sym6')
        self.assertEqual(self.dataset.wavelet, 'qshift3')
        self.assertEqual(self.dataset.level, 1)

    def test_powder_data_retrieval(self):
        """ Test the size of the output from PowderDiffractionDataset.powder_data """
        full_shape = (len(self.dataset.time_points), self.dataset.px_radius.size)

        full_data = self.dataset.powder_data(timedelay = None)
        self.assertSequenceEqual(full_data.shape, full_shape)

        time_slice = self.dataset.powder_data(timedelay = self.dataset.time_points[0])
        self.assertSequenceEqual(time_slice.shape, self.dataset.px_radius.shape)

    def test_recomputing_angular_average(self):
        """ Test that recomputing the angular average multiple times will work. This also
        tests resizing all powder data multiple times. """
        self.dataset.compute_angular_averages(center = (34, 56))
        self.dataset.compute_baseline(first_stage = 'sym6', wavelet = 'qshift1')
        self.dataset.compute_angular_averages(center = (45, 45), normalized = False)
        self.dataset.compute_baseline(first_stage = 'sym5', wavelet = 'qshift2')
        self.dataset.compute_angular_averages(center = (34, 56), angular_bounds = (15.3, 187))
        self.dataset.compute_baseline(first_stage = 'sym6', wavelet = 'qshift1')
    
    def test_powder_eq(self):
        """ Test PowderDiffractionDataset.powder_eq() """
        with self.subTest('bgr = False'):
            eq = self.dataset.powder_eq()
            self.assertSequenceEqual(eq.shape, self.dataset.px_radius.shape)

        with self.subTest('bgr = True'):
            self.dataset.compute_baseline(first_stage = 'sym6', wavelet = 'qshift3', mode = 'constant')
            eq = self.dataset.powder_eq(bgr = True)
            self.assertSequenceEqual(eq.shape, self.dataset.px_radius.shape)
        
        with self.subTest('No data before time-zero'):
            self.dataset.shift_time_zero(1 + abs(min(self.dataset.time_points)))
            eq = self.dataset.powder_eq()
            self.assertSequenceEqual(eq.shape, self.dataset.px_radius.shape)
            self.assertTrue(np.allclose(eq, np.zeros_like(eq)))

    def tearDown(self):
        self.dataset.close()
        del self.dataset
        with suppress(OSError):
            os.remove('test.hdf5')