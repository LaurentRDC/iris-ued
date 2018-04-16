# -*- coding: utf-8 -*-
from .. import McGillRawDataset, AbstractRawDataset, check_raw_bounds
from ..meta import ExperimentalParameter
import numpy as np
import unittest

class TestRawDataset(AbstractRawDataset):

    test        = ExperimentalParameter('test', int, default = 0)
    resolution  = ExperimentalParameter('resolution', tuple, (16,16))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.time_points = list(range(0, 10))
        self.scans       = list(range(1, 3))

    @check_raw_bounds
    def raw_data(self, timedelay, scan = 1): 
        return np.zeros((self.resolution), dtype = np.uint8)

class TestAbstractRawDataset(unittest.TestCase):

    def test_abstract_methods(self):
        """ Test that instantiation of AbstractRawDataset 
        raises an error """
        with self.assertRaises(TypeError):
            AbstractRawDataset('')
    
    def test_minimal_methods(self):
        """ 
        Test implementing the minimal methods:

        * raw_data
        """
        TestRawDataset()    

    def test_data_bounds(self):
        """ Test that a ValueError is raised if ``timedelay`` or ``scan`` are out-of-bounds. """
        test_dataset = TestRawDataset()

        with self.assertRaises(ValueError):
            test_dataset.raw_data(timedelay = 20, scan = 1)
        
        with self.assertRaises(ValueError):
            test_dataset.raw_data(timedelay = 5, scan = -1)
    
    def test_experimental_parameters(self):
        """ Test the behavior of the ExperimentalParameter descriptor """
        
        test_dataset = TestRawDataset()

        with self.subTest('Default value'):
            self.assertEqual(test_dataset.test, 0)
        
        with self.subTest('Changing value'):
            test_dataset.test = 1
            self.assertEqual(test_dataset.test, 1)
        
        with self.subTest('Parameter type'):
            with self.assertRaises(TypeError):
                test_dataset.test = 'test'
    
    def test_valid_metadata(self):
        """ Test that the class attribute 'valid_metadata' is working as intended """
        
        self.assertIn('test', TestRawDataset.valid_metadata)
        self.assertLessEqual(AbstractRawDataset.valid_metadata, TestRawDataset.valid_metadata)
    
    def test_init_metadata(self):
        """ Test that metadata is recorded correctly inside __init__ and
        that invalid metadata is ignored. """
        test_dataset = TestRawDataset(metadata = {'test': 5, 
                                                  'fluence': -2,
                                                  'random_attr': None})
        self.assertEqual(test_dataset.test, 5)
        self.assertEqual(test_dataset.fluence, -2)

        # Invalid metadata should be ignored.
        self.assertFalse(hasattr(test_dataset, 'random_attr'))