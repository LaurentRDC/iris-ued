
import tempfile
import unittest
from itertools import repeat

import numpy as np
import os

from ..processing import diff_avg, process, uint_subtract_safe
from .. import DiffractionDataset
from .dummies import dummy_raw_dataset


class TestProcess(unittest.TestCase):
    """ Test process() """

    def setUp(self):
        self.dataset = dummy_raw_dataset()
    
    def test_process_defaults(self):
        destination = os.path.join(tempfile.gettempdir(), 'test_process.hdf5')
        
        filename = process(self.dataset, destination = destination, beamblock_rect = [0,0,0,0], processes = 2)

        with DiffractionDataset(name = filename, mode = 'r') as processed:
            self.assertSequenceEqual(self.dataset.nscans, processed.nscans)
        
        os.remove(filename)

    def test_process_no_align(self):
        destination = os.path.join(tempfile.gettempdir(), 'test_process.hdf5')
        
        filename = process(self.dataset, destination = destination, 
                           beamblock_rect = [0,0,0,0], processes = 2, align = False)

        with DiffractionDataset(name = filename, mode = 'r') as processed:
            self.assertSequenceEqual(self.dataset.nscans, processed.nscans)

        os.remove(filename)

class TestDiffAvg(unittest.TestCase):
    """ Test the diff_avg routine """

    def test_trivial(self):
        """ test streaming_diff_avg on a stream of zeros """
        stream = repeat(np.zeros((128, 128), dtype = np.uint16), times = 10)
        avg, err = diff_avg(images = stream, valid_mask = None, weights = repeat(1, times = 10))

        self.assertTrue(np.allclose(avg, np.zeros_like(avg)))
        self.assertEqual(avg.dtype, np.float)

        self.assertTrue(np.allclose(err, np.zeros_like(err)))
        self.assertEqual(err.dtype, np.float)
    
    def test_constant(self):
        """ Test streaming_diff_avg on a stream of ones """
        stream = repeat(np.ones((128, 128), dtype = np.uint16), times = 10)
        avg, err = diff_avg(images = stream, valid_mask = None)

        self.assertTrue(np.allclose(avg, np.ones_like(avg)))
        self.assertTrue(np.allclose(err, np.zeros_like(err)))
    
    def test_running_error(self):
        """ Test that the streaming calculation of the error is correct """
        pass

class TestUintSubtractSafe(unittest.TestCase):
    """ Test the uint_subtract_safe function """

    def test_trivial(self):
        """ Test uint_subtract_safe on arrays of zeros """
        arr = np.zeros((64, 64), dtype = np.uint16)
        self.assertTrue(np.allclose(arr, uint_subtract_safe(arr, arr)))
    
    def test_overflow(self):
        """ Test uint_subtract_safe in case where normal subtraction would loop around """
        arr1 = np.zeros((64, 64), dtype = np.uint16)
        arr2 = np.ones_like(arr1)

        sub = uint_subtract_safe(arr1, arr2)
        self.assertTrue(np.allclose(sub, arr1))

if __name__ == '__main__':
    unittest.main()
