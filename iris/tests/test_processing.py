
from itertools import repeat
from ..processing import streaming_diff_avg, diff_avg
import numpy as np
import unittest

class TestStreamingDiffAvg(unittest.TestCase):
    """ Test the streaming_diff_avg routine """

    def test_trivial(self):
        """ test streaming_diff_avg on a stream of zeros """
        stream = repeat(np.zeros((128, 128), dtype = np.uint16), times = 10)
        avg, err = streaming_diff_avg(images = stream, valid_mask = None, weights = repeat(1, times = 10))

        self.assertTrue(np.allclose(avg, np.zeros_like(avg)))
        self.assertEqual(avg.dtype, np.float)

        self.assertTrue(np.allclose(err, np.zeros_like(err)))
        self.assertEqual(err.dtype, np.float)
    
    def test_constant(self):
        """ Test streaming_diff_avg on a stream of ones """
        stream = repeat(np.ones((128, 128), dtype = np.uint16), times = 10)
        avg, err = streaming_diff_avg(images = stream, valid_mask = None)

        self.assertTrue(np.allclose(avg, np.ones_like(avg)))
        self.assertTrue(np.allclose(err, np.zeros_like(err)))

if __name__ == '__main__':
    unittest.main()