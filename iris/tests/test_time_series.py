import numpy as n
from ..time_series import autocorr1d
import unittest
import matplotlib.pyplot as plt

n.random.seed(23)

class TestAutocorrelation(unittest.TestCase):

    def test_trivial(self):
        arr = n.random.rand(20, 30)
        autocorr = autocorr1d(arr, axis = 0)

        self.assertSequenceEqual(autocorr.shape, arr.shape)
    
    def test_white_noise(self):
        """ Test autocorrelation on white noise """
        # The idea for this test came from the documentation
        # of scipy.signal.fftconvolve
        signal = n.random.randn(1024)
        autocorr = autocorr1d(signal)

        self.assertEqual(len(signal)/2 - 1, n.argmax(autocorr))



if __name__ == '__main__':
    unittest.main()