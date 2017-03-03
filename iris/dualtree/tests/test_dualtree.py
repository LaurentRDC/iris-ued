from dualtree import dualtree, idualtree, approx_rec, detail_rec, dualtree_max_level, baseline
from dualtree._wavelets import dualtree_wavelet, dualtree_first_stage, kingsbury99, kingsbury99_fs, ALL_QSHIFT, ALL_FIRST_STAGE, ALL_COMPLEX_WAV

import numpy as n
import pywt
import unittest

n.random.seed(23)


##############################################################################
###             COMPLEX WAVELET 
##############################################################################

class TestComplexWavelets(unittest.TestCase):

    def setUp(self):
        self.array = n.sin(n.arange(0, 10, step = 0.01))
    
    def test_first_stage(self):
        """ Test of perfect reconstruction of first stage wavelets. """
        for wavelet in ALL_FIRST_STAGE:
            for wav in dualtree_first_stage(wavelet):
                # Using waverec and wavedec instead of dwt and idwt because parameters
                # don't need as much parsing.
                self.assertTrue(n.allclose( self.array, pywt.waverec(pywt.wavedec(self.array, wav), wav) ))
    
    def test_kingsbury99_fs(self):
        """ Test for perfect reconstruction """
        for wav in kingsbury99_fs():
            a, d = pywt.dwt(data = self.array, wavelet = wav)
            rec = pywt.idwt(cA = a, cD = d, wavelet = wav)
            self.assertTrue(n.allclose(self.array, rec))
    
    def test_kingsbury99(self):
        """ Test for perfect reconstruction """
        for wav in kingsbury99():
            a, d = pywt.dwt(data = self.array, wavelet = wav)
            rec = pywt.idwt(cA = a, cD = d, wavelet = wav)
            self.assertTrue(n.allclose(self.array, rec))

##############################################################################
###           DUAL-TREE COMPLEX WAVELET TRANSFORM
##############################################################################

class TestDualTree(object):
    """ Skeleton for 1D and 2D testing. Tests are run from subclasses. """
    
    def test_perfect_reconstruction_level_0(self):
        coeffs = dualtree(data = self.array, level = 0)
        reconstructed = idualtree(coeffs = coeffs)
        self.assertTrue(n.allclose(self.array, reconstructed))
    
    def test_perfect_reconstruction_level_1(self):
        for first_stage in ALL_FIRST_STAGE:
            coeffs = dualtree(data = self.array, level = 1, first_stage = first_stage)
            reconstructed = idualtree(coeffs = coeffs, first_stage = first_stage)
            self.assertTrue(n.allclose(self.array, reconstructed))
    
    def test_perfect_reconstruction_multilevel(self):
        for first_stage in ALL_FIRST_STAGE:
            for wavelet in ALL_COMPLEX_WAV:
                for level in range(1, dualtree_max_level(data = self.array, first_stage = first_stage, wavelet = wavelet)):
                    coeffs = dualtree(data = self.array, level = level, first_stage = first_stage, wavelet = wavelet)
                    reconstructed = idualtree(coeffs = coeffs, first_stage = first_stage, wavelet = wavelet)
                    self.assertTrue(n.allclose(self.array, reconstructed))
    
    def test_dt_approx_and_detail_rec(self):
        for first_stage in ALL_FIRST_STAGE:
            for wavelet in ALL_COMPLEX_WAV:
                low_freq = approx_rec(array = self.array, level = 'max', first_stage = first_stage, wavelet = wavelet)
                high_freq = detail_rec(array = self.array, level = 'max', first_stage = first_stage, wavelet = wavelet)
                self.assertTrue(n.allclose(self.array, low_freq + high_freq))
    
    def test_axis(self):
        for axis in range(0, self.array.ndim):
            coeffs = dualtree(data = self.array, level = 2, axis = axis)
            reconstructed = idualtree(coeffs = coeffs, axis = axis)
            self.assertTrue(n.allclose(self.array, reconstructed))
    
    def test_axis_limits(self):
        with self.assertRaises(ValueError):
            coeffs = dualtree(data = self.array, level = 1, axis = self.array.ndim)

# Actual tests

class Test1D(TestDualTree, unittest.TestCase):
    def setUp(self):
        self.array = n.random.random(size = (100,))

class Test2D(TestDualTree, unittest.TestCase):
    def setUp(self):
        self.array = n.random.random(size = (50,50))

class Test3D(TestDualTree, unittest.TestCase):
    def setUp(self):
        self.array = n.random.random(size = (10,10,10))

class TestBaseline(unittest.TestCase):
    def setUp(self):
        self.array = n.random.random(size = (99,))

    def test_shape(self):
        background = baseline(array = self.array, max_iter = 100)
        self.assertTrue(self.array.shape == background.shape)
    
    def test_baseline_along_axis(self):
        array = n.random.random(size = (99, 99))
        background = baseline(array = array, max_iter = 100, axis = 0)
        self.assertSequenceEqual(array.shape, background.shape)
    
if __name__ == '__main__':
    unittest.main()