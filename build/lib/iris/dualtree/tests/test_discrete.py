
from dualtree import baseline_dwt, denoise_dwt
import numpy as n
import unittest

class Test2D(unittest.TestCase):
    def setUp(self):
        self.array = n.zeros(shape = (100, 100), dtype = n.float)

class Test1D(unittest.TestCase):
    def setUp(self):
        self.array = n.zeros(shape = (100,), dtype = n.float)


class TestEdgeCases(object):

    def test_dimensions(self):
        self.assertRaises(Exception, baseline_dwt, {'data':n.zeros(shape = (3,3,3), dtype = n.uint), 'max_iter': 10, 'level': 1})

    def test_zero_level(self):
        # Since all function are based on approx_rec_dwt, we only need to test level = 0 for approx_rec_dwt
        self.assertTrue(n.allclose(self.array, baseline_dwt(self.array, max_iter = 1, level = 0, wavelet = 'db1')))

class TestEdgeCases1D(Test1D, TestEdgeCases): pass

class TestEdgeCases2D(Test2D, TestEdgeCases): pass



class TestTrivial(object):

    def test_baseline_dwt(self):
        self.assertTrue(n.allclose(self.array, baseline_dwt(self.array, max_iter = 10)))

    def test_denoise_dwt(self):
        self.assertTrue(n.allclose(self.array, denoise_dwt(self.array)))

class TestTrivial1D(Test1D, TestTrivial): pass

class TestTrivial2D(Test2D, TestTrivial): pass



class Testdenoise_dwt(object):

    def test_random(self):
        noisy = self.array + 0.05*n.random.random(size = self.array.shape)
        self.assertTrue(n.allclose(self.array, denoise_dwt( noisy, level = 'max', wavelet = 'db1' ), atol = 0.05))

class Testdenoise_dwt1D(Test1D, Testdenoise_dwt): pass

class Testdenoise_dwt2D(Test2D, Testdenoise_dwt): pass

if __name__ == '__main__':
    unittest.main()