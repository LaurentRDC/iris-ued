
from iris.wavelet import approx_rec, baseline, denoise, enhance
import numpy as n
import unittest

class TestTrivialArray(unittest.TestCase):
    
    def setUp(self):
        self.array = n.zeros(shape = (100, 100), dtype = n.float)
    
    def test_baseline(self):
        self.assertTrue(n.sum(self.array - baseline(self.array, max_iter = 10)) == 0)

    def test_denoise(self):
        self.assertTrue(n.sum(self.array - denoise(self.array)) == 0)
    
    def test_enhance(self):
        self.assertTrue(n.sum(self.array - enhance(self.array)) == 0)

if __name__ == '__main__':
    unittest.main()