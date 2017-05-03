
from ..utils import angular_average, find_center, average_tiff
from tempfile import TemporaryDirectory
import numpy as n
from scipy.ndimage import gaussian_filter
from os.path import join
from skimage.io import imsave
import unittest
from warnings import filterwarnings

class TestAverageTiff(unittest.TestCase):
    
    def test_trivial(self):
        """ Test average_tiff on a bunch of arrays of zeros """
        with TemporaryDirectory() as tempdir:
            for _ in range(5):
                imsave(join(tempdir, '_.tif'), 
                       arr = n.zeros((128, 128), dtype = n.uint16), 
                       plugin = 'tifffile')
            
            avg = average_tiff(tempdir, '*.tif')
        
        self.assertTrue(n.allclose(avg, n.zeros_like(avg)))
    
    def test_with_background(self):
        """ Test average_tiff on a bunch of arrays of ones, with
        a background of ones """
        with TemporaryDirectory() as tempdir:
            for _ in range(5):
                imsave(join(tempdir, '_.tif'), 
                       arr = n.full((128, 128), fill_value = 2, dtype = n.uint16), 
                       plugin = 'tifffile')
            
            avg = average_tiff(tempdir, '*.tif', background = n.ones((128, 128)))
        
        self.assertTrue(n.allclose(avg, n.ones_like(avg)))
        
if __name__ == '__main__':
    unittest.main()