
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
    

class TestAngularAverage(unittest.TestCase):

    def test_trivial_array(self):
        image = n.zeros(shape = (256, 256), dtype = n.float)
        center = (image.shape[0]/2, image.shape[1]/2)
        s, i, e = angular_average(image, center, (0,0,0,0))
        self.assertTrue(i.sum() == 0)
        self.assertTrue(len(s) == len(i) == len(e))
    
    def test_ring_no_beamblock(self):
        image = n.zeros(shape = (256, 256), dtype = n.float)
        xc, yc = (128, 128)
        extent = n.arange(0, image.shape[0])
        xx, yy = n.meshgrid(extent, extent)
        rr = n.sqrt((xx - xc)**2 + (yy - yc)**2)
        image[n.logical_and(24 < rr,rr < 26)] = 1

        s, i, e = angular_average(image, (xc, yc), (0,0,0,0))
        self.assertTrue(i.max() == image.max())
    
    def test_ring_with_beamblock(self):
        image = n.zeros(shape = (256, 256), dtype = n.float)
        xc, yc = (128, 128)
        extent = n.arange(0, image.shape[0])
        xx, yy = n.meshgrid(extent, extent)
        rr = n.sqrt((xx - xc)**2 + (yy - yc)**2)
        image[n.logical_and(24 < rr,rr < 26)] = 1
        beamblock_rect = (120, 136, 120, 136)

        s, i, e = angular_average(image, (xc, yc), beamblock_rect)

class TestFindCenter(unittest.TestCase):

    @unittest.skip('Not useful right now')
    def test_trivial_array(self):
        image = n.zeros(shape = (512, 512), dtype = n.float)
        xc, yc = find_center(image, guess_center = (256, 256), radius = 15, window_size = 10)
        self.assertTrue(0 <= xc < 512)
        self.assertTrue(0 <= yc < 512)
    
    def test_find_center_wrong_guess(self):
        """ Finding the center on an image, without reducing the image size """
        image = n.zeros(shape = (512, 512), dtype = n.float)
        xc, yc = (258, 254)
        extent = n.arange(0, image.shape[0])
        xx, yy = n.meshgrid(extent, extent)
        rr = n.sqrt((xx - xc)**2 + (yy - yc)**2)
        image[rr == 50] = 10
        image = gaussian_filter(image, 3)

        corr_x, corr_y = find_center(image, guess_center = (255, 251), 
                                    radius = 50, window_size = 10)
        self.assertEqual(yc, corr_y)
        self.assertEqual(xc, corr_x)
    
    def test_perfect_guess(self):
        """ Using a perfect center guess """
        image = n.zeros(shape = (512, 512), dtype = n.float)
        xc, yc = (232, 255)
        extent = n.arange(0, image.shape[0])
        xx, yy = n.meshgrid(extent, extent)
        rr = n.sqrt((xx - xc)**2 + (yy - yc)**2)
        image[rr == 25] = 10
        image = gaussian_filter(image, 3)

        corr_x, corr_y = find_center(image, guess_center = (xc, yc), radius = 25)
        self.assertEqual(yc, corr_y)
        self.assertEqual(xc, corr_x)

if __name__ == '__main__':
    unittest.main()