
import unittest
from ..utils import find_center
from skimage.filters import gaussian
import numpy as np

@unittest.skip('Not useful right now')
class TestFindCenter(unittest.TestCase):


    def test_trivial_array(self):
        image = np.zeros(shape = (512, 512), dtype = np.float)
        xc, yc = find_center(image, guess_center = (256, 256), radius = 15, window_size = 10)
        self.assertTrue(0 <= xc < 512)
        self.assertTrue(0 <= yc < 512)
    
    def test_find_center_wrong_guess(self):
        """ Finding the center on an image, without reducing the image size """
        image = np.zeros(shape = (512, 512), dtype = np.float)
        xc, yc = (258, 254)
        extent = np.arange(0, image.shape[0])
        xx, yy = np.meshgrid(extent, extent)
        rr = np.sqrt((xx - xc)**2 + (yy - yc)**2)
        image[rr == 50] = 10
        image = gaussian(image, 3)

        corr_x, corr_y = find_center(image, guess_center = (255, 251), 
                                    radius = 50, window_size = 10)
        self.assertEqual(yc, corr_y)
        self.assertEqual(xc, corr_x)
    
    def test_perfect_guess(self):
        """ Using a perfect center guess """
        image = np.zeros(shape = (512, 512), dtype = np.float)
        xc, yc = (232, 255)
        extent = np.arange(0, image.shape[0])
        xx, yy = np.meshgrid(extent, extent)
        rr = np.sqrt((xx - xc)**2 + (yy - yc)**2)
        image[rr == 25] = 10
        image = gaussian(image, 3)

        corr_x, corr_y = find_center(image, guess_center = (xc, yc), radius = 25)
        self.assertEqual(yc, corr_y)
        self.assertEqual(xc, corr_x)

if __name__ == '__main__':
    unittest.main()