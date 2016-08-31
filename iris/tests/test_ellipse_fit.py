
from iris.io import read
import matplotlib.pyplot as plt
import numpy as n
from os.path import dirname, join
from skimage.draw import circle_perimeter
from uediff import diffshow
import unittest

from iris.ellipse_fit import ellipse_fit, ellipse_center, ring_mask, diffraction_center

class TestEllipseFit(unittest.TestCase):

    def test_on_perfect_circle(self):
        """ Test ellipse_fit on perfectly circular data points. """
        radius = 1
        x = n.linspace(-radius, radius, num = 100)
        y = n.sqrt(radius**2 - x**2)

        # Extend to complete circle
        x, y = n.hstack( (x,x) ), n.hstack( (y, -y) )

        a,b,c,d,e,f = ellipse_fit(x = x, y = y)
        self.assertAlmostEqual(b, 0, places = 3)    # Excentricity of the ellipse
        self.assertAlmostEqual(-d/(2*a), 0, places = 3)    # center
        self.assertAlmostEqual(-e/(2*c), 0, places = 3)    # center
    
    def test_on_offcenter_circle(self):
        radius = 1
        xc, yc = 0.4, 0.6
        x = n.linspace(-radius, radius, num = 100)
        y = n.sqrt(radius**2 - x**2)

        # Extend to complete circle and move to  center
        x, y = n.hstack( (x,x) ), n.hstack( (y, -y) )
        x, y = x + xc, y + yc

        a,b,c,d,e,f = ellipse_fit(x = x, y = y)
        self.assertAlmostEqual(b, 0, places = 3)    # Excentricity of the ellipse
        self.assertAlmostEqual(-d/(2*a), xc, places = 3)    # center
        self.assertAlmostEqual(-e/(2*c), yc, places = 3)    # center
    
    def test_on_noisy_circle(self):
        n.random.seed(23)   #For reproducible results

        radius = 1
        xc, yc = 0.4, 0.6
        x = n.linspace(-radius, radius, num = 1000)
        y = n.sqrt(radius**2 - x**2)

        # Extend to complete circle and move to  center
        x, y = n.hstack( (x,x) ), n.hstack( (y, -y) )
        x, y = x + xc, y + yc
        x, y = x + 0.05*n.random.random(size = x.shape), y + 0.05*n.random.random(size = y.shape)

        a,b,c,d,e,f = ellipse_fit(x = x, y = y)
        self.assertAlmostEqual(-d/(2*a), xc, places = 1)    # center
        self.assertAlmostEqual(-e/(2*c), yc, places = 1)    # center

class TestDiffractionCenter(unittest.TestCase):

    def setUp(self):
        self.image = read(join(dirname(__file__), 'test_diff_picture.tif'))
        self.mask1 = ring_mask(self.image.shape, center = (990, 940), inner_radius = 215, outer_radius = 280)
        self.mask2 = ring_mask(self.image.shape, center = (1000, 950), inner_radius = 210, outer_radius = 290)
    
    @unittest.expectedFailure
    def test_stability(self):
        xc1, yc1 = diffraction_center(self.image, mask = self.mask1)
        xc2, yc2 = diffraction_center(self.image, mask = self.mask2)
        print('Center 1: {}, {}'.format(xc1, yc1))
        print('Center 2: {}, {}'.format(xc2, yc2))
        self.assertAlmostEqual(xc1, xc2, places = 0)
        self.assertAlmostEqual(yc1, yc2, places = 0)
    
if __name__ == '__main__':
    unittest.main()