
from iris.io import read
import matplotlib.pyplot as plt
import numpy as n
from os.path import dirname, join
from skimage.draw import circle_perimeter
from uediff import diffshow
import unittest

from ..ellipse_fit import ellipse_fit, ellipse_center

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
    
if __name__ == '__main__':
    unittest.main()