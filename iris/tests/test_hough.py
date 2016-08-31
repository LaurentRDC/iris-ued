# -*- coding: utf-8 -*-

import unittest

from uediff import diffshow
import numpy as n
from os.path import dirname, join
from skimage.draw import circle, circle_perimeter

from iris.hough import _binary_edge, _candidate_centers, diffraction_center
from iris.io import read

class TestEdgeDetection(unittest.TestCase):

    def setUp(self):
        self.shape = (2048, 2048)
        self.center = self.shape[0]/2, self.shape[1]/2
        self.image = n.zeros(shape = (2048, 2048), dtype = n.float)
        rr, cc = circle(1000, 1000, radius = 200)
        self.image[rr, cc] = 1
    
    def test_binary_edge(self):
        edge = _binary_edge(self.image)
        diffshow(edge)

class TestEdgeDetectionWithNoise(TestEdgeDetection):
    
    def setUp(self):
        super().setUp()
        self.image += 0.05*n.random.random(size = self.image.shape)
        

class TestHoughOnPerfectImages(unittest.TestCase):

    def setUp(self):
        self.shape = (500,500)      # Small size for quick tests
        self.center = int(self.shape[0]/2), int(self.shape[1]/2)
        self.image = n.zeros(shape = self.shape, dtype = n.float)
        for radius in [100, 150, 200]:
            rr, cc = circle_perimeter(self.center[0], self.center[1], radius = radius)
            self.image[rr, cc] = 1
    
    @unittest.skip('Too long')
    def test_diffraction_center(self):
        xc, yc = diffraction_center(self.image, min_rad = 50)
        ic, jc = self.center
        # Test equality within 1px
        self.assertAlmostEqual(xc, ic, places = 0)
        self.assertAlmostEqual(yc, jc, places = 0)

class TestHoughOnNoisyImages(TestHoughOnPerfectImages):

    def setUp(self):
        super().setUp()
        self.image += 0.05*n.random.random(size = self.image.shape)
    
    @unittest.expectedFailure
    def test_diffraction_center(self):
        super().test_diffraction_center()

class TestHoughOnDiffractionImages(unittest.TestCase):

    def setUp(self):
        self.image = read(join(dirname(__file__), 'test_diff_picture.tif'))
        self.shape = self.image.shape
        self.center = (1000, 1000)
    
    @unittest.expectedFailure
    def test_diffraction_center(self):
        xc, yc = diffraction_center(self.image, min_rad = 150)
        ic, jc = self.center
        print(xc, yc)
        self.assertAlmostEqual(xc, ic, places = 0)
        self.assertAlmostEqual(yc, jc, places = 0)
    
if __name__ == '__main__':
    #unittest.main()
    shape = (500,500)      # Small size for quick tests
    center = int(shape[0]/2), int(shape[1]/2)
    image = n.zeros(shape = shape, dtype = n.float)
    for radius in [100, 150, 200]:
        rr, cc = circle_perimeter(center[0], center[1], radius = radius)
        image[rr, cc] = 1
    image += 0.05*n.random.random(size = image.shape)

    center = diffraction_center(image, min_rad = 50)
    print(center)