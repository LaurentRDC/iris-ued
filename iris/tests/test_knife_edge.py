
from ..knife_edge import cdf, knife_edge
from math import sqrt, log
import numpy as n
import unittest

class TestCDFFaussian(unittest.TestCase):

    def test_limits(self):
        """ Test that the cdf function converges to one """
        self.assertAlmostEqual(cdf(-1000, amplitude = 1, std = 1, center = 0, offset = 0), 0)
        self.assertAlmostEqual(cdf(1000, amplitude = 1, std = 1, center = 0, offset = 0), 1)
    
class TestKnifeEdge(unittest.TestCase):
    
    def test_trivial_case(self):
        """ Test that passing perfect data returns the right FWHM """
        x = n.linspace(0, 20, num = 50)
        y = cdf(x, amplitude = 1, std = 2, center = 10, offset = 0)

        correct_fwhm = 2 * sqrt(2 * log(2)) * 2 # fwhm from std of 2
        
        fwhm = knife_edge(x, y)
        self.assertAlmostEqual(fwhm, correct_fwhm, places = 3)
    
    def test_fit_parameters_recovery(self):
        """ Test that fit parameters can recovered if a perfect function is provided"""
        x = n.linspace(-20, 20, num = 50)
        correct_params = {'amplitude': 1, 'std': 2, 'center': -5, 'offset':4}
        y = cdf(x, **correct_params)
        
        params = dict()
        _ = knife_edge(x, y, fit_parameters = params)

        # At the time of this writing, unittest.TestCase does not provide
        # an assertDictAlmostEqual method
        for key in correct_params:
            self.assertAlmostEqual(correct_params[key], params[key], places = 3)