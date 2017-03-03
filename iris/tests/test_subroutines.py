import numpy as n
from scipy.ndimage import fourier_shift
from skimage import data
from ..subroutines import diff_avg, diff_align, powder_align, shift_image
import unittest

class TestDiffAvg(unittest.TestCase):

    def test_trivial(self):
        """ Averaging identical pictures """
        arr = n.ones(shape = (256, 256, 30), dtype = n.float)
        avg, err = diff_avg(arr, mad = True, mad_dist = 1)

        self.assertSequenceEqual(avg.shape, (256, 256))
        self.assertSequenceEqual(err.shape, (256, 256))

        self.assertTrue(n.allclose(arr[:,:,0], avg))
        self.assertTrue(n.allclose(err, n.zeros_like(err)))
    
    def test_no_weights_no_mad(self):
        """ Show that diff_avg can be equivalent to numpy.mean() """
        arr = n.ones(shape = (256, 256, 30), dtype = n.float)
        avg, err = diff_avg(arr, weights = n.ones((30,)), mad = False)

        self.assertTrue(n.allclose(n.mean(arr, axis = 2), avg))
    
class TestDiffAlign(unittest.TestCase):

    def test_trivial(self):
        """ Test alignment of identical images """
        aligned = diff_align([data.camera() for _ in range(5)])
        
        self.assertEqual(len(aligned), 5)
        self.assertSequenceEqual(data.camera().shape, aligned[0].shape)
    
    def test_misaligned_canned_images(self):
        """ shift images from skimage.data by entire pixels """
        misaligned = [data.camera()] + [shift_image(data.camera(), (1,-3)) for _ in range(5)]
        aligned = diff_align(misaligned)

        # TODO: how to determine how close they are?

class TestPowderAlign(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()