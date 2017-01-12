
import numpy as n
from uediff.parallel import parallel_map, parallel_sum
import unittest

def identity(obj, *args, **kwargs):
    """ ignores args and kwargs """
    return obj

class ParallelMapTest(unittest.TestCase):

    def test_trivial_map_no_args(self):
        integers = list(range(0,10))
        result = parallel_map(identity, integers)
        self.assertEqual(integers, result)
    
    def test_trivial_map_with_args_and_kwargs(self):
        integers = list(range(0,10))
        result = parallel_map(identity, integers, args = (1,), kwargs = {'test' : True})
        self.assertEqual(result, integers)

class ParallelSumTest(unittest.TestCase):

    def test_trivial_sum(self):
        integers = list(range(0,11))
        serial = sum(integers)
        parallel = parallel_sum(sum, integers)
        self.assertEqual(serial, parallel)
    
    def test_numpy_sum(self):
        arrays = [n.ones((10,10)) for i in range(10)]
        serial = sum(arrays)
        parallel = parallel_sum(sum, arrays)
        self.assertTrue(n.allclose(serial, parallel))

if __name__ == '__main__':
    unittest.main()