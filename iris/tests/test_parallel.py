
import numpy as n
from ..optimizations import pmap
import unittest

def identity(obj, *args, **kwargs):
    """ ignores args and kwargs """
    return obj

class PMapTest(unittest.TestCase):

    def test_trivial_map_no_args(self):
        integers = list(range(0,10))
        result = parallel_map(identity, integers)
        self.assertEqual(integers, result)
    
    def test_trivial_map_with_args_and_kwargs(self):
        integers = list(range(0,10))
        result = parallel_map(identity, integers, args = (1,), kwargs = {'test' : True})
        self.assertEqual(result, integers)

if __name__ == '__main__':
    unittest.main()