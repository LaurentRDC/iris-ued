
import numpy as n
from ..optimizations import pmap
import unittest

def identity(obj, *args, **kwargs):
    """ ignores args and kwargs """
    return obj

class PMapTest(unittest.TestCase):

    def test_trivial_map_no_args(self):
        integers = list(range(0,10))
        result = pmap(identity, integers)
        self.assertEqual(integers, result)
    
    def test_map_one_process(self):
        """ Test pmap with a single process, which does not use multiprocessing.Pool.map """
        integers = list(range(0,10))
        result = pmap(identity, integers, processes = 1)
        self.assertEqual(integers, list(result))
    
    def test_trivial_map_with_args_and_kwargs(self):
        integers = list(range(0,10))
        result = pmap(identity, integers, args = (1,), kwargs = {'test' : True})
        self.assertEqual(result, integers)
    
    def test_on_generator(self):
        """ Test pmap on an input generator """
        integers = range(0, 10)
        self.assertEqual(list(integers), list(pmap(identity, integers, processes = 2)))
        self.assertEqual(list(integers), list(pmap(identity, integers, processes = 1)))

if __name__ == '__main__':
    unittest.main()