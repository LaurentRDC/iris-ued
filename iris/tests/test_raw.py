
from .. import parse_tagfile
from .dummies import dummy_raw_dataset
from os.path import join, dirname
import unittest

PARSED_METADATA = {'exposure': 15.0,
                   'fluence': 13.0,
                   'energy': 90.0,
                   'current': None,
                   'esizex': None,
                   'esizey': None,
                   'lsizex': None,
                   'lsizey': None}

class TestParseTagfile(unittest.TestCase):
    
    def setUp(self):
        self.metadata = parse_tagfile(join(dirname(__file__), 'tagfile.txt'))
    
    def test_values(self):
        """ Test that the result of parse_tagfile is correct """
        self.assertDictEqual(self.metadata, PARSED_METADATA)

class TestRawDataset(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()