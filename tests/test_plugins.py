# -*- coding: utf-8 -*-
import unittest
from pathlib import Path
from tempfile import gettempdir

from iris import AbstractRawDataset, DiffractionDataset
from iris.plugins import load_plugin

TEST_PLUGIN_PATH = Path(__file__).parent / "plugin_fixture.py"
BROKEN_PLUGIN_PATH = Path(__file__).parent / "broken_plugin.py"


class TestPlugin(unittest.TestCase):
    def setUp(self):
        load_plugin(TEST_PLUGIN_PATH)

    def test_experimental_parameters(self):
        """ Test that arbitrary experimental parameters can be manipulated """
        from iris.plugins import TestRawDatasetPlugin

        test = TestRawDatasetPlugin()

        test.is_useful = False
        self.assertFalse(test.is_useful)

        test.is_useful = True
        self.assertTrue(test.is_useful)

    def test_reduction(self):
        """ Test that data reduction works """
        from iris.plugins import TestRawDatasetPlugin

        test = TestRawDatasetPlugin()
        test.scans = [1, 2]
        test.time_points = [-1, 0, 1]

        temp_file = Path(gettempdir()) / "plugin_test.hdf5"
        with DiffractionDataset.from_raw(test, filename=temp_file, mode="w") as dataset:
            self.assertEqual(test.temperature, dataset.temperature)
            # Assert that extra metadata is not kept
            self.assertTrue(hasattr(test, "is_useful"))
            self.assertFalse(hasattr(dataset, "is_useful"))


class TestBrokenPlugin(unittest.TestCase):
    def test_loading_broken_plugin(self):
        """ Test that exceptions are caught when loading a broken plug-in. """
        load_plugin(BROKEN_PLUGIN_PATH)


if __name__ == "__main__":
    unittest.main()
