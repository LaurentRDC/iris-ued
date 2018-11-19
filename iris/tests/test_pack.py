# -*- coding: utf-8 -*-
import unittest
from pathlib import Path
from tempfile import gettempdir

from . import TestRawDataset
from ..raw import AbstractRawDataset
from ..pack import CompactRawDataset


class TestCompactRawDataset(unittest.TestCase):
    def test_metadata(self):
        """ Test that valid metadata is preserved in CompactRawDataset. """
        test_dataset = TestRawDataset()

        path = CompactRawDataset.pack(
            test_dataset, Path(gettempdir()) / "test.hdf5", mode="w"
        )

        with CompactRawDataset(path) as packed:
            for key, val in test_dataset.metadata.items():
                # Only check valid metadata
                if key not in CompactRawDataset.valid_metadata:
                    continue
                self.assertEqual(getattr(packed, key), val)

    def test_valid_metadata(self):
        """ Test that the class attribute 'valid_metadata' is the same for AbstractRawDataset and CompactRawDataset """
        self.assertEqual(
            CompactRawDataset.valid_metadata, AbstractRawDataset.valid_metadata
        )

    def test_raw_data(self):
        """ Test that raw data can be decompressed """
        test_dataset = TestRawDataset()

        path = CompactRawDataset.pack(
            test_dataset, Path(gettempdir()) / "test.hdf5", mode="w"
        )

        resolution = test_dataset.raw_data(1, 1).shape
        with CompactRawDataset(path) as packed:
            for scan in test_dataset.scans:
                for timedelay in test_dataset.time_points:
                    self.assertEqual(packed.raw_data(timedelay, scan).shape, resolution)


if __name__ == "__main__":
    unittest.main()
