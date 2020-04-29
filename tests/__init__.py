# -*- coding: utf-8 -*-
from iris import AbstractRawDataset, check_raw_bounds
from iris.meta import ExperimentalParameter
import numpy as np
import unittest


class TestRawDataset(AbstractRawDataset):
    """ Class for using raw datasets in tests """

    test = ExperimentalParameter("test", int, default=0)
    resolution = ExperimentalParameter("resolution", tuple, (16, 16))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.time_points = list(range(0, 10))
        self.scans = list(range(1, 3))

    @check_raw_bounds
    def raw_data(self, timedelay, scan=1):
        return np.ones((self.resolution), dtype=np.uint8)


if __name__ == "__main__":
    unittest.main()
