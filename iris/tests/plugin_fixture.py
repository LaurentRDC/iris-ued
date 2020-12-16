# -*- coding: utf-8 -*-
"""
Test plug-in
============
"""

from iris import AbstractRawDataset, ExperimentalParameter
import numpy as np


class TestRawDatasetPlugin(AbstractRawDataset):

    display_name = "Minimal raw dataset v1"

    temperature = ExperimentalParameter("temperature", ptype=float, default=500)
    is_useful = ExperimentalParameter("is_useful", ptype=bool, default=True)

    # We don't want pytest to collect this class as a test
    # https://stackoverflow.com/a/63430765
    __test__ = False

    def __init__(self, source=None, metadata=dict()):
        # Metadata can be filled as a dictionary before
        # initialization. # Attributes which are not
        # ExperimentalParameters are ignored.
        metadata.update(
            {"temperature": 100, "exposure": 1, "this_will_be_ignored": True}
        )
        super().__init__(source, metadata)

        self.time_points = tuple(range(0, 3))
        self.resolution = (64, 64)
        self.scans = tuple(range(1, 3))
        self.notes = "This is a test"
        self.is_useful = False

    def raw_data(self, *args, **kwargs):
        return np.random.random(size=self.resolution)
