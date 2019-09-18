# -*- coding: utf-8 -*-
"""
Example Plugin
==============
"""

from iris import AbstractRawDataset, ExperimentalParameter
import numpy as np


class MinimalRawDataset(AbstractRawDataset):
    """
    Minimal raw dataset interface. This is a showcase of what is possible with iris plug-ins.

    This file should be placed as-is in the :file:`~\iris_plugins` folder 
    (:file:`C:\\Users\\Username\\iris_plugins` on Windows)

    Minimal full implementation is only the :meth:`raw_data` method.
    """
    # Optionally, a display name can be added as a class attribute
    # This affects GUI displays only. By default, the class name is used.
    display_name = "Minimal raw dataset v1"

    # While the temperature metadata is already implemented in AbstractRawDataset,
    # it can be overriden in subclasses
    temperature = ExperimentalParameter("temperature", ptype=float, default=500)

    # New metadata that was not defined in AbstractRawDataset can also be added
    is_useful = ExperimentalParameter("is_useful", ptype=bool, default=True)

    def __init__(self, source=None, metadata=dict()):
        # Metadata can be filled as a dictionary before
        # initialization. # Attributes which are not
        # ExperimentalParameters are ignored.
        metadata.update(
            {"temperature": 100, "exposure": 1, "this_will_be_ignored": True}
        )
        super().__init__(source, metadata)

        # Metadata can also be changed attribute by attribute
        # Attributes which are not ExperimentalParameters
        self.time_points = tuple(range(0, 10))
        self.resolution = (512, 512)
        self.scans = tuple(range(1, 10))
        self.notes = "This is only an example"
        self.is_useful = False

    def raw_data(self, *args, **kwargs):
        return np.random.random(size=(512, 512))
