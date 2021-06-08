# -*- coding: utf-8 -*-
from pathlib import Path
from tempfile import gettempdir
from contextlib import redirect_stdout
import io

from iris import load_plugin, AbstractRawDataset, DiffractionDataset

TEST_PLUGIN_PATH = Path(__file__).parent / "plugin_fixture.py"
BROKEN_PLUGIN_PATH = Path(__file__).parent / "broken_plugin.py"


def test_plugin_experimental_parameters():
    """Test that arbitrary experimental parameters can be manipulated"""
    load_plugin(TEST_PLUGIN_PATH)
    from iris.plugins import TestRawDatasetPlugin

    test = TestRawDatasetPlugin()

    test.is_useful = False
    assert not test.is_useful

    test.is_useful = True
    assert test.is_useful


def test_plugin_reduction():
    """Test that data reduction works"""
    load_plugin(TEST_PLUGIN_PATH)
    from iris.plugins import TestRawDatasetPlugin

    test = TestRawDatasetPlugin()
    test.scans = [1, 2]
    test.time_points = [-1, 0, 1]

    temp_file = Path(gettempdir()) / "plugin_test.hdf5"
    with DiffractionDataset.from_raw(test, filename=temp_file, mode="w") as dataset:
        assert test.temperature == dataset.temperature
        # Assert that extra metadata is not kept
        assert hasattr(test, "is_useful")
        assert not hasattr(dataset, "is_useful")


def test_loading_broken_plugin():
    """Test that exceptions are caught when loading a broken plug-in."""
    f = io.StringIO()
    with redirect_stdout(f):
        load_plugin(BROKEN_PLUGIN_PATH)
