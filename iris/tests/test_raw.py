# -*- coding: utf-8 -*-
import numpy as np

from . import TestRawDataset
from iris import AbstractRawDataset
from iris.meta import ExperimentalParameter
import pytest


def test_raw_abstract_methods():
    """Test that instantiation of AbstractRawDataset
    raises an error"""
    with pytest.raises(TypeError):
        AbstractRawDataset("")


def test_raw_minimal_methods():
    """
    Test implementing the minimal methods:

    * raw_data
    """
    TestRawDataset()


def test_raw_data_bounds():
    """Test that a ValueError is raised if ``timedelay`` or ``scan`` are out-of-bounds."""
    test_dataset = TestRawDataset()

    with pytest.raises(ValueError):
        test_dataset.raw_data(timedelay=20, scan=1)

    with pytest.raises(ValueError):
        test_dataset.raw_data(timedelay=5, scan=-1)


def test_raw_experimental_parameters():
    """Test the behavior of the ExperimentalParameter descriptor"""

    test_dataset = TestRawDataset()

    assert test_dataset.test == 0

    test_dataset.test = 1
    assert test_dataset.test == 1

    with pytest.raises(TypeError):
        test_dataset.test = "test"


def test_raw_valid_metadata():
    """Test that the class attribute 'valid_metadata' is working as intended"""

    assert "test" in TestRawDataset.valid_metadata
    assert AbstractRawDataset.valid_metadata <= TestRawDataset.valid_metadata


def test_raw_init_metadata():
    """Test that metadata is recorded correctly inside __init__ and
    that invalid metadata is ignored."""
    test_dataset = TestRawDataset(
        metadata={"test": 5, "fluence": -2, "random_attr": None}
    )
    assert test_dataset.test == 5
    assert test_dataset.fluence == -2

    # Invalid metadata should be ignored.
    assert not hasattr(test_dataset, "random_attr")
