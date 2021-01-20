import sys
import os
from contextlib import suppress
from itertools import repeat
from tempfile import gettempdir

import numpy as np
from numpy.random import random

from crystals import Crystal
from iris import DiffractionDataset
from iris.lowlevel import InternalDatasets, SWMR_AVAILABLE
from skued import nfold, autocenter
from pathlib import Path
from . import TestRawDataset
from time import time
import pytest

np.random.seed(23)


def test_creation_from_raw_default(tmp_path):
    """ Test that DiffractionDataset.from_raw() works with default settigns """
    raw = TestRawDataset()

    with DiffractionDataset.from_raw(
        raw, filename=tmp_path / "test.hdf5", mode="w"
    ) as dataset:
        assert np.asarray(dataset).shape == raw.resolution + (len(raw.time_points),)


def test_creation_from_raw_alignment(tmp_path):
    """ Test that DiffractionDataset.from_raw(..., align = True) does not throw any errors """
    raw = TestRawDataset()

    with DiffractionDataset.from_raw(
        raw, filename=tmp_path / "test.hdf5", align=True
    ) as dataset:
        assert np.asarray(dataset).shape == raw.resolution + (len(raw.time_points),)


def test_creation_from_raw_multiprocess(tmp_path):
    """ Test that DiffractionDataset.from_raw(..., processes = 2) does not throw any errors """
    raw = TestRawDataset()

    with DiffractionDataset.from_raw(
        raw,
        filename=tmp_path / "test.hdf5",
        align=False,
        processes=2,
    ) as dataset:
        assert np.asarray(dataset).shape == raw.resolution + (len(raw.time_points),)


def test_creation_from_collection(tmp_path):
    """ Test the creation of a DiffractionDataset from a collection of patterns """
    patterns = repeat(random(size=(256, 256)), 10)
    metadata = {"fluence": 10, "energy": 90}

    with DiffractionDataset.from_collection(
        patterns,
        filename=tmp_path / "test.hdf5",
        time_points=list(range(10)),
        metadata=metadata,
    ) as dataset:

        assert np.asarray(dataset).shape == (256, 256, 10)
        assert dataset.metadata["fluence"] == metadata["fluence"]
        assert dataset.metadata["energy"] == metadata["energy"]
        assert list(dataset.time_points) == list(map(float, range(10)))


@pytest.fixture
def dataset():
    patterns = list(repeat(random(size=(256, 256)), 5))
    metadata = {"fluence": 10, "energy": 90}

    with DiffractionDataset.from_collection(
        patterns,
        filename=Path(gettempdir()) / "test.hdf5",
        time_points=range(5),
        metadata=metadata,
    ) as dset:
        setattr(dset, "patterns", patterns)
        yield dset


def test_symmetrization(dataset):
    """ Test correctness of symmetrization operation """
    r, c = dataset.center
    before = np.array(dataset)
    symmetrized = np.array(before, copy=True)
    for index, _ in enumerate(dataset.time_points):
        # Note the difference in the center format for nfold: (col, row)
        symmetrized[:, :, index] = nfold(
            before[:, :, index], mod=3, center=(c, r), mask=dataset.mask
        )

    dataset.symmetrize(mod=3)
    after = np.array(dataset)

    assert np.allclose(symmetrized, after)


@pytest.mark.skipif(not SWMR_AVAILABLE, reason="Parallel execution is not available")
def test_symmetrization_parallel(dataset):
    """ Test correctness of symmetrization operation in parallel mode """
    r, c = dataset.center
    before = np.array(dataset)
    symmetrized = np.array(before, copy=True)
    for index, _ in enumerate(dataset.time_points):
        symmetrized[:, :, index] = nfold(
            before[:, :, index], mod=3, center=(c, r), mask=dataset.mask
        )

    dataset.symmetrize(mod=3, processes=2)
    after = np.array(dataset)

    assert np.allclose(symmetrized, after)


def test_data(dataset):
    """ Test that data stored in DiffractionDataset is correct """
    for time, pattern in zip(list(dataset.time_points), getattr(dataset, "patterns")):
        assert np.allclose(dataset.diffraction_pattern(time), pattern)


def test_time_series(dataset):
    """ Test that the DiffractionDataset.time_series method is working """

    r1, r2, c1, c2 = 100, 120, 45, 57
    stack = np.stack(getattr(dataset, "patterns"), axis=-1)
    ts = np.mean(stack[r1:r2, c1:c2], axis=(0, 1))

    assert np.allclose(dataset.time_series([r1, r2, c1, c2], relative=False), ts)

    assert np.allclose(
        dataset.time_series([r1, r2, c1, c2], relative=True), np.ones_like(ts)
    )
