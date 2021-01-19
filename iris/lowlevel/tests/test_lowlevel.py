import sys
import os
from contextlib import suppress
from itertools import repeat
from tempfile import gettempdir

import numpy as np
from numpy.random import random
import npstreams as ns

from crystals import Crystal
from .. import (
    LowLevelDataset,
    InternalDatasets,
    MissingTimePointWarning,
    SWMR_AVAILABLE,
    IOMode,
)
from skued import (
    ArbitrarySelection,
    RectSelection,
    DiskSelection,
    RingArcSelection,
    RingSelection,
)
from warnings import catch_warnings, simplefilter
from pathlib import Path
import pytest

np.random.seed(23)


def double(im):
    return im * 2


@pytest.fixture
def fname():
    f = Path(gettempdir()) / "test.hdf5"
    yield f
    with suppress(OSError):
        os.remove(f)


def test_lowlevel_creation_from_collection(fname):
    """ Test the creation of a LowLevelDataset from a collection of patterns """
    patterns = repeat(random(size=(256, 256)), 10)
    metadata = {"fluence": 10, "energy": 90}

    with LowLevelDataset.from_collection(
        patterns,
        filename=fname,
        time_points=list(range(-5, 5)),
        metadata=metadata,
        mode=IOMode.Overwrite,
    ) as dataset:

        assert dataset.get_dataset(InternalDatasets.Intensity).shape == (256, 256, 10)
        assert dataset.metadata["fluence"] == metadata["fluence"]
        assert dataset.metadata["energy"] == metadata["energy"]
        assert list(dataset.time_points) == list(map(float, range(-5, 5)))


@pytest.fixture
def dataset():
    patterns = list(repeat(random(size=(256, 256)), 10))
    metadata = {"fluence": 10, "energy": 90}
    filename = Path(gettempdir()) / "test.hdf5"
    dset = LowLevelDataset.from_collection(
        patterns,
        filename=filename,
        time_points=range(-5, 5),
        metadata=metadata,
        mode=IOMode.Overwrite,
    )
    setattr(dset, "patterns", patterns)
    yield dset
    dset.close()
    del dset
    with suppress(OSError):
        os.remove(filename)


def test_lowlevel_file_modes(dataset):
    """ Successively open and close the same dataset with different file modes. """
    fname = dataset.filename
    metadata = dataset.metadata
    dataset.close()

    with LowLevelDataset(fname, mode=IOMode.ReadOnly) as dset:
        assert metadata == dset.metadata
        assert dset.mode == IOMode.ReadOnly

    for mode in ("r+", "a"):
        with LowLevelDataset(fname, mode=IOMode(mode)) as dset:
            assert metadata == dset.metadata
            assert dset.mode == IOMode.ReadWrite

    # Reopen dataset so it can be deleted
    dataset = LowLevelDataset(fname)


def test_lowlevel_write_access(dataset):
    """ Check that certain operations respect write access """
    fname = dataset.filename
    metadata = dataset.metadata
    dataset.close()

    with LowLevelDataset(fname, mode=IOMode.ReadOnly) as dset:
        with pytest.raises(IOError):
            dset.diff_apply(func=double)

    # Reopen dataset so it can be deleted
    dataset = LowLevelDataset(fname, IOMode.ReadOnly)


def test_lowlevel_diff_apply(dataset):
    """ Test that the diff_apply method works as expected """
    before = np.array(dataset.get_dataset(InternalDatasets.Intensity))
    dataset.diff_apply(lambda arr: arr * 2)
    after = np.array(dataset.get_dataset(InternalDatasets.Intensity))
    assert np.allclose(2 * before, after)

    with pytest.raises(TypeError):
        dataset.diff_apply(None)


@pytest.mark.skipif(not SWMR_AVAILABLE, reason="Parallel execution is not available")
def test_lowlevel_diff_apply_parallel(dataset):
    """ Test that the diff_apply method works as expected in parallel mode """
    before = np.array(dataset.get_dataset(InternalDatasets.Intensity))
    dataset.diff_apply(double, processes=2)
    after = np.array(dataset.get_dataset(InternalDatasets.Intensity))
    assert np.allclose(2 * before, after)

    with pytest.raises(TypeError):
        dataset.diff_apply(None)


def test_lowlevel_mask_modifications(dataset):
    """ test that modifying the mask from a LowLevelDataset works as expected """
    old_mask = dataset.mask
    func = lambda m: np.logical_not(m)
    new_mask = func(old_mask)
    dataset.mask = new_mask
    assert np.allclose(dataset.mask, new_mask)

    # Wrong type
    func2 = lambda m: m.astype(np.float)
    with pytest.raises(TypeError):
        dataset.mask = func2(dataset.mask)

    # Wrong shape
    func3 = lambda m: np.zeros((m.shape[0] // 2, m.shape[1] // 2), dtype=np.bool)
    with pytest.raises(ValueError):
        dataset.mask = func3(dataset.mask)


def test_lowlevel_diffraction_pattern(dataset):
    """ Test that data stored in LowLevelDataset is correct """
    for time, pattern in zip(list(dataset.time_points), getattr(dataset, "patterns")):
        assert np.allclose(dataset.diffraction_pattern(timedelay=time), pattern)


def test_lowlevel_time_zero_shift(dataset):
    """ Test that data changed with time_zero_shift() """
    unshifted = np.array(dataset.time_points)
    dataset.time_zero_shift = -100  # time zero is before the start of the data
    with catch_warnings():
        simplefilter("ignore", category=MissingTimePointWarning)
        assert dataset.get_time_index(0) == 0
    shifted = np.array(dataset.time_points)

    assert not np.allclose(unshifted, shifted)
    assert np.allclose(unshifted + 100, shifted)


def test_lowlevel_equilibrium_pattern(dataset):
    """ test that LowLevelDataset.equilibrium_pattern() returns the correct array """
    dataset.time_zero_shift = -10  # time zero is before the start of the data
    with catch_warnings():
        simplefilter("ignore", category=MissingTimePointWarning)
        assert dataset.get_time_index(0) == 0
    assert np.allclose(
        dataset.equilibrium_pattern,
        np.array(dataset.get_dataset(InternalDatasets.Intensity)[:, :, 0]),
    )

    dataset.time_zero_shift = 20  # time zero is after the start of the data
    with catch_warnings():
        simplefilter("ignore", category=MissingTimePointWarning)
        assert dataset.get_time_index(0) == len(dataset.time_points) - 1
    eq = ns.average(getattr(dataset, "patterns"), axis=2)
    assert np.allclose(dataset.equilibrium_pattern, eq)


def test_lowlevel_time_series(dataset):
    """ Test that the LowLevelDataset.time_series method is working """

    r1, r2, c1, c2 = 100, 120, 45, 57
    stack = np.stack(getattr(dataset, "patterns"), axis=-1)
    ts = np.mean(stack[r1:r2, c1:c2], axis=(0, 1))

    assert np.allclose(
        np.mean(dataset.time_series(r1, r2, c1, c2, relative=False), axis=(0, 1)), ts
    )

    t0_index = dataset.get_time_index(0)
    assert np.allclose(
        np.mean(dataset.time_series(r1, r2, c1, c2, relative=True), axis=(0, 1)),
        ts / np.mean(ts[:t0_index]),
    )
