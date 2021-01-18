import sys
import os
from contextlib import suppress
from itertools import repeat
from tempfile import gettempdir

import numpy as np
from numpy.random import random
import npstreams as ns

from crystals import Crystal
from .. import LowLevelDataset, InternalDatasets, SWMR_AVAILABLE, IOMode
from skued import (
    ArbitrarySelection,
    RectSelection,
    DiskSelection,
    RingArcSelection,
    RingSelection,
)
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


def test_lowlevel_dataset_metadata(dataset):
    """ Test that the property 'metadata' is working correctly"""
    metadata = dataset.metadata
    for required in LowLevelDataset.valid_metadata:
        assert required in metadata
    assert "filename" in metadata


def test_lowlevel_notes(dataset):
    """ Test that updating the notes works as intended """
    dataset.notes = "test notes"
    assert dataset.notes == "test notes"
    dataset.notes = "different notes"
    assert dataset.notes == "different notes"


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


def test_lowlevel_mask_apply(dataset):
    """ test that LowLevelDataset.mask_apply method works as expected """
    old_mask = dataset.mask
    func = lambda m: np.logical_not(m)
    new_mask = func(old_mask)
    dataset.mask_apply(func)
    assert np.allclose(dataset.mask, new_mask)

    with pytest.raises(TypeError):
        dataset.mask_apply(None)

    func2 = lambda m: m.astype(np.float)
    with pytest.raises(TypeError):
        dataset.mask_apply(func2)

    func3 = lambda m: np.zeros((m.shape[0] // 2, m.shape[1] // 2), dtype=np.bool)
    with pytest.raises(ValueError):
        dataset.mask_apply(func3)


def test_lowlevel_diffraction_pattern(dataset):
    """ Test that data stored in LowLevelDataset is correct """
    for time, pattern in zip(list(dataset.time_points), getattr(dataset, "patterns")):
        assert np.allclose(dataset.diffraction_pattern(timedelay=time), pattern)


def test_lowlevel_time_zero_shift(dataset):
    """ Test that data changed with time_zero_shift() """
    unshifted = np.array(dataset.time_points)
    dataset.shift_time_zero(100)
    assert dataset.get_time_index(0) == 0
    shifted = np.array(dataset.time_points)

    assert not np.allclose(unshifted, shifted)
    assert np.allclose(unshifted + 100, shifted)


def test_lowlevel_equilibrium_pattern(dataset):
    """ test that LowLevelDataset.equilibrium_pattern() returns the correct array """
    dataset.shift_time_zero(10)
    assert dataset.get_time_index(0) == 0
    assert np.allclose(
        dataset.equilibrium_pattern,
        np.array(dataset.get_dataset(InternalDatasets.Intensity)[:, :, 0]),
    )

    dataset.shift_time_zero(-20)
    assert dataset.get_time_index(0) == len(dataset.time_points) - 1
    eq = ns.average(getattr(dataset, "patterns"), axis=2)
    assert np.allclose(dataset.equilibrium_pattern, eq)


def test_lowlevel_time_series(dataset):
    """ Test that the LowLevelDataset.time_series method is working """

    r1, r2, c1, c2 = 100, 120, 45, 57
    stack = np.stack(getattr(dataset, "patterns"), axis=-1)
    ts = np.mean(stack[r1:r2, c1:c2], axis=(0, 1))

    assert np.allclose(dataset.time_series([r1, r2, c1, c2], relative=False), ts)

    t0_index = dataset.get_time_index(0)
    assert np.allclose(
        dataset.time_series([r1, r2, c1, c2], relative=True),
        ts - np.mean(ts[:t0_index]),
    )


def test_lowlevel_time_series_selection(dataset):
    """Test that the LowLevelDataset.time_series_selection
    method is working as expected"""
    mask = np.random.choice(
        [True, False], size=dataset.get_dataset(InternalDatasets.Intensity).shape[0:2]
    )
    selection = ArbitrarySelection(mask)

    stack = np.stack((im[selection] for im in getattr(dataset, "patterns")), axis=-1)
    ts = np.mean(stack, axis=(0, 1))

    assert np.allclose(dataset.time_series_selection(selection, relative=False), ts)

    assert np.allclose(
        dataset.time_series_selection(selection, relative=True), ts - np.mean(ts)
    )


def test_lowlevel_selection_rect(dataset):
    """Comparison of LowLevelDataset.time_series vs
    LowLevelDataset.time_series_selection with
    LowLevelDataset.selection_rect"""

    r1, r2, c1, c2 = 100, 120, 45, 57
    selection = RectSelection(
        dataset.get_dataset(InternalDatasets.Intensity).shape[0:2], r1, r2, c1, c2
    )

    ts = dataset.time_series([r1, r2 + 1, c1, c2 + 1], relative=False)
    tsbm = dataset.time_series_selection(selection, relative=False)
    assert np.allclose(ts, tsbm)

    ts = dataset.time_series([r1, r2 + 1, c1, c2 + 1], relative=True)
    tsbm = dataset.time_series_selection(selection, relative=True)
    assert np.allclose(ts, tsbm)


def test_lowlevel_selection_disk_nonrelative(dataset):
    """Test relative LowLevelDataset.time_series_selection with LowLevelDataset.selection_disk"""
    selection = DiskSelection(
        dataset.get_dataset(InternalDatasets.Intensity).shape[0:2],
        center=(120, 200),
        radius=10,
    )

    def mod(im):
        im[selection] = 5
        return im

    dataset.diff_apply(mod)

    ts = dataset.time_series_selection(selection, relative=False)
    assert np.allclose(ts, np.full_like(ts, fill_value=5))


def test_lowlevel_selection_disk_relative(dataset):
    """Test LowLevelDataset.time_series_selection with LowLevelDataset.selection_disk"""
    selection = DiskSelection(
        dataset.get_dataset(InternalDatasets.Intensity).shape[0:2],
        center=(120, 200),
        radius=10,
    )

    def mod(im):
        im[selection] = 5
        return im

    dataset.diff_apply(mod)

    ts = dataset.time_series_selection(selection, relative=True)
    assert np.allclose(ts, np.zeros_like(ts))


def test_lowlevel_selection_ring(dataset):
    """Test LowLevelDataset.time_series_selection with LowLevelDataset.selection_ring"""
    selection = RingSelection(
        dataset.get_dataset(InternalDatasets.Intensity).shape[0:2],
        center=(120, 200),
        inner_radius=10,
        outer_radius=20,
    )

    def mod(im):
        im[selection] = 2
        return im

    dataset.diff_apply(mod)

    ts = dataset.time_series_selection(selection, relative=False)
    assert np.allclose(ts, np.full_like(ts, fill_value=2))


def test_lowlevel_selection_ring_relative(dataset):
    """Test relative LowLevelDataset.time_series_selection with LowLevelDataset.selection_ring"""
    selection = RingSelection(
        dataset.get_dataset(InternalDatasets.Intensity).shape[0:2],
        center=(120, 200),
        inner_radius=10,
        outer_radius=20,
    )

    def mod(im):
        im[selection] = 1
        return im

    dataset.diff_apply(mod)

    ts = dataset.time_series_selection(selection, relative=True)
    assert np.allclose(ts - ts.max(), np.zeros_like(ts))
