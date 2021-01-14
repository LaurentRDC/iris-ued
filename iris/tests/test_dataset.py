import sys
import os
from contextlib import suppress
from itertools import repeat
from tempfile import gettempdir

import numpy as np
from numpy.random import random

from crystals import Crystal
from iris import DiffractionDataset, PowderDiffractionDataset
from iris.dataset import SWMR_AVAILABLE
from skued import nfold, RectSelection, DiskSelection, RingSelection, ArbitrarySelection
from pathlib import Path
from . import TestRawDataset
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


def test_creation_from_raw_default(fname):
    """ Test that DiffractionDataset.from_raw() works with default settigns """
    raw = TestRawDataset()

    with DiffractionDataset.from_raw(raw, filename=fname, mode="w") as dataset:
        assert dataset.diffraction_group["intensity"].shape == raw.resolution + (
            len(raw.time_points),
        )


def test_creation_from_raw_alignment(fname):
    """ Test that DiffractionDataset.from_raw(..., align = True) does not throw any errors """
    raw = TestRawDataset()

    with DiffractionDataset.from_raw(
        raw, filename=fname, align=True, mode="w"
    ) as dataset:
        assert dataset.diffraction_group["intensity"].shape == raw.resolution + (
            len(raw.time_points),
        )


def test_creation_from_raw_multiprocess(fname):
    """ Test that DiffractionDataset.from_raw(..., processes = 2) does not throw any errors """
    raw = TestRawDataset()

    with DiffractionDataset.from_raw(
        raw, filename=fname, align=False, processes=2, mode="w"
    ) as dataset:
        assert dataset.diffraction_group["intensity"].shape == raw.resolution + (
            len(raw.time_points),
        )


def test_creation_from_collection(fname):
    """ Test the creation of a DiffractionDataset from a collection of patterns """
    patterns = repeat(random(size=(256, 256)), 10)
    metadata = {"fluence": 10, "energy": 90}

    with DiffractionDataset.from_collection(
        patterns,
        filename=fname,
        time_points=list(range(10)),
        metadata=metadata,
        dtype=np.float16,
        mode="w",
    ) as dataset:

        assert dataset.diffraction_group["intensity"].shape == (256, 256, 10)
        assert dataset.diffraction_group["intensity"].dtype == np.float16
        assert dataset.fluence == metadata["fluence"]
        assert dataset.energy == metadata["energy"]
        assert list(dataset.time_points) == list(map(float, range(10)))


@pytest.fixture
def dataset():
    patterns = list(repeat(random(size=(256, 256)), 5))
    metadata = {"fluence": 10, "energy": 90}
    filename = Path(gettempdir()) / "test.hdf5"
    dset = DiffractionDataset.from_collection(
        patterns,
        filename=filename,
        time_points=range(5),
        metadata=metadata,
        mode="w",
    )
    setattr(dset, "patterns", patterns)
    yield dset
    dset.close()
    del dset
    with suppress(OSError):
        os.remove(filename)


def test_file_modes(dataset):
    """ Successively open and close the same dataset with different file modes. """
    fname = dataset.filename
    metadata = dataset.metadata
    dataset.close()

    with DiffractionDataset(fname, mode="r") as dset:
        assert metadata == dset.metadata
        assert dset.mode == "r"

    for mode in ("r+", "a"):
        with DiffractionDataset(fname, mode=mode) as dset:
            assert metadata == dset.metadata
            assert dset.mode == "r+"

    with pytest.raises(OSError):
        DiffractionDataset(fname, mode="x")

    # Reopen dataset so it can be deleted
    dataset = DiffractionDataset(fname, mode="r")


def test_write_access(dataset):
    """ Check that certain operations respect write access """
    fname = dataset.filename
    metadata = dataset.metadata
    dataset.close()

    with DiffractionDataset(fname, mode="r") as dset:
        with pytest.raises(IOError):
            dset.symmetrize(mod=3, center=(0, 0))

    # Reopen dataset so it can be deleted
    dataset = DiffractionDataset(fname, mode="r")


def test_dataset_metadata(dataset):
    """ Test that the property 'metadata' is working correctly"""
    metadata = dataset.metadata
    for required in DiffractionDataset.valid_metadata:
        assert required in metadata
    assert "filename" in metadata


def test_notes(dataset):
    """ Test that updating the notes works as intended """
    dataset.notes = "test notes"
    assert dataset.notes == "test notes"
    dataset.notes = "different notes"
    assert dataset.notes == "different notes"


def test_diff_apply(dataset):
    """ Test that the diff_apply method works as expected """
    before = np.array(dataset.diffraction_group["intensity"])
    dataset.diff_apply(lambda arr: arr * 2)
    after = np.array(dataset.diffraction_group["intensity"])
    assert np.allclose(2 * before, after)

    with pytest.raises(TypeError):
        dataset.diff_apply(None)


@pytest.mark.skipif(not SWMR_AVAILABLE, reason="Parallel execution is not available")
def test_diff_apply_parallel(dataset):
    """ Test that the diff_apply method works as expected in parallel mode """
    before = np.array(dataset.diffraction_group["intensity"])
    dataset.diff_apply(double, processes=2)
    after = np.array(dataset.diffraction_group["intensity"])
    assert np.allclose(2 * before, after)

    with pytest.raises(TypeError):
        dataset.diff_apply(None)


def test_symmetrization(dataset):
    """ Test correctness of symmetrization operation """
    before = np.array(dataset.diffraction_group["intensity"])
    symmetrized = np.array(before, copy=True)
    for index, _ in enumerate(dataset.time_points):
        symmetrized[:, :, index] = nfold(
            before[:, :, index], mod=3, center=(63, 65), mask=dataset.valid_mask
        )

    dataset.symmetrize(mod=3, center=(63, 65))
    after = np.array(dataset.diffraction_group["intensity"])

    assert np.allclose(symmetrized, after)
    assert dataset.center == (
        63,
        65,
    ), "Diffraction center was not properly set after symmetrization"


@pytest.mark.skipif(not SWMR_AVAILABLE, reason="Parallel execution is not available")
def test_symmetrization_parallel(dataset):
    """ Test correctness of symmetrization operation in parallel mode """
    before = np.array(dataset.diffraction_group["intensity"])
    symmetrized = np.array(before, copy=True)
    for index, _ in enumerate(dataset.time_points):
        symmetrized[:, :, index] = nfold(
            before[:, :, index], mod=3, center=(63, 65), mask=dataset.valid_mask
        )

    dataset.symmetrize(mod=3, center=(63, 65), processes=2)
    after = np.array(dataset.diffraction_group["intensity"])

    assert np.allclose(symmetrized, after)


def test_data(dataset):
    """ Test that data stored in DiffractionDataset is correct """
    for time, pattern in zip(list(dataset.time_points), getattr(dataset, "patterns")):
        assert np.allclose(dataset.diff_data(time), pattern)


def test_time_zero_shift(dataset):
    """ Test that data changed with time_zero_shift() """
    unshifted = np.array(dataset.time_points)
    dataset.shift_time_zero(100)
    shifted = np.array(dataset.time_points)

    assert not np.allclose(unshifted, shifted)
    assert np.allclose(unshifted + 100, shifted)


def test_diff_eq(dataset):
    """ test that DiffractionDataset.diff_eq() returns the correct array """
    dataset.shift_time_zero(10)
    assert np.allclose(dataset.diff_eq(), np.zeros(dataset.resolution))

    dataset.shift_time_zero(-20)
    eq = np.mean(np.stack(getattr(dataset, "patterns"), axis=-1), axis=2)
    assert np.allclose(dataset.diff_eq(), eq)


def test_time_series(dataset):
    """ Test that the DiffractionDataset.time_series method is working """

    r1, r2, c1, c2 = 100, 120, 45, 57
    stack = np.stack(getattr(dataset, "patterns"), axis=-1)
    ts = np.mean(stack[r1:r2, c1:c2], axis=(0, 1))

    assert np.allclose(dataset.time_series([r1, r2, c1, c2], relative=False), ts)

    assert np.allclose(dataset.time_series([r1, r2, c1, c2], relative=True), ts)


def test_time_series_selection(dataset):
    """Test that the DiffractionDataset.time_series_selection
    method is working as expected"""
    mask = np.random.choice([True, False], size=dataset.resolution)
    selection = ArbitrarySelection(mask)

    stack = np.stack(getattr(dataset, "patterns"), axis=-1)
    ts = np.mean(stack[selection], axis=(0, 1))

    assert np.allclose(dataset.time_series_selection(selection, relative=False), ts)

    assert np.allclose(dataset.time_series_selection(selection, relative=True), ts)


def test_selection_rect(dataset):
    """Comparison of DiffractionDataset.time_series vs
    DiffractionDataset.time_series_selection with
    DiffractionDataset.selection_rect"""

    r1, r2, c1, c2 = 100, 120, 45, 57
    selection = RectSelection(dataset.resolution, r1, r2, c1, c2)

    ts = dataset.time_series([r1, r2 + 1, c1, c2 + 1], relative=False)
    tsbm = dataset.time_series_selection(selection, relative=False)
    assert np.allclose(ts, tsbm)

    ts = dataset.time_series([r1, r2 + 1, c1, c2 + 1], relative=True)
    tsbm = dataset.time_series_selection(selection, relative=True)
    assert np.allclose(ts, tsbm)


def test_selection_disk_relative(dataset):
    """Test relative DiffractionDataset.time_series_selection with DiffractionDataset.selection_disk"""
    selection = DiskSelection(dataset.resolution, center=(120, 200), radius=10)

    # We modify the dataset so that within the selection, only zeroes are found
    # Note that HDF5 does not support fancy boolean indexing, so we must change the
    # content image-by-image.
    for index, _ in enumerate(dataset.time_points):
        arr = dataset.diffraction_group["intensity"][:, :, index]
        arr[selection] = 0
        dataset.diffraction_group["intensity"][:, :, index] = arr

    ts = dataset.time_series_selection(selection, relative=False)
    assert np.allclose(ts, np.zeros_like(ts))


def test_selection_disk_nonrelative(dataset):
    """Test DiffractionDataset.time_series_selection with DiffractionDataset.selection_disk"""
    selection = DiskSelection(dataset.resolution, center=(120, 200), radius=10)

    for index, _ in enumerate(dataset.time_points):
        arr = dataset.diffraction_group["intensity"][:, :, index]
        arr[selection] = 1
        dataset.diffraction_group["intensity"][:, :, index] = arr

    ts = dataset.time_series_selection(selection, relative=True)
    assert np.allclose(ts, np.ones_like(ts))


def test_selection_ring(dataset):
    """Test DiffractionDataset.time_series_selection with DiffractionDataset.selection_ring"""
    selection = RingSelection(
        dataset.resolution, center=(120, 200), inner_radius=10, outer_radius=20
    )

    # We modify the dataset so that within the selection, only zeroes are found
    # Note that HDF5 does not support fancy boolean indexing, so we must change the
    # content image-by-image.
    for index, _ in enumerate(dataset.time_points):
        arr = dataset.diffraction_group["intensity"][:, :, index]
        arr[selection] = 0
        dataset.diffraction_group["intensity"][:, :, index] = arr

    ts = dataset.time_series_selection(selection, relative=False)
    assert np.allclose(ts, np.zeros_like(ts))


def test_selection_ring_relative(dataset):
    """Test relative DiffractionDataset.time_series_selection with DiffractionDataset.selection_ring"""
    selection = RingSelection(
        dataset.resolution, center=(120, 200), inner_radius=10, outer_radius=20
    )

    for index, _ in enumerate(dataset.time_points):
        arr = dataset.diffraction_group["intensity"][:, :, index]
        arr[selection] = 1
        dataset.diffraction_group["intensity"][:, :, index] = arr

    ts = dataset.time_series_selection(selection, relative=True)
    assert np.allclose(ts, np.ones_like(ts))


@pytest.fixture
def powder_dataset():
    patterns = list(repeat(random(size=(128, 128)), 5))
    metadata = {"fluence": 10, "energy": 90}
    filename = Path(gettempdir()) / "test.hdf5"
    diff_dataset = DiffractionDataset.from_collection(
        patterns,
        filename=filename,
        time_points=range(5),
        metadata=metadata,
        mode="w",
    )
    dset = PowderDiffractionDataset.from_dataset(diff_dataset, center=(23, 45))
    setattr(dset, "patterns", patterns)
    yield dset
    dset.close()
    del dset
    with suppress(OSError):
        os.remove(filename)


def test_powder_baseline_attributes(powder_dataset):
    """Test that the attributes related to baseline have correct defaults and are
    set to the correct values after computation"""
    assert powder_dataset.first_stage is ""
    assert powder_dataset.wavelet is ""
    assert powder_dataset.level == 0
    assert powder_dataset.niter == 0

    powder_dataset.compute_baseline(
        first_stage="sym6", wavelet="qshift3", level=1, mode="periodic"
    )

    assert powder_dataset.first_stage == "sym6"
    assert powder_dataset.wavelet == "qshift3"
    assert powder_dataset.level == 1


def test_powder_calq(powder_dataset):
    """ Test scattering vector calibration """
    crystal = Crystal.from_database("vo2-m1")
    powder_dataset.powder_calq(crystal, (10, 100), [(1, 0, 0), (2, 0, 0)])

    # Check that shapes match
    assert powder_dataset.powder_eq().shape == powder_dataset.scattering_vector.shape
    # Check that scattering_vector is strictly increasing
    assert np.all(np.greater(np.diff(powder_dataset.scattering_vector), 0))


def test_powder_baseline_limits(powder_dataset):
    """ Test that the baseline is never less than 0, and the baseline-subtracted data is never negative. """

    powder_dataset.compute_baseline(
        first_stage="sym6", wavelet="qshift3", level=1, mode="periodic"
    )

    # Test that the baseline is always positive
    baseline = powder_dataset.powder_baseline(None)
    assert np.all(np.greater_equal(baseline, 0))

    data_bgr = powder_dataset.powder_data(None, bgr=True)
    assert np.all(np.greater_equal(data_bgr, 0))


def test_powder_data_retrieval(powder_dataset):
    """ Test the size of the output from PowderDiffractionDataset.powder_data """
    full_shape = (len(powder_dataset.time_points), powder_dataset.px_radius.size)

    full_data = powder_dataset.powder_data(timedelay=None)
    assert full_data.shape == full_shape

    time_slice = powder_dataset.powder_data(timedelay=powder_dataset.time_points[0])
    assert time_slice.shape == powder_dataset.px_radius.shape


def test_recomputing_angular_average(powder_dataset):
    """Test that recomputing the angular average multiple times will work. This also
    tests resizing all powder data multiple times."""
    powder_dataset.compute_angular_averages(center=(34, 56))
    powder_dataset.compute_baseline(first_stage="sym6", wavelet="qshift1")
    powder_dataset.compute_angular_averages(center=(45, 45), normalized=False)
    powder_dataset.compute_baseline(first_stage="sym5", wavelet="qshift2")
    powder_dataset.compute_angular_averages(center=(34, 56), angular_bounds=(15.3, 187))
    powder_dataset.compute_baseline(first_stage="sym6", wavelet="qshift1")


def test_powder_eq(powder_dataset):
    """ Test PowderDiffractionDataset.powder_eq() """
    eq = powder_dataset.powder_eq()
    assert eq.shape == powder_dataset.px_radius.shape

    powder_dataset.compute_baseline(
        first_stage="sym6", wavelet="qshift3", mode="constant"
    )
    eq = powder_dataset.powder_eq(bgr=True)
    assert eq.shape == powder_dataset.px_radius.shape

    powder_dataset.shift_time_zero(1 + abs(min(powder_dataset.time_points)))
    eq = powder_dataset.powder_eq()
    assert eq.shape == powder_dataset.px_radius.shape
    assert np.allclose(eq, np.zeros_like(eq))
