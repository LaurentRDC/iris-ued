import sys
import os
from contextlib import suppress
from itertools import repeat
from tempfile import gettempdir

import numpy as np
from numpy.random import random

from crystals import Crystal
from iris import PowderDiffractionDataset, DiffractionDataset
from pathlib import Path
import pytest

np.random.seed(23)


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
    """Test scattering vector calibration"""
    crystal = Crystal.from_database("vo2-m1")
    powder_dataset.powder_calq(crystal, (10, 100), [(1, 0, 0), (2, 0, 0)])

    # Check that shapes match
    assert powder_dataset.powder_eq().shape == powder_dataset.scattering_vector.shape
    # Check that scattering_vector is strictly increasing
    assert np.all(np.greater(np.diff(powder_dataset.scattering_vector), 0))


def test_powder_baseline_limits(powder_dataset):
    """Test that the baseline is never less than 0, and the baseline-subtracted data is never negative."""

    powder_dataset.compute_baseline(
        first_stage="sym6", wavelet="qshift3", level=1, mode="periodic"
    )

    # Test that the baseline is always positive
    baseline = powder_dataset.powder_baseline(None)
    assert np.all(np.greater_equal(baseline, 0))

    data_bgr = powder_dataset.powder_data(None, bgr=True)
    assert np.all(np.greater_equal(data_bgr, 0))


def test_powder_data_retrieval(powder_dataset):
    """Test the size of the output from PowderDiffractionDataset.powder_data"""
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
    """Test PowderDiffractionDataset.powder_eq()"""
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
