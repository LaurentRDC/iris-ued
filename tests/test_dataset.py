import os.path
import sys
import unittest
from contextlib import suppress
from itertools import repeat
from tempfile import gettempdir

import numpy as np
from numpy.random import random

from crystals import Crystal
from iris import DiffractionDataset, PowderDiffractionDataset
from iris.dataset import SWMR_AVAILABLE
from skued import nfold, RectSelection, DiskSelection, RingSelection, ArbitrarySelection

from . import TestRawDataset

np.random.seed(23)


def double(im):
    return im * 2


class TestDiffractionDatasetCreation(unittest.TestCase):
    def setUp(self):
        self.fname = os.path.join(gettempdir(), "test.hdf5")

    def test_from_raw_default(self):
        """ Test that DiffractionDataset.from_raw() works with default settigns """
        raw = TestRawDataset()

        with DiffractionDataset.from_raw(raw, filename=self.fname, mode="w") as dataset:
            self.assertSequenceEqual(
                dataset.diffraction_group["intensity"].shape,
                raw.resolution + (len(raw.time_points),),
            )

    def test_from_raw_alignment(self):
        """ Test that DiffractionDataset.from_raw(..., align = True) does not throw any errors """
        raw = TestRawDataset()

        with DiffractionDataset.from_raw(
            raw, filename=self.fname, align=True, mode="w"
        ) as dataset:
            self.assertSequenceEqual(
                dataset.diffraction_group["intensity"].shape,
                raw.resolution + (len(raw.time_points),),
            )

    def test_from_raw_multiprocess(self):
        """ Test that DiffractionDataset.from_raw(..., processes = 2) does not throw any errors """
        raw = TestRawDataset()

        with DiffractionDataset.from_raw(
            raw, filename=self.fname, align=False, processes=2, mode="w"
        ) as dataset:
            self.assertSequenceEqual(
                dataset.diffraction_group["intensity"].shape,
                raw.resolution + (len(raw.time_points),),
            )

    def test_from_collection(self):
        """ Test the creation of a DiffractionDataset from a collection of patterns """
        patterns = repeat(random(size=(256, 256)), 10)
        metadata = {"fluence": 10, "energy": 90}

        with DiffractionDataset.from_collection(
            patterns,
            filename=self.fname,
            time_points=list(range(10)),
            metadata=metadata,
            dtype=np.float16,
            mode="w",
        ) as dataset:

            self.assertSequenceEqual(
                dataset.diffraction_group["intensity"].shape, (256, 256, 10)
            )
            self.assertEqual(dataset.diffraction_group["intensity"].dtype, np.float16)
            self.assertEqual(dataset.fluence, metadata["fluence"])
            self.assertEqual(dataset.energy, metadata["energy"])
            self.assertSequenceEqual(tuple(dataset.time_points), list(range(10)))

    def tearDown(self):
        with suppress(OSError):
            os.remove(self.fname)


class TestDiffractionDataset(unittest.TestCase):
    def setUp(self):
        self.patterns = list(repeat(random(size=(256, 256)), 5))
        self.metadata = {"fluence": 10, "energy": 90}
        self.dataset = DiffractionDataset.from_collection(
            self.patterns,
            filename=os.path.join(gettempdir(), "test.hdf5"),
            time_points=range(5),
            metadata=self.metadata,
            mode="w",
        )

    def tearDown(self):
        fname = self.dataset.filename
        self.dataset.close()
        del self.dataset
        with suppress(OSError):
            os.remove(fname)

    def test_file_modes(self):
        """ Successively open and close the same dataset with different file modes. """
        fname = self.dataset.filename
        metadata = self.dataset.metadata
        self.dataset.close()

        with self.subTest("Read-only mode"):
            with DiffractionDataset(fname, mode="r") as dset:
                self.assertEqual(metadata, dset.metadata)
                self.assertEqual(dset.mode, "r")

        with self.subTest("Read/write modes"):
            for mode in ("r+", "a"):
                with DiffractionDataset(fname, mode=mode) as dset:
                    self.assertEqual(metadata, dset.metadata)
                    self.assertEqual(dset.mode, "r+")

            with self.assertRaises(OSError):
                DiffractionDataset(fname, mode="x")

        # Reopen dataset so it can be deleted
        self.dataset = DiffractionDataset(fname, mode="r")

    def test_dataset_metadata(self):
        """ Test that the property 'metadata' is working correctly"""
        metadata = self.dataset.metadata
        for required in DiffractionDataset.valid_metadata:
            self.assertIn(required, metadata)
        self.assertIn("filename", metadata)

    def test_notes(self):
        """ Test that updating the notes works as intended """
        self.dataset.notes = "test notes"
        self.assertEqual(self.dataset.notes, "test notes")
        self.dataset.notes = "different notes"
        self.assertEqual(self.dataset.notes, "different notes")

    def test_diff_apply(self):
        """ Test that the diff_apply method works as expected """
        with self.subTest("Applying an operation"):
            before = np.array(self.dataset.diffraction_group["intensity"])
            self.dataset.diff_apply(lambda arr: arr * 2)
            after = np.array(self.dataset.diffraction_group["intensity"])
            self.assertTrue(np.allclose(2 * before, after))

        with self.subTest("Checking for callable"):
            with self.assertRaises(TypeError):
                self.dataset.diff_apply(None)

    @unittest.skipIf(not SWMR_AVAILABLE, reason="Parallel execution is not available")
    def test_diff_apply_parallel(self):
        """ Test that the diff_apply method works as expected in parallel mode """
        with self.subTest("Applying an operation"):
            before = np.array(self.dataset.diffraction_group["intensity"])
            self.dataset.diff_apply(double, processes=2)
            after = np.array(self.dataset.diffraction_group["intensity"])
            self.assertTrue(np.allclose(2 * before, after))

        with self.subTest("Checking for callable"):
            with self.assertRaises(TypeError):
                self.dataset.diff_apply(None)

    def test_resolution(self):
        """ Test that dataset resolution is correct """
        self.assertSequenceEqual(self.patterns[0].shape, self.dataset.resolution)

    def test_symmetrization(self):
        """ Test correctness of symmetrization operation """
        before = np.array(self.dataset.diffraction_group["intensity"])
        symmetrized = np.array(before, copy=True)
        for index, _ in enumerate(self.dataset.time_points):
            symmetrized[:, :, index] = nfold(
                before[:, :, index], mod=3, center=(63, 65)
            )

        self.dataset.symmetrize(mod=3, center=(63, 65))
        after = np.array(self.dataset.diffraction_group["intensity"])

        self.assertTrue(np.allclose(symmetrized, after))
        self.assertEqual(
            self.dataset.center,
            (63, 65),
            "Diffraction center was not properly set after symmetrization",
        )

    @unittest.skipIf(not SWMR_AVAILABLE, reason="Parallel execution is not available")
    def test_symmetrization_parallel(self):
        """ Test correctness of symmetrization operation in parallel mode """
        before = np.array(self.dataset.diffraction_group["intensity"])
        symmetrized = np.array(before, copy=True)
        for index, _ in enumerate(self.dataset.time_points):
            symmetrized[:, :, index] = nfold(
                before[:, :, index], mod=3, center=(63, 65)
            )

        self.dataset.symmetrize(mod=3, center=(63, 65), processes=2)
        after = np.array(self.dataset.diffraction_group["intensity"])

        self.assertTrue(np.allclose(symmetrized, after))

    def test_symmetrization_shape(self):
        """ Test that dataset symmetrization raises no errors """
        with self.subTest("No smoothing"):
            self.dataset.symmetrize(mod=2, center=(128, 128))
            self.assertSequenceEqual(self.patterns[0].shape, self.dataset.resolution)

        with self.subTest("with smoothing"):
            self.dataset.symmetrize(mod=2, center=(128, 128), kernel_size=5)
            self.assertSequenceEqual(self.patterns[0].shape, self.dataset.resolution)

    def test_data(self):
        """ Test that data stored in DiffractionDataset is correct """
        for time, pattern in zip(list(self.dataset.time_points), self.patterns):
            self.assertTrue(np.allclose(self.dataset.diff_data(time), pattern))

    def test_time_zero_shift(self):
        """ Test that data changed with time_zero_shift() """
        unshifted = np.array(self.dataset.time_points)
        self.dataset.shift_time_zero(100)
        shifted = np.array(self.dataset.time_points)

        self.assertFalse(np.allclose(unshifted, shifted))
        self.assertTrue(np.allclose(unshifted + 100, shifted))

    def test_diff_eq(self):
        """ test that DiffractionDataset.diff_eq() returns the correct array """
        with self.subTest("No data before time-zero"):
            self.dataset.shift_time_zero(10)
            self.assertTrue(
                np.allclose(self.dataset.diff_eq(), np.zeros(self.dataset.resolution))
            )

        with self.subTest("All data before time-zero"):
            self.dataset.shift_time_zero(-20)
            eq = np.mean(np.stack(self.patterns, axis=-1), axis=2)
            self.assertTrue(np.allclose(self.dataset.diff_eq(), eq))

    def test_time_series(self):
        """ Test that the DiffractionDataset.time_series method is working """

        r1, r2, c1, c2 = 100, 120, 45, 57
        stack = np.stack(self.patterns, axis=-1)
        ts = np.mean(stack[r1:r2, c1:c2], axis=(0, 1))

        with self.subTest("Non-relative time-series"):
            self.assertTrue(
                np.allclose(
                    self.dataset.time_series([r1, r2, c1, c2], relative=False), ts
                )
            )

        with self.subTest("Relative time-series"):
            self.assertTrue(
                np.allclose(
                    self.dataset.time_series([r1, r2, c1, c2], relative=True), ts
                )
            )

    def test_time_series_selection(self):
        """ Test that the DiffractionDataset.time_series_selection 
        method is working as expected """
        mask = np.random.choice([True, False], size=self.dataset.resolution)
        selection = ArbitrarySelection(mask)

        stack = np.stack(self.patterns, axis=-1)
        ts = np.mean(stack[selection], axis=(0, 1))

        with self.subTest("Non-relative time-series"):
            self.assertTrue(
                np.allclose(
                    self.dataset.time_series_selection(selection, relative=False), ts
                )
            )

        with self.subTest("Relative time-series"):
            self.assertTrue(
                np.allclose(
                    self.dataset.time_series_selection(selection, relative=True), ts
                )
            )

    def test_selection_rect(self):
        """ Comparison of DiffractionDataset.time_series vs 
        DiffractionDataset.time_series_selection with 
        DiffractionDataset.selection_rect """

        r1, r2, c1, c2 = 100, 120, 45, 57
        selection = RectSelection(self.dataset.resolution, r1, r2, c1, c2)

        with self.subTest("Non-relative"):
            ts = self.dataset.time_series([r1, r2+1, c1, c2+1], relative=False)
            tsbm = self.dataset.time_series_selection(selection, relative=False)
            self.assertTrue(np.allclose(ts, tsbm))

        with self.subTest("Relative"):
            ts = self.dataset.time_series([r1, r2+1, c1, c2+1], relative=True)
            tsbm = self.dataset.time_series_selection(selection, relative=True)
            self.assertTrue(np.allclose(ts, tsbm))

    def test_selection_disk(self):
        """ Test DiffractionDataset.time_series_selection with 
        DiffractionDataset.selection_disk """
        selection = DiskSelection(self.dataset.resolution, center=(120, 200), radius=10)

        with self.subTest("Non-relative"):
            # We modify the dataset so that within the selection, only zeroes are found
            # Note that HDF5 does not support fancy boolean indexing, so we must change the
            # content image-by-image.
            for index, _ in enumerate(self.dataset.time_points):
                arr = self.dataset.diffraction_group["intensity"][:, :, index]
                arr[selection] = 0
                self.dataset.diffraction_group["intensity"][:, :, index] = arr

            ts = self.dataset.time_series_selection(selection, relative=False)
            self.assertTrue(np.allclose(ts, np.zeros_like(ts)))

        with self.subTest("Relative"):
            for index, _ in enumerate(self.dataset.time_points):
                arr = self.dataset.diffraction_group["intensity"][:, :, index]
                arr[selection] = 1
                self.dataset.diffraction_group["intensity"][:, :, index] = arr

            ts = self.dataset.time_series_selection(selection, relative=True)
            self.assertTrue(np.allclose(ts, np.ones_like(ts)))

    def test_selection_ring(self):
        """ Test DiffractionDataset.time_series_selection with 
        DiffractionDataset.selection_ring """
        selection = RingSelection(
            self.dataset.resolution, center=(120, 200), inner_radius=10, outer_radius=20
        )

        with self.subTest("Non-relative"):
            # We modify the dataset so that within the selection, only zeroes are found
            # Note that HDF5 does not support fancy boolean indexing, so we must change the
            # content image-by-image.
            for index, _ in enumerate(self.dataset.time_points):
                arr = self.dataset.diffraction_group["intensity"][:, :, index]
                arr[selection] = 0
                self.dataset.diffraction_group["intensity"][:, :, index] = arr

            ts = self.dataset.time_series_selection(selection, relative=False)
            self.assertTrue(np.allclose(ts, np.zeros_like(ts)))

        with self.subTest("Relative"):
            for index, _ in enumerate(self.dataset.time_points):
                arr = self.dataset.diffraction_group["intensity"][:, :, index]
                arr[selection] = 1
                self.dataset.diffraction_group["intensity"][:, :, index] = arr

            ts = self.dataset.time_series_selection(selection, relative=True)
            self.assertTrue(np.allclose(ts, np.ones_like(ts)))


class TestPowderDiffractionDataset(unittest.TestCase):
    def setUp(self):
        self.patterns = list(repeat(random(size=(128, 128)), 5))
        self.metadata = {"fluence": 10, "energy": 90}
        diff_dataset = DiffractionDataset.from_collection(
            self.patterns,
            filename="test.hdf5",
            time_points=range(5),
            metadata=self.metadata,
            mode="w",
        )
        self.dataset = PowderDiffractionDataset.from_dataset(
            diff_dataset, center=(23, 45)
        )

    def test_baseline_attributes(self):
        """ Test that the attributes related to baseline have correct defaults and are
        set to the correct values after computation """
        self.assertIs(self.dataset.first_stage, "")
        self.assertIs(self.dataset.wavelet, "")
        self.assertEqual(self.dataset.level, 0)
        self.assertEqual(self.dataset.niter, 0)

        self.dataset.compute_baseline(
            first_stage="sym6", wavelet="qshift3", level=1, mode="periodic"
        )

        self.assertEqual(self.dataset.first_stage, "sym6")
        self.assertEqual(self.dataset.wavelet, "qshift3")
        self.assertEqual(self.dataset.level, 1)

    def test_powder_calq(self):
        """ Test scattering vector calibration """
        crystal = Crystal.from_database("vo2-m1")
        self.dataset.powder_calq(crystal, (10, 100), [(1, 0, 0), (2, 0, 0)])

        # Check that shapes match
        self.assertTupleEqual(
            self.dataset.powder_eq().shape, self.dataset.scattering_vector.shape
        )
        # Check that scattering_vector is strictly increasing
        self.assertTrue(np.all(np.greater(np.diff(self.dataset.scattering_vector), 0)))

    def test_baseline_limits(self):
        """ Test that the baseline is never less than 0, and the baseline-subtracted data is never negative. """

        self.dataset.compute_baseline(
            first_stage="sym6", wavelet="qshift3", level=1, mode="periodic"
        )

        # Test that the baseline is always positive
        baseline = self.dataset.powder_baseline(None)
        self.assertTrue(np.all(np.greater_equal(baseline, 0)))

        data_bgr = self.dataset.powder_data(None, bgr=True)
        self.assertTrue(np.all(np.greater_equal(data_bgr, 0)))

    def test_powder_data_retrieval(self):
        """ Test the size of the output from PowderDiffractionDataset.powder_data """
        full_shape = (len(self.dataset.time_points), self.dataset.px_radius.size)

        full_data = self.dataset.powder_data(timedelay=None)
        self.assertSequenceEqual(full_data.shape, full_shape)

        time_slice = self.dataset.powder_data(timedelay=self.dataset.time_points[0])
        self.assertSequenceEqual(time_slice.shape, self.dataset.px_radius.shape)

    def test_recomputing_angular_average(self):
        """ Test that recomputing the angular average multiple times will work. This also
        tests resizing all powder data multiple times. """
        self.dataset.compute_angular_averages(center=(34, 56))
        self.dataset.compute_baseline(first_stage="sym6", wavelet="qshift1")
        self.dataset.compute_angular_averages(center=(45, 45), normalized=False)
        self.dataset.compute_baseline(first_stage="sym5", wavelet="qshift2")
        self.dataset.compute_angular_averages(
            center=(34, 56), angular_bounds=(15.3, 187)
        )
        self.dataset.compute_baseline(first_stage="sym6", wavelet="qshift1")

    def test_powder_eq(self):
        """ Test PowderDiffractionDataset.powder_eq() """
        with self.subTest("bgr = False"):
            eq = self.dataset.powder_eq()
            self.assertSequenceEqual(eq.shape, self.dataset.px_radius.shape)

        with self.subTest("bgr = True"):
            self.dataset.compute_baseline(
                first_stage="sym6", wavelet="qshift3", mode="constant"
            )
            eq = self.dataset.powder_eq(bgr=True)
            self.assertSequenceEqual(eq.shape, self.dataset.px_radius.shape)

        with self.subTest("No data before time-zero"):
            self.dataset.shift_time_zero(1 + abs(min(self.dataset.time_points)))
            eq = self.dataset.powder_eq()
            self.assertSequenceEqual(eq.shape, self.dataset.px_radius.shape)
            self.assertTrue(np.allclose(eq, np.zeros_like(eq)))

    def tearDown(self):
        fname = self.dataset.filename
        self.dataset.close()
        del self.dataset
        with suppress(OSError):
            os.remove(fname)
