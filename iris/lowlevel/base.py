# -*- coding: utf-8 -*-
"""
Low-level interface to HDF5 files
"""

import enum
from math import sqrt
from collections import OrderedDict
from functools import wraps, lru_cache  # TODO: cached_property in py38+
from pathlib import Path
from warnings import warn
from skued import (
    autocenter,
    nfold,
    ArbitrarySelection,
    Selection,
    azimuthal_average,
    powder_calq,
    baseline_dt,
    dt_max_level,
)
from scipy.ndimage import gaussian_filter

import h5py
import npstreams as ns
import numpy as np

from ..meta import HDF5ExperimentalParameter, MetaHDF5Dataset

# Whether or not single-writer multiple-reader (SWMR) mode is available
# See http://docs.h5py.org/en/latest/swmr.html for more information
SWMR_AVAILABLE = h5py.version.hdf5_version_tuple > (1, 10, 0)


@enum.unique
class IOMode(enum.Enum):
    ReadOnly = "r"
    ReadWrite = "r+"
    Overwrite = "w"

    @classmethod
    def _missing_(cls, value):
        if value in {"x", "a", "w-"}:
            return cls("r+")
        return super()._missing_(value)


def write_access_needed(f):
    """ Ensure that write access has been granted before using a method. """

    @wraps(f)
    def newf(self, *args, **kwargs):
        if self.mode != IOMode.ReadWrite:
            raise PermissionError(
                f"The dataset {self.filename} has not been opened with write access."
            )
        return f(self, *args, **kwargs)

    return newf


def update_dependents(f):
    """ Recompute the dependent quantities following a transformation, i.e. equilibrium pattern. """

    @wraps(f)
    def newf(self, *args, **kwargs):
        r = f(self, *args, **kwargs)
        self._recompute_equilibrium_pattern()
        self._recenter()
        return r

    return newf


@enum.unique
class InternalDatasets(enum.Enum):
    Intensity = "/processed/intensity"
    Mask = "/valid_mask"
    Equilibrium = "/processed/equilibrium"
    TimeDelays = "/time_points"
    Q = "/wavevector"


class LowLevelDataset(h5py.File, metaclass=MetaHDF5Dataset):
    """
    Low-level interface between :mod:`iris` and ``HDF5``. This class implements
    all the basic operations on diffraction patterns and time-series.

    .. versionadded:: 5.3.0

    """

    # Subclasses can add more experimental parameters like those below
    # The types must be representable by h5py
    center = HDF5ExperimentalParameter("center", tuple, default=(0, 0))
    acquisition_date = HDF5ExperimentalParameter("acquisition_date", str, default="")
    energy = HDF5ExperimentalParameter("energy", float, default=90)  # keV
    pump_wavelength = HDF5ExperimentalParameter("pump_wavelength", int, default=800)
    fluence = HDF5ExperimentalParameter("fluence", float, default=0)
    time_zero_shift = HDF5ExperimentalParameter("time_zero_shift", float, default=0)
    temperature = HDF5ExperimentalParameter("temperature", float, default=293)
    exposure = HDF5ExperimentalParameter("exposure", float, default=1)  # seconds
    scans = HDF5ExperimentalParameter("scans", tuple, default=(1,))
    camera_length = HDF5ExperimentalParameter("camera_length", float, default=0.23)
    pixel_width = HDF5ExperimentalParameter("pixel_width", float, default=14e-6)
    aligned = HDF5ExperimentalParameter("aligned", bool, default=False)
    normalized = HDF5ExperimentalParameter("normalized", bool, default=False)
    notes = HDF5ExperimentalParameter("notes", str, default="")

    def __init__(self, filename, mode=IOMode.ReadOnly, **kwargs):
        super().__init__(name=Path(filename), mode=IOMode(mode).value, **kwargs)

    @classmethod
    def from_collection(
        cls,
        patterns,
        filename,
        time_points,
        metadata,
        valid_mask=None,
        dtype=None,
        ckwargs=None,
        callback=None,
        **kwargs,
    ):
        """
        Create a LowLevelDataset from a collection of diffraction patterns and metadata.

        Parameters
        ----------
        patterns : iterable of ndarray or ndarray
            Diffraction patterns. These should be in the same order as ``time_points``. Note that
            the iterable can be a generator, in which case it will be consumed.
        filename : str or path-like
            Path to the assembled DiffractionDataset.
        time_points : array_like, shape (N,)
            Time-points of the diffraction patterns, in picoseconds.
        metadata : dict
            Valid keys are contained in ``DiffractionDataset.valid_metadata``.
        valid_mask : ndarray or None, optional
            Boolean array that evaluates to True on valid pixels. This information is useful in
            cases where a beamblock is used.
        ckwargs : dict, optional
            HDF5 compression keyword arguments. Refer to ``h5py``'s documentation for details.
            Default is to use the `lzf` compression pipeline.
        callback : callable or None, optional
            Callable that takes an int between 0 and 99. This can be used for progress update when
            ``patterns`` is a generator and involves large computations.
        kwargs
            Keywords are passed to ``h5py.File`` constructor.
            Default is file-mode 'x', which raises error if file already exists.
            Default libver is 'latest'.

        Returns
        -------
        dataset : DiffractionDataset
        """
        if "libver" not in kwargs:
            kwargs["libver"] = "latest"

        if callback is None:
            callback = lambda _: None

        time_points = np.array(time_points).reshape(-1)

        if ckwargs is None:
            ckwargs = {"compression": "lzf", "shuffle": True, "fletcher32": True}

        first, patterns = ns.peek(patterns)
        resolution = first.shape

        if valid_mask is None:
            valid_mask = np.ones(first.shape, dtype=np.bool)

        callback(0)
        mode = kwargs.pop("mode", "x")
        with cls(filename, mode=mode, **kwargs) as file:

            # Note that keys not associated with an ExperimentalParameter
            # descriptor will not be recorded in the file.
            metadata.pop("time_points", None)
            for key, val in metadata.items():
                if key not in cls.valid_metadata:
                    continue
                setattr(file, key, val)

            # Record time-points as a dataset; then, changes to it will be reflected
            # in other dimension scales
            times = file.create_dataset(
                name=InternalDatasets.TimeDelays.value, data=time_points, dtype=np.float
            )
            mask = file.create_dataset(
                name=InternalDatasets.Mask.value, data=valid_mask, dtype=np.bool
            )

            dset = file.create_dataset(
                name=InternalDatasets.Intensity.value,
                shape=resolution + (len(time_points),),
                dtype=first.dtype,
                chunks=True,
                **ckwargs,
            )

            file.create_dataset(
                name=InternalDatasets.Equilibrium.value,
                shape=resolution,
                dtype=np.float,  # Involves division; cannot be integer
                chunks=True,
                **ckwargs,
            )

            # Making use of the H5DS dimension scales
            # http://docs.h5py.org/en/latest/high/dims.html
            times.make_scale("time-delay")
            dset.dims[2].attach_scale(times)

            # At each iteration, we flush the changes to file
            # If this is not done, data can be accumulated in memory (>5GB)
            # until this loop is done.
            for index, pattern in enumerate(patterns):
                dset.write_direct(pattern, dest_sel=np.s_[:, :, index])
                file.flush()
                callback(round(100 * index / np.size(time_points)))

            file._recompute_equilibrium_pattern()
            file._recenter()

        callback(100)

        # Now that the file exists, we can switch to read/write mode
        kwargs["mode"] = "r+"
        return cls(filename, **kwargs)

    @property
    def mode(self):
        return IOMode(super().mode)

    @property
    def filename(self):
        return Path(super().filename)

    @property
    def time_points(self):
        return np.array(self.get_dataset(InternalDatasets.TimeDelays))

    @write_access_needed
    @update_dependents
    def shift_time_zero(self, shift):
        """
        Insert a shift in time points. Reset the shift by setting it to zero. Shifts are
        not consecutive, so that calling `shift_time_zero(20)` twice will not result
        in a shift of 40ps.

        Parameters
        ----------
        shift : float
            Shift [ps]. A positive value of `shift` will move all time-points forward in time,
            whereas a negative value of `shift` will move all time-points backwards in time.

        Raises
        ------
        PermissionError: if the dataset has not been opened with write access.
        """
        differential = shift - self.time_zero_shift
        self.time_zero_shift = shift
        time_points = self.time_points
        self.get_dataset(InternalDatasets.TimeDelays).write_direct(
            time_points + differential
        )

    @property
    def metadata(self):
        """ Dictionary of the dataset's metadata. Dictionary is sorted alphabetically by keys."""
        meta = {k: getattr(self, k) for k in self.valid_metadata}
        meta["filename"] = self.filename
        meta["time_points"] = tuple(self.time_points)
        meta.update(self.compression_params)

        # Ordered dictionary by keys is easiest to inspect
        return OrderedDict(sorted(meta.items(), key=lambda t: t[0]))

    @property
    def compression_params(self):
        """ Compression options in the form of a dictionary """
        dataset = self.get_dataset(InternalDatasets.Intensity)
        ckwargs = dict(
            compression=dataset.compression,
            fletcher32=dataset.fletcher32,
            shuffle=dataset.shuffle,
            chunks=True if dataset.chunks else None,
        )
        if dataset.compression_opts:  # could be None
            ckwargs.update(dataset.compression_opts)
        return ckwargs

    def get_dataset(self, dataset: InternalDatasets):
        return self[InternalDatasets(dataset).value]

    def get_time_index(self, timedelay):
        """
        Returns the index of the closest available time-point.

        Parameters
        ----------
        timdelay : float
            Time-delay [ps]

        Returns
        -------
        tp : index
            Index of the Time-point closest to `timedelay` [ps]
        """
        # time_index cannot be cast to int() if np.argwhere returns an empty array
        # catch the corresponding TypeError
        try:
            time_index = int(np.argwhere(self.time_points == float(timedelay)))
        except TypeError:
            time_index = np.argmin(np.abs(self.time_points - float(timedelay)))
            warn(
                f"Time-delay {timedelay}ps not available. Using closest-timedelay {self.time_points[time_index]}ps instead"
            )
        return time_index

    @property
    def equilibrium_pattern(self):
        return np.array(self.get_dataset(InternalDatasets.Equilibrium))

    @write_access_needed
    def _recompute_equilibrium_pattern(self):
        intensity_dset = self.get_dataset(InternalDatasets.Intensity)
        t0_index = self.get_time_index(0)

        # If there are no available data before time-zero, np.mean()
        # will return an array of NaNs; instead, return the first diffraction pattern
        # so that autocenter() doesn't error out
        if t0_index == 0:
            diff_eq = self.get_dataset(InternalDatasets.Intensity)[:, :, 0]
        else:
            diff_eq = ns.average(
                (intensity_dset[:, :, i] for i in range(t0_index)), axis=2
            )

        # Division might introduce infs and nans
        diff_eq[:] = np.nan_to_num(diff_eq, copy=False)
        np.minimum(diff_eq, 2 ** 16 - 1, out=diff_eq)

        self.get_dataset(InternalDatasets.Equilibrium).write_direct(
            source=diff_eq, source_sel=np.s_[:, :], dest_sel=np.s_[:, :]
        )

    @write_access_needed
    def _recenter(self):
        """ Recalculate the center of the diffraction pattern """
        image = self.equilibrium_pattern
        r, c = autocenter(im=image, mask=self.mask)

        # Note that for backwards-compatibility, the center
        # coordinates need to be stored as (col, row)
        self.center = (c, r)

    @property
    def mask(self):
        return np.array(self.get_dataset(InternalDatasets.Mask))

    def time_series(self, rect: (int, int, int, int), relative: bool = False, out=None):
        """
        Integrated intensity over time inside bounds.

        Parameters
        ----------
        rect : 4-tuple of ints
            Bounds of the region in px. Bounds are specified as [row1, row2, col1, col2]
        relative : bool, optional
            If True, data is returned relative to the average of all diffraction patterns
            before photoexcitation.
        out : ndarray or None, optional
            1-D ndarray in which to store the results. The shape
            should be compatible with ``(len(time_points),)``

        Returns
        -------
        out : ndarray, ndim 1
        """
        dset = self.get_dataset(InternalDatasets.Intensity)

        r1, r2, c1, c2 = rect
        if out is None:
            out = np.empty(shape=(dset.shape[2]), dtype=np.float)

        np.mean(np.array(dset[r1:r2, c1:c2, :]), axis=(0, 1), out=out)

        if relative:
            out -= np.mean(self.equilibrium_pattern[r1:r2, c1:c2])
        return out

    def time_series_selection(self, selection, relative=False, out=None):
        """
        Integrated intensity over time according to some arbitrary selection. This
        is a generalization of the ``DiffractionDataset.time_series`` method, which
        is much faster, but limited to rectangular selections.

        Parameters
        ----------
        selection : skued.Selection or ndarray, dtype bool, shape (N,M)
            A selection mask that dictates the regions to integrate in each scattering patterns.
            In the case `selection` is an array, an ArbirarySelection will be used. Performance
            may be degraded. Selection mask evaluating to ``True`` in the regions to integrate.
            The selection must be the same shape as one scattering pattern (i.e. two-dimensional).
        relative : bool, optional
            If True, data is returned relative to the average of all diffraction patterns
            before photoexcitation.
        out : ndarray or None, optional
            1-D ndarray in which to store the results. The shape
            should be compatible with ``(len(time_points),)``

        Returns
        -------
        out : ndarray, ndim 1

        Raises
        ------
        ValueError : if the shape of `mask` does not match the scattering patterns.

        See also
        --------
        time_series : integrated intensity in a rectangle.
        """
        if not isinstance(selection, Selection):
            selection = ArbitrarySelection(selection)

        resolution = self.get_dataset(InternalDatasets.Intensity).shape[0:2]
        if selection.shape != resolution:
            raise ValueError(
                f"selection mask shape {selection.shape} does not match scattering pattern shape {resolution}"
            )

        if out is None:
            out = np.zeros(shape=(len(self.time_points),), dtype=np.float)

        # For performance reasons, we want to know what is the largest bounding box that
        # fits this selection. Otherwise, all data must be loaded from disk, all the time.
        r1, r2, c1, c2 = selection.bounding_box
        reduced_selection = np.asarray(selection)[r1:r2, c1:c2, None]

        # There is no way to select data from HDF5 using arbitrary boolean mask
        # Therefore, we must iterate through all time-points.
        dataset = self.get_dataset(InternalDatasets.Intensity)
        placeholder = np.empty(shape=(r2 - r1, c2 - c1, 1), dtype=dataset.dtype)
        for index, _ in enumerate(self.time_points):
            dataset.read_direct(placeholder, source_sel=np.s_[r1:r2, c1:c2, index])

            out[index] = np.mean(placeholder[reduced_selection])

        if relative:
            out -= np.mean(self.equilibrium_pattern[selection])

        return out

    def diffraction_pattern(self, timedelay, relative=False, out=None):
        """
        Returns diffraction pattern at a specific time-delay.

        Parameters
        ----------
        timdelay : float
            Timedelay [ps].
        relative : bool, optional
            If True, data is returned relative to the average of all diffraction patterns
            before photoexcitation.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid making intermediate copies.

        Returns
        -------
        arr : ndarray
            Time-delay data. If ``out`` is provided, ``arr`` is a view
            into ``out``.

        Raises
        ------
        ValueError: If timedelay does not exist.
        """
        dataset = self.get_dataset(InternalDatasets.Intensity)
        if out is None:
            out = np.empty(tuple(dataset.shape[0:2]), dtype=dataset.dtype)

        out = out[:, :, None]
        time_index = self.get_time_index(timedelay)
        dataset.read_direct(out, source_sel=np.s_[:, :, time_index])
        out = np.squeeze(out)

        if relative:
            out -= self.equilibrium_pattern

        return out

    @write_access_needed
    @update_dependents
    def diff_apply(self, func, callback=None, processes=1):
        """
        Apply a function to each diffraction pattern possibly in parallel. The diffraction patterns
        will be modified in-place.

        .. warning::
            This is an irreversible in-place operation.

        Parameters
        ----------
        func : callable
            Function that takes in an array (diffraction pattern) and returns an
            array of the exact same shape, with the same data-type.
        callback : callable or None, optional
            Callable that takes an int between 0 and 99. This can be used for progress update.
        processes : int or None, optional
            Number of parallel processes to use. If ``None``, all available processes will be used.
            In case Single Writer Multiple Reader mode is not available, ``processes`` is ignored.

        Raises
        ------
        TypeError : if `func` is not a proper callable
        PermissionError: if the dataset has not been opened with write access.
        """
        if not callable(func):
            raise TypeError(f"Expected a callable argument, but received {type(func)}")

        if callback is None:
            callback = lambda _: None

        # We implement parallel diff apply in a separate method
        # because single-threaded diff apply can be written with a
        # placeholder array
        if SWMR_AVAILABLE and (processes != 1):
            return self._diff_apply_parallel(func, callback, processes)

        ntimes = self.get_dataset(InternalDatasets.TimeDelays).shape[0]
        dset = self.get_dataset(InternalDatasets.Intensity)

        # Create a placeholder numpy array where to load and store the results
        placeholder = np.empty(
            shape=tuple(dset.shape[0:2]) + (1,), dtype=dset.dtype, order="C"
        )

        for index, time_point in enumerate(self.time_points):
            dset.read_direct(placeholder, source_sel=np.s_[:, :, index])
            dset.write_direct(
                func(np.squeeze(placeholder)), dest_sel=np.s_[:, :, index]
            )
            callback(int(100 * index / ntimes))

    @write_access_needed
    @update_dependents
    def _diff_apply_parallel(self, func, callback, processes):
        """
        Apply a function to each diffraction pattern in parallel. The diffraction patterns
        will be modified in-place. This method is not supposed to be called directly.

        Raises
        ------
        PermissionError: if the dataset has not been opened with write access.
        """
        ntimes = len(self.time_points)
        dset = self.get_dataset(InternalDatasets.Intensity)

        transformed = ns.pmap(
            _apply_diff,
            self.time_points,
            processes=processes,
            ntotal=ntimes,
            kwargs={"fname": self.filename, "func": func},
        )

        # We need to switch SWMR mode ON
        # Note that it cannot be turned OFF
        self.swmr_mode = True

        for index, im in enumerate(transformed):
            dset.write_direct(im, dest_sel=np.s_[:, :, index])
            dset.flush()
            callback(int(100 * index / ntimes))

    @write_access_needed
    def mask_apply(self, func):
        """
        Modify the diffraction pattern mask ``m`` to ``func(m)``

        Parameters
        ----------
        func : callable
            Function that takes in the diffraction pattern mask, which evaluates to
            ``True`` on valid pixels, and returns an array of the exact same shape,
            with the same data-type.

        Raises
        ------
        TypeError : if `func` is not a proper callable.
        TypeError : if the result of ``func(m)`` is not boolean.
        ValueError: if the result of ``func(m)`` does not have the right shape.
        PermissionError: if the dataset has not been opened with write access.
        """
        if not callable(func):
            raise TypeError(f"Expected a callable argument, but received {type(func)}")

        old_mask = self.mask

        r = func(old_mask)
        if r.dtype != np.bool:
            raise TypeError(f"Diffraction pattern masks must be boolean, not {r.dtype}")
        if r.shape != old_mask.shape:
            raise ValueError(
                f"Expected diffraction pattern mask with shape {old_mask.shape}, but got {r.shape}"
            )
        dset = self.get_dataset(InternalDatasets.Mask)
        dset.write_direct(r)


# Functions to be passed to pmap must not be local functions
def _apply_diff(timedelay, fname, func):
    with LowLevelDataset(
        fname, mode=IOMode.ReadOnly, libver="latest", swmr=True
    ) as dset:
        im = dset.diffraction_pattern(timedelay=timedelay)
    return func(im)


def _symmetrize(im, mod, center, mask, kernel_size):
    im = nfold(im, mod=mod, center=center, mask=mask)
    if kernel_size is None:
        return im
    return gaussian_filter(im, order=0, sigma=kernel_size, mode="nearest")