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
)
from warnings import catch_warnings, simplefilter
from scipy.ndimage import gaussian_filter

import h5py
import npstreams as ns
import numpy as np


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


def update_center(f):
    """ Recompute the dependent quantities following a transformation, i.e. equilibrium pattern. """

    @wraps(f)
    def newf(self, *args, **kwargs):
        r = f(self, *args, **kwargs)
        self._recenter()
        return r

    return newf


def update_equilibrium_pattern(f):
    """ Recompute the dependent quantities following a transformation, i.e. equilibrium pattern. """

    @wraps(f)
    def newf(self, *args, **kwargs):
        r = f(self, *args, **kwargs)
        self._recompute_equilibrium_pattern()
        return r

    return newf


@enum.unique
class InternalDatasets(enum.Enum):
    Intensity = "/intensity"
    Mask = "/mask"
    Equilibrium = "/equilibrium"
    TimeDelays = "/time_points"
    ScatteringVectors = "/scattvectors"


@enum.unique
class RequiredGroups(enum.Enum):
    Root = "/"
    Metadata = "/metadata"  # location of user-defined metadata only


@enum.unique
class RequiredMetadata(enum.Enum):
    Center = "center"
    TimeZeroShift = "time_zero_shift"


class LowLevelDataset(h5py.File):
    """
    Low-level interface between :mod:`iris` and ``HDF5``. This class implements
    all the basic operations on diffraction patterns and time-series.

    .. versionadded:: 6.0.0

    Parameters
    ----------
    filename: Path-like
        Path to the file.
    mode : IOMode, optional
        File IO mode.
    kwargs
        Keyword arguments are passed to the :class:`h5py.File` constructor.
    """

    def __init__(self, filename, mode=IOMode.ReadOnly, **kwargs):
        super().__init__(name=Path(filename), mode=IOMode(mode).value, **kwargs)

    @classmethod
    def from_collection(
        cls,
        patterns,
        filename,
        time_points,
        mask=None,
        metadata=None,
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
            Path to the assembled LowLevelDataset. Default behavior is to overwrite the file at ``filename``;
            to prevent this, pass the ``mode='x'`` argument.
        time_points : array_like, shape (N,)
            Time-points of the diffraction patterns, in picoseconds.
        mask : ndarray or None, optional
            Boolean array that evaluates to True on valid pixels. This information is useful in
            cases where a beamblock is used.
        metadata : dict, optional
            User-defined metadata. Keys must be strings. Values must be types representable by HDF5.
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

        if mask is None:
            mask = np.ones(first.shape, dtype=np.bool)

        callback(0)
        mode = kwargs.pop("mode", IOMode.Overwrite)
        with cls(filename, mode=mode, **kwargs) as file:

            for required_group in RequiredGroups:
                file.require_group(required_group.value)

            if metadata:
                for key, val in metadata.items():
                    file[RequiredGroups.Metadata.value].attrs[key] = val

            file.attrs[RequiredMetadata.Center.value] = (0, 0)  # placeholder
            file.attrs[RequiredMetadata.TimeZeroShift.value] = 0

            # Record time-points as a dataset; then, changes to it will be reflected
            # in other dimension scales
            times = file.create_dataset(
                name=InternalDatasets.TimeDelays.value, data=time_points, dtype=np.float
            )
            mask = file.create_dataset(
                name=InternalDatasets.Mask.value, data=mask, dtype=np.bool
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
                **ckwargs,
            )

            file.create_dataset(
                name=InternalDatasets.ScatteringVectors.value,
                shape=resolution + (3,),
                dtype=np.float,
            )

            # At each iteration, we flush the changes to file
            # If this is not done, data can be accumulated in memory (>5GB)
            # until this loop is done.
            for index, pattern in enumerate(patterns):
                dset.write_direct(pattern, dest_sel=np.s_[:, :, index])
                callback(round(100 * index / np.size(time_points)))

            file._recompute_equilibrium_pattern()
            file._recenter()

        callback(100)

        # Now that the file exists, we can switch to read/write mode
        kwargs["mode"] = IOMode.ReadWrite
        return cls(filename, **kwargs)

    @property
    def mode(self):
        return IOMode(super().mode)

    @property
    def filename(self):
        return Path(super().filename)

    @property
    def time_points(self):
        return (
            np.array(self.get_dataset(InternalDatasets.TimeDelays))
            - self.time_zero_shift
        )

    @property
    def center(self):
        """ Center of diffraction patterns in [row, col] format."""
        return self.attrs[RequiredMetadata.Center.value]

    @center.setter
    @write_access_needed
    def center(self, center):
        center = tuple(center)
        # TODO: check bounds?
        self.attrs[RequiredMetadata.Center.value] = center

    @property
    def time_zero_shift(self):
        return self.attrs[RequiredMetadata.TimeZeroShift.value]

    @time_zero_shift.setter
    @write_access_needed
    @update_equilibrium_pattern
    def time_zero_shift(self, val: float):
        self.attrs[RequiredMetadata.TimeZeroShift.value] = val

    @property
    def metadata(self):
        """ Dictionary of the dataset's user-defined metadata. Dictionary is sorted alphabetically by keys."""
        meta = dict(self.get_group(RequiredGroups.Metadata).attrs)
        return OrderedDict(sorted(meta.items(), key=lambda t: t[0]))

    def get_group(self, gp: RequiredGroups):
        return self[gp.value]

    def get_dataset(self, dataset: InternalDatasets):
        return self[InternalDatasets(dataset).value]

    def get_time_index(self, timedelay: float):
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
                f"Time-delay {timedelay}ps not available. Using closest-timedelay {self.time_points[time_index]}ps instead.",
                category=MissingTimePointWarning,
                stacklevel=2,
            )
        return time_index

    @property
    def equilibrium_pattern(self):
        return np.array(self.get_dataset(InternalDatasets.Equilibrium))

    @write_access_needed
    def _recompute_equilibrium_pattern(self):
        intensity_dset = self.get_dataset(InternalDatasets.Intensity)
        with catch_warnings():
            simplefilter("ignore", category=MissingTimePointWarning)
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
        self.center = (r, c)

    @property
    def mask(self):
        return np.array(self.get_dataset(InternalDatasets.Mask))

    @mask.setter
    @write_access_needed
    def mask(self, new):
        old_mask = self.mask
        if new.dtype != old_mask.dtype:
            raise TypeError(
                f"Diffraction pattern masks must be boolean, not {new.dtype}"
            )
        if new.shape != old_mask.shape:
            raise ValueError(
                f"Expected diffraction pattern mask with shape {old_mask.shape}, but got {new.shape}"
            )
        self.get_dataset(InternalDatasets.Mask).write_direct(new)

    def time_series(self, r1, r2, c1, c2, relative: bool = False, out=None):
        """
        Diffracted intensity over time inside bounds.

        Parameters
        ----------
        r1, r2 : ints
            Row bounds of the region in px.
        c1, c2 : int
            Column bounds of the region in px.
        relative : bool, optional
            If True, data is returned relative to the average of all diffraction patterns
            before photoexcitation.
        out : ndarray or None, optional
            1-D ndarray in which to store the results. The shape
            should be compatible with ``(len(time_points),)``

        Returns
        -------
        out : ndarray, ndim 2
        """
        dset = self.get_dataset(InternalDatasets.Intensity)
        r1, r2 = sorted([r1, r2])
        c1, c2 = sorted([c1, c2])

        if out is None:
            out = np.empty(shape=(r2 - r1, c2 - c1, dset.shape[2]), dtype=np.float)

        dset.read_direct(out, source_sel=np.s_[r1:r2, c1:c2, :])

        if relative:
            out /= self.equilibrium_pattern[r1:r2, c1:c2, None]
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
    @update_equilibrium_pattern
    @update_center
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
            self._diff_apply_parallel(func, callback, processes)
            # Important to return from this function
            # so that dependent quantities are updated.
            return

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


# Functions to be passed to pmap must not be local functions
def _apply_diff(timedelay, fname, func):
    with LowLevelDataset(
        fname, mode=IOMode.ReadOnly, libver="latest", swmr=True
    ) as dset:
        im = dset.diffraction_pattern(timedelay=timedelay)
    return func(im)


class MissingTimePointWarning(UserWarning):
    """ Class of warning when requesting data from a time-point that is missing. """

    pass
