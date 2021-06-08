# -*- coding: utf-8 -*-
"""
Diffraction dataset types
"""
from sys import platform
from collections import OrderedDict
from collections.abc import Callable
from functools import lru_cache, partial, wraps
from math import sqrt
from warnings import warn

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

import npstreams as ns
from skued import (
    __version__,
    nfold,
    autocenter,
    ArbitrarySelection,
    Selection,
)

from .meta import HDF5ExperimentalParameter, MetaHDF5Dataset

# Whether or not single-writer multiple-reader (SWMR) mode is available
# See http://docs.h5py.org/en/latest/swmr.html for more information
SWMR_AVAILABLE = h5py.version.hdf5_version_tuple > (1, 10, 0)


class MigrationWarning(UserWarning):
    """Warning class for warnings involving the migration of datasets to a newer version."""

    pass


class MigrationError(Exception):
    """Thrown if a particular dataset requires migration."""

    pass


def write_access_needed(f):
    """Ensure that write access has been granted before using a method."""

    @wraps(f)
    def newf(self, *args, **kwargs):
        if self.mode != "r+":
            raise PermissionError(
                f"The dataset {self.filename} has not been opened with write access."
            )
        return f(self, *args, **kwargs)

    return newf


def update_center(f):
    """Recompute the dependent quantities following a transformation, i.e. equilibrium pattern."""

    @wraps(f)
    def newf(self, *args, **kwargs):
        r = f(self, *args, **kwargs)
        self._autocenter()
        return r

    return newf


def update_equilibrium_pattern(f):
    """Recompute the dependent quantities following a transformation, i.e. equilibrium pattern."""

    @wraps(f)
    def newf(self, *args, **kwargs):
        r = f(self, *args, **kwargs)
        self._recompute_diff_eq()  # It is assumed that diff_eq caches the result
        return r

    return newf


class DiffractionDataset(h5py.File, metaclass=MetaHDF5Dataset):
    """
    Abstraction of an HDF5 file to represent diffraction datasets.
    """

    _diffraction_group_name = "/processed"
    _exp_params_group_name = "/"

    # Subclasses can add more experimental parameters like those below
    # The types must be representable by h5py
    center = HDF5ExperimentalParameter("center", tuple, default=(0, 0))
    acquisition_date = HDF5ExperimentalParameter(
        "acquisition_date", str, default=""
    )  # Acquisition date, no specific format
    energy = HDF5ExperimentalParameter("energy", float, default=90)  # keV
    pump_wavelength = HDF5ExperimentalParameter(
        "pump_wavelength", int, default=800
    )  # nanometers
    fluence = HDF5ExperimentalParameter(
        "fluence", float, default=0
    )  # milliJoules / centimeters ^ 2
    time_zero_shift = HDF5ExperimentalParameter(
        "time_zero_shift", float, default=0
    )  # picoseconds
    temperature = HDF5ExperimentalParameter(
        "temperature", float, default=293
    )  # kelvins
    exposure = HDF5ExperimentalParameter("exposure", float, default=1)  # seconds
    scans = HDF5ExperimentalParameter("scans", tuple, default=(1,))
    camera_length = HDF5ExperimentalParameter(
        "camera_length", float, default=0.23
    )  # meters
    pixel_width = HDF5ExperimentalParameter(
        "pixel_width", float, default=14e-6
    )  # meters
    aligned = HDF5ExperimentalParameter("aligned", bool, default=False)
    normalized = HDF5ExperimentalParameter("normalized", bool, default=False)
    notes = HDF5ExperimentalParameter("notes", str, default="")

    def __init__(self, *args, **kwargs):
        # Secret option to skip the checks below
        # This is only useful when building a dataset
        # Don't use it
        skip_checks = kwargs.pop("skip_checks", False)

        super().__init__(*args, **kwargs)

        if not skip_checks:
            self._migration_checks()

    def _migration_checks(self):
        """
        Migration checks should be performed here. As iris has evolved beyond v5.0.0,
        new requirements have emerged. If the file is opened with writing
        permissions, we can migrate silently.
        """
        if self.center == (0, 0):  # default
            if self.mode == "r+":
                self._autocenter()
            else:
                warn(
                    "".join(
                        [
                            f"The center of diffraction for the dataset {self.filename} is missing.",
                            "Open it with writing permissions so it can be calculated.",
                            "This warning will become an error in future versions of iris.",
                        ]
                    ),
                    category=MigrationWarning,
                    stacklevel=2,
                )
        if "equilibrium" not in self.diffraction_group:
            if self.mode == "r+":
                self._recompute_diff_eq()
            else:
                warn(
                    "".join(
                        [
                            f"The equilibrium diffraction pattern for the dataset {self.filename} is missing.",
                            "Open it with writing permissions so it can be calculated.",
                            "This warning will become an error in future versions of iris.",
                        ]
                    ),
                    category=MigrationWarning,
                    stacklevel=2,
                )

    def __repr__(self):
        rep = f"< {type(self).__name__} object with following metadata: "
        for key, val in self.metadata.items():
            rep += f"\n    {key}: {val}"

        return rep + " >"

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
        Create a DiffractionDataset from a collection of diffraction patterns and metadata.

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
        dtype : dtype or None, optional
            Patterns will be cast to ``dtype``. If None (default), ``dtype`` will be set to the same
            data-type as the first pattern in ``patterns``.
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
        if "mode" not in kwargs:
            kwargs["mode"] = "x"

        if "libver" not in kwargs:
            kwargs["libver"] = "latest"

        # H5py will raise an exception if arrays are not contiguous
        # patterns = map(np.ascontiguousarray, iter(patterns))

        if callback is None:
            callback = lambda _: None

        time_points = np.array(time_points).reshape(-1)

        if ckwargs is None:
            ckwargs = {"compression": "lzf", "shuffle": True, "fletcher32": True}
        ckwargs[
            "chunks"
        ] = True  # For some reason, if no chunking, writing to disk is SLOW

        first, patterns = ns.peek(patterns)
        if dtype is None:
            dtype = first.dtype
        resolution = first.shape

        if valid_mask is None:
            valid_mask = np.ones(first.shape, dtype=bool)

        callback(0)
        with cls(filename, skip_checks=True, **kwargs) as file:

            # Note that keys not associated with an ExperimentalParameter
            # descriptor will not be recorded in the file.
            metadata.pop("time_points", None)
            for key, val in metadata.items():
                if key not in cls.valid_metadata:
                    continue
                setattr(file, key, val)

            # Record time-points as a dataset; then, changes to it will be reflected
            # in other dimension scales
            gp = file.experimental_parameters_group
            times = gp.create_dataset("time_points", data=time_points, dtype=float)
            mask = gp.create_dataset("valid_mask", data=valid_mask, dtype=bool)

            pgp = file.diffraction_group
            dset = pgp.create_dataset(
                name="intensity",
                shape=resolution + (len(time_points),),
                dtype=dtype,
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

            file._autocenter()
            file._recompute_diff_eq()

        callback(100)

        # Now that the file exists, we can switch to read/write mode
        kwargs["mode"] = "r+"
        return cls(filename, **kwargs)

    @classmethod
    def from_raw(
        cls,
        raw,
        filename,
        exclude_scans=None,
        valid_mask=None,
        processes=1,
        callback=None,
        align=True,
        normalize=True,
        ckwargs=None,
        dtype=None,
        **kwargs,
    ):
        """
        Create a DiffractionDataset from a subclass of AbstractRawDataset.

        Parameters
        ----------
        raw : AbstractRawDataset instance
            Raw dataset instance.
        filename : str or path-like
            Path to the assembled DiffractionDataset.
        exclude_scans : iterable of ints or None, optional
            Scans to exclude from the processing. Default is to include all scans.
        valid_mask : ndarray or None, optional
            Boolean array that evaluates to True on valid pixels. This information is useful in
            cases where a beamblock is used.
        processes : int or None, optional
            Number of Processes to spawn for processing. Default is number of available
            CPU cores.
        callback : callable or None, optional
            Callable that takes an int between 0 and 99. This can be used for progress update.
        align : bool, optional
            If True (default), raw images will be aligned on a per-scan basis.
        normalize : bool, optional
            If True, images within a scan are normalized to the same integrated diffracted intensity.
        ckwargs : dict or None, optional
            HDF5 compression keyword arguments. Refer to ``h5py``'s documentation for details.
        dtype : dtype or None, optional
            Patterns will be cast to ``dtype``. If None (default), ``dtype`` will be set to the same
            data-type as the first pattern in ``patterns``.
        kwargs
            Keywords are passed to ``h5py.File`` constructor.
            Default is file-mode 'x', which raises error if file already exists.

        Returns
        -------
        dataset : DiffractionDataset

        See Also
        --------
        open_raw : open raw datasets by guessing the appropriate format based on available plug-ins.

        Raises
        ------
        IOError
            If the filename is already associated with a file.
        """
        if callback is None:
            callback = lambda _: None

        if exclude_scans is None:
            exclude_scans = set([])

        if valid_mask is None:
            valid_mask = np.ones(shape=raw.resolution, dtype=bool)

        metadata = raw.metadata.copy()
        metadata["scans"] = tuple(set(raw.scans) - set(exclude_scans))
        metadata["aligned"] = align
        metadata["normalized"] = normalize

        # Assemble the metadata
        kwargs.update(
            {
                "ckwargs": ckwargs,
                "valid_mask": valid_mask,
                "metadata": metadata,
                "time_points": raw.time_points,
                "dtype": dtype,
                "callback": callback,
                "filename": filename,
            }
        )

        reduced = raw.reduced(
            exclude_scans=exclude_scans,
            align=align,
            normalize=normalize,
            mask=np.logical_not(valid_mask),
            processes=processes,
            dtype=dtype,
        )

        return cls.from_collection(patterns=reduced, **kwargs)

    @write_access_needed
    @update_center
    @update_equilibrium_pattern
    def diff_apply(self, func, callback=None, processes=1):
        """
        Apply a function to each diffraction pattern possibly in parallel. The diffraction patterns
        will be modified in-place.

        .. warning::
            This is an irreversible in-place operation.

        .. versionadded:: 5.0.3

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

            .. versionadded:: 5.0.6

        Raises
        ------
        TypeError
            if `func` is not a proper callable
        PermissionError
            if the dataset has not been opened with write access.
        """
        if not callable(func):
            raise TypeError(f"Expected a callable argument, but received {type(func)}")

        if callback is None:
            callback = lambda _: None

        ntimes = len(self.time_points)
        dset = self.diffraction_group["intensity"]

        # We implement parallel diff apply in a separate method
        # because single-threaded diff apply can be written with a
        # placeholder array
        if SWMR_AVAILABLE and (processes != 1):
            transformed = ns.pmap(
                _apply_diff,
                self.time_points,
                processes=processes,
                ntotal=ntimes,
                kwargs=dict(fname=self.filename, func=func),
            )

            # We need to switch SWMR mode ON
            # Note that it cannot be turned OFF
            self.swmr_mode = True

            for index, im in enumerate(transformed):
                dset.write_direct(im, dest_sel=np.s_[:, :, index])
                dset.flush()
                callback(int(100 * index / ntimes))
        else:
            # Create a placeholder numpy array where to load and store the results
            placeholder = np.empty(shape=self.resolution, dtype=dset.dtype, order="C")

            for index, _ in enumerate(self.time_points):
                # NOTE: Using dset.read_direct was causing problems because
                #       the destination had shape (N,N), but read_direct wanted a
                #       destination of shape (N,N,1). This is a new behavior since h5py 3.*
                placeholder[:] = dset[:, :, index]
                placeholder[:] = func(placeholder)
                dset.write_direct(placeholder, dest_sel=np.s_[:, :, index])
                callback(int(100 * index / ntimes))

    @write_access_needed
    @update_center
    @update_equilibrium_pattern
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
        TypeError
            if `func` is not a proper callable, or if the result of ``func(m)`` is not boolean.
        ValueError
            if the result of ``func(m)`` does not have the right shape.
        PermissionError
            if the dataset has not been opened with write access.
        """
        if not callable(func):
            raise TypeError(f"Expected a callable argument, but received {type(func)}")

        old_mask = func(self.valid_mask)

        r = func(old_mask)
        if r.dtype != bool:
            raise TypeError(f"Diffraction pattern masks must be boolean, not {r.dtype}")
        if r.shape != old_mask.shape:
            raise ValueError(
                f"Expected diffraction pattern mask with shape {old_mask.shape}, but got {r.shape}"
            )
        self.experimental_parameters_group["valid_mask"][:] = func(self.valid_mask)

    @write_access_needed
    @update_equilibrium_pattern
    def symmetrize(
        self, mod, center=None, kernel_size=None, callback=None, processes=1
    ):
        """
        Symmetrize diffraction images based on n-fold rotational symmetry.

        .. warning::
            This is an irreversible in-place operation.

        Parameters
        ----------
        mod : int
            Fold symmetry number.
        center : array-like, shape (2,) or None
            Coordinates of the center (in pixels). If None (default), the center will be automatically
            determined.
        kernel_size : float or None, optional
            If not None, every diffraction pattern will be smoothed with a gaussian kernel.
            `kernel_size` is the standard deviation of the gaussian kernel in units of pixels.
        callback : callable or None, optional
            Callable that takes an int between 0 and 99. This can be used for progress update.
        processes : int or None, optional
            Number of parallel processes to use. If ``None``, all available processes will be used.
            In case Single Writer Multiple Reader mode is not available, ``processes`` is ignored.

            .. versionadded:: 5.0.6

        Raises
        ------
        ValueError
            if ``mod`` is not a divisor of 360.
        PermissionError
            if the dataset has not been opened with write access.

        See Also
        --------
        diff_apply : apply an operation to each diffraction pattern one-by-one
        """
        if center is None:
            center = self.center
        # Due to possibility of parallel operation,
        # we can't use lambdas or local functions
        # Therefore, we define _symmetrize below and use it here
        apply = partial(
            _symmetrize,
            mod=mod,
            center=center,
            mask=self.valid_mask,
            kernel_size=kernel_size,
        )
        self.diff_apply(apply, callback=callback, processes=processes)

    @property
    def metadata(self):
        """Dictionary of the dataset's metadata. Dictionary is sorted alphabetically by keys."""
        meta = {k: getattr(self, k) for k in self.valid_metadata}
        meta["filename"] = self.filename
        meta["time_points"] = tuple(self.time_points)
        meta.update(self.compression_params)

        # Ordered dictionary by keys is easiest to inspect
        return OrderedDict(sorted(meta.items(), key=lambda t: t[0]))

    @property
    def valid_mask(self):
        """Array that evaluates to True on valid pixels (i.e. not on beam-block, not hot pixels, etc.)"""
        return np.array(self.experimental_parameters_group["valid_mask"])

    @property
    def invalid_mask(self):
        """Array that evaluates to True on invalid pixels (i.e. on beam-block, hot pixels, etc.)"""
        return np.logical_not(self.valid_mask)

    @property
    def time_points(self):
        # Time-points are not treated as metadata because
        return np.array(self.experimental_parameters_group["time_points"])

    @property
    def resolution(self):
        """Resolution of diffraction patterns (px, px)"""
        intensity_shape = self.diffraction_group["intensity"].shape
        return tuple(intensity_shape[0:2])

    @write_access_needed
    @update_equilibrium_pattern
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
        PermissionError
            if the dataset has not been opened with write access.
        """
        differential = shift - self.time_zero_shift
        self.time_zero_shift = shift
        self.experimental_parameters_group["time_points"][:] = (
            self.time_points + differential
        )

    def _get_time_index(self, timedelay):
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

    def diff_eq(self):
        """
        Returns the averaged diffraction pattern for all times before photoexcitation.
        In case no data is available before photoexcitation, an array of zeros is returned.

        If the dataset was opened with writing access, the result of this function is
        cached to file. It will be recomputed as needed.

        Time-zero can be adjusted using the ``shift_time_zero`` method.

        Returns
        -------
        I : ndarray, ndim 2
            Diffracted intensity [counts]
        """
        try:
            # The reason this diffraction group might not exist is because
            # this dataset was not part of the initial iris v5 format.
            # it was added in 5.3.0. Therefore, we need to be prepared
            # in case it does not exist in older DiffractionDatasets
            return np.array(self.diffraction_group["equilibrium"])

        # Soon this will become a real error
        except KeyError:
            # Only with write access can diff_eq be cached.
            if self.mode == "r+":
                self._recompute_diff_eq()
                return np.array(self.diffraction_group["equilibrium"])

            # Otherwise, it needs to be calculated from scratch
            intensity = self.diffraction_group["intensity"]
            t0_index = np.argmin(np.abs(self.time_points))

            # If there are no available data before time-zero, np.mean()
            # will return an array of NaNs; instead, return zeros.
            if t0_index == 0:
                return np.zeros(shape=self.resolution, dtype=intensity.dtype)

            return ns.average((intensity[:, :, i] for i in range(t0_index)), axis=2)

    @write_access_needed
    def _recompute_diff_eq(self):
        """Calculate and store the equilibrium diffraction pattern."""

        intensity = self.diffraction_group["intensity"]
        t0_index = np.argmin(np.abs(self.time_points))

        # If there are no available data before time-zero, np.mean()
        # will return an array of NaNs; instead, return zeros.
        if t0_index == 0:
            diff_eq = np.zeros(shape=self.resolution, dtype=float)
        else:
            diff_eq = ns.average((intensity[:, :, i] for i in range(t0_index)), axis=2)

        eq_dset = self.diffraction_group.require_dataset(
            name="equilibrium", shape=diff_eq.shape, dtype=float
        )
        eq_dset[:] = diff_eq

    def diff_data(self, timedelay, relative=False, out=None):
        """
        Returns diffraction data at a specific time-delay.

        Parameters
        ----------
        timdelay : float or None
            Timedelay [ps]. If None, the entire block is returned.
        relative : bool, optional
            If True, data is returned relative to the average of all diffraction patterns
            before photoexcitation.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.

        Returns
        -------
        arr : ndarray
            Time-delay data. If ``out`` is provided, ``arr`` is a view
            into ``out``.

        Raises
        ------
        ValueError
            If timedelay does not exist.
        """
        dataset = self.diffraction_group["intensity"]

        if timedelay is None:
            if out is None:
                out = np.empty_like(dataset)
            dataset.read_direct(out)

        else:
            time_index = self._get_time_index(timedelay)
            if out is None:
                out = np.empty(self.resolution, dtype=dataset.dtype)
            # NOTE: Using dataset.read_direct was causing problems because
            #       the destination had shape (N,N), but read_direct wanted a
            #       destination of shape (N,N,1). This is a new behavior since h5py 3.*
            out[:] = dataset[:, :, time_index]

        if relative:
            out -= self.diff_eq()
            out /= self.diff_eq()

            # Division might introduce infs and nans
            out[:] = np.nan_to_num(out, copy=False)
            np.minimum(out, 2 ** 16 - 1, out=out)

        return out

    def time_series(self, rect, relative=False, out=None):
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

        See also
        --------
        time_series_selection : intensity integration using arbitrary selections.
        """
        x1, x2, y1, y2 = rect
        data = self.diffraction_group["intensity"][x1:x2, y1:y2, :]
        if relative:
            data -= self.diff_eq()[x1:x2, y1:y2, None]
        return np.mean(data, axis=(0, 1), out=out)

    def time_series_selection(self, selection, relative=False, out=None):
        """
        Integrated intensity over time according to some arbitrary selection. This
        is a generalization of the ``DiffractionDataset.time_series`` method, which
        is much faster, but limited to rectangular selections.

        .. versionadded:: 5.2.1

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
        ValueError
            if the shape of ``mask`` does not match the scattering patterns.

        See also
        --------
        time_series : integrated intensity in a rectangle.
        """
        if not isinstance(selection, Selection):
            selection = ArbitrarySelection(selection)

        if selection.shape != self.resolution:
            raise ValueError(
                f"selection mask shape {selection.shape} does not match scattering pattern shape {self.resolution}"
            )

        if out is None:
            out = np.zeros(shape=(len(self.time_points),), dtype=float)

        # For performance reasons, we want to know what is the largest bounding box that
        # fits this selection. Otherwise, all data must be loaded from disk, all the time.
        r1, r2, c1, c2 = selection.bounding_box
        reduced_selection = np.asarray(selection)[r1:r2, c1:c2]

        # There is no way to select data from HDF5 using arbitrary boolean mask
        # Therefore, we must iterate through all time-points.
        dataset = self.diffraction_group["intensity"]
        placeholder = np.empty(shape=(r2 - r1, c2 - c1), dtype=dataset.dtype)
        for index, _ in enumerate(self.time_points):
            # NOTE: Using dataset.read_direct was causing problems because
            #       the destination had shape (N,N), but read_direct wanted a
            #       destination of shape (N,N,1). This is a new behavior since h5py 3.*
            placeholder[:] = dataset[r1:r2, c1:c2, index]

            out[index] = np.mean(placeholder[reduced_selection])

        if relative:
            out -= np.mean(self.diff_eq()[selection])

        return out

    @write_access_needed
    def _autocenter(self):
        """
        Determine the diffraction pattern center automatically.

        .. versionadded:: 5.3.0

        Raises
        ------
        PermissionError
            if the dataset has not been opened with write access.
        ValueError
            If all pixels that are deemed valid have zero intensity.
        """
        intensity = self.diffraction_group["intensity"]
        image = ns.average(intensity[:, :, i] for i in range(intensity.shape[2]))
        if np.allclose(image * self.valid_mask, 0):
            raise ValueError(
                "There is not enough data to determine a center; all valid pixels have zero intensity."
            )
        r, c = autocenter(im=image, mask=self.valid_mask)

        # Note that for backwards-compatibility, the center
        # coordinates need to be stored as (col, row)
        self.center = (c, r)

    @property
    def experimental_parameters_group(self):
        return self.require_group(name=self._exp_params_group_name)

    @property
    def diffraction_group(self):
        return self.require_group(name=self._diffraction_group_name)

    @property
    def compression_params(self):
        """Compression options in the form of a dictionary"""
        dataset = self.diffraction_group["intensity"]
        ckwargs = dict()
        ckwargs["compression"] = dataset.compression
        ckwargs["fletcher32"] = dataset.fletcher32
        ckwargs["shuffle"] = dataset.shuffle
        ckwargs["chunks"] = True if dataset.chunks else None
        if dataset.compression_opts:  # could be None
            ckwargs.update(dataset.compression_opts)
        return ckwargs


# Functions to be passed to pmap must not be local functions
def _apply_diff(timedelay, fname, func):
    with DiffractionDataset(fname, mode="r", libver="latest", swmr=True) as dset:
        im = dset.diff_data(timedelay)
    return func(im)


def _symmetrize(im, mod, center, mask, kernel_size):
    im = nfold(im, mod=mod, center=center, mask=mask)
    if kernel_size is None:
        return im
    return gaussian_filter(im, order=0, sigma=kernel_size, mode="nearest")
