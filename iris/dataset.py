# -*- coding: utf-8 -*-
"""
Diffraction dataset types
"""
from collections import OrderedDict
from collections.abc import Callable
from functools import lru_cache, partial
from math import sqrt
from warnings import warn

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

from npstreams import average, itercopy, peek, pmap
from skued import (
    azimuthal_average,
    baseline_dt,
    combine_masks,
    electron_wavelength,
    ialign,
    mask_from_collection,
    nfold,
    powder_calq,
)
from skued.baseline import dt_max_level

from .meta import HDF5ExperimentalParameter, MetaHDF5Dataset

# Whether or not single-writer multiple-reader (SWMR) mode is available
# See http://docs.h5py.org/en/latest/swmr.html for more information
SWMR_AVAILABLE = h5py.version.hdf5_version_tuple > (1, 10, 0)


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

    def __repr__(self):
        rep = "< {} object with following metadata: ".format(type(self).__name__)
        for key, val in self.metadata.items():
            rep += "\n    {key}: {val}".format(key=key, val=val)

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
        **kwargs
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

        first, patterns = peek(patterns)
        if dtype is None:
            dtype = first.dtype
        resolution = first.shape

        if valid_mask is None:
            valid_mask = np.ones(first.shape, dtype=np.bool)

        callback(0)
        with cls(filename, **kwargs) as file:

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
            times = gp.create_dataset("time_points", data=time_points, dtype=np.float)
            mask = gp.create_dataset("valid_mask", data=valid_mask, dtype=np.bool)

            pgp = file.diffraction_group
            dset = pgp.create_dataset(
                name="intensity",
                shape=resolution + (len(time_points),),
                dtype=dtype,
                **ckwargs
            )

            # Making use of the H5DS dimension scales
            # http://docs.h5py.org/en/latest/high/dims.html
            dset.dims.create_scale(times, "time-delay")
            dset.dims[2].attach_scale(times)

            # At each iteration, we flush the changes to file
            # If this is not done, data can be accumulated in memory (>5GB)
            # until this loop is done.
            for index, pattern in enumerate(patterns):
                dset.write_direct(
                    pattern, source_sel=np.s_[:, :], dest_sel=np.s_[:, :, index]
                )
                file.flush()
                callback(round(100 * index / np.size(time_points)))

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
        **kwargs
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
        IOError : If the filename is already associated with a file.
        """
        if callback is None:
            callback = lambda _: None

        if exclude_scans is None:
            exclude_scans = set([])

        if valid_mask is None:
            valid_mask = np.ones(shape=raw.resolution, dtype=np.bool)

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
        TypeError : if `func` is not a proper callable
        """
        if not callable(func):
            raise TypeError(
                "Expected a callable argument, but received {}".format(type(func))
            )

        if callback is None:
            callback = lambda _: None

        # We implement parallel diff apply in a separate method
        # because single-threaded diff apply can be written with a
        # placeholder array
        if SWMR_AVAILABLE and (processes != 1):
            return self._diff_apply_parallel(func, callback, processes)

        ntimes = len(self.time_points)
        dset = self.diffraction_group["intensity"]

        # Create a placeholder numpy array where to load and store the results
        placeholder = np.empty(shape=self.resolution, dtype=dset.dtype, order="C")

        for index, time_point in enumerate(self.time_points):
            dset.read_direct(
                placeholder, source_sel=np.s_[:, :, index], dest_sel=np.s_[:, :]
            )
            placeholder[:] = func(placeholder)
            dset.write_direct(
                placeholder, source_sel=np.s_[:, :], dest_sel=np.s_[:, :, index]
            )
            callback(int(100 * index / ntimes))

        self.diff_eq.cache_clear()

    def _diff_apply_parallel(self, func, callback, processes):
        """
        Apply a function to each diffraction pattern in parallel. The diffraction patterns
        will be modified in-place. This method is not supposed to be called directly.

        .. versionadded:: 5.0.6
        """
        ntimes = len(self.time_points)
        dset = self.diffraction_group["intensity"]

        transformed = pmap(
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
            dset.write_direct(im, source_sel=np.s_[:, :], dest_sel=np.s_[:, :, index])
            callback(int(100 * index / ntimes))
        self.diff_eq.cache_clear()

    def symmetrize(self, mod, center, kernel_size=None, callback=None, processes=1):
        """
        Symmetrize diffraction images based on n-fold rotational symmetry.

        .. warning::
            This is an irreversible in-place operation.

        Parameters
        ----------
        mod : int
            Fold symmetry number. 
        center : array-like, shape (2,) or None
            Coordinates of the center (in pixels). If None, the data is symmetrized around the
            center of the images.
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
        ValueError: if ``mod`` is not a divisor of 360.
    
        See Also
        --------
        diff_apply : apply an operation to each diffraction pattern one-by-one
        """
        # Due to possibility of parallel operation,
        # we can't use lambdas or local functions
        # Therefore, we define _symmetrize below and use it here
        apply = partial(
            _symmetrize,
            mod=mod,
            center=center,
            mask=self.invalid_mask,
            kernel_size=kernel_size,
        )
        self.diff_apply(apply, callback=callback, processes=processes)
        self.center = center

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
    def valid_mask(self):
        """ Array that evaluates to True on valid pixels (i.e. not on beam-block, not hot pixels, etc.) """
        return np.array(self.experimental_parameters_group["valid_mask"])

    @property
    def invalid_mask(self):
        """ Array that evaluates to True on invalid pixels (i.e. on beam-block, hot pixels, etc.) """
        return np.logical_not(self.valid_mask)

    @property
    def time_points(self):
        # Time-points are not treated as metadata because
        return np.array(self.experimental_parameters_group["time_points"])

    @property
    def resolution(self):
        """ Resolution of diffraction patterns (px, px) """
        intensity_shape = self.diffraction_group["intensity"].shape
        return tuple(intensity_shape[0:2])

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
        """
        differential = shift - self.time_zero_shift
        self.time_zero_shift = shift
        self.experimental_parameters_group["time_points"][:] = (
            self.time_points + differential
        )
        self.diff_eq.cache_clear()

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
                "Time-delay {}ps not available. Using \
                 closest-timedelay {}ps instead".format(
                    timedelay, self.time_points[time_index]
                )
            )
        return time_index

    @lru_cache(maxsize=1)
    def diff_eq(self):
        """ 
        Returns the averaged diffraction pattern for all times before photoexcitation. 
        In case no data is available before photoexcitation, an array of zeros is returned.
        The result of this function is cached to minimize overhead.

        Time-zero can be adjusted using the ``shift_time_zero`` method.

        Returns
        -------
        I : ndarray, shape (N,)
            Diffracted intensity [counts]
        """
        dset = self.diffraction_group["intensity"]
        t0_index = np.argmin(np.abs(self.time_points))
        b4t0_slice = dset[:, :, :t0_index]

        # If there are no available data before time-zero, np.mean()
        # will return an array of NaNs; instead, return zeros.
        if t0_index == 0:
            return np.zeros(shape=self.resolution, dtype=dset.dtype)

        # To be able to use lru_cache, we cannot have an `out` parameter
        return np.mean(b4t0_slice, axis=2)

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
            dataset.read_direct(
                out, source_sel=np.s_[:, :, time_index], dest_sel=np.s_[:, :]
            )

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
        """
        x1, x2, y1, y2 = rect
        data = self.diffraction_group["intensity"][x1:x2, y1:y2, :]
        if relative:
            data -= self.diff_eq()[x1:x2, y1:y2]
        return np.mean(data, axis=(0, 1), out=out)

    @property
    def experimental_parameters_group(self):
        return self.require_group(name=self._exp_params_group_name)

    @property
    def diffraction_group(self):
        return self.require_group(name=self._diffraction_group_name)

    @property
    def compression_params(self):
        """ Compression options in the form of a dictionary """
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


class PowderDiffractionDataset(DiffractionDataset):
    """ 
    Abstraction of HDF5 files for powder diffraction datasets.
    """

    _powder_group_name = "/powder"

    angular_bounds = HDF5ExperimentalParameter(
        "angular_bounds", tuple, default=(0, 360)
    )
    first_stage = HDF5ExperimentalParameter(
        "powder_baseline_first_stage", str, default=""
    )
    wavelet = HDF5ExperimentalParameter("powder_baseline_wavelet", str, default="")
    level = HDF5ExperimentalParameter("powder_baseline_level", int, default=0)
    niter = HDF5ExperimentalParameter("powder_baseline_niter", int, default=0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure that all required powder groups exist
        maxshape = (len(self.time_points), sqrt(2 * max(self.resolution) ** 2))
        for name in {"intensity", "baseline"}:
            if name not in self.powder_group:
                self.powder_group.create_dataset(
                    name=name,
                    shape=maxshape,
                    maxshape=maxshape,
                    dtype=np.float,
                    fillvalue=0.0,
                    **self.compression_params
                )

        # Radius from center in units of pixels
        shape = self.powder_group["intensity"].shape
        placeholder = np.arange(0, shape[-1])
        if "px_radius" not in self.powder_group:
            self.powder_group.create_dataset(
                "px_radius", data=placeholder, maxshape=(maxshape[-1],), dtype=np.float
            )

        # Radius from center in units of inverse angstroms
        if "scattering_vector" not in self.powder_group:
            self.powder_group.create_dataset(
                "scattering_vector",
                data=placeholder,
                maxshape=(maxshape[-1],),
                dtype=np.float,
            )

    @classmethod
    def from_dataset(
        cls, dataset, center, normalized=True, angular_bounds=None, callback=None
    ):
        """
        Transform a DiffractionDataset instance into a PowderDiffractionDataset. This requires
        computing the azimuthal averages as well.

        Parameters
        ----------
        dataset : DiffractionDataset
            DiffractionDataset instance.
        center : 2-tuple or None, optional
            Center of the diffraction patterns. If None (default), the dataset
            attribute will be used instead.
        normalized : bool, optional
            If True, each pattern is normalized to its integral. Default is False.
        angular_bounds : 2-tuple of float or None, optional
            Angle bounds are specified in degrees. 0 degrees is defined as the positive x-axis. 
            Angle bounds outside [0, 360) are mapped back to [0, 360).
        callback : callable or None, optional
            Callable of a single argument, to which the calculation progress will be passed as
            an integer between 0 and 100.
        
        Returns
        -------
        powder : PowderDiffractionDataset
        """
        fname = dataset.filename
        dataset.close()

        powder_dataset = cls(fname, mode="r+")
        powder_dataset.compute_angular_averages(
            center, normalized, angular_bounds, callback
        )
        return powder_dataset

    @property
    def powder_group(self):
        return self.require_group(self._powder_group_name)

    @property
    def px_radius(self):
        """ Pixel-radius of azimuthal average """
        return np.array(self.powder_group["px_radius"])

    @property
    def scattering_vector(self):
        """ Array of scattering vector norm :math:`|q|` [:math:`1/\AA`] """
        return np.array(self.powder_group["scattering_vector"])

    def shift_time_zero(self, *args, **kwargs):
        """
        Shift time-zero uniformly across time-points.

        Parameters
        ----------
        shift : float
            Shift [ps]. A positive value of `shift` will move all time-points forward in time,
            whereas a negative value of `shift` will move all time-points backwards in time.
        """
        self.powder_eq.cache_clear()
        return super().shift_time_zero(*args, **kwargs)

    def powder_calq(self, crystal, peak_indices, miller_indices):
        """
        Determine the scattering vector q corresponding to a polycrystalline diffraction pattern
        and a known crystal structure.

        For best results, multiple peaks (and corresponding Miller indices) should be provided; the
        absolute minimum is two.

        Parameters
        ----------
        crystal : skued.Crystal instance
            Crystal that gave rise to the diffraction data.
        peak_indices : n-tuple of ints
            Array index location of diffraction peaks. For best
            results, peaks should be well-separated. More than two peaks can be used.
        miller_indices : iterable of 3-tuples
            Indices associated with the peaks of ``peak_indices``. More than two peaks can be used.
            E.g. ``indices = [(2,2,0), (-3,0,2)]``
        
        Raises
        ------
        ValueError : if the number of peak indices does not match the number of Miller indices.
        ValueError : if the number of peaks given is lower than two.
        """
        I = self.powder_eq()
        q = powder_calq(
            I=I,
            crystal=crystal,
            peak_indices=peak_indices,
            miller_indices=miller_indices,
        )

        self.powder_group["scattering_vector"].resize(I.shape)
        self.powder_group["scattering_vector"].write_direct(q)

    @lru_cache(maxsize=2)  # with and without background
    def powder_eq(self, bgr=False):
        """ 
        Returns the average powder diffraction pattern for all times before photoexcitation. 
        In case no data is available before photoexcitation, an array of zeros is returned.

        Parameters
        ----------
        bgr : bool
            If True, background is removed.

        Returns
        -------
        I : ndarray, shape (N,)
            Diffracted intensity [counts]
        """
        t0_index = np.argmin(np.abs(self.time_points))
        b4t0_slice = self.powder_group["intensity"][:t0_index, :]

        # If there are no available data before time-zero, np.mean()
        # will return an array of NaNs; instead, return zeros.
        if t0_index == 0:
            return np.zeros_like(self.px_radius)

        if not bgr:
            return np.mean(b4t0_slice, axis=0)

        bg = self.powder_group["baseline"][:t0_index, :]
        return np.mean(b4t0_slice - bg, axis=0)

    def powder_data(self, timedelay, bgr=False, relative=False, out=None):
        """
        Returns the angular average data from scan-averaged diffraction patterns.

        Parameters
        ----------
        timdelay : float or None
            Time-delay [ps]. If None, the entire block is returned.
        bgr : bool, optional
            If True, background is removed.
        relative : bool, optional
            If True, data is returned relative to the average of all diffraction patterns
            before photoexcitation.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.
        
        Returns
        -------
        I : ndarray, shape (N,) or (N,M)
            Diffracted intensity [counts]
        """
        dataset = self.powder_group["intensity"]

        if timedelay is None:
            if out is None:
                out = np.empty_like(dataset)
            dataset.read_direct(out)

        else:
            time_index = self._get_time_index(timedelay)
            if out is None:
                out = np.empty_like(self.px_radius)
            dataset.read_direct(out, source_sel=np.s_[time_index, :], dest_sel=np.s_[:])

        if bgr:
            out -= self.powder_baseline(timedelay)

        if relative:
            out -= self.powder_eq(bgr=bgr)

        return out

    def powder_baseline(self, timedelay, out=None):
        """ 
        Returns the baseline data. 

        Parameters
        ----------
        timdelay : float or None
            Time-delay [ps]. If None, the entire block is returned.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.
        
        Returns
        -------
        out : ndarray
            If a baseline hasn't been computed yet, the returned
            array is an array of zeros.
        """
        try:
            dataset = self.powder_group["baseline"]
        except KeyError:
            return np.zeros_like(self.px_radius)

        if timedelay is None:
            if out is None:
                out = np.empty_like(dataset)
            dataset.read_direct(out)

        else:
            time_index = self._get_time_index(timedelay)
            if out is None:
                out = np.empty_like(self.px_radius)
            dataset.read_direct(out, source_sel=np.s_[time_index, :], dest_sel=np.s_[:])

        return out

    def powder_time_series(
        self, rmin, rmax, bgr=False, relative=False, units="pixels", out=None
    ):
        """
        Average intensity over time.
        Diffracted intensity is integrated in the closed interval [rmin, rmax]

        Parameters
        ----------
        rmin : float
            Lower scattering vector bound [1/A]
        rmax : float
            Higher scattering vector bound [1/A]. 
        bgr : bool, optional
            If True, background is removed. Default is False.
        relative : bool, optional
            If True, data is returned relative to the average of all diffraction patterns
            before photoexcitation.
        units : str, {'pixels', 'momentum'}
            Units of the bounds rmin and rmax.
        out : ndarray or None, optional
            1-D ndarray in which to store the results. The shape
            should be compatible with (len(time_points),)
        
        Returns
        -------
        out : ndarray, shape (N,)
            Average diffracted intensity over time.
        """
        # In some cases, it is easier
        if units not in {"pixels", "momentum"}:
            raise ValueError(
                "``units`` must be either 'pixels' or 'momentum', not {}".format(units)
            )
        abscissa = self.px_radius if units == "pixels" else self.scattering_vector

        i_min, i_max = (
            np.argmin(np.abs(rmin - abscissa)),
            np.argmin(np.abs(rmax - abscissa)),
        )
        i_max += 1  # Python slices are semi-open by design, therefore i_max + 1 is used
        trace = np.array(self.powder_group["intensity"][:, i_min:i_max])
        if bgr:
            trace -= np.array(self.powder_group["baseline"][:, i_min:i_max])

        if relative:
            trace -= self.powder_eq(bgr=bgr)[i_min:i_max]

        if out is not None:
            return np.mean(axis=1, out=out)
        return np.mean(trace, axis=1).reshape(-1)

    def compute_baseline(self, first_stage, wavelet, max_iter=50, level=None, **kwargs):
        """
        Compute and save the baseline computed based on the dual-tree complex wavelet transform. 
        All keyword arguments are passed to scikit-ued's `baseline_dt` function.

        Parameters
        ----------
        first_stage : str, optional
            Wavelet to use for the first stage. See :func:`skued.available_first_stage_filters` for a list of suitable arguments
        wavelet : str, optional
            Wavelet to use in stages > 1. Must be appropriate for the dual-tree complex wavelet transform.
            See :func:`skued.available_dt_filters` for possible values.
        max_iter : int, optional

        level : int or None, optional
            If None (default), maximum level is used.
        """
        block = self.powder_data(timedelay=None, bgr=False)

        baseline_kwargs = {
            "array": block,
            "max_iter": max_iter,
            "level": level,
            "first_stage": first_stage,
            "wavelet": wavelet,
            "axis": 1,
        }
        baseline_kwargs.update(**kwargs)

        baseline = np.ascontiguousarray(
            baseline_dt(**baseline_kwargs)
        )  # In rare cases this wasn't C-contiguous

        # The baseline dataset is guaranteed to exist after compte_angular_averages was called.
        self.powder_group["baseline"].resize(baseline.shape)
        self.powder_group["baseline"].write_direct(baseline)

        if level == None:
            level = dt_max_level(
                data=self.px_radius, first_stage=first_stage, wavelet=wavelet
            )

        self.level = level
        self.first_stage = first_stage
        self.wavelet = wavelet
        self.niter = max_iter

        self.powder_eq.cache_clear()

    def compute_angular_averages(
        self,
        center=None,
        normalized=False,
        angular_bounds=None,
        trim=True,
        callback=None,
    ):
        """ 
        Compute the angular averages.
        
        Parameters
        ----------
        center : 2-tuple or None, optional
            Center of the diffraction patterns. If None (default), the dataset
            attribute will be used instead.
        normalized : bool, optional
            If True, each pattern is normalized to its integral.
        angular_bounds : 2-tuple of float or None, optional
            Angle bounds are specified in degrees. 0 degrees is defined as the positive x-axis. 
            Angle bounds outside [0, 360) are mapped back to [0, 360).
        trim : bool, optional
            If True, leading/trailing zeros - possibly due to masks - are trimmed.
        callback : callable or None, optional
            Callable of a single argument, to which the calculation progress will be passed as
            an integer between 0 and 100.
        """
        # TODO: allow to cut away regions
        if not any([self.center, center]):
            raise RuntimeError(
                "Center attribute must be either saved in the dataset \
                                as an attribute or be provided."
            )

        if callback is None:
            callback = lambda i: None

        if center is not None:
            self.center = center

        # Because it is difficult to know the angular averaged data's shape in advance,
        # we calculate it first and store it next
        callback(0)
        results = list()
        for index, timedelay in enumerate(self.time_points):
            px_radius, avg = azimuthal_average(
                self.diff_data(timedelay),
                center=self.center,
                mask=self.invalid_mask,
                angular_bounds=angular_bounds,
                trim=False,
            )

            # px_radius is not stored but used once
            results.append(avg)
            callback(int(100 * index / len(self.time_points)))

        # Concatenate arrays for intensity and error
        # If trimming is enabled, there might be a problem where
        # different averages are trimmed to different length
        # therefore, we trim to the most restrictive bounds
        if trim:
            bounds = [_trim_bounds(I) for I in results]
            min_bound = max(min(bound) for bound in bounds)
            max_bound = min(max(bound) for bound in bounds)
            results = [I[min_bound:max_bound] for I in results]
            px_radius = px_radius[min_bound:max_bound]

        rintensity = np.stack(results, axis=0)

        if normalized:
            rintensity /= np.sum(rintensity, axis=1, keepdims=True)

        # We allow resizing. In theory, an angular averave could never be
        # longer than the diagonal of resolution
        self.powder_group["intensity"].resize(rintensity.shape)
        self.powder_group["intensity"].write_direct(rintensity)

        self.powder_group["px_radius"].resize(px_radius.shape)
        self.powder_group["px_radius"].write_direct(px_radius)

        # Use px_radius as placeholder for scattering_vector until calibration
        self.powder_group["scattering_vector"].resize(px_radius.shape)
        self.powder_group["scattering_vector"].write_direct(px_radius)

        self.powder_group["baseline"].resize(rintensity.shape)
        self.powder_group["baseline"].write_direct(np.zeros_like(rintensity))

        self.powder_eq.cache_clear()
        callback(100)


def _trim_bounds(arr):
    """ Returns the bounds which would be used in numpy.trim_zeros but also trimmming nans"""
    first = 0
    for i in arr:
        if i != 0.0:
            break
        else:
            first = first + 1
    last = len(arr)
    for i in arr[::-1]:
        if i != 0.0:
            break
        else:
            last = last - 1
    return first, last
