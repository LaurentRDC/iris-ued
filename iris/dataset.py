# -*- coding: utf-8 -*-
"""
Diffraction dataset types
"""
from sys import platform
from collections import OrderedDict
from contextlib import AbstractContextManager
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
    ArbitrarySelection,
    Selection,
)

from .lowlevel import LowLevelDataset, SWMR_AVAILABLE, IOMode, InternalDatasets


def lowlevel_write(f):
    @wraps(f)
    def newf(self, *args, **kwargs):
        filename = self._lowlevel.filename
        self._lowlevel.close()
        with LowLevelDataset(filename=filename, mode=IOMode.ReadWrite) as dset:
            self._lowlevel = dset
            r = f(self, *args, **kwargs)
        self._lowlevel = LowLevelDataset(filename=filename, mode=IOMode.ReadOnly)
        return r

    return newf


class DiffractionDataset(AbstractContextManager):
    """
    Abstraction of an HDF5 file to represent diffraction datasets.
    """

    def __init__(self, filename):
        self._lowlevel = LowLevelDataset(filename=filename, mode=IOMode.ReadOnly)

    def __exit__(self, *args, **kwargs):
        return self._lowlevel.close()

    def __repr__(self):
        rep = f"< {type(self).__name__} object with following metadata: "
        for key, val in self._lowlevel.metadata.items():
            rep += f"\n    {key}: {val}"

        return rep + " >"

    def __array__(self):
        dset = self._lowlevel.get_dataset(InternalDatasets.Intensity)
        arr = np.empty(shape=dset.shape, dtype=dset.dtype)
        dset.read_direct(arr)
        return arr

    @classmethod
    @wraps(LowLevelDataset.from_collection)
    def from_collection(cls, *args, **kwargs):
        with LowLevelDataset.from_collection(*args, **kwargs) as dset:
            fname = dset.filename
        return cls(filename=fname)

    @classmethod
    def from_raw(
        cls,
        raw,
        filename,
        exclude_scans=None,
        mask=None,
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
        mask : ndarray or None, optional
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

        if mask is None:
            mask = np.ones(shape=raw.resolution, dtype=np.bool)

        metadata = raw.metadata.copy()
        metadata["scans"] = tuple(set(raw.scans) - set(exclude_scans))
        metadata["aligned"] = align
        metadata["normalized"] = normalize

        # Assemble the metadata
        kwargs.update(
            {
                "ckwargs": ckwargs,
                "mask": mask,
                "metadata": metadata,
                "time_points": raw.time_points,
                "callback": callback,
                "filename": filename,
            }
        )

        reduced = raw.reduced(
            exclude_scans=exclude_scans,
            align=align,
            normalize=normalize,
            mask=np.logical_not(mask),
            processes=processes,
            dtype=dtype,
        )

        return cls.from_collection(patterns=reduced, **kwargs)

    @lowlevel_write
    def symmetrize(self, mod, callback=None, processes=1):
        """
        Symmetrize diffraction images based on n-fold rotational symmetry.

        .. warning::
            This is an irreversible in-place operation.

        Parameters
        ----------
        mod : int
            Fold symmetry number.
        callback : callable or None, optional
            Callable that takes an int between 0 and 99. This can be used for progress update.
        processes : int or None, optional
            Number of parallel processes to use. If ``None``, all available processes will be used.
            In case Single Writer Multiple Reader mode is not available, ``processes`` is ignored.

        Raises
        ------
        ValueError: if ``mod`` is not a divisor of 360.
        PermissionError: if the dataset has not been opened with write access.
        """
        r, c = self.center
        return self._lowlevel.diff_apply(
            partial(nfold, mod=mod, center=(c, r), mask=self.mask),
            callback=callback,
            processes=processes,
        )

    @property
    @wraps(LowLevelDataset.metadata)
    def metadata(self):
        return self._lowlevel.metadata

    @property
    @wraps(LowLevelDataset.mask)
    def mask(self):
        return self._lowlevel.mask

    @property
    @wraps(LowLevelDataset.center)
    def center(self):
        return self._lowlevel.center

    @property
    @wraps(LowLevelDataset.time_points)
    def time_points(self):
        return self._lowlevel.time_points

    @property
    @wraps(LowLevelDataset.equilibrium_pattern)
    def equilibrium_pattern(self):
        return self._lowlevel.equilibrium_pattern

    @wraps(LowLevelDataset.diffraction_pattern)
    def diffraction_pattern(self, *args, **kwargs):
        return self._lowlevel.diffraction_pattern(*args, **kwargs)

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
        return np.mean(
            self._lowlevel.time_series(*rect, relative=relative), axis=(0, 1)
        )


def _symmetrize(im, mod, center, mask):
    return nfold(im, mod=mod, center=center, mask=mask)
