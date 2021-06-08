from functools import lru_cache, partial, wraps
from math import sqrt

import numpy as np

from npstreams import peek, pmap
from skued import (
    __version__,
    azimuthal_average,
    baseline_dt,
    autocenter,
    powder_calq,
    ArbitrarySelection,
    Selection,
)
from skued.baseline import dt_max_level

from .meta import HDF5ExperimentalParameter, MetaHDF5Dataset

from .dataset import DiffractionDataset, write_access_needed


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

        # TODO: stop writing datasets in __init__
        #       since the GUI will only open datasets in read-mode by default

        # Ensure that all required powder groups exist
        maxshape = (len(self.time_points), sqrt(2 * max(self.resolution) ** 2))
        for name in {"intensity", "baseline"}:
            if name not in self.powder_group:
                self.powder_group.create_dataset(
                    name=name,
                    shape=maxshape,
                    maxshape=maxshape,
                    dtype=float,
                    fillvalue=0.0,
                    **self.compression_params,
                )

        # Radius from center in units of pixels
        shape = self.powder_group["intensity"].shape
        placeholder = np.arange(0, shape[-1])
        if "px_radius" not in self.powder_group:
            self.powder_group.create_dataset(
                "px_radius", data=placeholder, maxshape=(maxshape[-1],), dtype=float
            )

        # Radius from center in units of inverse angstroms
        if "scattering_vector" not in self.powder_group:
            self.powder_group.create_dataset(
                "scattering_vector",
                data=placeholder,
                maxshape=(maxshape[-1],),
                dtype=float,
            )

    @classmethod
    def from_dataset(
        cls, dataset, center=None, normalized=True, angular_bounds=None, callback=None
    ):
        """
        Transform a DiffractionDataset instance into a PowderDiffractionDataset. This requires
        computing the azimuthal averages as well.

        Parameters
        ----------
        dataset : DiffractionDataset
            DiffractionDataset instance.
        center : 2-tuple or None, optional
            Center of the diffraction patterns. If None (default), center will be automatically-determined.
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
        """Pixel-radius of azimuthal average"""
        return np.array(self.powder_group["px_radius"])

    @property
    def scattering_vector(self):
        """Array of scattering vector norm :math:`|q|` [:math:`1/\\AA`]"""
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

    @write_access_needed
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
        ValueError
            if the number of peak indices does not match the number of Miller indices, or if the
            number of peaks given is lower than two.
        IOError
            If the filename is already associated with a file.
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
            dataset.read_direct(out, source_sel=np.s_[time_index, :])

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
            dataset.read_direct(out, source_sel=np.s_[time_index, :])

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
                f"``units`` must be either 'pixels' or 'momentum', not {units}"
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

    @write_access_needed
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

        Raises
        ------
        IOError
            If the filename is already associated with a file.
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

    @write_access_needed
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
            attribute will be used instead. If that is not possible, the center will
            be automatically determined. See :meth:`DiffractionDataset.autocenter`.
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

        Raises
        ------
        IOError
            If the filename is already associated with a file.
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

        if center is None:
            center = self.center

        # Because it is difficult to know the angular averaged data's shape in advance,
        # we calculate it first and store it next
        callback(0)
        results = list()
        for index, timedelay in enumerate(self.time_points):
            px_radius, avg = azimuthal_average(
                self.diff_data(timedelay),
                center=center,
                mask=self.valid_mask,
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
    """Returns the bounds which would be used in numpy.trim_zeros but also trimmming nans"""
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
