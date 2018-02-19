"""
@author: Laurent P. Rene de Cotret
"""
from functools import lru_cache, partial
from itertools import repeat
from math import sqrt
from warnings import warn

import h5py
import numpy as np
from cached_property import cached_property
from scipy.signal import detrend

from npstreams import average, peek, pmap, itercopy
from npstreams.stats import _ivar
from skued import (electron_wavelength, baseline_dt, azimuthal_average, ialign, 
                   mask_from_collection, combine_masks)
from skued.baseline import dt_max_level

from .meta import HDF5ExperimentalParameter, MetaHDF5Dataset


class DiffractionDataset(h5py.File, metaclass = MetaHDF5Dataset):
    """
    Abstraction of an HDF5 file to represent diffraction datasets.
    """
    _diffraction_group_name = '/processed'
    _exp_params_group_name  = '/'

    energy          = HDF5ExperimentalParameter('energy',          float, default = 90)       # keV
    pump_wavelength = HDF5ExperimentalParameter('pump_wavelength', int,   default = 800)      # nanometers
    fluence         = HDF5ExperimentalParameter('fluence',         float, default = 0)        # milliJoules / centimeters ^ 2
    time_zero_shift = HDF5ExperimentalParameter('time_zero_shift', float, default = 0)        # picoseconds
    temperature     = HDF5ExperimentalParameter('temperature',     float, default = 293)      # kelvins
    exposure        = HDF5ExperimentalParameter('exposure',        float, default = 1)        # seconds
    scans           = HDF5ExperimentalParameter('scans',           tuple, default = (1,))
    camera_length   = HDF5ExperimentalParameter('camera_length',   float, default = 0.23)     # meters
    pixel_width     = HDF5ExperimentalParameter('pixel_width',     float, default = 14e-6)    # meters
    aligned         = HDF5ExperimentalParameter('aligned',         bool,  default = False)
    normalized      = HDF5ExperimentalParameter('normalized',      bool,  default = False)
    notes           = HDF5ExperimentalParameter('notes',           str,   default = '')

    def __repr__(self):
        return '< DiffractionDataset object. \
                  Acquisition date : {}, \n \
                  fluence {} mj/cm**2 >'.format(self.acquisition_date, self.fluence)
    
    @classmethod
    def from_collection(cls, patterns, filename, time_points, metadata, valid_mask = None, 
                        dtype = None, ckwargs = None, callback = None, **kwargs):
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
        
        Returns
        -------
        dataset : DiffractionDataset
        """
        # H5py will raise an exception if arrays are not contiguous
        patterns = map(np.ascontiguousarray, iter(patterns))

        if callback is None: 
            callback = lambda _: None

        if 'mode' not in kwargs:
            kwargs['mode'] = 'x'    #safest mode

        time_points = np.array(time_points).reshape(-1)

        if ckwargs is None:
            ckwargs = {'compression': 'lzf', 
                       'shuffle'    : True, 
                       'fletcher32' : True}
        ckwargs['chunks'] = True # For some reason, if no chunking, writing to disk is SLOW

        first, patterns = peek(patterns)
        if dtype is None:
            dtype = first.dtype
        resolution = first.shape

        if valid_mask is None:
            valid_mask = np.ones(first.shape, dtype = np.bool)

        callback(0)
        with cls(filename, **kwargs) as file:

            # Note that keys not associated with an ExperimentalParameter
            # descriptor will not be recorded in the file.
            metadata.pop('time_points', None)
            for key, val in metadata.items():
                if key not in cls.valid_metadata:
                    continue
                setattr(file, key, val)
            
            # Record time-points as a dataset; then, changes to it will be reflected
            # in other dimension scales
            gp = file.experimental_parameters_group
            times = gp.create_dataset('time_points', data = time_points, dtype = np.float)
            mask = gp.create_dataset('valid_mask', data = valid_mask, dtype = np.bool)

            pgp = file.diffraction_group
            dset = pgp.create_dataset(name = 'intensity', 
                                      shape = resolution + (len(time_points), ), 
                                      dtype = dtype, **ckwargs)
            
            # Making use of the H5DS dimension scales
            # http://docs.h5py.org/en/latest/high/dims.html
            dset.dims.create_scale(times, 'time-delay')
            dset.dims[2].attach_scale(times)

            # At each iteration, we flush the changes to file
            # If this is not done, data can be accumulated in memory (>5GB)
            # until this loop is done.
            for index, pattern in enumerate(patterns):
                dset.write_direct(pattern, source_sel = np.s_[:,:], dest_sel = np.s_[:,:,index])
                file.flush()
                callback(round(100 * index / np.size(time_points)))

        callback(100)
        return cls(filename)

    @classmethod
    def from_raw(cls, raw, filename, exclude_scans = set([]), valid_mask = None, 
                 processes = 1, callback = None, align = True, normalize = True, 
                 clip = None, ckwargs = dict(), dtype = None, **kwargs):
        """
        Create a DiffractionDataset from a subclass of AbstractRawDataset.

        Parameters
        ----------
        raw : RawDataset
            Raw dataset instance.
        filename : str or path-like
            Path to the assembled DiffractionDataset.
        exclude_scans : iterable of ints, optional
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
            If True, images are normalized according to their total electron count.
        clip : iterable or None, optional
            If not None, all image are clipped to values between `min(clip)` and `max(clip)`.
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

        Raises
        ------
        IOError : If the filename is already associated with a file.
        """
        if callback is None: 
            callback = lambda _: None
        
        if valid_mask is None:
            valid_mask = np.ones(shape = raw.resolution, dtype = np.bool)

        valid_scans = tuple(set(raw.scans) - set(exclude_scans))
        
        metadata = raw.metadata.copy()
        metadata['scans']       = valid_scans
        metadata['aligned']     = align
        metadata['normalized']  = normalize

        # Calculate a mask from a scan
        # to catch dead pixels, for example
        images = (raw.raw_data(timedelay, scan = valid_scans[0]) for timedelay in raw.time_points)
        coll_mask = mask_from_collection(images, std_thresh = 3)
        invalid_mask = combine_masks(np.logical_not(valid_mask), coll_mask)
        valid_mask = np.logical_not(invalid_mask)
        callback(1)

        # Assemble the metadata
        kwargs.update({'ckwargs'    : ckwargs, 
                       'valid_mask' : valid_mask, 
                       'metadata'   : metadata,
                       'time_points': raw.time_points,
                       'dtype'      : dtype,
                       'callback'   : callback,
                       'filename'   : filename})
        
        map_kwargs = {'raw'         : raw,
                      'clip'        : clip,
                      'valid_scans' : valid_scans,
                      'align'       : align,
                      'normalize'   : normalize,
                      'invalid_mask': np.logical_not(valid_mask)}

        reduced = pmap(_raw_combine, raw.time_points, 
                       kwargs = map_kwargs,
                       processes = processes,
                       ntotal = len(raw.time_points))

        return cls.from_collection(patterns = reduced, **kwargs)
        
    @property
    def metadata(self):
        """ Dictionary of the dataset's metadata. """
        meta = {k:getattr(self, k) for k in self.valid_metadata}
        meta['filename'] = self.filename
        meta['time_points'] = tuple(self.time_points)
        meta.update(self.compression_params)
        return meta
    
    @cached_property
    def valid_mask(self):
        """ Array that evaluates to True on valid pixels (i.e. not on beam-block, not hot pixels, etc.) """
        return np.array(self.experimental_parameters_group['valid_mask'])
    
    @cached_property
    def invalid_mask(self):
        """ Array that evaluates to True on invalid pixels (i.e. on beam-block, hot pixels, etc.) """
        return np.logical_not(self.valid_mask)

    @property
    def time_points(self):
        # Time-points are not treated as metadata because
        return np.array(self.experimental_parameters_group['time_points'])

    @property
    def resolution(self):
        """ Resolution of diffraction patterns (px, px) """
        intensity_shape = self.diffraction_group['intensity'].shape
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
        self.experimental_parameters_group['time_points'][:] = self.time_points + differential
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
            warn('Time-delay {}ps not available. Using \
                 closest-timedelay {}ps instead'.format(timedelay, self.time_points[time_index]))
        return time_index

    @lru_cache(maxsize = 1)
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
        dset = self.diffraction_group['intensity']
        t0_index = np.argmin(np.abs(self.time_points))
        b4t0_slice = dset[:, :, :t0_index]

        # If there are no available data before time-zero, np.mean()
        # will return an array of NaNs; instead, return zeros.
        if t0_index == 0:
            return np.zeros(shape = self.resolution, dtype = dset.dtype)
        
        # To be able to use lru_cache, we cannot have an `out` parameter
        return np.mean(b4t0_slice, axis = 2)

    def diff_data(self, timedelay, relative = False, out = None):
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
        dataset = self.diffraction_group['intensity']

        if timedelay is None:
            if out is None:
                out = np.empty_like(dataset)
            dataset.read_direct(out)

        else:
            time_index = self._get_time_index(timedelay)
            if out is None:
                out = np.empty(self.resolution, dtype = dataset.dtype)
            dataset.read_direct(out, source_sel = np.s_[:,:, time_index], dest_sel = np.s_[:,:])
        
        if relative:
            out -= self.diff_eq()
            out /= self.diff_eq()
            out[:] = np.nan_to_num(out, copy = False)

        return out
    
    def time_series(self, rect, relative = False, out = None):
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
        data = self.diffraction_group['intensity'][x1:x2, y1:y2, :]
        if relative:
            data -= self.diff_eq()[x1:x2, y1:y2]
        return np.mean(data, axis = (0,1), out = out)
    
    @property
    def experimental_parameters_group(self):
        return self.require_group(name = self._exp_params_group_name)
    
    @property
    def diffraction_group(self):
        return self.require_group(name = self._diffraction_group_name)
    
    @cached_property
    def compression_params(self):
        """ Compression options in the form of a dictionary """
        dataset = self.diffraction_group['intensity']
        ckwargs = dict()
        ckwargs['compression'] = dataset.compression
        ckwargs['fletcher32'] = dataset.fletcher32
        ckwargs['shuffle'] = dataset.shuffle
        ckwargs['chunks'] = True if dataset.chunks else None
        if dataset.compression_opts: #could be None
            ckwargs.update(dataset.compression_opts)
        return ckwargs

def _raw_combine(timedelay, raw, valid_scans, align, normalize, clip, invalid_mask):
    
    images = map(partial(raw.raw_data, timedelay), valid_scans)
    if clip:
        images = map(partial(iclip, low = min(clip), high = max(clip)), images)

    if align:
        images = ialign(images, mask = invalid_mask)
    
    if normalize:
        valid = np.logical_not(invalid_mask)
        images, images2 = itercopy(images, copies = 2)

        # Compute the total intensity of first image
        # This will be the reference point
        first2, images2 = peek(images2)
        initial_weight = np.sum(first2[valid])
        weights = (initial_weight/np.sum(image[valid]) for image in images2)
    else:
        weights = None

    return average(images, weights = weights)

def iclip(im, low, high):
    """ Clip images in-place """
    np.clip(im, low, high, out = im)
    return im

def avg_and_error(arrays, axis = -1, ddof = 0, weights = None, ignore_nan = False):
    """
    Calculate the average and error in a stream conccurently.

    Parameters
    ----------
    arrays : iterable of ndarrays
        Arrays to be combined. This iterable can also a generator.
    axis : int, optional
        Reduction axis. Default is to combine the arrays in the stream as if 
        they had been stacked along a new axis, then compute the standard error along this new axis.
        If None, arrays are flattened. If `axis` is an int larger that
        the number of dimensions in the arrays of the stream, standard error is computed
        along the new axis.
    ddof : int, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is one.
    weights : iterable of ndarray, iterable of floats, or None, optional
        Iterable of weights associated with the values in each item of `arrays`. 
        Each value in an element of `arrays` contributes to the standard error 
        according to its associated weight. The weights array can either be a float
        or an array of the same shape as any element of `arrays`. If weights=None, 
        then all data in each element of `arrays` are assumed to have a weight equal to one.
    ignore_nan : bool, optional
        If True, NaNs are set to zero weight. Default is propagation of NaNs.
    
    Returns
    -------
    avg : `~numpy.ndarray`
        Weighted average. 
    err : `~numpy.ndarray`
        Weighted standard deviation.
    """
    avg, sq_avg, swgt = last(_ivar(arrays, axis, weights, ignore_nan))
    std = np.sqrt((sq_avg - avg**2) * (swgt / (swgt - ddof)))
    return avg, std

class PowderDiffractionDataset(DiffractionDataset):
    """ 
    Abstraction of HDF5 files for powder diffraction datasets.
    """
    _powder_group_name = '/powder'

    center =         HDF5ExperimentalParameter('center',                      tuple, default = (0,0))
    angular_bounds = HDF5ExperimentalParameter('angular_bounds',              tuple, default = (0, 360)) 
    first_stage =    HDF5ExperimentalParameter('powder_baseline_first_stage', str,   default = '')
    wavelet =        HDF5ExperimentalParameter('powder_baseline_wavelet',     str,   default = '')
    level =          HDF5ExperimentalParameter('powder_baseline_level',       int,   default = 0)
    niter =          HDF5ExperimentalParameter('powder_baseline_niter',       int,   default = 0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure that all required powder groups exist
        maxshape = (len(self.time_points), sqrt(2*max(self.resolution)**2))
        for name in {'intensity', 'baseline',}:
            if name not in self.powder_group:
                self.powder_group.create_dataset(name = name, shape = maxshape, maxshape = maxshape, 
                                                 dtype = np.float, fillvalue = 0.0, **self.compression_params)
        
        if 'px_radius' not in self.powder_group:
            self.powder_group.create_dataset('px_radius', shape = (maxshape[-1],), maxshape = (maxshape[-1],),
                                             dtype = np.float, fillvalue = 0.0)
    
    @classmethod
    def from_dataset(cls, dataset, center, normalized = True, angular_bounds = None, callback = None):
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

        powder_dataset = cls(fname, mode = 'r+')
        powder_dataset.compute_angular_averages(center, normalized, angular_bounds, callback)
        return powder_dataset

    @property
    def powder_group(self):
        return self.require_group(self._powder_group_name)
    
    @property
    def px_radius(self):
        """ Pixel-radius of azimuthal average """
        return np.array(self.powder_group['px_radius'])

    @property
    def scattering_angle(self):
        """ Array of scattering angle :math:`2 \Theta` """
        # TODO: cache
        return self.px_radius
        #return np.arctan(self.px_radius * self.pixel_width / self.camera_length)

    @property
    def scattering_vector(self):
        """ Array of scattering vector norm :math:`|G|` [:math:`1/\AA`] """
        # Scattering vector norm is defined as G = 4 pi sin(theta)/wavelength
        # TODO: cache
        return 4*np.pi*np.sin(self.scattering_angle/2) / electron_wavelength(self.energy)

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
    
    @lru_cache(maxsize = 2) # with and without background
    def powder_eq(self, bgr = False):
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
        b4t0_slice = self.powder_group['intensity'][:t0_index, :]

        # If there are no available data before time-zero, np.mean()
        # will return an array of NaNs; instead, return zeros.
        if t0_index == 0:
            return np.zeros_like(self.px_radius)

        if not bgr:
            return np.mean(b4t0_slice, axis = 0)
        
        bg = self.powder_group['baseline'][:t0_index, :]
        return np.mean(b4t0_slice - bg, axis = 0)

    def powder_data(self, timedelay, bgr = False, relative = False, out = None):
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
        dataset = self.powder_group['intensity']

        if timedelay is None:
            if out is None:
                out = np.empty_like(dataset)
            dataset.read_direct(out)

        else:
            time_index = self._get_time_index(timedelay)
            if out is None:
                out = np.empty_like(self.px_radius)
            dataset.read_direct(out, source_sel = np.s_[time_index,:], dest_sel = np.s_[:])

        if bgr:
            out -= self.powder_baseline(timedelay)
        
        if relative:
            out -= self.powder_eq(bgr = bgr)

        return out     
    
    def powder_baseline(self, timedelay, out = None):
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
            dataset = self.powder_group['baseline']
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
            dataset.read_direct(out, source_sel = np.s_[time_index,:], dest_sel = np.s_[:]) 
        
        return out
    
    def powder_time_series(self, rmin, rmax, bgr = False, relative = False, units = 'pixels', out = None):
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
        if units not in {'pixels', 'momentum'}:
            raise ValueError("``units`` must be either 'pixels' or 'momentum', not {}".format(units))
        abscissa = self.px_radius if units == 'pixels' else self.scattering_vector
        
        i_min, i_max = np.argmin(np.abs(rmin - abscissa)), np.argmin(np.abs(rmax - abscissa))
        i_max += 1 # Python slices are semi-open by design, therefore i_max + 1 is used
        trace = np.array(self.powder_group['intensity'][:, i_min:i_max])
        if bgr :
            trace -= np.array(self.powder_group['baseline'][:, i_min:i_max])
        
        if relative:
            trace -= self.powder_eq(bgr = bgr)[i_min:i_max]
        
        if out is not None:
            return np.mean(axis = 1, out = out)
        return np.mean(trace, axis = 1).reshape(-1)
    
    def compute_baseline(self, first_stage, wavelet, max_iter = 50, level = None, **kwargs):
        """
        Compute and save the baseline computed from the dualtree package. All keyword arguments are
        passed to scikit-ued's `baseline_dt` function.

        Parameters
        ----------
        first_stage : str, optional
            Wavelet to use for the first stage. See dualtree.ALL_FIRST_STAGE for a list of suitable arguments
        wavelet : str, optional
            Wavelet to use in stages > 1. Must be appropriate for the dual-tree complex wavelet transform.
            See dualtree.ALL_COMPLEX_WAV for possible
        max_iter : int, optional

        level : int or None, optional
            If None (default), maximum level is used.
        """
        block = self.powder_data(timedelay = None, bgr = False)
        trend = block - detrend(block, axis = 1)

        baseline_kwargs = {'array': block - trend,
                           'max_iter': max_iter, 'level': level, 
                           'first_stage': first_stage, 'wavelet': wavelet,
                           'axis': 1}
        baseline_kwargs.update(**kwargs)
        
        baseline = np.ascontiguousarray(trend + baseline_dt(**baseline_kwargs)) # In rare cases this wasn't C-contiguous
        
        # The baseline dataset is guaranteed to exist after compte_angular_averages was called. 
        self.powder_group['baseline'].resize(baseline.shape) 
        self.powder_group['baseline'].write_direct(baseline) 
        
        if level == None:
            level = dt_max_level(data = self.px_radius, first_stage = first_stage, wavelet = wavelet)
            
        self.level = level
        self.first_stage = first_stage
        self.wavelet = wavelet
        self.niter = max_iter
    
    def compute_angular_averages(self, center = None, normalized = False, angular_bounds = None, callback = None):
        """ 
        Compute the angular averages.
        
        Parameters
        ----------
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
        """
        # TODO: allow to cut away regions
        if not any([self.center, center]):
            raise RuntimeError('Center attribute must be either saved in the dataset \
                                as an attribute or be provided.')
        
        if callback is None:
            callback = lambda i: None
        
        if center is not None:
            self.center = center
        
        if angular_bounds is not None:
            self.angular_bounds = angular_bounds
        else:
            self.angular_bounds = (0, 360)

        # Because it is difficult to know the angular averaged data's shape in advance, 
        # we calculate it first and store it next
        callback(0)
        results = list()
        for index, timedelay in enumerate(self.time_points):
            px_radius, avg = azimuthal_average(self.diff_data(timedelay), 
                                               center = self.center, 
                                               mask = np.logical_not(self.valid_mask), 
                                               angular_bounds = angular_bounds)
            
            results.append((px_radius, avg))
            callback(int(100*index / len(self.time_points)))
        
        # Concatenate arrays for intensity and error
        rintensity = np.stack([I for _, I in results], axis = 0)

        if normalized:
            rintensity /= np.sum(rintensity, axis = 1, keepdims = True)
        
        # We allow resizing. In theory, an angular averave could never be 
        # longer than the diagonal of resolution
        self.powder_group['intensity'].resize(rintensity.shape)
        self.powder_group['intensity'].write_direct(rintensity)
        
        # We store the raw px radius and infer other measurements (e.g. diffraction angle) from it
        self.powder_group['px_radius'].resize(px_radius.shape)
        self.powder_group['px_radius'].write_direct(px_radius)
        
        self.powder_group['intensity'].dims.create_scale(self.powder_group['px_radius'], 'radius [px]')
        self.powder_group['intensity'].dims[0].attach_scale(self.powder_group['px_radius'])
        self.powder_group['intensity'].dims[1].attach_scale(self.experimental_parameters_group['time_points'])
        
        self.powder_group['baseline'].resize(rintensity.shape)
        self.powder_group['baseline'].write_direct(np.zeros_like(rintensity))

        callback(100)
