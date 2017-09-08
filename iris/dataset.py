"""
@author: Laurent P. Rene de Cotret
"""
from collections import namedtuple
from contextlib import suppress
from functools import lru_cache, partial
from math import sqrt
from os import listdir
from os.path import isdir, isfile, join
from tempfile import TemporaryFile
from warnings import warn

import h5py
import numpy as np
from cached_property import cached_property
from scipy.signal import detrend
from skimage.io import imread

from npstreams import iaverage, ipipe, last, peek, pmap_unordered, ipipe
from skued import electron_wavelength
from skued.baseline import baseline_dt, dt_max_level
from skued.image import azimuthal_average, ialign

# Centralized location where possible metadata is
# listed. For convenience, we also list the units
# Note that HDF5 cannot deal with None, so default cannot
# be None
DatasetMetadata = namedtuple('DatasetMetadata', ['type', 'default', 'units'])
VALID_DATASET_METADATA = {'fluence':         DatasetMetadata(float, default = 0.0, units = 'mj/cm^2'),
                          'energy':          DatasetMetadata(float, default = 90, units = 'keV'),
                          'time_zero_shift': DatasetMetadata(float, default = 0.0, units = 'ps'),
                          'nscans':          DatasetMetadata(tuple, default = (1,), units = 'count'),
                          'acquisition_date':DatasetMetadata(str, default = '', units = None),
                          'temperature':     DatasetMetadata(float, default = 293, units = 'K'),
                          'exposure':        DatasetMetadata(float, default = 0, units = 's'),
                          'notes':           DatasetMetadata(str, default = '', units = None),
                          'pixel_width':     DatasetMetadata(float, default = 14e-6, units = 'm'),
                          'camera_distance': DatasetMetadata(float, default = 0.2235, units = 'm')}

VALID_POWDER_METADATA = {'center':              DatasetMetadata(tuple, default = (0,0), units = 'px'),
                         'angular_bounds':      DatasetMetadata(tuple, default = (0, 360), units = 'deg'),
                         # Powder baseline attributes
                         'first_stage':      DatasetMetadata(str, default = '', units = None),
                         'wavelet':          DatasetMetadata(str, default = '', units = None),
                         'level':            DatasetMetadata(int, default = 0, units = None),
                         'niter':            DatasetMetadata(int, default = 0, units = None)}

class DatasetMetadataDescriptor(object):
    """ Descriptor to experimental parameters. """
    def __init__(self, name, output, default = None):
        """ 
        Parameters
        ----------
        name : str
        output : callable
            Callable to format output.
            e.g. numpy.array, tuple, float, ...
        default : object or None, optional
            Default value returned. If None, exception is raised.
        """
        self.name = name
        self.output = output
        self.default = default
    
    def __get__(self, instance, cls):
        value = instance.experimental_parameters_group.attrs.get(self.name, default = self.default)
        return self.output(value) if value is not None else None
    
    def __set__(self, instance, value):
        if (value is None) and (self.default is not None):
            value = self.default
        instance.experimental_parameters_group.attrs[self.name] = value
    
    def __delete__(self, instance):
        del instance.experimental_parameters_group.attrs[self.name]

def _raw_combine(raw, valid_scans, align, timedelay):
    order = list(raw.time_points).index(timedelay)
    images = map(partial(raw.raw_data, timedelay), valid_scans)
    pipe = iaverage(ialign(images)) if align else iaverage(images)
    return order, last(pipe)

class DiffractionDataset(h5py.File):
    """
    Abstraction of an HDF5 file to represent diffraction datasets.
    """
    _diffraction_group_name = '/processed'
    _exp_params_group_name = '/'

    valid_metadata = frozenset(VALID_DATASET_METADATA.keys())

    def __init__(self, *args, **kwargs):           
        super().__init__(*args, **kwargs)

        # Dynamically set attributes
        for name, metadata in VALID_DATASET_METADATA.items():
            setattr(type(self), name, DatasetMetadataDescriptor(name, metadata.type, metadata.default))

    def __repr__(self):
        return '< DiffractionDataset object. \
                  Sample type: {}, \n \
                  Acquisition date : {}, \n \
                  fluence {} mj/cm**2 >'.format(self.sample_type,self.acquisition_date, self.fluence)
    
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
            The dictionary can contain the following keys:

            * ``'fluence'`` (`float`): photoexcitation fluence [mJ/cm^2]
            * ``'energy'`` (`float`): electron energy [keV]
            * ``'time_zero_shift'`` (`float`): Shift between time-zero and points in `time_points`.
            * ``'notes'`` (`str`): User notes. Can be any string
            * ``'acquisition_date'`` (`str`): Dataset acquisition date. Can be any string
            * ``'nscans'`` (`iterable`): Scans from which the dataset was assembled, e.g. ``(1,2,3,4,5,10)``.
            * ``'current'`` (`float`): Electron beam current [pA].
            * ``'exposure'`` (`float`): Diffraction pattern integration time, or photographic exposure [s].
            * ``'temperature'`` (`float`): sample temperature [K].
            * ``'time_zero_shift'`` (`float`): time-zero shift between the input `time_points` and the 'real' time-zero [ps].
            * ``'pixel_width'`` (`float`): width of a single pixel [m].
            * ``'camera_distance'`` (`float`): distance between the electron detector and sample [m].
        
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
        patterns = iter(patterns)

        if callback is None: 
            callback = lambda _: None

        if 'mode' not in kwargs:
            kwargs['mode'] = 'x'    #safest mode

        time_points = np.array(time_points).reshape(-1)

        # Fill-in missing metadata with default
        for key in (set(VALID_DATASET_METADATA.keys()) - set(metadata.keys())):
            metadata[key] = VALID_DATASET_METADATA[key].default

        if ckwargs is None:
            ckwargs = {'compression': 'lzf', 'shuffle': True, 'fletcher32': True}
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
                setattr(file, key, val)
            
            # Record time-points as a dataset; then, changes to it will be reflected
            # in other dimension scales
            gp = file.experimental_parameters_group
            times = gp.create_dataset('time_points', data = time_points, dtype = np.float)
            mask = gp.create_dataset('valid_mask', data = valid_mask, dtype = np.bool)

            pgp = file.diffraction_group
            dset = pgp.create_dataset(name = 'intensity', shape = resolution + (len(time_points), ), 
                                      dtype = dtype, **ckwargs)
            pgp.create_dataset(name = 'diff_eq', shape = resolution, dtype = dtype, fillvalue = 0.0, **ckwargs)
            
            # Making use of the H5DS dimension scales
            # http://docs.h5py.org/en/latest/high/dims.html
            dset.dims.create_scale(times, 'time-delay')
            dset.dims[2].attach_scale(times)

            for index, pattern in enumerate(patterns):
                dset.write_direct(pattern, source_sel = np.s_[:,:], dest_sel = np.s_[:,:,index])
                callback(round(100 * index / np.size(time_points)))

        callback(100)
        return cls(filename)

    @classmethod
    def from_raw(cls, raw, filename, exclude_scans = set([]), valid_mask = None, 
                 processes = 1, callback = None, align = True, ckwargs = dict(), 
                 dtype = None, **kwargs):
        """
        Create a DiffractionDataset from a subclass of RawDatasetBase.

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
            Callable that takes an int between 0 and 99. This can be used for progress update when
            ``patterns`` is a generator and involves large computations.
        align : bool, optional
            If True (default), raw images will be aligned on a per-scan basis.
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
        TypeError: if ``raw`` is not an instance of RawDatasetBase
        """
        if callback is None: 
            callback = lambda _: None

        valid_scans = tuple(set(raw.nscans) - set(exclude_scans))
        metadata = raw.metadata.copy()
        metadata['nscans'] = valid_scans
        ntimes = len(raw.time_points)

        kwargs.update({'ckwargs': ckwargs, 
                       'valid_mask': valid_mask, 
                       'metadata': metadata,
                       'time_points': raw.time_points,
                       'dtype': dtype})
        
        # It is much safer and faster to first store reduced data into
        # a numpy array
        # TODO: memmap?
        stack = np.empty(shape = raw.resolution + (ntimes,), dtype = dtype)

        # TODO: include dtype in _raw_combine
        reduced = pmap_unordered(_raw_combine, raw.time_points, 
                                 args = (raw, valid_scans, align), 
                                 processes = processes,
                                 ntotal = ntimes)
        
        callback(0)
        for progress, (index, pattern) in enumerate(reduced, start = 1):
            stack[:,:,index] = pattern
            callback(round(100 * progress / ntimes))  

        # We squeeze all patterns because numpy.split() doesn't
        # remove the 3rd dimensions (axis = 2)
        patterns = np.split(stack, ntimes, axis = 2)
        return cls.from_collection(patterns = ipipe(np.ascontiguousarray, np.squeeze, patterns), 
                                   filename = filename, **kwargs)
        
    @property
    def metadata(self):
        """ Dictionary of the dataset's metadata """
        metadata = dict()
        for attr in (VALID_DATASET_METADATA.keys() | {'filename', 'time_points'}):
            metadata[attr] = getattr(self, attr)
        metadata.update(self.compression_params)
        return metadata
    
    @cached_property
    def valid_mask(self):
        """ Array that evaluates to True on valid pixels (i.e. not on beam-block, not hot pixels) """
        return np.array(self.experimental_parameters_group['valid_mask'])

    @property
    def time_points(self):
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
            time_index = int(np.argwhere(np.array(self.time_points) == float(timedelay)))
        except TypeError:
            time_index = np.argmin(np.abs(np.array(self.time_points) - float(timedelay)))
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

class PowderDiffractionDataset(DiffractionDataset):
    """ 
    Abstraction of HDF5 files for powder diffraction datasets.
    """
    _powder_group_name = '/powder'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for name, metadata in VALID_POWDER_METADATA.items():
            setattr(type(self), name, DatasetMetadataDescriptor(name, metadata.type, metadata.default))        

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
    def metadata(self):
        powder_meta = dict()
        for attr in VALID_POWDER_METADATA:
            powder_meta[attr] = getattr(self, attr)
        powder_meta.update(super().metadata)
        return powder_meta

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
        return np.arctan(self.px_radius * self.pixel_width / self.camera_distance)

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
    
    def powder_time_series(self, qmin, qmax, bgr = False, relative = False, out = None):
        """
        Average intensity in a scattering angle range, over time.
        Diffracted intensity is integrated in the closed interval [qmin, qmax]

        Parameters
        ----------
        qmin : float
            Lower scattering vector bound [1/A]
        qmax : float
            Higher scattering vector bound [1/A]. 
        bgr : bool, optional
            If True, background is removed. Default is False.
        relative : bool, optional
            If True, data is returned relative to the average of all diffraction patterns
            before photoexcitation.
        out : ndarray or None, optional
            1-D ndarray in which to store the results. The shape
            should be compatible with (len(time_points),)
        
        Returns
        -------
        out : ndarray, shape (N,)
            Average diffracted intensity over time.
        """
        i_min, i_max = np.argmin(np.abs(qmin - self.scattering_vector)), np.argmin(np.abs(qmax - self.scattering_vector))
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
    
    def compute_angular_averages(self, center = None, normalized = True, angular_bounds = None, 
                                 trim = True, callback = None):
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
        trim : bool, optional
            If True (default), leading and trailing zeroes due to pixel mask are trimmed.
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