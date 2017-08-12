"""
@author: Laurent P. Rene de Cotret
"""
from contextlib import suppress
from functools import partial
from math import sqrt
from os import listdir
from os.path import isdir, isfile, join
from warnings import warn

import h5py
import numpy as np
from scipy.signal import detrend
from skimage.io import imread

from npstreams import iaverage, ipipe, last, peek, pmap
from skued import electron_wavelength
from skued.baseline import baseline_dt, dt_max_level
from skued.image import angular_average, ialign

from .optimizations import cached_property


class ExperimentalParameter(object):
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

class DiffractionDataset(h5py.File):
    """
    Abstraction of an HDF5 file to represent diffraction datasets.
    """
    _processed_group_name = '/processed'
    _exp_params_group_name = '/'

    required_metadata = frozenset({'fluence', 'energy', 'time_points'})
    optional_metadata = frozenset({'nscans', 'acquisition_date', 'temperature', 
                                   'exposure', 'time_zero_shift', 'notes',
                                   'pixel_width', 'camera_distance'})
    
    # Experimental parameters as descriptors
    # TODO: use self.attrs.modify to ensure type?
    #       http://docs.h5py.org/en/latest/high/attr.html
    nscans = ExperimentalParameter('nscans', tuple, default = (1,))
    acquisition_date = ExperimentalParameter('acquisition_date', str, default = '')
    fluence = ExperimentalParameter('fluence', float)
    current = ExperimentalParameter('current', float, 0)
    temperature = ExperimentalParameter('temperature', float, default = 293)
    exposure = ExperimentalParameter('exposure', float, 0)
    energy = ExperimentalParameter('energy', float, default = 90)
    time_zero_shift = ExperimentalParameter('time_zero_shift', float, default = 0.0)
    notes = ExperimentalParameter('notes', str, default = '')
    pixel_width = ExperimentalParameter('pixel_width', float, default = 14e-6)
    camera_distance = ExperimentalParameter('camera_distance', float, default = 0.2235)

    def __repr__(self):
        return '< DiffractionDataset object. \
                  Sample type: {}, \n \
                  Acquisition date : {}, \n \
                  fluence {} mj/cm**2 >'.format(self.sample_type,self.acquisition_date, self.fluence)
    
    @classmethod
    def from_collection(cls, patterns, filename, time_points, metadata, 
                        valid_mask = None, ckwargs = {'chunks': True}, 
                        fkwargs = {'mode': 'x'}, callback = None):
        """
        Create a DiffractionDataset from a collection of diffraction patterns and metadata.

        Parameters
        ----------
        patterns : iterable of ndarray
            Diffraction patterns. These should be in the same order as ``time_points``. Note that
            the iterable can be a generator, in which case it will be consumed.
        filename : str or path-like
            Path to the assembled DiffractionDataset.
        time_points : array_like, shape (N,)
            Time-points of the diffraction patterns, in picoseconds.
        metadata : dictionary
            The dictionary must contain the following keys:

            * fluence : float
            * energy : float

            The following keys are optional:

            * notes : str
            * acquisition_date : str
            * nscans : iterable of ints
            * current : float
            * exposure : float
            * temperature : float
            * time_zero_shift : float
            * pixel_width : float
            * camera_distance : float

            Any other key will not be recorded in the DiffractionDataset. 
        
        valid_mask : ndarray or None, optional
            Boolean array that evaluates to True on valid pixels. This information is useful in
            cases where a beamblock is used.
        ckwargs : dict, optional
            HDF5 compression keyword arguments. Refer to ``h5py``'s documentation for details.
        fkwargs : dict, optional
            File keyword-arguments. These keywords are passed to ``h5py.File`` constructor. 
            Default is file-mode 'x', which raises error if file already exists.
        callback : callable or None, optional
            Callable that takes an int between 0 and 99. This can be used for progress update when
            ``patterns`` is a generator and involves large computations.
        
        Returns
        -------
        dataset : DiffractionDataset

        Raises
        ------
        ValueError: if required metadata is not complete.
        """
        if callback is None: callback = lambda _: None
        time_points = np.array(time_points).reshape(-1)

        # Required metadata must be present, and cannot be None (since HDF5 cannot deal with it.)
        required_keys = set(cls.required_metadata) - set(['time_points'])
        try:
            assert required_keys <= metadata.keys()
        except AssertionError:
            raise ValueError('Required metadata {} not all present'.format(cls.required_metadata))

        first, patterns = peek(patterns)
        dtype = first.dtype
        resolution = first.shape

        if valid_mask is None:
            valid_mask = np.ones(first.shape, dtype = np.bool)
        
        with cls(filename, **fkwargs) as file:

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

            pgp = file.processed_measurements_group
            dset = pgp.create_dataset(name = 'intensity', shape = resolution + (len(time_points), ), 
                                     dtype = first.dtype, **ckwargs)
            
            # Making use of the H5DS dimension scales
            # http://docs.h5py.org/en/latest/high/dims.html
            dset.dims.create_scale(times, 'time-delay')
            dset.dims[2].attach_scale(times)
            
            for index, pattern in enumerate(patterns):
                dset.write_direct(pattern, source_sel = np.s_[:,:], dest_sel = np.s_[:,:,index])
                callback(round(100 * index / np.size(time_points)))

        return cls(filename)

    @classmethod
    def from_raw(cls, raw, filename, exclude_scans = set([]), valid_mask = None, 
                 processes = 1, callback = None, align = True, ckwargs = dict()):
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
        align : bool, optional
            If True (default), raw images will be aligned on a per-scan basis.
        ckwargs : dict or None, optional
            HDF5 compression keyword arguments. Refer to ``h5py``'s documentation for details.
        
        Returns
        -------
        dataset : DiffractionDataset

        Raises
        ------
        IOError : If the filename is already associated with a file.
        TypeError: if ``raw`` is not an instance of RawDatasetBase
        """
        valid_scans = tuple(set(raw.nscans) - set(exclude_scans))

        ipatterns = list()
        for time_point in raw.time_points:
            arrays = map(partial(raw.raw_data, time_point, bgr = True), valid_scans) 
            pipeline = ipipe(iaverage, ialign, arrays) if align else iaverage(arrays)
            ipatterns.append(pipeline)

        # TODO: align all patterns with each other?
        patterns = pmap(last, ipatterns, processes = processes)
        
        return cls.from_collection(patterns = patterns, 
                                   filename = filename,
                                   time_points = raw.time_points,
                                   metadata = raw.metadata, 
                                   valid_mask = valid_mask)
        
    @property
    def metadata(self):
        """ Dictionary of the dataset's metadata """
        metadata = dict()
        for attr in (self.required_metadata | self.optional_metadata | {'filename'}):
            metadata[attr] = getattr(self, attr)
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
        intensity_shape = self.processed_measurements_group['intensity'].shape
        return tuple(intensity_shape[0:2])
    
    def shift_time_zero(self, shift):
        """
        Shift time-zero uniformly across time-points.

        Parameters
        ----------
        shift : float
            Shift [ps]. A positive value of `shift` will move all time-points forward in time,
            whereas a negative value of `shift` will move all time-points backwards in time.
        """
        self.time_zero_shift = shift
        self.experimental_parameters_group['time_points'][:] = self.time_points + shift
    
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

    def equilibrium(self, out = None):
        """ 
        Returns the averaged diffraction pattern for all times before photoexcitation. 
        In case no data is available before photoexcitation, an array of zeros is returned.

        Time-zero can be adjusted using the ``shift_time_zero`` method.

        Parameters
        ----------
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.

        Returns
        -------
        I : ndarray, shape (N,)
            Diffracted intensity [counts]
        """
        dset = self.processed_measurements_group['intensity']
        t0_index = np.argmin(np.abs(self.time_points))
        b4t0_slice = dset[:, :, :t0_index]

        # If there are no available data before time-zero, np.mean()
        # will return an array of NaNs; instead, return zeros.
        if t0_index == 0:
            return np.zeros(shape = self.resolution, dtype = dset.dtype)
        
        return np.mean(b4t0_slice, axis = 2, out = out)

    def data(self, timedelay, relative = False, out = None):
        """
        Returns data at a specific time-delay.

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
        arr : ndarray or None
            Time-delay data. If out is provided, None is returned.
        
        Raises
        ------
        ValueError
            If timedelay does not exist.
        """
        dataset = self.processed_measurements_group['intensity']

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
            out -= self.equilibrium()

        return out
    
    def time_series(self, rect, out = None):
        """
        Integrated intensity over time inside bounds.

        Parameters
        ----------
        rect : 4-tuple of ints
            Bounds of the region in px. Bounds are specified as [row1, row2, col1, col2]
        out : ndarray or None, optional
            1-D ndarray in which to store the results. The shape
            should be compatible with ``(len(time_points),)``
        
        Returns
        -------
        out : ndarray, ndim 1
        """
        x1, x2, y1, y2 = rect
        data = self.processed_measurements_group['intensity'][x1:x2, y1:y2, :]
        return np.mean(data, axis = (0,1), out = out)
    
    @property
    def experimental_parameters_group(self):
        return self.require_group(name = self._exp_params_group_name)
    
    @property
    def processed_measurements_group(self):
        return self.require_group(name = self._processed_group_name)
    
    @property
    def pumpoff_pictures_group(self):
        return self.require_group(name = self._pumpoff_pictures_group_name)
    
    @cached_property
    def compression_params(self):
        """ Compression options in the form of a dictionary """
        dataset = self.processed_measurements_group['intensity']
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

    required_metadata = DiffractionDataset.required_metadata | frozenset({'center'})
    optional_metadata = DiffractionDataset.optional_metadata | frozenset({'first_stage', 'wavelet', 'level', 'niter', 'angular_bounds'}) 

    center = ExperimentalParameter('center', tuple)
    first_stage = ExperimentalParameter(name = 'powder_wavelet_baseline_first_stage', output = str, default = None)
    wavelet = ExperimentalParameter(name = 'powder_baseline_wavelet', output = str, default = None)
    level = ExperimentalParameter(name = 'powder_baseline_transform_level', output = int, default = 0)
    niter = ExperimentalParameter(name = 'powder_baseline_niter', output = int, default = 0)
    angular_bounds = ExperimentalParameter('powder_average_angular_bounds', tuple, default = (0, 360))

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
        return np.arctan(self.px_radius * self.pixel_width / self.camera_distance)

    @property
    def scattering_vector(self):
        """ Array of scattering vector norm :math:`|G|` [:math:`1/\AA`] """
        # Scattering vector norm is defined as G = 4 pi sin(theta)/wavelength
        # TODO: cache
        return 4*np.pi*np.sin(self.scattering_angle/2) / electron_wavelength(self.energy)

    def powder_equilibrium(self, bgr = False, out = None):
        """ 
        Returns the average powder diffraction pattern for all times before photoexcitation. 
        In case no data is available before photoexcitation, an array of zeros is returned.

        Parameters
        ----------
        bgr : bool
            If True, background is removed.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.

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
            return np.mean(b4t0_slice, axis = 0, out = out)
        
        bg = self.powder_group['baseline'][:t0_index, :]
        return np.mean(b4t0_slice - bg, axis = 0, out = out)

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
            out -= self.powder_equilibrium(bgr = bgr)

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
    
    def powder_time_series(self, qmin, qmax, bgr = False, out = None):
        """
        Average intensity in a scattering angle range, over time.
        Diffracted intensity is integrated in the closed interval [smin, smax]

        Parameters
        ----------
        qmin : float
            Lower scattering vector bound [1/A]
        qmax : float
            Higher scattering vector bound [1/A]. 
        bgr : bool, optional
            If True, background is removed. Default is False.
        out : ndarray or None, optional
            1-D ndarray in which to store the results. The shape
            should be compatible with (len(time_points),)
        
        Returns
        -------
        out : ndarray, shape (N,)
            Average diffracted intensity over time.
        """
        # Python slices are semi-open by design, therefore i_max + 1 is used
        # so that the integration interval is closed.
        i_min, i_max = np.argmin(np.abs(smin - self.scattering_vector)), np.argmin(np.abs(smax - self.scattering_vector))
        trace = np.array(self.powder_group['intensity'][:, i_min:i_max + 1])
        if bgr :
            trace -= np.array(self.powder_group['baseline'][:, i_min:i_max + 1])
        
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
    
    def compute_angular_averages(self, center = None, normalized = True, angular_bounds = None, callback = None):
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
            radius, avg = angular_average(self.data(timedelay), 
                                          center = self.center, 
                                          mask = np.logical_not(self.valid_mask), 
                                          angular_bounds = angular_bounds)
            results.append((radius, avg))
            callback(int(100*index / len(self.time_points)))
        
        # Concatenate arrays for intensity and error
        rintensity = np.stack([I for _, I in results], axis = 0)

        if normalized:
            rintensity /= np.sum(rintensity, axis = 1, keepdims = True)
        
        # We allow resizing. In theory, an angular averave could never be 
        # longer than the diagonal of resolution
        maxshape = (len(self.time_points), sqrt(2*max(self.resolution)**2))
        if 'intensity' not in self.powder_group:
            # We allow resizing. In theory, an angualr averave could never be longer than the resolution
            # Only chunked datasets are resizeable
            self.powder_group.create_dataset(name = 'intensity', data = rintensity, maxshape = maxshape)
        else:
            self.powder_group['intensity'].resize(rintensity.shape)
            self.powder_group['intensity'].write_direct(rintensity)
        
        # We store the raw px radius and infer other measurements (e.g. diffraction angle) from it
        px_radius = results[0][0]
        if 'px_radius' not in self.powder_group:
            self.powder_group.create_dataset(name = 'px_radius', data = px_radius, maxshape = (maxshape[-1],))
        else:
            self.powder_group['px_radius'].resize(px_radius.shape)
            self.powder_group['px_radius'].write_direct(px_radius)
        
        self.powder_group['intensity'].dims.create_scale(self.powder_group['px_radius'], 'radius [px]')
        self.powder_group['intensity'].dims[0].attach_scale(self.powder_group['px_radius'])
        self.powder_group['intensity'].dims[1].attach_scale(self.experimental_parameters_group['time_points'])
        
        if 'baseline' not in self.powder_group:
            self.powder_group.create_dataset(name = 'baseline', shape = rintensity.shape, 
                                             maxshape = maxshape, fillvalue = 0.0)
        else:
            self.powder_group['baseline'].resize(rintensity.shape)
            self.powder_group['baseline'].write_direct(np.zeros_like(rintensity))

        callback(100)


class SinglePictureDataset: pass
