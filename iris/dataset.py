"""
@author: Laurent P. Rene de Cotret
"""
import h5py
import numpy as np
from os.path import join
from scipy.signal import detrend
from skimage.io import imread
from skued.image_analysis import angular_average
from skued.baseline import baseline_dt, dt_max_level
from tempfile import gettempdir

from .utils import scattering_length
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
        instance.experimental_parameters_group.attrs[self.name] = value
    
    def __delete__(self, instance):
        del instance.experimental_parameters_group.attrs[self.name]

class DiffractionDataset(h5py.File):
    """
    Abstraction of an HDF5 file to represent diffraction datasets.
    """

    _processed_group_name = '/processed'
    _exp_params_group_name = '/'
    _pumpoff_pictures_group_name = '/pumpoff'
    
    # Experimental parameters as descriptors
    # TODO: use self.attrs.modify to ensure type?
    #       http://docs.h5py.org/en/latest/high/attr.html
    nscans = ExperimentalParameter('nscans', tuple)
    time_points = ExperimentalParameter('time_points', tuple)
    acquisition_date = ExperimentalParameter('acquisition_date', str, default = '')
    fluence = ExperimentalParameter('fluence', float)
    current = ExperimentalParameter('current', float)
    exposure = ExperimentalParameter('exposure', float)
    energy = ExperimentalParameter('energy', float, default = 90)
    resolution = ExperimentalParameter('resolution', tuple, default = (2048, 2048))
    sample_type = ExperimentalParameter('sample_type', str)
    time_zero_shift = ExperimentalParameter('time_zero_shift', float, default = 0.0)
    notes = ExperimentalParameter('notes', str, default = '')

    def __init__(self, name, mode = 'r', **kwargs):
        """
        Keyword arguments are passed to the h5py.File constructor.

        Parameters
        ----------
        name : str
        mode : str, {'w', 'r' (default), 'r+', 'a', 'w-'}
        """
        super().__init__(name = name, mode = mode, **kwargs)
    
    def __repr__(self):
        return '< DiffractionDataset object. \
                  Sample type: {}, \n \
                  Acquisition date : {}, \n \
                  fluence {} mj/cm**2 >'.format(self.sample_type,self.acquisition_date, self.fluence)
    
    @property
    def metadata(self):
        """ Dictionary of the dataset's metadata """
        return dict(self.attrs.items())
    
    @cached_property
    def valid_mask(self):
        """ Array that evaluates to True on valid pixels (i.e. not on beam-block, not hot pixels) """
        try: 
            return np.array(self.experimental_parameters_group['valid_mask'])
        except: #Legacy
            x1,x2,y1,y2 = self.beamblock_rect
            valid_mask = np.ones(self.resolution, dtype = np.bool)
            valid_mask[y1:y2, x1:x2] = False
            return valid_mask
    
    @property
    def corrected_time_points(self):
        """ Time points corrected for time-zero shift. """
        return tuple(np.array(self.time_points) - self.time_zero_shift)
    
    def shift_time_zero(self, shift):
        """
        Shift time-zero uniformly across time-points.

        Parameters
        ----------
        shift : float
            Shift [ps].
        """
        self.time_zero_shift = shift
    
    def approx_time_point(self, timedelay):
        """ Returns the closest available time-point 
        
        Parameters
        ----------
        timdelay : float
            Time-delay [ps]
        
        Returns
        -------
        tp : float
            Time-point closest to `timedealay` [ps]
        """
        index = np.argmin(np.abs(np.array(self.time_points) - timedelay))
        return self.time_points[index]

    def averaged_data(self, timedelay, out = None):
        """
        Returns data at a specific time-delay.

        Parameters
        ----------
        timdelay : float or None
            Timedelay [ps]. If None, the entire block is returned.
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
            # time_index cannot be cast to int() if np.argwhere returns an empty array
            # catch the corresponding TypeError
            try:
                time_index = int(np.argwhere(np.array(self.corrected_time_points) == float(timedelay)))
            except TypeError:
                potential_timedelay_index = np.argmin(np.abs(np.array(self.corrected_time_points) - float(timedelay)))
                potential_timedelay = self.corrected_time_points[potential_timedelay_index]
                raise ValueError('Time-delay {} ps does not exist. Did you mean {} ps?'.format(timedelay, potential_timedelay))

            if out is None:
                out = np.empty(self.resolution)
            dataset.read_direct(out, source_sel = np.s_[:,:, time_index], dest_sel = np.s_[:,:])

        return out

    def averaged_error(self, timedelay, out = None):
        """ 
        Returns error in measurement.

        Parameters
        ----------
        timdelay : float or None
            Timedelay [ps]. If None, the entire block is returned.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.
        
        Returns
        -------
        arr : ndarray or None
            Time-delay error. If out is provided, None is returned.

        Raises
        ------
        ValueError
            If timedelay does not exist.
        """
        dataset = self.processed_measurements_group['error']

        if timedelay is None:
            if out is None:
                out = np.empty_like(dataset)
            dataset.read_direct(out)

        else:
            # time_index cannot be cast to int() if np.argwhere returns an empty array
            # catch the corresponding TypeError
            try:
                time_index = int(np.argwhere(np.array(self.corrected_time_points) == float(timedelay)))
            except TypeError:
                potential_timedelay_index = np.argmin(np.abs(np.array(self.corrected_time_points) - float(timedelay)))
                potential_timedelay = self.corrected_time_points[potential_timedelay_index]
                raise ValueError('Time-delay {} does not exist.'.format(timedelay))

            if out is None:
                out = np.empty(self.resolution)
            dataset.read_direct(out, source_sel = np.s_[:,:, time_index], dest_sel = np.s_[:,:])
        
        return out
    
    def time_series(self, rect, out = None):
        """
        Integrated intensity over time inside bounds.

        Parameters
        ----------
        rect : 4-tuple of ints
            Bounds of the region in px.
        out : ndarray or None, optional
            1-D ndarray in which to store the results. The shape
            should be compatible with (len(time_points),)
        
        Returns
        -------
        out : ndarray, ndim 1
        """
        x1, x2, y1, y2 = rect
        data = self.processed_measurements_group['intensity'][y1:y2, x1:x2, :]  # Numpy axes are transposed
        return np.sum(data, axis = (0,1), out = out)

    def pumpoff_data(self, scan, out = None):
        """
        Returns a pumpoff picture from a specific scan.

        Parameters
        ----------
        scan : int or None
            If None, the entire block (i.e. for all scans) is returned.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.

        Returns
        -------
        arr : ndarray or None
            Pump-off picture. If out is provided, None is returned.
        """
        #Scans start at 1, ndarray indices start at 0
        dataset = self.pumpoff_pictures_group['pumpoff_pictures']
        if scan:
            if out is None:
                out = np.empty(self.resolution)
            dataset.read_direct(out, source_sel = np.s_[:,:,scan - 1], dest_sel = np.s_[:,:])
        else:
            if out is None:
                out = np.empty_like(dataset)
            dataset.read_direct(out)
        
        return out
       
    @property
    def background_pumpon(self):
        return np.array(self.processed_measurements_group['background_pumpon'])
    
    @property
    def background_pumpoff(self):
        return np.array(self.processed_measurements_group['background_pumpoff'])
    
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

    center = ExperimentalParameter('center', tuple)
    first_stage = ExperimentalParameter(name = 'powder_wavelet_baseline_first_stage', output = str)
    wavelet = ExperimentalParameter(name = 'powder_baseline_wavelet', output = str)
    level = ExperimentalParameter(name = 'powder_baseline_transform_level', output = int)
    baseline_removed = ExperimentalParameter(name = 'powder_baseline_removed', output = bool, default = False)

    @property
    def powder_group(self):
        return self.require_group(self._powder_group_name)
    
    @property
    def scattering_length(self):
        return np.array(self.powder_group['scattering_length'])

    def powder_data(self, timedelay, bgr = False, out = None):
        """
        Returns the angular average data from scan-averaged diffraction patterns.

        Parameters
        ----------
        timdelay : float or None
            Time-delay [ps]. If None, the entire block is returned.
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
        dataset = self.powder_group['intensity']

        if timedelay is None:
            if out is None:
                out = np.empty_like(dataset)
            dataset.read_direct(out)

        else:
            try:
                time_index = np.argwhere(np.array(self.corrected_time_points) == float(timedelay))
            except TypeError:
                potential_timedelay_index = np.argmin(np.abs(np.array(self.corrected_time_points) - float(timedelay)))
                potential_timedelay = self.corrected_time_points[potential_timedelay_index]
                raise ValueError('Time-delay {} ps does not exist. Did you mean {} ps?'.format(timedelay, potential_timedelay))
            if out is None:
                out = np.empty_like(self.scattering_length)
            dataset.read_direct(out, source_sel = np.s_[time_index,:], dest_sel = np.s_[:])

        if bgr:
            out -= self.baseline(timedelay)
        return out     
    
    def powder_error(self, timedelay, out = None):
        """
        Returns the angular average error from scan-averaged diffraction patterns.

        Parameters
        ----------
        timdelay : float or None
            Time-delay [ps]. If None, the entire block is returned.
        out : ndarray or None, optional
            If an out ndarray is provided, h5py can avoid
            making intermediate copies.
        
        Returns
        -------
        out : ndarray, shape (N,)
            Error in diffracted intensity [counts].
        """
        dataset = self.powder_group['error']

        if timedelay is None:
            if out is None:
                out = np.empty_like(dataset)
            dataset.read_direct(out)
        
        else:
            try:
                time_index = np.argwhere(np.array(self.corrected_time_points) == float(timedelay))
            except TypeError:
                potential_timedelay_index = np.argmin(np.abs(np.array(self.corrected_time_points) - float(timedelay)))
                potential_timedelay = self.corrected_time_points[potential_timedelay_index]
                raise ValueError('Time-delay {} ps does not exist. Did you mean {} ps?'.format(timedelay, potential_timedelay))
            if out is None:
                out = np.empty_like(self.scattering_length)
            dataset.read_direct(out, source_sel = np.s_[time_index,:], dest_sel = np.s_[:])  
        return out

    def baseline(self, timedelay, out = None):
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
        if not self.baseline_removed:
            return np.zeros_like(self.scattering_length)

        dataset = self.powder_group['baseline']

        if timedelay is None:
            if out is None:
                out = np.empty_like(dataset)
            dataset.read_direct(out)
        
        else:
            try:
                time_index = np.argwhere(np.array(self.corrected_time_points) == float(timedelay))
            except TypeError:
                potential_timedelay_index = np.argmin(np.abs(np.array(self.corrected_time_points) - float(timedelay)))
                potential_timedelay = self.corrected_time_points[potential_timedelay_index]
                raise ValueError('Time-delay {} ps does not exist. Did you mean {} ps?'.format(timedelay, potential_timedelay))
            if out is None:
                out = np.empty_like(self.scattering_length)
            dataset.read_direct(out, source_sel = np.s_[time_index,:], dest_sel = np.s_[:]) 
        
        return out
    
    def powder_time_series(self, smin, smax, bgr = False, out = None):
        """
        Integrated intensity in a scattering angle range, over time.
        Diffracted intensity is integrated in the closed interval [smin, smax]

        Parameters
        ----------
        smin : float
            Lower scattering angle bound [rad/A]
        smax : float
            Higher scattering angle bound [rad/A]. 
        bgr : bool, optional
            If True, background is removed. Default is False.
        out : ndarray or None, optional
            1-D ndarray in which to store the results. The shape
            should be compatible with (len(time_points),)
        
        Returns
        -------
        out : ndarray, shape (N,)
            Integrated diffracted intensity over time.
        """
        # Python slices are semi-open by design, therefore i_max + 1 is used
        # so that the integration interval is closed.
        i_min, i_max = np.argmin(np.abs(smin - self.scattering_length)), np.argmin(np.abs(smax - self.scattering_length))
        trace = np.array(self.powder_group['intensity'][:, i_min:i_max + 1])
        if bgr :
            trace -= np.array(self.powder_group['baseline'][:, i_min:i_max + 1])
        
        if out is not None:
            return np.sum(axis = 1, out = out)
        return np.squeeze(np.sum(trace, axis = 1))
    
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
        if 'baseline' not in self.powder_group:
            self.powder_group.create_dataset(name = 'baseline', data = baseline, 
                                             maxshape = (len(self.time_points), max(self.resolution)),
                                             **self.compression_params)
        else:
            self.powder_group['baseline'].resize(baseline.shape)
            self.powder_group['baseline'].write_direct(baseline)
        
        if level == None:
            level = dt_max_level(data = self.scattering_length, first_stage = first_stage, wavelet = wavelet)
            
        self.level = level
        self.first_stage = first_stage
        self.wavelet = wavelet
        self.baseline_removed = True
    
    def compute_angular_averages(self, center = None, callback = None):
        """ 
        Compute the angular averages.
        
        Parameters
        ----------
        center : 2-tuple or None, optional
            Center of the diffraction patterns. If None (default), the dataset
            attribute will be used instead.
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
        
        self.sample_type = 'powder'

        # Because it is difficult to know the angular averaged data's shape in advance, 
        # we calculate it first and store it next
        callback(0)
        results = list()
        extras = dict()
        for index, timedelay in enumerate(self.time_points):
            extras.clear()
            radius, avg = angular_average(self.averaged_data(timedelay), center = self.center, 
                                          mask = np.logical_not(self.valid_mask), extras = extras) 
            results.append((radius, avg, extras['error']))
            callback(int(100*index / len(self.time_points)))
        
        # Concatenate arrays for intensity and error
        rintensity = np.stack([I for _, I, _ in results], axis = 0)
        rerror =  np.stack([e for _, _, e in results], axis = 0)
        
        # We allow resizing. In theory, an angular averave could never be longer than the resolution
        maxshape = (len(self.time_points), max(self.resolution))
        if 'intensity' not in self.powder_group:
            # We allow resizing. In theory, an angualr averave could never be longer than the resolution
            # Only chunked datasets are resizeable
            self.powder_group.create_dataset(name = 'intensity', data = rintensity, maxshape = maxshape, chunks = True)
            self.powder_group.create_dataset(name = 'error', data = rerror,  maxshape = maxshape, chunks = True)
        else:
            self.powder_group['intensity'].resize(rintensity.shape)
            self.powder_group['intensity'].write_direct(rintensity)
            self.powder_group['error'].resize(rerror.shape)
            self.powder_group['error'].write_direct(rerror)
        
        # TODO: variable pixel_width and camera distance in the future
        px_radius = results[0][0]
        s_length = scattering_length(px_radius, energy = self.energy)
        if 'scattering_length' not in self.powder_group:
            self.powder_group.create_dataset(name = 'scattering_length', data = s_length, 
                                             maxshape = (max(self.resolution),), chunks = True)
        else:
            self.powder_group['scattering_length'].resize(s_length.shape)
            self.powder_group['scattering_length'].write_direct(s_length)

        self.baseline_removed = False   # baseline is not valid anymore

        callback(100)

class SinglePictureDataset(DiffractionDataset):

    nscans = (1,)
    time_points = (0,)
    acquisition_date = ''
    fluence = 0.0
    current = 0.0
    exposure = 0.0
    energy = 90
    sample_type = 'single_crystal'
    time_zero_shift = 0.0
    notes = ''

    def __init__(self, path):
        """ 
        Create a single-picture dataset from an image path
        
        Parameters
        ----------
        path : str or path-like object
            Absolute path to the image.
        """
        self.path = path
        self.image = imread(path).astype(np.float)

    def __repr__(self):
        return "< Single picture DiffractionDataset instance \n from file {}>".format(self.path)
    
    @property
    def metadata(self):
        return dict({'time_points': (0,), 'nscans': (1,)})

    @property
    def valid_mask(self):
        return np.ones_like(self.image, dtype = np.bool)

    @property
    def resolution(self):
        return self.image.shape
    
    def averaged_data(self, *args, **kwargs):
        return self.image
    
    def averaged_error(self, *args, **kwargs):
        return np.zeros_like(self.image)