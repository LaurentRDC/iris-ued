# -*- coding: utf-8 -*-
"""
Diffraction Dataset container object
------------------------------------
This object is meant to be an abstraction of a processed folder. It contains
the averaged images, backgrounds, substrate diffraction pattern, experimental 
parameters, etc.

@author: Laurent P. René de Cotret
"""
import numpy as n
import h5py
import shutil

from .pattern import Pattern
from .hough import diffraction_center
from .io import read, save, cast_to_16_bits, ImageNotFoundError
from .raw import RawDataset

import os.path

class cached_property(object):
    """
    Decorator that minimizes computations of class attributes by caching
    the attribute values if it ever accessed. Attrbutes are calculated once.
    
    This descriptor should be used for computationally-costly attributes that
    don't change much.
    """
    _missing = object()
    
    def __init__(self, attribute, name = None):      
        self.attribute = attribute
        self.__name__ = name or attribute.__name__
    
    def __get__(self, instance, owner = None):
        if instance is None:
            return self
        value = instance.__dict__.get(self.__name__, self._missing)
        if value is self._missing:
            value = self.attribute(instance)
            instance.__dict__[self.__name__] = value
        return value

def electron_wavelength(kV, units = 'meters'):
    """ 
    Returns the relativistic wavelength of an electron.
        
    Parameters
    ----------
    kV : float
        Voltage in kilovolts of the instrument
    units : str, optional
        Acceptable units are 'meters' or 'm', and 'angstroms' or 'A'.
    """
    m = 9.109*10**(-31)     #in kg
    e = 1.602*10**(-19)     #in C
    c = 299792458*(10**10)  #in m/s
    h = 6.63*10**(-34)      #in J*s
    V = kV * 1000
    
    #from: en.wikipedia.org/wiki/Electron_diffraction#Wavelength_of_electrons
    wavelength_meters = n.sqrt((h**2*c**2)/(e*V*(2*m*c**2+e*V)))
    if units == 'meters' or units == 'm':
        return wavelength_meters
    elif units == 'angstroms' or units.lower() == 'a':
        return wavelength_meters*(10**10)
    else:
        raise ValueError('Invalid units')

def scattering_length(radius, energy, pixel_width = 14e-6, camera_distance = 0.2235):
    """
    Returns the scattering length s = G/4pi for an array of radius data in pixels.
    
    Parameters
    ----------
    radius : array-like, shape (N,)
        Radius from center of diffraction pattern, in units of pixels.
    pixel_width : numerical
        CCD pixel width in meters.
    energy : numerical
        Electron energy in keV.
        
    Notes
    -----
    Default values for pixel width and camera distance correspond to experimental
    values for the Siwick diffractometer as of April 2016.
    """
    
    radius = n.array(radius) * pixel_width
    diffraction_half_angle = n.arctan(radius/camera_distance)/2
    return n.sin(diffraction_half_angle)/electron_wavelength(energy, units = 'angstroms')

def radial_average(image, center, beamblock_rect, mask = None, return_error = False):
    """
    This function returns a radially-averaged pattern computed from a TIFF image.
    
    Parameters
    ----------
    image : ndarray
        image data from the diffractometer.
    center : array-like shape (2,) or None
        [x,y] coordinates of the center (in pixels). If None, the center of the image
        will be determined via the Hough transform.
    name : str
        String identifier for the output RadialCurve
    beamblock_rect : Tuple, shape (4,)
        Tuple containing x- and y-bounds (in pixels) for the beamblock mask
        mast_rect = (x1, x2, y1, y2)
    mask : ndarray or None, optional
        Array of booleans that evaluates to False on pixels that should be discarded.
        If None (default), all pixels are treated as valid (except for beamblock)
    return_error :  bool, optional
        If True, returns the square-root-N error on the counts.
        
    Returns
    -------
    radius : ndarray, shape (N,)
    intensity : ndarray, shape (N,)
    error : ndarray, shape (N,)
        Returned only if return_error is True
    
    Raises
    ------
    ValueError
        If 'mask' is an array of a different shape than the image shape.
    
    See also
    --------
    iris.hough.diffraction_center
        Find the center of polycrystalline diffraction images using the Hough transform.
    """    
    #preliminaries
    if mask is None:
        mask = n.ones_like(image, dtype = n.bool)
    elif mask.shape != image.shape:
        raise ValueError("'mask' array (shape {}) must have the same shape as 'image' (shape {})".format(mask.shape, image.shape))
    
    if center is None:
        center = diffraction_center(image, beamblock = beamblock_rect)
    
    image = image.astype(n.float)
    mask = mask.astype(n.bool)
    
    xc, yc = center     #Center coordinates
    x1, x2, y1, y2 = beamblock_rect     
    
    #Create meshgrid and compute radial positions of the data
    X, Y = n.meshgrid(n.arange(0, image.shape[0]), n.arange(0, image.shape[1]))
    R = n.sqrt( (X - xc)**2 + (Y - yc)**2 )
    
    #radii beyond r_max don't fit a full circle within the image
    image_edge_values = n.array([R[0,:], R[-1,:], R[:,0], R[:,-1]])
    r_max = image_edge_values.min()           #Maximal radius that fits completely in the image
    
    # Find the smallest circle that completely fits inside the mask rectangle
    r_min = min([ n.sqrt((xc - x1)**2 + (yc - y1)**2), n.sqrt((xc - x2)**2 + (yc - y2)**2) ])

    # Replace all values in the image corresponding to beamblock or other irrelevant
    # data by 0: this way, it will never count in any calculation because image
    # values are used as weights in numpy.bincount
    # Create a composite mask that uses the pixels mask, beamblock mask, and maximum/minimum
    # radii     
    composite_mask = n.array(mask, dtype = n.bool)      # start from the pixel mask
    composite_mask[R > r_max] = False
    composite_mask[R < r_min] = False
    composite_mask[x1:x2, y1:y2] = False
    image[ n.logical_not(composite_mask) ] = 0    # composite_mask is false where pixels should be disregarded
    
    #Radial average
    px_bin = n.bincount(R.ravel().astype(n.int), weights = image.ravel())
    r_bin = n.bincount(R.ravel().astype(n.int))  
    radial_intensity = px_bin/r_bin
    
    # Compute the error as square-root-N
    radial_intensity_error = n.sqrt(px_bin)/r_bin
    
    # Only return values with radius between r_min and r_max
    radius = n.unique(R.ravel().astype(n.int))      # Round up to integer number of pixel
    r_max_index = n.argmin(n.abs(r_max - radius))
    r_min_index = n.argmin(n.abs(r_min - radius))
    
    if return_error:
        return radius[r_min_index + 1:r_max_index], radial_intensity[r_min_index + 1:r_max_index], radial_intensity_error[r_min_index + 1:r_max_index]
    else:
        return radius[r_min_index + 1:r_max_index], radial_intensity[r_min_index + 1:r_max_index]



class DiffractionDataset(object):
    """ 
    Container object for Ultrafast Electron Diffraction Datasets from the Siwick 
    Research group.

    Attributes
    ----------
    directory : str
        Absolute path to the dataset directory
    mode : str, {'polycrystalline', 'single crystal'} or None
        Data type.
    
    Attributes (experimental parameters)
    ------------------------------------
    resolution : tuple of ints
        CCD resolution
    acquisition_date : str
        Acquisition date from the folder name as a string of the form: '2016.01.06.15.35'
    fluence : float
        Laser fluence in mJ/cm**2
    exposure : float
        Exposure time in seconds
    energy : float
        Electron energy in keV
    current : float
        Electron beam current. Not sure what units.
    time_points : list of strings
        List of data time points, taken from the name of the image files.
    
    Methods
    -------
    image
        Access time-delay averaged diffraction patterns
    
    image_series
        Access to the ensemble of time-delay averaged diffraction patterns
        in a single ndarray.
    
    pattern
    
    pattern_series
    
    baseline
    
    baseline_series
        
    intensity_noise
        RMS intensity of diffraction pictures before photoexcitation
    
    stability_diagnostic
        Plots the overall diffracted intensity of the pumpoff pictures, which
        are taken before every delay scans.
        
    Special methods
    ---------------
    DiffractionDataset[time] returns the diffraction image of time delay 'time'.
    'time' can be an integer, float, or string.
    
    """
    
    _time_group_template = 'timedelay_{}'
    _baseline_group_template = 'baseline_{}'
    
    def __init__(self, directory):
        """
        Parameters
        ----------
        directory : str
            Absolute path to the dataset directory
        """
        # Check that the directory is a 'processed' directory.
        if directory.endswith('processed'):
            self.directory = directory
        else:
            self.directory = os.path.join(directory, 'processed')
    
    @property
    def time_points(self):            
        # get time points. Filename look like:
        # data_timedelay_-10.00_pumpon.tif
        time_list = [f.split('_')[2] for f in self._time_filenames]
        time_list.sort(key = float)
        return time_list
        
    # Access to data ----------------------------------------------------------
    
    def image(self, time, reduced_memory = False):
        """
        Returns the image of the processed pictures.
        
        Parameters
        ----------
        time : str, numerical
            Time delay value of the image.
        reduced_memory : bool, optional
            If False (default), the image array is returned as array of floats.
            If True, image is returned as an array of int16.
            
        Notes
        -----
        reduced_memory = True is good for displaying, but not for computations.
        """
        time = str(float(time))
        try:
            image = read(os.path.join(self.directory, 'data_timedelay_{0}_average_pumpon.tif'.format(time)))
        except ImageNotFoundError:
            raise ImageNotFoundError('Available time points : {0}'.format(repr(self.time_points)))
        
        if reduced_memory:
            return cast_to_16_bits(image)
        else:
            return image
    
    def image_series(self, reduced_memory = False):
        """
        Returns a stack of time delay images, with time as the first axis.
        
        Parameters
        ----------
        reduced_memory : bool, optional
            If False (default), the image series array is returned as array of floats.
            If True, image series is returned as an array of int16. This is
            best for displaying, but introduces computation errors.
        
        Returns
        -------
        times : ndarray, shape (N,)
            Array of time-delay values
        series : ndarray, ndim 3
            Array where the first axis is time. 
        """
        images = list()
        for time in self.time_points:
            images.append(self.image(time, reduced_memory))
            
        return n.array(list(map(float, self.time_points))), n.array(images)
    
    # -------------------------------------------------------------------------
    # Properties read from the experimental parameters file
    
    def _read_experimental_parameter(self, key):
        """
        Reads an experimental parameter from the DiffractionDataset's
        experimental parameter file.
        
        Parameters
        ----------
        key : str
            Name of the parameter
        """
        with open(self._exp_params_filename, 'r') as exp_params:
            for line in exp_params:
                if line.startswith(key): 
                    value = line.split('=')[-1]
                    break
        
        value = value.replace(' ','')
        value = value.replace('s','')                   # For exposure values with units
        if key == 'Acquisition date': 
            return value.strip('\n')
        else:
            try:
                return float(value)
            except: #Value might be an invalid number. E.g. 'BLANK'
                return 0.0

    @property
    def resolution(self):
        # Get the shape of the first image in self._time_filenames
        fn = os.path.join(self.directory, self._time_filenames[0])
        return read(fn).shape
    
    @property
    def fluence(self):
        return self._read_experimental_parameter('Fluence')
    
    @property
    def current(self):
        return self._read_experimental_parameter('Current')
    
    @property
    def exposure(self):
        return self._read_experimental_parameter('Exposure')
    
    @property
    def energy(self):
        return self._read_experimental_parameter('Energy')
        
    @property    
    def acquisition_date(self):
        return self._read_experimental_parameter('Acquisition date')
        
    # Shortcut to files -------------------------------------------------------
    
    @property
    def _results_filename(self):
        return os.path.join(self.directory, 'results.hdf5')
    
    @property
    def _exp_params_filename(self):
        return os.path.join(self.directory, 'experimental_parameters.txt')
    
    @property
    def _pumpoff_filenames(self):
        return [os.path.join(self._pumpoff_directory, f) 
                for f in os.listdir(self._pumpoff_directory)
                if os.path.isfile(os.path.join(self._pumpoff_directory, f)) 
                and f.endswith('pumpoff.tif')]
                
    @property
    def _time_filenames(self):
        return [f for f in os.listdir(self.directory) 
                if os.path.isfile(os.path.join(self.directory, f)) 
                and f.startswith('data_timedelay_') 
                and f.endswith('_pumpon.tif')]
    
    @property
    def _pumpoff_background(self):
        return read(os.path.join(self.directory, 'background_average_pumpoff.tif'))
    
    @property
    def _pumpon_background(self):
        return read(os.path.join(self.directory, 'background_average_pumpon.tif'))

    @property
    def _pumpoff_directory(self):
        return os.path.join(self.directory, 'pumpoff pictures')
    
    @property
    def _substrate(self):
        return read(os.path.join(self.directory, 'substrate.tif'))
    
    # Noise analysis ----------------------------------------------------------
    
    def _pumpoff_intensity_stability(self):
        """ Plots the total intensity of the pumpoff pictures. """
        overall_intensity = list()
        for fn in self._pumpoff_filenames:
            overall_intensity.append( read(os.path.join(self._pumpoff_directory, fn)).sum() )
        return n.array(overall_intensity)/overall_intensity[0]

    # Master HDF5 results file ------------------------------------------------
    @property
    def master_file(self):
        """
        Returns
        -------
        master : h5py.File object
        """
        # Create file if it doesn't exist, otherwise open for read/write
        return h5py.File(self._results_filename, 'a', libver = 'latest')
    
    def _group(self, timedelay, name_template):
        """
        Returns HDF5 group of time-delay. Basis for time_group and baseline_group methods.
        
        Parameters
        ----------
        timedelay : str or numerical
            Pump-probe time-delay.
        name_template : str
            formatable string. E.g. 'timedelay_{}', 'background_{}'
        
        Returns
        -------
        group : HDF5.Group object
        """
        timedelay = str(float(timedelay))
        group_name = name_template.format(timedelay)
        if not group_name in self.master_file:
            return self.master_file.create_group(group_name)
        else:
            return self.master_file[group_name]
        
    def time_group(self, timedelay):
        """
        Returns the HDF5 group associated with a certain time. If the group doesn't exist,
        it will be created.
        
        Parameters
        ----------
        timedelay : str or numerical
            Pump-probe time-delay.
        
        Returns
        -------
        group : HDF5.Group object
        """
        return self._group(timedelay, self._time_group_template)
    
    def baseline_group(self, timedelay):
        """
        Returns the HDF5 group associated with a certain time. If the group doesn't exist,
        it will be created.
        
        Parameters
        ----------
        timedelay : str or numerical
            Pump-probe time-delay.
        
        Returns
        -------
        group : HDF5.Group object
        """
        return self._group(timedelay, self._baseline_group_template)
        
    def baseline(self, time):
        pass
        
    def baseline_series(self):
        """
        Returns a list of time-delay inelastic scattering background data.
        
        Returns
        -------
        patterns : list of Pattern objects
        """
        out = list()
        for time in self.time_points:
            out.append(self.baseline(time))
        return out
        
    def baseline_fit(self, positions, mode = 'dynamic', **kwargs):
        """
        Fits an inelastic scattering background to the data. The fits are applied 
        to each time point independently. Results are then exported to the master
        HDF5 file.
        
        Parameters
        ----------
        positions : list of floats or None
            x-data positions of where radial averages are known to be purely background. 
            If None, the positions will be automatically determined. For an unassisted
            fit, set positions = [].
        mode : str, {'dynamic' (default), 'static'}
            Background fit mode. If 'dynamic', an inelastic background is fit to each pattern
            independently. If 'static', the first inelastic background fit for the first time-point 
            is used for all time-points.
        
        *args : optional arguments
            Can be any argument accepted by iris.pattern.Pattern.baseline. Most
            useful are max_iter and wavelet.
        
        Raises
        ------
        ValueError 
            If mode argument is invalid
        """
        backgrounds = list()
        if mode == 'dynamic':
            for time in self.time_points:
                backgrounds.append( (time, self.pattern(time).baseline(background_regions = positions, level = 'max', **kwargs)) )
        elif mode == 'static':
            # TODO: have static background be the average of all time points before time-zero
            static_background = self.pattern(self.time_points[0]).baseline(background_regions = positions, level = 'max', **kwargs)
            for time in self.time_points:
                backgrounds.append( (time, static_background))
        else:
            raise ValueError("'mode' argument {} invalid. Possible values are ['dynamic', 'static']".format(mode))
        
        self.export(backgrounds, group = self.baseline_group)
    
    def _record_parameters(self):
        """
        Records experimental parameters in the master HDF5 file.
        
        Parameters
        ----------
        opened_file : h5py.File object or None, optional
            If None (default), the master HDF5 file will be opened and closed
            automatically after recording parameters.
        """
        with self.master_file as opened_file:
            opened_file.attrs['acquisition date'] = self.acquisition_date
            opened_file.attrs['resolution'] = self.resolution
            opened_file.attrs['fluence'] = self.fluence
            opened_file.attrs['energy'] = self.energy
            opened_file.attrs['exposure'] = self.exposure
            opened_file.attrs['current'] = self.current
        
    def export(self, results, group):
        """
        Parameters
        ----------
        results : list
            List of tuples containing a time delay (str) and a pattern (Pattern)
        group : callable
            Method that takes a single argument (timedelay). Results will be stored
            in this family of groups.
        
        Examples
        --------
        >>> d = DiffractionDataset(...)
        >>> background_results = zip(d.time_points, d.baseline_series)
        >>> d.export(background_results, group = d.baseline_group)
        
        Notes
        -----
        This function will overwrite existing radial averages.
        """   
        self._record_parameters()
        
        for (timedelay, pattern) in results:
            pattern.save(group(timedelay))


class SingleCrystalDiffractionDataset(DiffractionDataset):
    """
    Diffraction dataset of single-crystal diffraction data. Inherits from
    DiffractionDataset.
    """
    def __init__(self, directory):
        """
        Parameters
        ----------
        directory : str
            Absolute path to the dataset directory
        """
        super().__init__(directory)
    
    def master_file(self):
        return super().master_file(polycrystalline = False)




class PowderDiffractionDataset(DiffractionDataset):
    """
    Diffraction dataset of polycrystalline diffraction data. Inherits from 
    DiffractionDataset.
    
    Methods
    -------       
    radial_peak_dynamics
        Get time-delay vs. intensity data for regions of a radial pattern.
        
    radial_average
        Radial average of a time delay picture.
    
    radial_average_series
        Radial average of all time delay pictures, saved in HDF5 and MATLAB formats
    """
    def __init__(self, directory):
        """
        Parameters
        ----------
        directory : str
            Absolute path to the dataset directory
        """
        super().__init__(directory)
    
    @property
    def center(self):
        return tuple(self.master_file.attrs['center'])
    
    @center.setter
    def center(self, value):
        self.master_file.attrs['center'] = value
        
    @property
    def beamblock_rect(self):
        return tuple(self.master_file.attrs['beamblock_rect'])
    
    @beamblock_rect.setter
    def beamblock_rect(self, value):
        self.master_file.attrs['beamblock_rect'] = value
    
    def radial_average(self, time, center, mask_rect = None):
        """
        Radial average of an image.
        
        Parameters
        ----------
        time : str, numerical, optional
            Time delay value of the image.
        center : array-like, shape (2,)
            [x,y] coordinates of the center (in pixels). If None, the center of 
            the image will be determined via the Hough transform.
        mask_rect : Tuple, shape (4,)
            Tuple containing x- and y-bounds (in pixels) for the beamblock mask
            mast_rect = (x1, x2, y1, y2)
        
        Returns
        -------
        Pattern object, type 'polycrystalline'
        """
        self.center = center
        self.beamblock_rect = mask_rect if not None else (0,0,0,0)
        
        xdata, intensity, error = radial_average(self.image(time), center, mask_rect, return_error = True)
        # Change x-data from pixels to scattering length
        s = scattering_length(xdata, self.energy)
        return Pattern(data = [s, intensity], error = error, name = str(time))
    
    def radial_average_series(self, center, mask_rect = None):
        """
        Radially averages and exports time delay data.
        
        Parameters
        ----------
        center : array-like, shape (2,)
            [x,y] coordinates of the center (in pixels)
        mask_rect : Tuple, shape (4,), optional
            Tuple containing x- and y-bounds (in pixels) for the beamblock mask
            mast_rect = (x1, x2, y1, y2)
        """
        results = list()
        for time in self.time_points:
            pattern = self.radial_average(time, center, mask_rect)
            results.append( (time, pattern) )
        self.export(results, group = self.time_group)
    
    def pattern(self, time = None):
        """
        Returns the radially-averaged pattern.
        
        Parameters
        ----------
        time : str, numerical, or None, optional
            Time delay value of the image. If None, the earliest measured pattern
            is returned
        
        Notes
        -----
        This function depends on the existence of an HDF5 file containing radial
        averages.
        """
        if time is None:
            time = self.time_points[0]
        time = str(float(time))
        
        return Pattern.from_hdf5(self.time_group(time))
    
    def pattern_series(self, background_subtracted = False):
        """
        Returns a list of time-delay radially-averaged data.
        
        Parameters
        ----------
        background_subtracted : bool
            If True, radial patterns will be background-subtracted before being 
            returned. False by default.
        
        Returns
        -------
        patterns : list of Pattern objects
        """
        out = list()
        for time in self.time_points:
            if background_subtracted:
                out.append(self.pattern(time) - self.baseline(time))
            else:
                out.append(self.pattern(time))
        return out
        
    def baseline(self, time = None):
        """
        Returns the inelastic scattering background of the radially-averaged pattern.
        
        The inelastic background is computed for the average of radial patterns
        before photoexcitation; however, the background pattern is saved in each 
        HDF5 time group for flexibility. 
        
        Parameters
        ----------
        time : str, numerical, or None. optional
            Time delay value of the image. If None (default), the earlier background
            is returned.
        
        Notes
        -----
        This function depends on the existence of an HDF5 file containing radial
        averages.
        """
        if time is None:
            time = self.time_points[0]
        time = str(float(time))
        
        return Pattern.from_hdf5(self.baseline_group(time))
    
    # Operations --------------------------------------------------------------
        
    def radial_peak_dynamics(self, edge, edge2 = None, subtract_background = False, background_dynamics = False, return_error = False):
        """
        Returns a pattern corresponding to the time-dynamics of a location in the 
        diffraction patterns. Think of it as looking at the time-evolution
        of a diffraction peak. This can be for data or for the background fits
        
        Parameters
        ----------
        edge : float
            Edge value in terms of xdata
        edge2 : float, optional
            If not None (default), the peak value is integrated between edge and
            edge2.
        subtract_background : bool, optional
            If True, inelastic scattering background is subtracted from the intensity data 
            before integration. Default is False.
        background_dynamics : bool, optional
            If True, the radial peak dynamics in the background fit is returned. In this case,
            the subtract_background parameter is ignored
        return_error: bool, optional
            If True, a stack of radial intensity errors is returned. Default is False.
        
        Returns
        -------
        time_values : ndarray, shape (N,)
            Time-delay values as an array
        intensity_series : ndarray, shape (N,M)
            Array of intensities. Each column corresponds to a time-delay. 
            Intensities are normalized to the maximal intensity over time.
        error_series : ndarray, shape (N,M)
            Returned if return_error is True (default is False)
        """
        if background_dynamics:
            subtract_background = False
            patterns = self.baseline_series()
        else:
            patterns = self.pattern_series()    
        
        scattering_length = patterns[0].xdata
        intensity_series = n.vstack( tuple( [pattern.data for pattern in patterns] ))
        error_series =  n.vstack( tuple( [pattern.error for pattern in patterns] ))
        
        if subtract_background:
            background_series = n.vstack( tuple(bg_pattern.data for bg_pattern in self.baseline_series()))
        else:
            background_series = n.zeros_like(intensity_series)

        time_values = n.array(list(map(float, self.time_points)))
        index = n.argmin(n.abs(scattering_length - edge))
        
        if edge2 is None:
            intensities = intensity_series[:, index] - background_series[:, index]
            errors = error_series[:, index]
        else:
            index2 = n.argmin(n.abs(scattering_length - edge2))
            intensities = (intensity_series[:, index:index2] - background_series[:, index:index2]).mean(axis = 1)
            errors = error_series[:, index:index2].mean(axis=1)
        
        # Normalize intensities
        if return_error:
            return time_values, intensities/intensities.max(), errors/intensities.max()
        else:
            return time_values, intensities/intensities.max()
    

class PowderDiffractionDiagnosticDataset(h5py.File, PowderDiffractionDataset, RawDataset):
    
    _scan_template = 'nscan_{}'
    _time_template = 'time_{}'
    
    @classmethod
    def scan_to_group(cls, scan_number):
        return cls._scan_template.format(int(scan_number))
    
    @classmethod
    def time_to_group(cls, timedelay):
        return cls._time_template.format(str(float(timedelay)))
        
    def __init__(self, filename, directory):
        h5py.File.__init__(name = filename, mode = 'r+', liver = 'latest')
        RawDataset.__init__(directory)
        self._calculate()
    
    def _calculate(self):
        """
        Radially average all images in the underlying RawDataset
        """
        for scan in self.nscans:
            self.create_group(self.scan_to_group(scan))
            for timedelay in self.time_points:
                radius, intensity, error = radial_average(image = self.raw_image(timedelay, scan) - self.pumpon_background, 
                                                          center = self.center, beamblock_rect = self.beamblock_rect, return_error = True)
                radav = Pattern([scattering_length(radius, self.energy), intensity], error = error)
                
                group = self.create_group(self.time_to_group(scan))
                radav.save(group)
            
        self.attrs['calculated'] = True
        
    def group(self, timedelay, scan):
        return self[self.scan_to_group(scan)][self.time_to_group(timedelay)]
    
    def pattern(self, timedelay, scan):
        """
        Returns the radial average at a certain time-delay and scan number.
        """
        return Pattern.from_hdf5(self.group(timedelay, scan))
    
    def pattern_series(self, timedelays, scans):
        """
        Returns the Pattern objects associated with various timedelays or scans.
        
        Parameters
        ----------
        timedelays : list of floats
        
        scans : list of ints
        """


        
class SinglePictureDataset(PowderDiffractionDataset):
    """
    Dummy diffraction dataset to perform operations on single images.
    """
    def __init__(self, filename):
        from tempfile import TemporaryDirectory
        self.temporary_directory = TemporaryDirectory()

        directory = os.path.join(self.temporary_directory.name, 'processed')
        os.mkdir(path = directory)
        
        # Create the illusion of a full-fledged dataset, but with a single timepoint and picture.
        shutil.copyfile(filename, os.path.join(directory, 'data_timedelay_0.0_average_pumpon.tif'))
        
        resolution = read(filename).shape
        save(n.zeros(shape = resolution), os.path.join(directory, 'background_average_pumpon.tif'))
        save(n.zeros(shape = resolution), os.path.join(directory, 'background_average_pumpoff.tif'))
        save(n.zeros(shape = resolution), os.path.join(directory, 'substrate.tif'))
        
        # Go on with our day with this 'temporary' dataset
        super().__init__(directory = directory)
    
    def __del__(self):
        self.temporary_directory.cleanup()
    
    # Overload  properties as placeholder values
    @property
    def fluence(self):
        return 0.0
    
    @property
    def current(self):
        return 0.0
    
    @property
    def exposure(self):
        return 0.0
        
    @property
    def energy(self):
        return 90
        
    @property    
    def acquisition_date(self):
        return '0.0.0.0.0'

if __name__ == '__main__':
   directory = 'K:\\test_folder'
   d = PowderDiffractionDataset(directory)