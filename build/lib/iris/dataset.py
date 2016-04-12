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

#Batch processing libraries
from curve import Curve
import os.path
from glob import glob
from tifffile import imread, imsave

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
        
def cast_to_16_bits(array):
    """ 
    Returns an array in int16 format. Array values below 0 are cast to 0,
    and array values above (2**16) - 1 are cast to (2**16) - 1.
    
    Parameters
    ----------
    array : array-like
        Array to be cast into 16 bits.
    
    Returns
    -------
    out : ndarray, dtype numpy.int16
    """
    array = n.asarray(array)
    array[ array < 0] = 0
    array[ array > (2**16) - 1] = (2**16) - 1
    return n.around(array, decimals = 0).astype(n.int16)
    
def read(filename):
    """ 
    Returns a ndarray from an image filename. 
    
    Parameters
    ----------
    filename : str
        Absolute path to the file.
    
    Returns
    -------
    out : ndarray, dtype numpy.float
        Numpy array from the image.
    """
    return imread(filename).astype(n.float)
    
def save(array, filename):
    """ 
    Saves a ndarray to an image filename. 
    
    Parameters
    ----------
    array : ndarray
        Image as an array of any type. The array will be cast to 16 bits.
    filename : str
        Absolute path to the file.
    
    See also
    --------        
    cast_to_16_bits
        For casting rules.
    """
    array = cast_to_16_bits(array)
    imsave(filename, array)

def radial_average(image, center, mask_rect):
    """
    This function returns a radially-averaged pattern computed from a TIFF image.
    
    Parameters
    ----------
    image : ndarray
        image data from the diffractometer.
    center : array-like, shape (2,)
        [x,y] coordinates of the center (in pixels)
    name : str
        String identifier for the output RadialCurve
    mask_rect : Tuple, shape (4,)
        Tuple containing x- and y-bounds (in pixels) for the beamblock mask
        mast_rect = (x1, x2, y1, y2)
        
    Returns
    -------
    curve.Curve object
    """
    
    #preliminaries
    image = image.astype(n.float)
    xc, yc = center     #Center coordinates
    
    #Create meshgrid and compute radial positions of the data
    X, Y = n.meshgrid(n.arange(0, image.shape[0]), n.arange(0, image.shape[1]))
    R = n.sqrt( (X - xc)**2 + (Y - yc)**2 )
    
    #radii beyond r_max don't fit a full circle within the image
    image_edge_values = n.array([R[0,:], R[-1,:], R[:,0], R[:,-1]])
    r_max = image_edge_values.min()           #Maximal radius that fits completely in the image

    # Replace all values in the image corresponding to beamblock or other irrelevant
    # data by 0: this way, it will never count in any calculation because image
    # values are used as weights in numpy.bincount
    image[R > r_max] = 0
    x1, x2, y1, y2 = mask_rect
    image[x1:x2, y1:y2] = 0
    
    # Find the smallest circle that completely fits inside the mask rectangle
    r_min = min([ n.sqrt((xc - x1)**2 + (yc - y1)**2), n.sqrt((xc - x2)**2 + (yc - y2)**2) ])
    
    #Radial average
    px_bin = n.bincount(R.ravel().astype(n.int), weights = image.ravel())
    r_bin = n.bincount(R.ravel().astype(n.int))  
    radial_intensity = px_bin/r_bin
    
    # Only return values with radius below r_max
    radius = n.unique(R.ravel().astype(n.int))
    r_max_index = n.argmin(n.abs(r_max - radius))
    r_min_index = n.argmin(n.abs(r_min - radius))
    
    return (radius[r_min_index + 1:r_max_index], radial_intensity[r_min_index + 1:r_max_index])

class DiffractionDataset(object):
    """ 
    Container object for Ultrafast Electron Diffraction Datasets from the Siwick 
    Research group.

    Attributes
    ----------
    directory : str
        Absolute path to the dataset directory
    radial_average_computed : bool
        If radial averages have been computed and are available as an HDF5 file,
        returns True. False otherwise.
    
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
    
    radial_pattern
        Access time-delay processed powder diffraction patterns
    
    radial_pattern_series
        Access to the ensemble of time-delay radially-averaged diffraction patterns.
        
    radial_average
        Radial average of a time delay picture.
    
    radial_average_series
        Radial average of all time delay pictures, saved in HDF5 and MATLAB formats
    
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
    
    def __init__(self, directory):
        """
        Parameters
        ----------
        directory : str
            Absolute path to the dataset directory
        """
        # Check that the directory is a 'processed' directory.
        if not directory.endswith('processed') and 'processed' in os.listdir(directory):
            directory = os.path.join(directory, 'processed')
        
        self.raw_directory = os.path.dirname(directory)
        self.directory = directory
    
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
        except IOError:
            print('Available time points : {0}'.format(repr(self.time_points)))
        
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
        
    def radial_pattern(self, time):
        """
        Returns the radially-averaged pattern.
        
        Parameters
        ----------
        time : str, numerical
            Time delay value of the image.
        
        Notes
        -----
        This function depends on the existence of an HDF5 file containing radial
        averages.
        """
        from h5py import File
        
        time = str(float(time))
        file = File(self._radial_average_filename, 'r')
        
        # Rebuild Curve object from saved data
        xdata = file['/{0}/xdata'.format(time)]
        intensity = file['/{0}/intensity'.format(time)]
        return Curve(xdata, intensity, name = time)
    
    def radial_pattern_series(self, background_subtracted = False):
        """
        Returns a list of time-delay radially-averaged data.
        
        Parameters
        ----------
        background_subtracted : bool
            If True, radial patterns will be background-subtracted before being 
            returned. False by default.
        
        Returns
        -------
        curves : list of curve.Curve objects
        """
        curves = list()
        for time in self.time_points:
            if background_subtracted:
                curves.append(self.radial_pattern(time) - self.inelastic_background(time))
            else:
                curves.append(self.radial_pattern(time))
        return curves
    
    def inelastic_background(self, time = 0.0):
        """
        Returns the inelastic scattering background of the radially-averaged pattern.
        
        The inelastic background is computed for the average of radial patterns
        before photoexcitation; however, the background curve is saved in each 
        HDF5 time group for flexibility. 
        
        Parameters
        ----------
        time : str, numerical, optional
            Time delay value of the image.
        
        Notes
        -----
        This function depends on the existence of an HDF5 file containing radial
        averages.
        """
        from h5py import File
        
        time = str(float(time))
        file = File(self._radial_average_filename, 'r')
        
        # Rebuild Curve object from saved data
        # see self._export_background_curves for the HDF5 group naming
        xdata = self._access_dataset(file, time, 'xdata')
        ydata = self._access_dataset(file, time, 'inelastic background')
        return Curve(xdata, ydata, name = time)
    
    def inelastic_background_series(self):
        """
        Returns a list of time-delay inelastic scattering background data.
        
        Returns
        -------
        curves : list of curve.Curve objects
        """
        curves = list()
        for time in self.time_points:
            curves.append(self.inelastic_background(time))
        return curves
    
    def peak_dynamics(self, rect):
        pass
        
    def radial_peak_dynamics(self, edge, edge2 = None):
        """
        Returns a curve corresponding to the time-dynamics of a location in the 
        diffraction patterns. Think of it as looking at the time-evolution
        of a diffraction peak.
        
        Parameters
        ----------
        edge : float
            Edge value in terms of xdata
        edge2 : float, optional
            If not None (default), the peak value is integrated between edge and
            edge2.
        
        Returns
        -------
        time_values : ndarray, shape (N,)
            Time-delay values as an array
        intensity_series : ndarray, shape (N,M)
            Array of intensities. Each column corresponds to a time-delay
        """
        curves = self.radial_pattern_series()    
        scattering_length = curves[0].xdata
        intensity_series = n.vstack( tuple( [curve.ydata for curve in curves] ))

        time_values = n.array(list(map(float, self.time_points)))
        index = n.argmin(n.abs(scattering_length - edge))
        
        if edge2 is None:
            intensities = intensity_series[:, index]
        else:
            index2 = n.argmin(n.abs(scattering_length - edge2))
            intensities = intensity_series[:, index:index2].mean(axis = 1)
        
        # Normalize intensities
        return time_values, intensities/intensities.max()
        
    @property
    def _radial_average_filename(self):
        return os.path.join(self.directory, 'radial_averages.hdf5')
    
    @property
    def _pumpoff_directory(self):
        return os.path.join(self.directory, 'pumpoff pictures')
    
    @property
    def radial_average_computed(self):
        return self._radial_average_filename in os.listdir(self.directory)
    
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
        #TODO: make _time_filenames return a list of absolute paths, not relative
        return [f for f in os.listdir(self.directory) 
                if os.path.isfile(os.path.join(self.directory, f)) 
                and f.startswith('data_timedelay_') 
                and f.endswith('_pumpon.tif')]
    
    @property
    def time_points(self):            
        # get time points. Filename look like:
        # data_timedelay_-10.00_pumpon.tif
        time_list = [f.split('_')[2] for f in self._time_filenames]
        time_list.sort(key = lambda x: float(x))
        return time_list
    
    @property
    def resolution(self):
        # Get the shape of the first image in self._time_filenames
        fn = os.path.join(self.directory, self._time_filenames[0])
        return read(fn).shape
    
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
        
    # -------------------------------------------------------------------------
    
    @property
    def _pumpoff_background(self):
        return read(os.path.join(self.directory, 'background_average_pumpoff.tif'))
    
    @property
    def _pumpon_background(self):
        return read(os.path.join(self.directory, 'background_average_pumpon.tif'))
    
    @property
    def _substrate(self):
        return read(os.path.join(self.directory, 'substrate.tif'))
    
    def intensity_noise(self):
        """
        RMS intensity noise for pictures before photoexcitation.
        """
        b4_time0 = n.dstack( tuple([self.image(time) for time in self.time_points if float(time) <= 0.0]) )
        return n.std(b4_time0, axis = -1)
    
    def _pumpoff_intensity_stability(self):
        """ Plots the total intensity of the pumpoff pictures. """
        overall_intensity = list()
        for fn in self._pumpoff_filenames:
            overall_intensity.append( imread(os.path.join(self._pumpoff_directory, fn)).sum() )
        return n.array(overall_intensity)/overall_intensity[0]
        
    def stability_diagnostic(self, time = '+0.00'):
        """ Plots the overall intensity of time-delay picture over nscan."""
        template = 'data.timedelay.' + time + '.nscan.*.pumpon.tif'
        filenames = [fn for fn in glob(os.path.join(self.raw_directory, template))]
        return n.array([read(filename).sum() for filename in filenames])
    
    def radial_average(self, time, center, mask_rect = None):
        """
        Radial average of a
        Parameters
        ----------
        image : ndarray
            image data from the diffractometer.
        center : array-like, shape (2,)
            [x,y] coordinates of the center (in pixels)
        name : str
            String identifier for the output RadialCurve
        mask_rect : Tuple, shape (4,)
            Tuple containing x- and y-bounds (in pixels) for the beamblock mask
            mast_rect = (x1, x2, y1, y2)
        """
        xdata, intensity = radial_average(self.image(time), center, mask_rect)
        
        # Change x-data from pixels to scattering length
        s = scattering_length(xdata, self.energy)
        return Curve(s, intensity, name = str(time), color = 'b')
    
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
            curve = self.radial_average(time, center, mask_rect)
            results.append( (time, curve) )
        self._export_curves(results)
    
    def inelastic_background_fit(self, positions):
        """
        Fits a biexponential function to the inelastic scattering background. The
        fit is applied to the average radial diffraction pattern before time 0.
        
        Parameters
        ----------
        positions : list of floats, optional
            x-data positions of where radial averages should be fit to. If a list
            is not provided, the positions will be automatically determined.
        
        See also
        --------
        curve.Curve.auto_inelastic_background
            Automatically determine the inelastic bakground using the continuous
            wavelet transform with Ricker mother wavelet.
        """        
        curves_before_photoexcitation = [self.radial_pattern(time) for time in self.time_points if float(time) < 0.0]
        average_ydata = sum([curve.ydata for curve in curves_before_photoexcitation])/len(curves_before_photoexcitation)
        average_curve = Curve(curves_before_photoexcitation[0].xdata, average_ydata)
        
        if positions is None:
            background_curve = average_curve.auto_inelastic_background()
        else:
            background_curve = average_curve.inelastic_background(positions)
            
        self._export_background_curves(background_curve)

    # -------------------------------------------------------------------------
    
    @staticmethod
    def _access_time_group(opened_file, timedelay):
        """
        Returns the HDF5 group associated with a certain time. If the group doesn't exist,
        it will be created.
        
        Parameters
        ----------
        opened_file : h5py.File object
            Opened file.
        timedelay : str or numerical
            Pump-probe time-delay.
        """
        
        timedelay = str(float(timedelay))
        return opened_file.require_group('/{0}'.format(timedelay))
    
    def _access_dataset(self, opened_file, timedelay, dataset_name, data = None):
        """
        Returns a dataset as a numpy array, or write a numpy array into the dataset.
        
        Parameters
        ----------
        opened_file : h5py.File object
            Opened file.
        timedelay : str or numerical
            Pump-probe time-delay.
        dataset_name : str
            Name of the dataset to be created
        data : ndarray or None
            If not None, data will be written in the dataset.
        
        Returns
        -------
        out : ndarray or None
            If data is None, out is an ndarray
        """
        # group.require_dataset cannot be used as the dataset shape is not known in advance
        group = self._access_time_group(opened_file, timedelay)
        
        if data is None: # Retrieve data in the form of an ndarray
            return n.array(group[dataset_name])
            
        else:
            del group[dataset_name] 
            group.create_dataset(dataset_name, dtype = n.float, data = data)
                
    def _export_curves(self, results):
        """
        Parameters
        ----------
        results : list
            List of tuples containing a time delay (str) and a curve (curve.Curve)
        
        Notes
        -----
        This function will overwrite existing radial averages.
        """
        from h5py import File
        
        with File(self._radial_average_filename, 'w', libver = 'latest') as f:       # Overwrite if it already exists
    
            # Attributes
            f.attrs['acquisition date'] = self.acquisition_date
            f.attrs['resolution'] = self.resolution
            f.attrs['fluence'] = self.fluence
            f.attrs['energy'] = self.energy
            f.attrs['exposure'] = self.exposure
            f.attrs['current'] = self.current
            
            #Iteratively create a group for each timepoint
            for item in results:
                timedelay, curve = item
                group = self._access_time_group(f, timedelay)
                group.attrs['time delay'] = timedelay
                
                #Add some data to the file
                self._access_dataset(f, timedelay, dataset_name = 'xdata', data = curve.xdata)
                self._access_dataset(f, timedelay, dataset_name = 'intensity', data = curve.ydata)
    
    def _export_background_curves(self, background_curve):
        """
        Exports the background curves. If background curves have been computed,
        a radial-average file already exists.
        
        Parameters
        ----------
        background_curve : curve.Curve object
            inelastic scattering background curve
        
        Notes
        -----
        This function will overwrite existing inelastic scattering background curves.
        """
        from h5py import File
        
        with File(self._radial_average_filename, 'r+', libver = 'latest') as f:       # Overwrite if it already exists
        
            #Iteratively visit groups for each timepoint
            for timedelay in self.time_points:
                self._access_dataset(f, timedelay, dataset_name = 'inelastic background', data = background_curve.ydata)


if __name__ == '__main__':
    directory = 'K:\\2012.11.09.19.05.VO2.270uJ.50Hz.70nm'
    d = DiffractionDataset(directory)