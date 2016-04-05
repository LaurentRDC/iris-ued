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
import matplotlib.pyplot as plt #For testing only

#Batch processing libraries
from curve import Curve
import os.path
import h5py
from tqdm import tqdm
from PIL import Image #TODO: use tifffile if possible

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
    return array.astype(n.int16)
    
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
    
    See also
    --------
    PIL.Image.open 
        For supported file types.
    """
    im = Image.open(filename)
    return n.array(im).astype(n.float)
    
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
    PIL.Image.open 
        For supported file types.
        
    cast_to_16_bits
        For casting rules.
    """
    array = cast_to_16_bits(array)
    im = Image.fromarray(array)
    im.save(filename)

def radial_average(image, center, mask_rect = None):
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
    RadialCurve object
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
    if mask_rect is None:
        image[:xc, :] = 0      #All poins above center of the image are disregarded (because of beamblock)
    else:
        x1, x2, y1, y2 = mask_rect
        image[x1:x2, y1:y2] = 0
    
    #Radial average
    px_bin = n.bincount(R.ravel().astype(n.int), weights = image.ravel())
    r_bin = n.bincount(R.ravel().astype(n.int))  
    radial_intensity = px_bin/r_bin
    
    return (n.unique(R.ravel().astype(n.int)), radial_intensity)

class DiffractionDataset(object):
    """ 
    Container object for Ultrafast Electron Diffraction Datasets from the Siwick 
    Research group.

    Attributes
    ----------
    directory : str
        Absolute path to the dataset directory
    processed_directory : str
        Absolute path to the processed files directory
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
    time_filenames : list of strings
        Filenames of the TIFF associated with time delay data. Filenames are
        relative to processed_directory
    substrate : ndarray
        CCD image of the substrate-only diffraction pattern. If no substrate image
        is in the dataset directory, substrate is an array of zeros.
        It is assumed that the substrate diffraction pattern was acquired without
        laser pumping. The 'pump-off' background is subtracted.
    pumpon_background : ndarray
        Average of the pumpon background images
    pumpoff_background : ndarray
        Average of the pumpoff backgorund images
    
    Methods
    -------
    image
        Access time-delay averaged diffraction patterns
    
    image_series
        Access to the ensemble of time-delay averaged diffraction patterns
        in a single ndarray.
    
    radial_pattern
        Access time-delay processed powder diffraction patterns
        
    radial_average
        Radial average of a time delay picture.
    
    radial_average_series
        Radial average of all time delay pictures, saved in HDF5 and MATLAB formats
        
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
            If True, image series is returned as an array of int16.
        
        Notes
        -----
        reduced_memory = True is good for displaying, but not for computations.
        """
        images = list()
        for time in self.time_points:
            images.append(self.image(time, reduced_memory))
            
        return n.array(images)
        
    def radial_pattern(self, time):
        """
        Returns the image of the processed pictures.
        
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
        fn = self.radial_average_filename + '.hdf5'
        file = File(fn, 'r')
        
        # Rebuild Curve object from saved data
        xdata = file['/{0}/xdata'.format(time)]
        intensity = file['/{0}/intensity'.format(time)]
        return Curve(xdata, intensity, name = time)
    
    def radial_pattern_series(self):
        """
        
        """
        curves = list()
        for time in self.time_points:
            curves.append(self.radial_pattern(time))
        
        return curves
    
    def peak_dynamics(self, index, index2 = None):
        """
        Returns a curve corresponding to the time-dynamics of a location in the 
        diffraction patterns. Think of it as looking at the time-evolution
        of a diffraction peak.
        
        Parameters
        ----------
        index : int
            Index of the xdata
        index2 : int, optional
            If not None (default), the peak value is integrated between index and
            index 2.
        """

        time_values = n.array(list(map(float, self.time_points)))
        
        intensity = list()
        for time_point in self.time_points:
            if index2 is not None:
                intensity.append(self.radial_pattern(time_point).ydata[index:index2].sum())
            else:
                intensity.append(self.radial_pattern(time_point).ydata[index])
        
        return time_values, n.array(intensity, dtype = n.float)        
        
    @property
    def radial_average_filename(self):
        return os.path.join(self.directory, 'radial_averages')
    
    @property
    def _exp_params_filename(self):
        return os.path.join(self.directory, 'experimental_parameters.txt')
        
    @property
    def time_filenames(self):
        return [f for f in os.listdir(self.directory) 
                if os.path.isfile(os.path.join(self.directory, f)) 
                and f.startswith('data_timedelay_') 
                and f.endswith('_pumpon.tif')]
    
    @property
    def time_points(self):            
        # get time points. Filename look like:
        # data_timedelay_-10.00_pumpon.tif
        time_list = [f.split('_')[2] for f in self.time_filenames]
        time_list.sort(key = lambda x: float(x))
        return time_list
    
    @property
    def resolution(self):
        # Get the shape of the first image in self.time_filenames
        fn = os.path.join(self.directory, self.time_filenames[0])
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
    def pumpoff_background(self):
        return read(os.path.join(self.directory, 'background.average.pumpoff.tif'))
    
    @property
    def pumpon_background(self):
        return read(os.path.join(self.directory, 'background.average.pumpon.tif'))
    
    @property
    def substrate(self):
        return read(os.path.join(self.directory, 'substrate.tif'))
    
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
        for time in tqdm(self.time_points):
            curve = self.radial_average(time, center, mask_rect)
            results.append( (time, curve) )
        self._export(results)
        
    def _export(self, results):
        """ 
        Export radially-averaged processed powder diffraction patterns in HDF5
        format and legacy MATLAB format. If the files already exist, they will be
        overwritten.
        
        Parameters
        ----------
        results : list
            List of tuples containing a time delay (str) and a curve (curve.Curve)
        """
        self._export_hdf5(self.radial_average_filename + '.hdf5', results)
        self._export_mat(self.radial_average_filename + '.mat', results)
        
    def _export_mat(self, filename, results):
        """
        Parameters
        ----------
        filename : str
            Absolute filename of the exported data
        results : list
            List of tuples containing a time delay (str) and a curve (curve.Curve)
        
        Notes
        -----
        scipy.io.savemat overwrites existing files.
        """
        from scipy.io import savemat
        
        mdict = dict()
        for item in results:
            timedelay, curve = item
            mdict[timedelay] = n.vstack( (curve.xdata, curve.ydata) )
        savemat(filename, mdict, appendmat = False)
    
    def _export_hdf5(self, filename, results):
        """
        Parameters
        ----------
        filename : str
            Absolute filename of the exported data
        results : list
            List of tuples containing a time delay (str) and a curve (curve.Curve)
        
        Notes
        -----
        This function will overwrite existing radial averages.
        """
        from h5py import File
        
        with File(filename, 'w', libver = 'latest') as f:       # Overwrite if it already exists
    
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
                
                #Create group and attribute
                group = f.create_group(timedelay)
                group.attrs['time delay'] = timedelay
                
                #Add some data to the file
                group.create_dataset(name = 'xdata', data = curve.xdata)
                group.create_dataset(name = 'intensity', data = curve.ydata)

# -----------------------------------------------------------------------------
#           TESTING FUNCTION FOR PLOTTING DYNAMIC DATA 
# -----------------------------------------------------------------------------

def plotTimeResolved(filename):    
    
    f = h5py.File(filename, 'r')
    times = f.keys()
    times.sort()
    datasets = [(f[time]['Radav ' + time], time) for time in times]
    
    #Plotting
    for dataset in datasets:
        data, time = dataset
        plt.plot(data[0], data[1], label = str(time))
        plt.legend()

if __name__ == '__main__':
    
    directory = 'K:\\2012.11.09.19.05.VO2.270uJ.50Hz.70nm'
    d = DiffractionDataset(directory)
    d.radial_average_series([1024,1024], None)