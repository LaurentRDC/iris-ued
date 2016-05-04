# -*- coding: utf-8 -*-

#Basics
import numpy as n
from scipy.signal import find_peaks_cwt, ricker

import scipy.optimize as opt
import wavelet

class Pattern(object):
    """
    This class represents both polycrystalline 1D patterns and 
    single-crystal 2D patterns.
    
    Attributes
    ----------
    xdata : ndarray or None
        Scattering length of the pattern, if radial pattern. If Pattern is a 2D
        image (as for single crystal data), xdata is None.
    data : ndarray, shape (M,N) or shape(M,)
        Image data as an array or intensity data from radially-averaged data.
    type : str {'polycrystalline', 'single crystal'}
        Data type.
    name : str
        Description.
    
    Special Methods
    ---------------
    This object overloads the following special methods:
    [__add__, __sub__, __copy__]
    
    Methods
    -------
    plot
        Diagnostic plotting tool
    
    inelastic_background
        Fits a biexponential inelastic scattering background to the data
    """
    def __init__(self, data, name = ''):
        """
        Parameters
        ----------
        data : ndarray ndim 2, or list
            If list or tuple of ndarrays, assumed to be a 1D radial pattern
        name : str, optional
            Descriptive name
        """
        self.name = name
        self.xdata = None
        self.data = None
        
        if isinstance(data, (list, tuple, iter)):
            self.xdata = n.asarray(data[0], dtype = n.float)
            self.data = n.asarray(data[1], dtype = n.float)
        else:
            self.data = n.asarray(data, dtype = n.float)
    
    @property
    def type(self):
        if self.data.ndim == 1:
            return 'polycrystalline'
        elif self.data.ndim == 2:
            return 'single crystal'
        else:
            raise TypeError
    
    def __sub__(self, pattern):
        if self.type == 'polycrystalline':
            return Pattern([self.xdata, self.data - n.interp(self.xdata, pattern.xdata, pattern.data)], name = self.name)
        elif self.type == 'single crystal':
            return Pattern(data = self.data - pattern.data, name = self.name)
    
    def __add__(self, pattern):
        if self.type == 'polycrystalline':
            return Pattern([self.xdata, self.data + n.interp(self.xdata, pattern.xdata, pattern.data)], name = self.name)
        elif self.type == 'single crystal':
            return Pattern(data = self.data + pattern.data, name = self.name)
    
    def __copy__(self):
        if self.type == 'polycrystalline':
            return Pattern([n.array(self.xdata), n.array(self.data)], name = self.name)
        elif self.type == 'single crystal':
            return Pattern(data = n.array(self.data), name = self.name)
    
    def plot(self):
        """ For debugging purposes. """
        from matplotlib.pyplot import imshow, plot
        if self.type == 'single crystal':
            imshow(self.data)
        elif self.type == 'polycrystalline':
            plot(self.xdata, self.data)
    
    def inelastic_background(self, xpoints, level = 10):
        """
        Method for inelastic background determination via wavelet decomposition.d
        
        Parameters
        ---------
        xpoints : list or None
            List of x-values at which the curve is entirely background. If None, 
            these points will be automatically determined using the continuous 
            wavelet transform.
        level : int, optional
            Wavelet decomposition level. A higher level implies a coarser approximation 
            to the baseline.
        
        Returns
        -------
        background : Curve object
            Background curve.
        """
        if self.type == 'single crystal':
            raise NotImplemented
        
        if xpoints is None:          # Find x-values between peaks
            xpoints = [self.xdata[int(i)] for i in self._background_guesses()]
        background_indices = [n.argmin(n.abs(self.xdata - xpoint)) for xpoint in xpoints]
        
        # Remove low frequency exponential trends
        exp_bg = self._exponential_baseline()
        
        # Remove background
        background_values = wavelet.baseline(array = self.data - exp_bg,
                                             max_iter = 200,
                                             level = level,
                                             wavelet = 'db10',
                                             background_regions = background_indices)
        
        return Pattern([self.xdata, background_values + exp_bg], 'IBG {0}'.format(self.name))
        
    def _exponential_baseline(self):
        """
        Fits an exponential decay to the curve, in order to remove low frequency baseline.
        
        Returns
        -------
        baseline : ndarray
            Same shape as ydata
        """
        # Only fit to the first and last data point, for speed.
        # What really matters here is that the fit is below the signal
        xpoints = n.array([self.xdata[0], self.xdata[-1]])
        
        def exponential(x, amplitude, decay):
            return amplitude*n.exp(decay*x)
        
        def positivity_constraint(params):
            """
            This function returns a positive value if the background is less than
            the data, everywhere.
            """
            return self.ydata - exponential(self.xdata, *params)
        
        def residuals(params):
            """
            This function is 0 if the background passes through all required xpoints
            """
            background = exponential(xpoints, *params)
            ypoints = n.interp(xpoints, self.xdata, self.data)     # interpolation of the data at the background xpoints
            return n.sum( (ypoints - background) ** 2 )
            
        constraints = {'type' : 'ineq', 'fun' : positivity_constraint}
        
        #Create initial guesses and fit
        guesses = (self.data[0], -1)
        results = opt.minimize(residuals, x0 = guesses, constraints = constraints, method = 'SLSQP', options = {'disp' : False, 'maxiter' : 1000})
        
        # Create inelastic background function 
        amp, dec = results.x
        return exponential(self.xdata, amp, dec)
                            
    def _background_guesses(self, data_values = False):
        """
        Finds the indices associated with local minima between peaks in ydata
        
        Parameters
        ----------
        data_values : bool, optional
            If False (default), indices of the location of background guesses
            is returned. If True, values of xdata of background guesses is returned
        
        Returns
        -------
        out : list
            Indices of the locations of background in data (if data_values is False)
            of data values of the background in data (if data_values is True).
        
        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(curve.ydata)
        >>> 
        >>> indices = curve._find_peaks()
        >>> for i in indices:
        ...    plt.axvline(i)
        ...
        >>>
        
        See also
        --------
        scipy.signal.find_peaks_cwt
        """
        if self.type == 'single crystal':
            raise NotImplemented
        
        widths = n.arange(1, len(self.data)/10)    # Max width determined with testing
        indices = find_peaks_cwt(-self.data, widths = widths, wavelet = ricker, 
                                 min_length = len(widths)/10, min_snr = 1.5)
        if data_values:
            return [self.xdata[int(i)] for i in indices]
        else:
            return indices
    
    def _find_peaks(self):
        """
        Finds the indices associated with peaks in ydata
        
        Returns
        -------
        peak_indices : list of ints
            Indices of the peaks in ydata.
        
        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(curve.ydata)
        >>> 
        >>> peak_indices = curve._find_peaks()
        >>> for i in peak_indices:
        ...    plt.axvline(i)
        ...
        >>>
        
        See also
        --------
        scipy.signal.find_peaks_cwt
        """
        if self.type == 'single crystal':
            raise NotImplemented
        
        widths = n.arange(1, len(self.data)/10)    # Max width determined with testing
        return find_peaks_cwt(self.data, widths = widths, wavelet = ricker, 
                              min_length = len(widths)/20, min_snr = 1.5)