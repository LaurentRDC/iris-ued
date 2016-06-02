# -*- coding: utf-8 -*-

#Basics
import numpy as n
from scipy.signal import find_peaks_cwt, ricker

import scipy.optimize as opt
from iris.wavelet import baseline
from warnings import warn


def biexponential(x, amplitude1, amplitude2, decay1, decay2, offset1, offset2, floor):
    """
    Returns a biexponential function evaluated over an array.
    
    Notes
    -----
    In case of fitting, there are 7 free parameters, and thus at least 7 points
    must be provided.
    """
    exp1 = amplitude1*n.exp(-decay1*(x - offset2))
    exp2 = amplitude2*n.exp(-decay2*(x - offset1))
    return exp1 + exp2 + floor

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
    [__add__ (+), __sub__ (-), __truediv__ (/), __copy__]
    
    Methods
    -------
    plot
        Diagnostic plotting tool
    
    baseline
        Determine a baseline using the discrete wavelet transform
    
    fft
        FFT of the pattern.
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
    
    def __truediv__(self, divisor):
        if self.type == 'polycrystalline':
            return Pattern(data = [self.xdata, self.data/divisor], name = self.name)
        elif self.type == 'single crystal':
            return Pattern(data = self.data/divisor, name = self.name)
    
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
    
    def fft(self):
        """
        Fast Fourier transform of the Pattern.
        
        Returns
        -------
        out : Pattern object
        """
        if self.type == 'polycrystalline':
            return Pattern([self.xdata, n.fft.fft(self.data)])
        elif self.type == 'single crystal':
            return Pattern(n.fft.fft2(self.data))
    
    def baseline(self, background_regions = [], level = None, max_iter = 1000, wavelet = 'sym4'):
        """
        Method for inelastic background determination via wavelet decomposition.d
        
        Parameters
        ---------
        background_regions : list or None, optional
            List of x-values at which the curve is entirely background. If None, 
            these points will be automatically determined using the continuous 
            wavelet transform. Default is empty list.
        level : int or None, optional
            Wavelet decomposition level. A higher level implies a coarser approximation 
            to the baseline. If None (default), level is automatically set to the maximum
            possible.
        max_iter : int, optional
            Number of iterations to perform. Default is 200, a good compromise between 
            speed and goodness-of-fit in most cases.
        wavelet : PyWavelet.Wavelet object or str, optional
            Wavelet with which to perform the algorithm. See PyWavelet documentation
            for available values. Default is 'db10'.
        
        Returns
        -------
        background : Pattern object
            Background curve.
        
        See also
        --------
        iris.wavelet.baseline
            Lower-level function for baseline determination from digital signals.
        """
        if self.type == 'single crystal':
            raise NotImplementedError
        
        if background_regions is None:          # Guess indices of xdata corresponding to background
            background_indices = self._background_guesses(data_values = False)
        else:
            background_indices = [n.argmin(n.abs(self.xdata - xpoint)) for xpoint in background_regions]
        
        # Remove background
        background_values = baseline(array = self.data,
                                     max_iter = max_iter,
                                     level = level,
                                     wavelet = wavelet,
                                     background_regions = background_indices)
        
        return Pattern([self.xdata, background_values], 'IBG {0}'.format(self.name))
        
    def inelastic_background_legacy(self, xpoints):
        """
        Legacy inelastic scattering background fit.
        
        Parameters
        ----------
        points : array-like, optional
            x-values of the points to fit to.
        """
        # Preliminaries
        xpoints = n.array(xpoints, dtype = n.float)
        
        def positivity_constraint(params):
            """
            This function returns a positive value if the background is less than
            the data, everywhere.
            """
            return self.data - biexponential(self.xdata, *params)
        
        def residuals(params):
            """
            This function is 0 if the background passes through all required xpoints
            """
            background = biexponential(xpoints, *params)
            ypoints = n.interp(xpoints, self.xdata, self.data)     # interpolation of the data at the background xpoints
            return n.sum( (ypoints - background) ** 2 )
            
        constraints = {'type' : 'ineq', 'fun' : positivity_constraint}
        
        #Create initial guesses
        # amp1, amp2, decay1, decay2, offset1, offset2, floor
        guesses = (75, 52, 7, 2,0,0,self.data.min())
        
        # Bounds detemined by logic and empiricism
        bounds = [(0, 1e3*self.data.max()), (0, 1e3*self.data.max()), (-100, 100),(-100, 100), (-1e4, 1e4), (-1e4, 1e4), (-1e4, 1e4)]
        
        # method = 'SLSQP', 
        results = opt.minimize(residuals, x0 = guesses, bounds = bounds, constraints = constraints, method = 'SLSQP', options = {'disp' : True, 'maxiter' : 1000})
        
        # Create inelastic background function 
        amp1, amp2, dec1, dec2, off1, off2, floor = results.x
        new_fit = biexponential(self.xdata, amp1, amp2, dec1, dec2, off1, off2, floor)
        return Pattern([self.xdata, new_fit], 'IBG {0}'.format(self.name))

        
    def _background_guesses(self, data_values = False):
        """
        Finds the indices associated with local minima between peaks.
        
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
        
        Raises
        ------
        NotImplementedError
            If instance is of type 'single crystal'
        
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
            Continuous wavelet transform applied to peak-finding in 1D signals.
        """
        if self.type == 'single crystal':
            raise NotImplementedError
        
        widths = n.arange(1, len(self.data)/10)    # Max width determined with testing
        indices = find_peaks_cwt(-self.data, widths = widths, wavelet = ricker, 
                                 min_length = len(widths)/10, min_snr = 1.5)
        if data_values:
            return [self.xdata[int(i)] for i in indices]
        else:
            return indices
    
    def _find_peaks(self):
        return self.peak_indices()
    
    def peak_indices(self):
        """
        Finds the indices associated with peaks in ydata
        
        Returns
        -------
        peak_indices : list of ints
            Indices of the peaks in ydata.
        
        Raises
        ------
        NotImplementedError
            If instance is of type 'single crystal'
        
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
            Continuous wavelet transform applied to peak-finding in 1D signals.
        """
        if self.type == 'single crystal':
            raise NotImplementedError
        
        widths = n.arange(1, len(self.data)/10)    # Max width determined with testing
        return find_peaks_cwt(self.data, widths = widths, wavelet = ricker, 
                              min_length = len(widths)/15, min_snr = 2)
    
    def show_peaks(self):
        """
        Plots the location of the peaks
        """
        from matplotlib.pyplot import figure, axvline
        
        if self.type == 'single crystal':
            raise NotImplementedError
        
        figure()
        self.plot()
        
        for index in self.peak_indices():
            axvline(x = self.xdata[index], color = 'r')