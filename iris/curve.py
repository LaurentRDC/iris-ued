# -*- coding: utf-8 -*-

#Basics
import matplotlib.pyplot as plt
import numpy as n
from scipy.signal import find_peaks_cwt, ricker
import scipy.optimize as opt
import wavelet

class Curve(object):
    """
    This class represents any radially averaged diffraction pattern or fit.
    
    Attributes
    ----------
    xdata : ndarray, shape (N,)
        Abscissa values of the curve.
    ydata : ndarray, shape (N,)
        Ordinate values of the curve.
    name : str
        String identifier.
    color : str
        Plotting color.
    
    Methods
    -------    
    inelastic_background
        Fits a biexponential inelastic scattering background to the curve
    """
    def __init__(self, xdata, ydata, name = '', color = 'b'):
        
        self.xdata = n.asarray(xdata, dtype = n.float)
        self.ydata = n.asarray(ydata, dtype = n.float)
        self.name = name
        
        #Plotting attributes
        self.color = color
    
    def __sub__(self, pattern):
        #Interpolate values so that substraction makes sense
        return Curve(self.xdata, self.ydata - n.interp(self.xdata, pattern.xdata, pattern.ydata), name = self.name, color = self.color)
    
    def __add__(self, pattern):
        #Interpolate values so that addition makes sense
        return Curve(self.xdata, self.ydata + n.interp(self.xdata, pattern.xdata, pattern.ydata), name = self.name, color = self.color)
    
    def __copy__(self):
        return Curve(self.xdata, self.ydata, self.name, self.color)
    
    def plot(self, show_peaks = False):
        """ Diagnostic tool. """        
        plt.figure()
        plt.title(self.name)
        plt.plot(self.xdata, self.ydata, self.color)
        
        if show_peaks:
            for peak_index in self._find_peaks():
                plt.axvline(self.xdata[peak_index], color = 'r')
    
    def inelastic_background(self, xpoints, level = 10):
        """
        Master method for inelastic background determination via wavelet decomposition.
        
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
        if xpoints is None:          # Find x-values between peaks
            xpoints = [self.xdata[int(i)] for i in self._between_peaks()]
        background_indices = [n.argmin(n.abs(self.xdata - xpoint)) for xpoint in xpoints]
        
        # Remove low frequency exponential trends
        exp_bg = self._exponential_baseline()
        
        # Remove background
        background_values = wavelet.baseline(array = self.ydata - exp_bg,
                                             max_iter = 200,
                                             level = level,
                                             wavelet = 'db10',
                                             background_regions = background_indices)
        
        return Curve(self.xdata, background_values + exp_bg, 'IBG {0}'.format(self.name), 'red')
    
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
            ypoints = n.interp(xpoints, self.xdata, self.ydata)     # interpolation of the data at the background xpoints
            return n.sum( (ypoints - background) ** 2 )
            
        constraints = {'type' : 'ineq', 'fun' : positivity_constraint}
        
        #Create initial guesses and fit
        guesses = (self.ydata[0], -1)
        results = opt.minimize(residuals, x0 = guesses, constraints = constraints, method = 'SLSQP', options = {'disp' : False, 'maxiter' : 1000})
        
        # Create inelastic background function 
        amp, dec = results.x
        return exponential(self.xdata, amp, dec)
                            
    def _between_peaks(self):
        """
        Finds the indices associated with local minima between peaks in ydata
        
        Returns
        -------
        indices : list of ints
            Indices of the throughs in ydata.
        
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
        widths = n.arange(1, len(self.ydata)/10)    # Max width determined with testing
        return find_peaks_cwt(-self.ydata, widths = widths, wavelet = ricker, 
                              min_length = len(widths)/10, min_snr = 1.5)
    
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
        widths = n.arange(1, len(self.ydata)/10)    # Max width determined with testing
        return find_peaks_cwt(self.ydata, widths = widths, wavelet = ricker, 
                              min_length = len(widths)/20, min_snr = 1.5)

if __name__ == '__main__':
    # Test curve
    directory = 'D:\\2016.04.20.15.15.VO2_4mW'
    from dataset import PowderDiffractionDataset
    d = PowderDiffractionDataset(directory)
    test = d.radial_pattern(0.0)
    background = test.inelastic_background(xpoints = [], level = 10)
    plt.plot(test.ydata)
    plt.plot(background.ydata)