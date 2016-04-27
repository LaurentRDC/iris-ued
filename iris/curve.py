# -*- coding: utf-8 -*-

#Basics
import matplotlib.pyplot as plt
import numpy as n
from scipy.signal import find_peaks_cwt, ricker
import scipy.optimize as opt
from scipy import interpolate
import wavelet

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
        
# -----------------------------------------------------------------------------
#           RADIAL CURVE CLASS
# -----------------------------------------------------------------------------

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
        
    auto_inelastic_background
        Fits a biexponential inelastic scattering background to the curve.
        Background values are automatically determined from peak locations.
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
    
    def plot(self, show_peaks = True):
        """ Diagnostic tool. """        
        plt.figure()
        plt.title(self.name)
        plt.plot(self.xdata, self.ydata, self.color)
        
        if show_peaks:
            for peak_index in self._find_peaks():
                plt.axvline(self.xdata[peak_index], color = 'r')
    
    def inelastic_background(self, xpoints):
        """
        Better inelastic scattering background fit.
        
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
            return self.ydata - biexponential(self.xdata, *params)
        
        def residuals(params):
            """
            This function is 0 if the background passes through all required xpoints
            """
            background = biexponential(xpoints, *params)
            ypoints = n.interp(xpoints, self.xdata, self.ydata)     # interpolation of the data at the background xpoints
            return n.sum( (ypoints - background) ** 2 )
            
        constraints = {'type' : 'ineq', 'fun' : positivity_constraint}
        
        #Create initial guesses
        # amp1, amp2, decay1, decay2, offset1, offset2, floor
        guesses = (10,10,-1,-1,0,0,self.ydata.min())
        
        # Bounds detemined by logic and empiricism
        bounds = [(0, 1e3*self.ydata.max()), (0, 1e3*self.ydata.max()), (-100, 100),(-100, 100), (-1e4, 1e4), (-1e4, 1e4), (-1e4, 1e4)]
        
        # method = 'SLSQP', 
        results = opt.minimize(residuals, x0 = guesses, bounds = bounds, constraints = constraints, method = 'SLSQP', options = {'disp' : True, 'maxiter' : 1000})
        
        # Create inelastic background function 
        amp1, amp2, dec1, dec2, off1, off2, floor = results.x
        new_fit = biexponential(self.xdata, amp1, amp2, dec1, dec2, off1, off2, floor)
        return Curve(self.xdata, new_fit, 'IBG {0}'.format(self.name), 'red')
    
    def auto_inelastic_background(self, mode = 'fit'):
        """
        Fits the inelastic background by first finding the location of diffraction 
        peaks, and automatically selecting points between peaks as background
        intensities.
        
        Parameters
        ----------
        mode : str {'fit' (default), 'spline', 'wavelet'}, optional
            Background fit mode.
            
        Returns
        -------
        out : Curve object
        """
        # Find indices between peaks
        xpoints = [self.xdata[int(i)] for i in self._between_peaks()]
        if mode == 'fit':
            return self.inelastic_background(xpoints)
        elif mode == 'spline':
            return self._spline_background(xpoints)
        elif mode == 'wavelet':
            return self._wavelet_background(xpoints)
    
    def _spline_background(self, xpoints):
        """
        Background fitting using a cubic spline interpolation.
        
        Parameters
        ----------
        points : array-like, optional
            x-values of the points to fit to.
        """
        # find the bacground values to interpolate from
        ypoints = n.interp(xpoints, self.xdata, self.ydata)
        
        tck = interpolate.splrep(xpoints, ypoints, s = 0)
        background = interpolate.splev(self.xdata, tck, der = 0)
        return Curve(self.xdata, background, 'IBG {0}'.format(self.name), 'red')
    
    def _wavelet_background(self, xpoints):
        """
        Perform background fitting using the discrete wavelet transform.
        """
        background_indices = [n.argmin(n.abs(self.xdata - xpoint)) for xpoint in xpoints]
        bg = wavelet.baseline(signal = self.ydata, niter = 10, level = 5, wavelet = 'db10', background_regions = background_indices)
        return Curve(self.xdata, bg, 'IBG {0}'.format(self.name), 'red')
    
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
                              min_length = len(widths)/20, min_snr = 1.0)
    
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
    background = test.auto_inelastic_background(mode = 'wavelet')
    plt.plot(test.ydata, 'b', background.ydata, 'r')