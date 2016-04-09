# -*- coding: utf-8 -*-

#Basics
import numpy as n
from scipy.signal import find_peaks_cwt, ricker
import scipy.optimize as opt

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
    cutoff
        Cuts off a part of the curve.
    
    inelastic_background
        Fits a biexponential inelastic background to the curve
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
    
    def plot(self):
        """ Diagnostic tool. """
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.title(self.name)
        plt.plot(self.xdata, self.ydata, self.color)

    def cutoff(self, cutoff):
        """ Cuts off a part of the pattern"""
        if isinstance(cutoff, list):
            cutoff = cutoff[0]
        
        cutoff_index = n.argmin(n.abs(self.xdata - cutoff))
        self.xdata = self.xdata[cutoff_index::] 
        self.ydata = self.ydata[cutoff_index::]
    
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
        
        def matching_background(params):
            """
            This function is 0 if the background passes through all required xpoints
            """
            background = biexponential(xpoints, *params)
            ypoints = n.interp(xpoints, self.xdata, self.ydata)     # interpolation of the data at the background xpoints
            return n.sum( (ypoints - background) ** 2 )
        
        constraints = {'type': 'ineq', 'fun' : positivity_constraint}
        
        #Create initial guesses
        # amp1, amp2, decay1, decay2, offset1, offset2, floor
        # TODO: find better guesses?
        guesses = (self.ydata.max(), self.ydata.max()/2, 10, 1, 0, 0, self.ydata.min())
        
        results = opt.minimize(matching_background, x0 = guesses, method = 'COBYLA', constraints = constraints, options = {'disp' : True, 'maxiter' : 300000})
        
        # Create inelastic background function 
        amp1, amp2, dec1, dec2, off1, off2, floor = results.x
        new_fit = biexponential(self.xdata, amp1, amp2, dec1, dec2, off1, off2, floor)
        return Curve(self.xdata, new_fit, 'IBG {0}'.format(self.name), 'red')
            
    
    def auto_inelastic_background(self):
        """
        Fits the inelastic background by first finding the location of diffraction 
        peaks, and automatically selecting points between peaks as background
        intensities.
        
        Returns
        -------
        out : Curve object
        """
        # Determine peak locations
        peak_indices = self._find_peaks()
        
        # Find indices between peaks
        between_peaks_indices = [(peak_indices[i] - peak_indices[i - 1])/2 for i in range(1, len(peak_indices))]
        between_peaks_indices.append(0)     # To fit the inelastic background with the bound
        xpoints = [self.xdata[int(i)] for i in between_peaks_indices]
        
        return self.inelastic_background(xpoints)
    
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
        ...    plt.axvline(i) for i in peak_indices
        ...
        >>>
        
        See also
        --------
        scipy.signal.find_peaks_cwt
        """
        widths = n.arange(1, len(self.ydata)/10)    # Max width determined with testing
        return find_peaks_cwt(self.ydata, widths = widths, wavelet = ricker, min_length = len(widths)/15)

if __name__ == '__main__':
    # Test curve
    import matplotlib.pyplot as plt
    directory = 'K:\\2012.11.09.19.05.VO2.270uJ.50Hz.70nm'
    from dataset import DiffractionDataset
    d = DiffractionDataset(directory)
    test = d.radial_pattern(0.0)