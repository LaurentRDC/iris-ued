# -*- coding: utf-8 -*-
"""
Wavelet decomposition paper
"""
from colorsys import hsv_to_rgb
import pywt
import numpy as n
from uediff.structure import crystalMaker
from uediff.instrumentation import gaussian
from uediff.diffsim.powder import powder_diffraction
from iris import pattern, dataset
import matplotlib.pyplot as plt

# Typical experimental time-points
# and other useful CONSTANTS
timepoints = n.arange(0, 30, dtype = n.float)
DEC_LEVEL = 10
NOISE_STD = 0.05
PLOT_MARKERSIZE = 2

def spectrum_colors(num_colors):
    """
    Generates a set of RGB colors corresponding to the visible spectrum.
    
    Parameters
    ----------
    num_colors : int
        number of colors to return
    
    Returns
    -------
    colors : list of (R,B,G) tuples.
    """
    # Hue values from 0 to 1
    hue_values = [i/num_colors for i in range(num_colors)]
    
    # Scale so that the maximum is 'purple':
    hue_values = [0.8*hue for hue in hue_values]
    
    colors = list()
    for hue in reversed(hue_values):
        colors.append(hsv_to_rgb(h = hue, s = 0.7, v = 0.9))
    return colors
    
def best_wavelet():
    """
    Determines what the best wavelet is for unassisted background removal for UED data
    """
    global composite, signal, noise, background, s
    VO2 = crystalMaker('M1')
    
    s = n.linspace(0.11, 0.8, 1000)
    background = 80*n.exp(-7*s) + 50*n.exp(-2*s)
    signal = 20*powder_diffraction(VO2, normalized = True, plot = False, scattering_length = s)[1]
    noise = n.random.normal(loc = 0, scale = NOISE_STD, size = s.shape)
    composite = pattern.Pattern([s, background + signal + noise])
    
    # Define figure-of-merit function
    def background_removal(wavelet):
        rec_background = composite.baseline(background_regions = [], max_iter = 100, level = None, wavelet = wavelet)
        return n.sqrt(n.mean(((rec_background.data - background.data)/background.data)**2))
    
    # Loop decomposition over wavelets
    residuals = list()
    for wavelet_id in pywt.wavelist():
        residuals.append( (wavelet_id, background_removal(wavelet_id)) )
    
    # Sort the list results
    residuals.sort(key = lambda tup: tup[1])   # Sort by residual sum, second item in the tuples of the list 'residuals'
    return residuals
    
def time_dependent_background(scatt_angle):
    """
    Returns an array of time-varying background.
    
    Parameters
    ----------
    scatt_angle : ndarray, shape (N,)
        Scattering angle 's'
    
    Returns
    -------
    background : ndarray, shape (N, M)
        Background array for which each row is associated with a specific timepoint
    """
    def amplitude(time_points):
        """ Returns monotonically decreasing amplitude over time. """
        return n.linspace(75, 70, num = len(time_points))
    
    def subamplitude(time_points):
        """ Returns monotonically increasing subamplitude over time. """
        return n.linspace(55, 60, num = len(time_points))
    
    def substrate_amp1(time_points):
        return n.linspace(0.7, 1, num = len(time_points))
        
    def substrate_amp2(time_points):
        return n.linspace(1, 0.7, num = len(time_points))
    
    def decay(time_points):
        return n.linspace(-7, -5, num = len(time_points))
    
    backgrounds = list()
    for time, amp, subamp, dec, subs1, subs2 in zip(timepoints, amplitude(timepoints), subamplitude(timepoints), decay(timepoints), substrate_amp1(timepoints), substrate_amp2(timepoints)):
        arr = amp*n.exp(dec*scatt_angle) + subamp*n.exp(-2*scatt_angle) + subs1*gaussian(scatt_angle, (scatt_angle.max() + scatt_angle.min())/2, (scatt_angle.max() - scatt_angle.min())/8) + subs2*gaussian(scatt_angle, (scatt_angle.max() + scatt_angle.min())/2.5, (scatt_angle.max() - scatt_angle.min())/8)
        backgrounds.append( pattern.Pattern([scatt_angle, arr], str(time)) )
    return backgrounds

def time_dependent_diffraction(scatt_angle):
    # Generate time dynamics
    # Simulate powder diffraction first
    R = crystalMaker('R')
    M1 = crystalMaker('M1')
    r_pattern = 20*powder_diffraction(R, plot = False, scattering_length = scatt_angle)[1]
    m1_pattern = 20*powder_diffraction(M1, plot = False, scattering_length = scatt_angle)[1]
    
    # Set the change timescale
    timescale = n.linspace(1, 0, len(timepoints))**2
    
    # Distribute the changes into Pattern objects
    dynamics = n.zeros(shape = (len(timepoints), len(scatt_angle)))
    dynamics += n.outer(timescale, m1_pattern)
    dynamics += n.outer(1 - timescale, r_pattern)
    return [pattern.Pattern([scatt_angle, row], '') for row in dynamics]

def plot_dynamics():
    s = n.linspace(0.11, 0.8, 10000)
    
    backgrounds = time_dependent_background(s)
    patterns = time_dependent_diffraction(s)
    
    colors = spectrum_colors(len(timepoints))
    for c, bg, diff_pattern in zip(colors, backgrounds, patterns):
        composite = diff_pattern + bg
        noise = n.random.normal(0.0, NOISE_STD, size = composite.data.shape)
        plt.plot(composite.xdata, composite.data + noise, color = c, marker = '.', markersize = PLOT_MARKERSIZE, linestyle = 'None')
    plt.xlabel('Scattering length (1/A)')
    plt.ylabel('Intensity (counts)')
    plt.xlim([s.min(), s.max()])

def figure_of_merit(array1, array2):
    """
    Returns the similarity between 2 arrays of the same length.
    
    Parameters
    ----------
    array1, array2 : ndarrays, shapes (N,)
    
    Returns
    -------
    FOM : float
    """
    # Root mean square
    return n.sqrt(n.mean((array1 - array2)**2))
    
# Good background guesses for Rutile AND monoclinic M1
# only valid for scattering_length = n.linspace(0.11, 0.8, 10000)
bg_regions = list(range(0,300)) + list(range(1000, 1200)) + list(range(2177, 2340)) + [4290, 5766] + list(range(8200, 8279)) + [9643, 9999]

def simulated_background_fit(background_indices = bg_regions):
    """
    Fits time-varying background to diffraction data and evaluates a figure of
    merit for each time point.
    """
    s = n.linspace(0.11, 0.8, 10000)
    
    # Determine background regions in terms of s
    background_regions = [s[i] for i in background_indices]
    backgrounds = time_dependent_background(s)
    
    # Generate main signal with gaussian noise on the order of 0.25 count
    signals = time_dependent_diffraction(s)
    
    colors = spectrum_colors(num_colors = len(timepoints))
    
    FOM_over_time = n.zeros_like(timepoints, dtype = n.float)
    fig = plt.figure()
    frame1 = fig.add_axes((0.1, 0.3, 0.8, 0.6)) #TODO: add inset for reconstruction figure of merit
    frame2 = fig.add_axes((0.1, 0.1, 0.8, 0.2))
    frame2.axhline(y = 0, color = 'k', linewidth = 2)
    for i, background, signal, c in zip(range(len(timepoints)), backgrounds, signals, colors):
        noise = pattern.Pattern([s, n.random.normal(0.0, NOISE_STD, size = s.shape)], '')
        composite = signal + background + noise
        wav_background = composite.baseline(background_regions = background_regions, max_iter = 200, level = None, wavelet = 'sym6')   # Use max level with level = None
        
        reconstructed = composite - wav_background
        residuals = pattern.Pattern([s, (reconstructed.data - (signal.data + noise.data))])
        
        # Set residuals to 0 in background regions
        for j in background_indices:
            residuals.data[j] = 0
        
        frame1.plot(reconstructed.xdata, reconstructed.data, color = c, marker = '.', markersize = PLOT_MARKERSIZE, linestyle = 'None')
        frame2.plot(residuals.xdata, residuals.data, color = c, marker = '.', markersize = PLOT_MARKERSIZE, linestyle = 'None')
        FOM_over_time[i] = figure_of_merit(reconstructed.data, (composite-background).data)
    
    # Plot formatting
    [label.set_visible(False) for label in frame1.get_xticklabels()]    # Hide xlabel ticks for the top plot
    frame1.set_ylabel('Intensity (counts)', fontsize = 20)
    frame1.set_xlim([s.min(), s.max()])
    frame1.set_ylim([0, 21])
    
    frame2.set_xlabel('Scattering length (1/A)', fontsize = 20)
    frame2.set_ylabel('Residuals (counts)', fontsize = 20)
    frame2.set_xlim([s.min(), s.max()])
    
    return FOM_over_time

def simulated_background_fit_unassisted():
    return simulated_background_fit(background_indices = [])

# -----------------------------------------------------------------------------
#           DEALING WITH REAL DATA
# -----------------------------------------------------------------------------
directory = 'K:\\2012.11.09.19.05.VO2.270uJ.50Hz.70nm'
d = dataset.PowderDiffractionDataset(directory)    


def track_background():
    """
    """
    global b4t0
    fig = plt.figure()
    colors = spectrum_colors(num_colors = len(d.time_points))
    
    # Build reference curve as an average of before-time-zero
    b4t0 = [d.inelastic_background(time) for time in d.time_points if float(time) < 0.0]
    reference = pattern.Pattern([b4t0[0].xdata, sum([curve.data for curve in b4t0])/len(b4t0)], name = '') # Average curve. Need __truediv__ to be overloaded in Pattern
    
    for time, c in zip(d.time_points, colors):
        if float(time) < 0.0:
            continue
        background = d.inelastic_background(time)
        diff = background - reference
        plt.plot(diff.xdata, diff.data, color = c, marker = '.', markersize = 2*PLOT_MARKERSIZE, linestyle = 'None')
    
if __name__ == '__main__':
    pass
    #wavelist = best_wavelet()
    FOM = simulated_background_fit_unassisted()