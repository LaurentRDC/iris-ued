# -*- coding: utf-8 -*-
"""
Wavelet decomposition paper
"""
from dualtree import baseline, baseline_dwt
from colorsys import hsv_to_rgb
import numpy as n
from os.path import dirname, join
import scipy.optimize as opt
from uediff.structure import InorganicCrystal
from uediff.instrumentation import gaussian
from uediff.diffsim.powder import powder_diffraction
from iris import pattern, dataset
import matplotlib.pyplot as plt

# Typical experimental time-points
# and other useful CONSTANTS
TIMEPOINTS = n.arange(0, 30, dtype = n.float)
DEFAULT_SCATT_LENGTH = n.linspace(0.11, 0.8, 10000)
NOISE_STD = 0.1
PLOT_MARKERSIZE = 2

DEFAULT_FIRST_STAGE = 'sym4'
DEFAULT_CMP_WAV = 'qshift4'

# Diffraction dataset from Morrison
directory = 'K:\\2012.11.09.19.05.VO2.270uJ.50Hz.70nm'
d = dataset.PowderDiffractionDataset(directory)

n.random.seed(23)

peaks_to_look_at = [1393, 1805, 2565, 2795, 4980, 5510, 7390, 8470]   # Based on s = n.linspace(0.11, 0.8, 1000)
MIN_PLOT, MAX_PLOT = 0.13, 0.74
MIN_PLOT_INDEX, MAX_PLOT_INDEX = n.argmin(n.abs(DEFAULT_SCATT_LENGTH - MIN_PLOT)), n.argmin(n.abs(DEFAULT_SCATT_LENGTH - MAX_PLOT)) # Plot things between these indices

def spectrum_colors(num_colors):
    """
    Generates a set of RGB colors corresponding to the visible spectrum.
    
    Parameters
    ----------
    num_colors : int or iterable
        number of colors to return. Alternatively, if num_colors is an object
        with the attribute __len__ (list, tuple, ndarray, ...), then the number
        of colors is deduced.
    
    Returns
    -------
    colors : list of (R,B,G) tuples.
    """
    if isinstance(num_colors, int):
        # Hue values from 0 to 1
        hue_values = [i/num_colors for i in range(num_colors)]
    else:
        # Scale iterable to be between 0 and 1
        num_colors = n.asarray(num_colors)
        num_colors -= num_colors.min()
        num_colors = num_colors / num_colors.max()
        hue_values = num_colors.tolist()
    
    # Scale so that the maximum is 'purple':
    hue_values = [0.8*hue for hue in hue_values]
    
    colors = list()
    for hue in reversed(hue_values):
        colors.append(hsv_to_rgb(h = hue, s = 0.7, v = 0.9))
    return colors

def red_colors(num_colors):
    """
    Generates a set of RGB colors corresponding to color, from low values to high.
    
    Parameters
    ----------
    num_colors : int or iterable
        number of colors to return. Alternatively, if num_colors is an object
        with the attribute __len__ (list, tuple, ndarray, ...), then the number
        of colors is deduced.
    
    Returns
    -------
    colors : list of (R,B,G) tuples.
    """
    if hasattr(num_colors, '__len__'):
        num_colors = len(num_colors)
    
    # Values from 0 to 1
    values = [i/num_colors for i in range(num_colors)]
    
    colors = list()
    for value in values:
        colors.append(hsv_to_rgb(h = 0, s = 0.7, v = value))
    return colors
        
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
        return n.linspace(0.8, 0.9, num = len(time_points))
        
    def substrate_amp2(time_points):
        return n.linspace(0.9, 0.8, num = len(time_points))
    
    def decay(time_points):
        return n.linspace(-7, -6.9, num = len(time_points))
    
    backgrounds = list()
    for time, amp, subamp, dec, subs1, subs2 in zip(TIMEPOINTS, amplitude(TIMEPOINTS), subamplitude(TIMEPOINTS), decay(TIMEPOINTS), substrate_amp1(TIMEPOINTS), substrate_amp2(TIMEPOINTS)):
        arr = amp*n.exp(dec*scatt_angle) + subamp*n.exp(-2*scatt_angle) + subs1*gaussian(scatt_angle, (scatt_angle.max() + scatt_angle.min())/2, (scatt_angle.max() - scatt_angle.min())/8) + subs2*gaussian(scatt_angle, (scatt_angle.max() + scatt_angle.min())/2.5, (scatt_angle.max() - scatt_angle.min())/8)
        backgrounds.append( pattern.Pattern([scatt_angle, arr], str(time)) )
    return backgrounds   

def generate_time_dependent_diffraction(scatt_angle):
    # Generate time dynamics
    # Simulate powder diffraction first
    R = InorganicCrystal.from_preset('R')
    M1 = InorganicCrystal.from_preset('M1')
    r_pattern = 20*powder_diffraction(R, plot = False, scattering_length = scatt_angle)[1]
    m1_pattern = 20*powder_diffraction(M1, plot = False, scattering_length = scatt_angle)[1]
    
    # Set the change timescale
    timescale = n.exp(-0.2*TIMEPOINTS)
    timescale /= timescale.max()
    
    # Distribute the changes into Pattern objects
    dynamics = n.zeros(shape = (len(TIMEPOINTS), len(scatt_angle)))
    dynamics += n.outer(timescale, m1_pattern)
    dynamics += n.outer(1 - timescale, r_pattern)
    
    patterns = [pattern.Pattern([scatt_angle, row], '') for row in dynamics]
    
    #Remove the simulation background, artifact from the simulation of peak widening
    patterns = [pat - pat.baseline() for pat in patterns]
    
    # Save as npy
    arr = n.vstack( tuple([pattern.data.flatten() for pattern in patterns]) )
    n.save(join(dirname(__file__),'time_dependent_diffraction.npy'), arr)

def time_dependent_diffraction(scatt_angle = None):
    s = DEFAULT_SCATT_LENGTH
    patterns = list()
    data = n.load(file = join(dirname(__file__),'time_dependent_diffraction.npy'))
    for row in data:
        patterns.append(pattern.Pattern(data = [s, row]))
    return patterns

def plot_dynamics():
    s = DEFAULT_SCATT_LENGTH
    
    backgrounds = time_dependent_background(s)
    patterns = time_dependent_diffraction(s)
    
    colors = spectrum_colors(TIMEPOINTS)
    for c, bg, diff_pattern in zip(colors, backgrounds, patterns):
        composite = diff_pattern + bg
        noise = n.random.normal(0.0, NOISE_STD, size = composite.data.shape)
        plt.plot(composite.xdata, composite.data + noise, color = c, marker = '.', markersize = PLOT_MARKERSIZE, linestyle = 'None')
    plt.xlabel('Scattering length (1/A)', fontsize = 20)
    plt.ylabel('Intensity (counts)', fontsize = 20)
    plt.xlim([s.min(), s.max()])

def simulated_background_fit(background_indices = []):
    """
    Fits time-varying background to diffraction data
    """
    s = DEFAULT_SCATT_LENGTH
    
    # Determine background regions in terms of s
    background_regions = [s[i] for i in background_indices]
    backgrounds = time_dependent_background(s) #static_background(s)
    
    # Generate main signal with gaussian noise on the order of 0.25 count
    signals = time_dependent_diffraction(s)
    
    colors = spectrum_colors(TIMEPOINTS)

    fig = plt.figure()
    frame1 = fig.add_axes((0.1, 0.3, 0.8, 0.6)) #TODO: add inset for reconstruction figure of merit
    frame2 = fig.add_axes((0.1, 0.1, 0.8, 0.2))
    frame2.axhline(y = 0, color = 'k', linewidth = 2)
        
    # Add vertical line to plot to indicate peak dynamic sthat will be investigated later
    for index in peaks_to_look_at:
        frame1.axvline(x = signals[0].xdata[index], color = 'k', linewidth = 2)
        
    for i, background, signal, c in zip(range(len(TIMEPOINTS)), backgrounds, signals, colors):
        noise = pattern.Pattern([s, n.random.normal(0.0, NOISE_STD, size = s.shape)], '')
        composite = signal + background + noise
        wav_background = composite.baseline(first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV, background_regions = background_regions, max_iter = 100)   
        #wav_background = composite.baseline(wavelet = 'sym7', background_regions = background_regions, max_iter = 100)
        reconstructed = composite - wav_background
        residuals = pattern.Pattern([s, (wav_background.data - background.data)/background.data])
        
        # Set residuals to 0 in background regions
        for j in background_indices:
            residuals.data[j] = 0
        
        frame1.plot(reconstructed.xdata, reconstructed.data, color = c, marker = '.', markersize = PLOT_MARKERSIZE, linestyle = 'None')
        frame2.plot(residuals.xdata, 100*residuals.data, color = c, marker = '.', markersize = PLOT_MARKERSIZE, linestyle = 'None')
    
    # Plot formatting
    [label.set_visible(False) for label in frame1.get_xticklabels()]    # Hide xlabel ticks for the top plot
    frame1.set_ylabel('Intensity (counts)', fontsize = 20)
    frame1.set_xlim([MIN_PLOT, MAX_PLOT])
    frame1.set_ylim([0, 21])
    
    frame2.set_xlabel('Scattering length (1/A)', fontsize = 20)
    frame2.set_ylabel('Residuals (%)', fontsize = 20)
    frame2.set_xlim([MIN_PLOT, MAX_PLOT])

def peak_dynamics():
    s = DEFAULT_SCATT_LENGTH
    backgrounds = time_dependent_background(s)
    signals = time_dependent_diffraction(s)
    noise = pattern.Pattern([s, n.random.normal(0.0, NOISE_STD, size = s.shape)], '')
    composites = [bg + sig + noise for bg, sig in zip(backgrounds, signals)]
    reconstructed_signals = [composite - composite.baseline(first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV) for composite in composites]
    
    # Track peaks
    indices = peaks_to_look_at
    colors = spectrum_colors(indices)
    
    # compute amplitudes
    ref_amplitudes = [abs(signals[-1].data[i] - signals[0].data[i]) for i in indices]
    
    # For fitting the time constant
    def exp(time, amp, constant, floor):
        return amp*n.exp(-constant*time) + floor

    time_constant_results = list()
    amplitude_results = list()

    for index, color in zip(indices, colors):
        change = n.asarray([sig.data[index - 2 : index + 3].mean() for sig in reconstructed_signals]) - reconstructed_signals[0].data[index - 2 : index + 3].mean()
        change = n.abs(change) # Flip so that the change is always in the same direction
        
        # Fit the time constant
        params , covariant_matrix = opt.curve_fit(exp, xdata = TIMEPOINTS, ydata = change)
        amplitude_results.append(change.max() - change.min())
        time_constant_results.append(params[1])
    
    # Build amplitude data
    print('Reference amplitudes: ', n.abs(ref_amplitudes))
    print('Reconstructed amplitudes: ', n.abs(amplitude_results))
    print('Average deviation of amplitude (%):', 1 - n.mean(n.abs(ref_amplitudes)/n.abs(amplitude_results)))
    
    # Plot formatting
    plt.xlabel('Time-delay (ps)', fontsize = 20)
    plt.ylabel('Absolute change in intensity (a. u.)', fontsize = 20)
    
    # Return analysis results
    print('Average time constant (ps):', n.mean(time_constant_results))
    print('Standard deviation (ps):', n.std(time_constant_results))
    return time_constant_results

def peak_dynamics_2():
    # Load/generate data
    s = DEFAULT_SCATT_LENGTH
    backgrounds = time_dependent_background(s)
    signals = time_dependent_diffraction(s)
    composites = [bg + sig for bg, sig in zip(backgrounds, signals)]
    reconstructed_signals = [composite - composite.baseline(first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV) for composite in composites]
    
    # Track peaks
    indices = peaks_to_look_at
    colors = spectrum_colors(indices)
    
    # For fitting the time constant
    def exp(time, amp, constant, floor):
        return amp*n.exp(-constant*time) + floor
    
    amp_diff = list()   # Difference in amplitude in %
    tc_diff = list()    # difference in time constant in %
    for peak_index in indices:
        transient = n.asarray([sig.data[peak_index] for sig in signals])
        rec_transient = n.asarray([sig.data[peak_index] for sig in reconstructed_signals])
        
        transient, rec_transient = n.abs(transient), n.abs(rec_transient)
        
        transient -= transient[0]
        rec_transient -= rec_transient[0]
        
        plt.plot(transient)
        
        # Fit the time constant
        params , _ = opt.curve_fit(exp, xdata = TIMEPOINTS, ydata = n.abs(transient))
        rec_params, _ = opt.curve_fit(exp, xdata = TIMEPOINTS, ydata = n.abs(rec_transient))
        
        amp_diff.append((params[1] - rec_params[1])/params[1])
        tc_diff.append( (params[0] - rec_params[0])/params[0])
    
    return amp_diff, tc_diff
    
def simulated_background_fit_unassisted():
    return simulated_background_fit(range(0, 30))
   
 
if __name__ == '__main__':
    #generate_time_dependent_diffraction(DEFAULT_SCATT_LENGTH)
    simulated_background_fit_unassisted()
    #amp_diff, tc_diff = peak_dynamics_2()
    #peak_dynamics()