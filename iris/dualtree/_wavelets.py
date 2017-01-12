"""
Extension of PyWavelets to complex wavelets suitable for the Dual-Tree Complex Wavelet Transform.

Author : Laurent P. René de Cotret

The wavelet coefficients are taken from Ivan Selesnick's Matlab code hosted at
http://eeweb.poly.edu/iselesni/WaveletSoftware/
"""
import numpy as n
from os.path import join, dirname
from pywt import Wavelet, wavelist
from warnings import warn

__all__ = ['dualtree_wavelet', 'dualtree_first_stage']

DATADIR = join(dirname(__file__), 'data')
ALL_QSHIFT = ('qshift1', 'qshift2', 'qshift3', 'qshift4', 'qshift5', 'qshift6')
ALL_COMPLEX_WAV = ('kingsbury99',) + ALL_QSHIFT
ALL_FIRST_STAGE = ('kingsbury99_fs',) + tuple([wav for wav in wavelist(kind = 'discrete') if wav != 'dmey'])

# lru_cache only exists as of Python 3.2
# In case it cannot be found, use a trivial decorator
def _trivial_decorator(func, *args, **kwargs):
    return func

try:
    from functools import lru_cache
except ImportError:
    warn('functools.lru_cache could not be found. Performance will be affected.', ImportWarning)
    lru_cache = _trivial_decorator

@lru_cache(maxsize = len(ALL_COMPLEX_WAV))
def dualtree_wavelet(name):
    """
    Returns a complex wavelet suitable for dual-tree cwt from a name.

    Parameters
    ----------
    name : str, {'qshift1', 'qshift2', 'qshift3', 'qshift4', 'kingsbury99'}
        Valid arguments can be found in dualtree.ALL_COMPLEX_WAV
    
    Returns
    -------
    real, imag : pywt.Wavelet objects.
    
    Raises
    ------
    ValueError
        If illegal wavelet name.
    """
    if name not in ALL_COMPLEX_WAV:
        raise ValueError('{} is not associated with any implemented complex wavelet. Possible choices are {}'.format(name, ALL_COMPLEX_WAV))
    
    # This function is simply for now
    if name == 'kingsbury99':
        return kingsbury99()
    
    return _qshift(name)

@lru_cache(maxsize = len(ALL_FIRST_STAGE))
def dualtree_first_stage(wavelet = 'kingsbury99_fs'):
    """
    Returns two wavelets to be used in the dual-tree complex wavelet transform, at the first stage.

    Parameters
    ----------
    wavelet : str or Wavelet
        Wavelet to be shifted for first-stage use. Valid arguments can be found in dualtree.ALL_FIRST_STAGE

    Return
    ------
    wav1, wav2 : Wavelet objects

    Raises
    ------
    ValueError
        If invalid first stage wavelet.
    """
    # Special case, preshifted
    if wavelet == 'kingsbury99_fs':
        return kingsbury99_fs()

    if not isinstance(wavelet, Wavelet):
        wavelet = Wavelet(wavelet)
    
    if wavelet.name not in ALL_FIRST_STAGE:
        raise ValueError('{} is an invalid first stage wavelet.'.format(wavelet.name))
    
    # extend filter bank with zeros
    filter_bank = [n.array(f, copy = True) for f in wavelet.filter_bank]
    for filt in filter_bank:
        extended = n.zeros( shape = (filt.shape[0] + 2,), dtype = n.float)
        extended[1:-1] = filt
        filt = extended

    # Shift deconstruction filters to one side, and reconstruction
    # to the other side
    shifted_fb = [n.array(f, copy = True) for f in wavelet.filter_bank]
    for filt in shifted_fb[::2]:    # Deconstruction filters
        filt = n.roll(filt, 1)
    for filt in shifted_fb[2::]:    # Reconstruction filters
        filt = n.roll(filt, -1)
    
    return Wavelet(name = wavelet.name, filter_bank = filter_bank), Wavelet(name = wavelet.name, filter_bank = shifted_fb)

@lru_cache(maxsize = len(ALL_QSHIFT))
def _qshift(name):
    """
    Returns a complex qshift wavelet by name.

    Parameters
    ----------
    name : str
        Wavelet to use. Valid arguments can be found in dualtree.ALL_QSHIFT
    
    Returns
    -------
    wav1, wav2 : pywt.Wavelet objects
        real and imaginary wavelet
    
    Raises
    ------ 
    ValueError 
        If illegal wavelet family name.
    """
    filters = ('h0a', 'h0b', 'g0a', 'g0b', 'h1a', 'h1b', 'g1a', 'g1b')
    
    filename = join(DATADIR, name + '.npz')
    with n.load(filename) as mat:
        try:
            (dec_real_low, dec_imag_low, rec_real_low, rec_imag_low, 
             dec_real_high, dec_imag_high, rec_real_high, rec_imag_high) = tuple([mat[k].flatten() for k in filters])
        except KeyError:
            raise ValueError('Wavelet does not define ({0}) coefficients'.format(', '.join(filters)))
    
    real_filter_bank = [dec_real_low, dec_real_high, rec_real_low, rec_real_high]
    imag_filter_bank = [dec_imag_low, dec_imag_high, rec_imag_low, rec_imag_high]

    return Wavelet(name = 'real:' + name, filter_bank = real_filter_bank), Wavelet(name = 'imag:' + name, filter_bank = imag_filter_bank)


#############################################################################################
#                           EXAMPLE COMPLEX WAVELETS FROM
#               http://eeweb.poly.edu/iselesni/WaveletSoftware/dt1D.html 
#############################################################################################
@lru_cache(maxsize = 1)
def kingsbury99_fs():
    """
    Returns a first-stage complex wavelet as published in [1]. 

    References
    ----------
    [1] Kingsbury, N. 'Image Processing with Complex Wavelets'. Philocophical Transactions of the Royal Society A pp. 2543-2560, September 1999
    """
    real_dec_lo = n.array([0, -0.08838834764832, 0.08838834764832, 0.69587998903400,0.69587998903400, 0.08838834764832, -0.08838834764832, 0.01122679215254, 0.01122679215254, 0])
    real_dec_hi = n.array([0, -0.01122679215254, 0.01122679215254, 0.08838834764832, 0.08838834764832, -0.69587998903400, 0.69587998903400, -0.08838834764832, -0.08838834764832, 0])
    real_rec_lo, real_rec_hi = real_dec_lo[::-1], real_dec_hi[::-1]

    imag_dec_lo = n.array([0.01122679215254, 0.01122679215254, -0.08838834764832, 0.08838834764832, 0.69587998903400, 0.69587998903400, 0.08838834764832, -0.08838834764832, 0, 0])
    imag_dec_hi = n.array([0, 0, -0.08838834764832, -0.08838834764832, 0.69587998903400, -0.69587998903400, 0.08838834764832, 0.08838834764832, 0.01122679215254, -0.01122679215254])
    imag_rec_lo, imag_rec_hi = imag_dec_lo[::-1], imag_dec_hi[::-1]

    real_fb = [real_dec_lo, real_dec_hi, real_rec_lo, real_rec_hi]
    imag_fb = [imag_dec_lo, imag_dec_hi, imag_rec_lo, imag_rec_hi]

    return Wavelet(name = 'real:', filter_bank = real_fb), Wavelet(name = 'imag:', filter_bank = imag_fb)

@lru_cache(maxsize = 1)
def kingsbury99():
    """
    Returns a late-stage complex wavelet as published in [1].

    References
    ----------
    [1] Kingsbury, N. 'Image Processing with Complex Wavelets'. Philocophical Transactions of the Royal Society A pp. 2543-2560, September 1999
    """
    real_dec_lo = n.array([ 0.03516384000000, 0, -0.08832942000000, 0.23389032000000, 0.76027237000000, 0.58751830000000, 0, -0.11430184000000, 0, 0])
    real_dec_hi = n.array([0, 0, -0.11430184000000, 0, 0.58751830000000, -0.76027237000000, 0.23389032000000, 0.08832942000000, 0, -0.03516384000000])
    real_rec_lo, real_rec_hi = real_dec_lo[::-1], real_dec_hi[::-1]

    imag_dec_lo = n.array([ 0, 0, -0.11430184000000, 0, 0.58751830000000, 0.76027237000000, 0.23389032000000, -0.08832942000000, 0, 0.03516384000000])
    imag_dec_hi = n.array([-0.03516384000000, 0, 0.08832942000000, 0.23389032000000, -0.76027237000000, 0.58751830000000, 0, -0.11430184000000, 0, 0])
    imag_rec_lo, imag_rec_hi = imag_dec_lo[::-1], imag_dec_hi[::-1]

    real_fb = [real_dec_lo, real_dec_hi, real_rec_lo, real_rec_hi]
    imag_fb = [imag_dec_lo, imag_dec_hi, imag_rec_lo, imag_rec_hi]

    return Wavelet(name = 'real:', filter_bank = real_fb), Wavelet(name = 'imag:', filter_bank = imag_fb)