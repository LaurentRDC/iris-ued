"""
Dual-tree complex wavelet transform (DTCWT) module.

Author: Laurent P. René de Cotret
"""
from itertools import cycle
import numpy as n
from pywt import dwt, idwt, dwt_max_level
from ._wavelets import dualtree_wavelet, dualtree_first_stage

__all__ = ['dualtree', 'idualtree', 'dualtree_max_level', 'approx_rec', 'detail_rec']

DEFAULT_MODE = 'constant'
DEFAULT_FIRST_STAGE = 'sym4'
DEFAULT_CMP_WAV = 'qshift4'

def dualtree(data, first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV, level = 'max', mode = DEFAULT_MODE, axis = -1):
    """
    Dual-tree complex wavelet transform, implemented from [1], along an axis. 

    Parameters
    ----------
    data: array_like
        Input data. Can be of any shape, but the transform can only be applied in 1D (i.e. along a single axis).
        The length along the axis must be even.
    first_stage : str, optional
        Wavelet to use for the first stage. See dualtree.ALL_FIRST_STAGE for a list of suitable arguments
    wavelet : str, optional
        Wavelet to use in stages > 1. Must be appropriate for the dual-tree complex wavelet transform.
        See dualtree.ALL_COMPLEX_WAV for possible
    level : int or 'max', optional
        Decomposition level (must be >= 0). If level is 'max' (default) then it
        will be calculated using the ``dualtree_max_level`` function.
    mode : str, optional
        Signal extension mode, see pywt.Modes.
    axis : int, optional
        Axis over which to compute the transform. Default is -1

    Returns
    -------
    [cA_n, cD_n, cD_n-1, ..., cD2, cD1] : list
        Ordered list of coefficients arrays
        where `n` denotes the level of decomposition. The first element
        (`cA_n`) of the result is approximation coefficients array and the
        following elements (`cD_n` - `cD_1`) are details coefficients arrays.
        Arrays have the same number of dimensions as the input.
    
    Raises
    ------
    ValueError
        Raised if axis argument is invalid (e.g. too large).
    
    Notes
    -----
    The implementation uses two tricks presented in [1]:
        `` Different first-stage wavelet ``
            The first level of the dual-tree complex wavelet transform involves a combo of shifted wavelets.
        
        `` Swapping of filters at each stage ``
            At each level > 1, the filters (separated into real and imaginary trees) are swapped.
    
    References
    ----------
    [1] Selesnick, I. W. et al. 'The Dual-tree Complex Wavelet Transform', IEEE Signal Processing Magazine pp. 123 - 151, November 2005.
    """
    data = n.asarray(data, dtype = n.float)/n.sqrt(2)

    if level == 'max':
        level = dualtree_max_level(data = data, first_stage = first_stage, wavelet = wavelet, axis = axis)
    elif level == 0:
        return [data]
    
    # Check axis bounds
    if axis > data.ndim - 1:
        raise ValueError('Input array has {} dimensions, but input axis is {}'.format(data.ndim, axis))
        
    real_wavelet, imag_wavelet = dualtree_wavelet(wavelet)
    real_first, imag_first = dualtree_first_stage(first_stage)
    
    real_coeffs = _single_tree_analysis(data = data, first_stage = real_first, wavelet = (real_wavelet, imag_wavelet), level = level, mode = mode, axis = axis)
    imag_coeffs = _single_tree_analysis(data = data, first_stage = imag_first, wavelet = (imag_wavelet, real_wavelet), level = level, mode = mode, axis = axis)

    # Combine coefficients into complex form
    # TODO: This is slow
    return [real + 1j*imag for real, imag in zip(real_coeffs, imag_coeffs)]

def idualtree(coeffs, first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV, mode = DEFAULT_MODE, axis = -1):
    """
    Inverse dual-tree complex wavelet transform, implemented from [1], along an axis.

    Parameters
    ----------
    coeffs : array_like
        Coefficients list [cAn, cDn, cDn-1, ..., cD2, cD1]
    first_stage : str, optional
        Wavelet to use for the first stage. See dualtree.ALL_FIRST_STAGE for a list of possible arguments.
    wavelet : str, optional
        Wavelet to use in stages > 1. Must be appropriate for the dual-tree complex wavelet transform.
        See dualtree.ALL_COMPLEX_WAV for possible arguments.
    mode : str, optional
        Signal extension mode, see pywt.Modes.
    axis : int, optional
        Axis over which to compute the inverse transform.
        
    Returns
    -------
    reconstructed : ndarray

    Raises
    ------
    ValueError
        If the input coefficients are too few
    
    Notes
    -----
    The implementation uses two tricks presented in [1]:
        `` Different first-stage wavelet ``
            The first level of the dual-tree complex wavelet transform involves a combo of shifted wavelets.
        
        `` Swapping of filters at each stage ``
            At each level > 1, the filters (separated into real and imaginary trees) are swapped.
    
    References
    ----------
    [1] Selesnick, I. W. et al. 'The Dual-tree Complex Wavelet Transform', IEEE Signal Processing Magazine pp. 123 - 151, November 2005.
    """
    if len(coeffs) < 1:
        raise ValueError("Coefficient list too short with {} elements (minimum 1 array required).".format(len(coeffs)))
    elif len(coeffs) == 1: # level 0 inverse transform
        real, imag = coeffs[0], coeffs[0]
    else:
        real_wavelet, imag_wavelet = dualtree_wavelet(wavelet)
        real_first, imag_first = dualtree_first_stage(first_stage)

        real = _single_tree_synthesis(coeffs = [coeff.real for coeff in coeffs], first_stage = real_first, 
                                      wavelet = (real_wavelet, imag_wavelet), mode = mode, axis = axis)
        imag = _single_tree_synthesis(coeffs = [coeff.imag for coeff in coeffs], first_stage = imag_first, 
                                      wavelet = (imag_wavelet, real_wavelet), mode = mode, axis = axis)
    
    return n.sqrt(2)*(real + imag)/2

def approx_rec(array, level, first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV, mode = DEFAULT_MODE, axis = -1):
    """
    Approximate reconstruction of a signal/image using the dual-tree approach.
    
    Parameters
    ----------
    array : array-like
        Array to be decomposed.
    level : int or 'max'
        Decomposition level. A higher level will result in a coarser approximation of
        the input array. If the level is higher than the maximum possible decomposition level,
        the maximum level is used. If 'max' (default), the maximum possible decomposition level is used.
    first_stage : str, optional
        First-stage wavelet to use. See dualtree.ALL_FIRST_STAGE for possible arguments.
    wavelet : str, optional
        Complex wavelet to use in late stages. See dualtree.ALL_COMPLEX_WAV for possible arguments.
    axis : int, optional
        Axis over which to compute the transform. Default is -1.
            
    Returns
    -------
    reconstructed : ndarray
        Approximated reconstruction of the input array.
    """
    coeffs = dualtree(data = array, first_stage = first_stage, wavelet = wavelet, level = level, mode = mode, axis = axis)
    app_coeffs, *det_coeffs = coeffs        # Python 3 only
    
    det_coeffs = [n.zeros_like(det, dtype = n.complex) for det in det_coeffs]
    reconstructed = idualtree(coeffs = [app_coeffs] + det_coeffs, first_stage = first_stage, wavelet = wavelet, mode = mode, axis = axis)
    return n.resize(reconstructed, new_shape = array.shape)

def detail_rec(array, level, first_stage = DEFAULT_FIRST_STAGE, wavelet = DEFAULT_CMP_WAV, axis = -1):
    """
    Detail reconstruction of a signal/image using the dual-tree approach.
    
    Parameters
    ----------
    array : array-like
        Array to be decomposed. Currently, only 1D arrays are supported. 2D support is planned.
    level : int or 'max'
        Decomposition level. 
        If None, the maximum possible decomposition level is used.
    first_stage : str, optional
        First-stage wavelet to use. See dualtree.ALL_FIRST_STAGE for possible arguments.
    wavelet : str or Wavelet object
        Complex wavelet to use in late stages. See dualtree.ALL_COMPLEX_WAV for possible arguments.
    axis : int, optional
        Axis over which to compute the transform. Default is -1.
            
    Returns
    -------
    reconstructed : ndarray
        Approximated reconstruction of the input array.

    See Also
    --------
    approx_rec 
    """
    coeffs = dualtree(data = array, first_stage = first_stage, wavelet = wavelet, level = level, mode = DEFAULT_MODE, axis = axis)
    app_coeffs = n.zeros_like(coeffs[0], dtype = n.complex) 
    reconstructed = idualtree(coeffs = [app_coeffs] + coeffs[1:], first_stage = first_stage, wavelet = wavelet, mode = DEFAULT_MODE, axis = axis)
    return n.resize(reconstructed, new_shape = array.shape)

def dualtree_max_level(data, first_stage, wavelet, axis = -1):
    """
    Returns the maximum decomposition level from the dual-tree complex wavelet transform.

    Parameters
    ----------
    data : ndarray
        Input data. Can be of any dimension.
    first_stage : str
        Wavelet used in the first stage of the dual-tree cwt. See pywt.wavelist() for suitable arguments.
    wavelet : str
        Dual-tree complex wavelet to use. Argument must be supported by dualtree_wavelet
    axis : int, optional
        Axis over which to compute the transform. Default is -1
        
    Returns
    -------
    max_level : int
    """
    real_wavelet, imag_wavelet = dualtree_wavelet(wavelet)
    return dwt_max_level(data_len = data.shape[axis], filter_len = max([real_wavelet.dec_len, imag_wavelet.dec_len]))

def _normalize_size_axis(approx, detail, axis):
    """ 
    Adjust the approximate coefficients' array size to that of the detail coefficients' array.
    
    Parameters
    ---------- 
    approx : ndarray
    detail: ndarray
    axis : int

    Returns
    -------
    ndarray
        Same shape as detail input.
    """
    if approx.shape[axis] == detail.shape[axis]:
        return approx
    # Swap axes to bring the specific axis to front, truncate array, and re-swap.
    # This is an extension of the 1D case:
    # >>> approx = approx[:-1] 
    return n.swapaxes( n.swapaxes(approx, axis, 0)[:-1] ,0, axis)

def _single_tree_analysis(data, first_stage, wavelet, level, mode, axis):
    """
    Single tree of the forward dual-tree complex wavelet transform.
    
    Parameters
    ----------
    data : ndarray, ndim 1
    first_stage : Wavelet object
    wavelet : 2-tuple of Wavelet object
    level : int
    mode : str
    axis : int

    Returns
    -------
    [cA_n, cD_n, cD_n-1, ..., cD2, cD1] : list
        Ordered list of coefficients arrays
        where `n` denotes the level of decomposition. The first element
        (`cA_n`) of the result is approximation coefficients array and the
        following elements (`cD_n` - `cD_1`) are details coefficients arrays.
    """
    approx, first_detail = dwt(data = data, wavelet = first_stage, mode = mode, axis = axis)    
    coeffs_list = [first_detail]
    for i, wav in zip(range(level - 1), cycle(wavelet)):
        approx, detail = dwt(data = approx, wavelet = wav, mode = mode, axis = axis)
        coeffs_list.append(detail)
    
    # Format list ot be compatible to PyWavelet's format. See pywt.wavedec documentation.
    coeffs_list.append(approx)
    return list(reversed(coeffs_list))

def _single_tree_synthesis(coeffs, first_stage, wavelet, mode, axis):
    """
    Single tree of the inverse dual-tree complex wavelet transform.

    Parameters
    ----------
    coeffs : list
    first_stage : Wavelet object
    wavelet : 2-tuple of Wavelet objects
    mode : str
    axis : int

    Returns
    -------
    reconstructed : ndarray, ndim 1
    """
    # Determine the level except first stage:
    # The order of wavelets depends on whether
    # the level is even or odd.
    level = len(coeffs) - 1
    approx, detail_coeffs, first_stage_detail = coeffs[0], coeffs[1:-1], coeffs[-1]

    # Set the right alternating order between real and imag wavelet
    if level % 2 == 1:
        wavelet = reversed(wavelet)
        
    # In the case of level = 1, coeffs[1:-1] is an empty list. Then, the following
    # loop is not run since zip() iterates as long as the shortest iterable.
    for detail, wav in zip(detail_coeffs, cycle(wavelet)):
        approx = _normalize_size_axis(approx, detail, axis = axis)
        approx = idwt(cA = approx, cD = detail, wavelet = wav, mode = mode, axis = axis)
    
    approx = _normalize_size_axis(approx, first_stage_detail, axis = axis)
    return idwt(cA = approx, cD = first_stage_detail, wavelet = first_stage, mode = mode, axis = axis)