"""
DUALTREE

Author: Laurent P. René de Cotret

Python implementation of the dual-tree complex wavelet transform, based on [1].

Functions
---------
dualtree, idualtree
    Dual-tree complex wavelet transform (and its inverse) implemented using PyWavelets. Implementation
    tricks from [1], such as first stage filtering and filter swapping at later stages, are also
    included.

approx_rec, detail_rec
    Decomposition and recomposition of signals using only approximate or detail coefficients.

dualtree_max_level
    Maximal decomposition level of the dual-tree complex wavelet transform.

baseline
    Baseline determination of signals using the dual-tree complex wavelet transform. Modified algorithm
    from [2].

baseline_dwt
    Baseline determination of signals using the discrete wavelet transform. Provided for comparison
    with the dual-tree equivalent 'baseline'. Modified algorithm from [2].

denoise
    Denoising of signals using the dual-tree complex wavelet transform.

denoise_dwt
    Denoising of signals using the discrete wavelet transform. Provided for comparison
    with the dual-tree equivalent 'denoise'.

dualtree_wavelet
    Pair of real and imaginary wavelet that forms a complex wavelet appropriate for the dual-tree
    complex wavelet transform.

dualtree_first_stage
    Pair of real and imaginary wavelet, shifted by one sample with respect to one another, forming a complex
    wavelet appropriate for first-stage filtering during the dual-tree complex wavelet transform.

References
----------
[1] Selesnick, I. W. et al. 'The Dual-tree Complex Wavelet Transform', IEEE Signal Processing Magazine pp. 123 - 151, November 2005.

[2] Galloway et al. 'An Iterative Algorithm for Background Removal in Spectroscopy by Wavelet Transforms', Applied Spectroscopy pp. 1370 - 1376, September 2009.
"""
from ._dtcwt import dualtree, idualtree, approx_rec, detail_rec, dualtree_max_level, DEFAULT_CMP_WAV, DEFAULT_FIRST_STAGE, DEFAULT_MODE
from ._algorithms import baseline, denoise
from ._discrete import baseline_dwt, denoise_dwt
from ._wavelets import dualtree_wavelet, dualtree_first_stage, ALL_FIRST_STAGE, ALL_COMPLEX_WAV

__all__ = ['dualtree', 'idualtree', 'approx_rec', 'detail_rec', 'dualtree_max_level', 'baseline', 
           'denoise', 'baseline_dwt', 'denoise_dwt', 'dualtree_wavelet', 'dualtree_first_stage']