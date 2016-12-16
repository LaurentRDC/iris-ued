"""
Author : Laurent P. Rene de Cotret

Ref.:
[1] NUMERICALLY STABLE DIRECT LEAST SQUARES FITTING OF ELLIPSES
"""
import numpy as n
from numpy.linalg import inv, eig

def ellipse_fit(x, y, circle_constraint = False):
    """
    Returns the ellipse parameters that fits data points. The special 
    case of a circle can also be treated.

    Parameters
    ----------
    x, y : ndarrays, shape (N,)
        Data points to be fitted.
    circle_constraint : bool, optional
        If True, a condition is added to fit a circle.
        Default is False.
    
    Returns
    -------
    a, b, c, d, e, f : numerical
        Conic section parameters for an ellipse.

    Raises
    ------
    RunTimeError
        If a solution cannot be found.

    Notes
    -----
    In the case of an ellipse, b**2 - 4*a*c < 0, while for a circle, b**2 - 4*a*c = 0.
    From the parameters, the center of the ellipse is given by (xc, yc) = (d/2, e/2) 
    """
    x, y = n.asarray(x, dtype = n.float), n.asarray(y, dtype = n.float)
    x, y = x.ravel(), y.ravel()

    # Translation of the code in Fig 2, ref [1]
    # quadratic part of the design matrix
    D1 = n.zeros(shape = (x.size, 3), dtype = n.float)
    D1[:,0] = x**2
    D1[:,1] = x*y
    D1[:,2] = y**2

    if circle_constraint:
        D1[:,1] = 0     # equivalent to forcing b = 0, or no x*y terms.

    # Linear part of the design matrix
    D2 = n.zeros(shape = (x.size, 3), dtype = n.float)
    D2[:,0] = x
    D2[:,1] = y
    D2[:,2] = 1

    # scatter matrix
    S1 = n.dot(D1.conj().transpose(), D1)      #S1
    S2 = n.dot(D1.conj().transpose(), D2)      #S2
    S3 = n.dot(D2.conj().transpose(), D2)      #S3

    C1 = n.array([ [0, 0, 2], [0, -1, 0], [2, 0, 0] ])
    
    # Solve 
    T = n.dot( -inv(S3), S2.conj().transpose() )
    M = S1 + n.dot(S2, T)
    M = n.dot(-inv(C1), M) 

    # Solution is the eigenvector associated with the minimal
    # nonzero eigenvalue of M.
    # To do this, we change the negative eigenvalues to +infinity
    # and find the minimal element of vals
    vals, vecs = eig(M)
    if vals.max() < 0: 
        raise RuntimeError('Unstable system?')
    vals[vals < 0] = n.inf
    min_pos_eig = vals.argmin()
    a1 = vecs[:, min_pos_eig].ravel()
    a2 = n.dot(T, a1) # Second half of coefficients

    # Ellipse coefficients from eq. 1
    a, b, c, d, e, f = tuple(a1) + tuple(a2)
    return a, b, c, d, e, f

def ellipse_center(x, y):
    """
    High-level function to find the center of an ellipse.

    Parameters
    ----------
    x, y : ndarrays, shape (N,)
        Data points to be fitted.
    
    Returns
    -------
    xc, yc : float
        Center position
    """
    a, b, c, d, e, f = ellipse_fit(x, y)
    return -d/(2*a), -e/(2*c)