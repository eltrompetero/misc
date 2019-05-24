# ===================================================================================== #
# Module for implementation of common numpy functions that do not work in jit environment.
# Author : Eddie Lee, edlee@alumni.princeton.edu
# ===================================================================================== #
import numpy as np
from numba import jit, njit


@njit(cache=True)
def ind_to_sub(n, ix):
    """Convert index from flattened upper triangular matrix to pair subindex.

    Parameters
    ----------
    n : int
        Dimension size of square array.
    ix : int
        Index to convert.

    Returns
    -------
    subix : tuple
        (i,j)
    """

    k = 0
    for i in range(n-1):
        for j in range(i+1,n):
            if k==ix:
                return (i,j)
            k += 1

@jit(nopython=True,cache=True)
def all(X,axis=0):
    """Right now, only implemented for two dimensional arrays.
    
    Params:
    -------
    X
    axis (int=0)
    """
    if axis==0:
        out = np.zeros(X.shape[1])==0
        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                out[i] &= X[j,i]!=0
    elif axis==1:
        out = np.zeros(X.shape[0])==0
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                out[i] &= X[i,j]!=0
    else:
        raise Exception("Only 2-d arrays accepted.")

    return out

@jit(nopython=True,cache=True)
def any(X,axis=0):
    """
    Right now, only implemented for two dimensional arrays.
    
    Params:
    -------
    X
    axis (int=0)
    """
    raise NotImplementedError
    if X.ndim==1:
        for x in X:
            if x:
                return True
        return False

    if axis==0:
        out = np.zeros(X.shape[1])==1
        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                if X[j,i]:
                    out[i] = True
                    break
    elif axis==1:
        out = np.zeros(X.shape[0])==0
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i,j]:
                    out[i] = True
                    break
    else:
        raise Exception("Only 2-d arrays accepted.")

    return out
