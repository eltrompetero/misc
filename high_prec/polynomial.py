from mpmath import mp
from mpmath import cos,sin,pi,legendre,mpf,matrix
import numpy as np


def tricomi_root(n,k):
    """To order n^4"""
    return (1-mpf(1)/mpf(8)*(1/n**2+1/n**3)) * cos(pi*(4*k-1)/(4*n+2))

def polish_root(n,approx_root,n_iters=None,tol=None):
    """Use Newton method to get close to root.
    """
    if n_iters is None:
        tol=10**(-mp.dps+5)
        n_iters=mp.dps
        
        dx=1.
        oldRoot=approx_root
        while abs(dx)>tol:
            newRoot=( oldRoot - legendre(n, oldRoot) * (oldRoot**2-1) / n / 
                                     (oldRoot*legendre(n, oldRoot)-legendre(n-1, oldRoot)) )
            dx=newRoot-oldRoot
            oldRoot=newRoot
        return newRoot
    
    else:
        oldRoot=approx_root
        for i in range(n_iters):
            newRoot=( oldRoot - legendre(n, oldRoot) * (oldRoot**2-1) / n / 
                                     (oldRoot*legendre(n, oldRoot)-legendre(n-1, oldRoot)) )
            oldRoot=newRoot
        return newRoot
    
def leggauss(n):
    """
    For Legendre-Gaussian quadrature.

    Parameters
    ----------
    n : int

    Returns
    -------
    roots : list of mpf
    weights : list of mpf
    """
    assert n>=2
    roots=[]
    weights=[]
    for k in range(n,n//2-1,-1):
        # Using Tricomi approximation to get high precision estimates of roots.
        roots.append( polish_root(n,tricomi_root(n,k)) )
        weights.append( 2 / legendre(n-1,roots[-1]) * (roots[-1]**2-1) / n**2 /
                       (roots[-1]*legendre(n, roots[-1]) - legendre(n-1, roots[-1])) )
    # roots are antisymmetric about 0
    for k in range(n//2-2,-1,-1):
        roots.append( -roots[k] )
        weights.append( weights[k] )
        
    return roots,weights
