# =============================================================================================== #
# Module for custom interpolation routines.
# Author: Eddie Lee
# 2018-04-24
# =============================================================================================== #
from __future__ import division
import numpy as np
from scipy.optimize import minimize
from scipy.special import binom as binomk



def binom_polyval(coeffs):
    order=len(coeffs)
    def p(n,coeffs=coeffs,order=order):
        t=np.zeros(len(n))
        for i,n_ in enumerate(n):
            t[i]=binomk(n_,np.arange(order)[::-1]).dot(coeffs)
        return t
    return p

def binom_polyfit(n,t,order):
    """
    Fit to polynomial expansion:
    a_n * binomk(x,n) + a_{n-1} * binomk(x,n-1) + ... + a_0

    Parameters
    ----------
    n : ndarray
        size of subgroup
    t : ndarray
        duration
    order : int
    """
    assert len(n)==len(t)
    assert order>0
    
    def cost(args):
        coeffs=args
        return ((binom_polyval(coeffs)(n)-t)**2).sum()
    
    return minimize(cost,np.zeros(order+1))['x']

def constrained_binom_polyfit(n,t,order,fixed_coeffs=(),return_full_output=False):
    """
    Parameters
    ----------
    n : ndarray
        size of subgroup
    t : ndarray
        duration
    order : int
    fixed_coeffs : list of twoples
        Each twople is the order that is fixed and the coeff value to fix it at.
    """
    if len(fixed_coeffs)>0:
        # Sort from highest order term to lowest.
        fixed_coeffs=sorted(fixed_coeffs,key=lambda x:x[0])[::-1]
        
        # Checks.
        # Cannot surpass highest order.
        for i,j in fixed_coeffs:
            assert i<=order
        # Only one constraint per power.
        assert len(np.unique([i[0] for i in fixed_coeffs]))==len(fixed_coeffs)
            
        fixed_coeffs=[(order-i-ix,j) for ix,(i,j) in enumerate(fixed_coeffs)]
    
    def cost(args):
        if len(fixed_coeffs)>0:
            coeffs=np.insert(args,*zip(*fixed_coeffs))
        else:
            coeffs=args
        return ((binom_polyval(coeffs)(n)-t)**2).sum()
    
    soln=minimize(cost,np.zeros(order+1-len(fixed_coeffs)))
    if return_full_output:
        if len(fixed_coeffs)>0:
            return np.insert(soln['x'],*zip(*fixed_coeffs))['x']
        return soln['x'],soln

    if len(fixed_coeffs)>0:
        return np.insert(soln['x'],*zip(*fixed_coeffs))
    return soln['x']
