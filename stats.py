# Module for helper functions with statistical analysis of data.
# Author: Eddie Lee
# 2017-01-17



import numpy as np
from numpy import fft




def acf(x,axis=-1,return_power=False):
    """
    Compute the autocorrelation function of a given time series. According to the Wiener-Khintchine theorem,
    the autocorrelation function and power spectrum are Fourier transform duals. The mean is subtracted
    <f(t)f(t+dt)>-<f(t)>^2
    2017-04-05

    Parameters
    ----------
    x : ndarray
    axis : int,-1
    return_power: bool,False
        If True, return power spectrum.
    """
    w = fft.fft(x-np.expand_dims(x.mean(axis=axis),axis),axis=axis)
    S = np.abs(w)**2
    # We know this must be real because the input signal is all real.
    acf = fft.ifft(S,axis=axis).real

    if x.ndim==1 or axis==0:
        if x.ndim>1:
            acf /= np.take(acf,[0],axis=axis)
        else:
            acf /= acf[0]

        acf = acf[:len(acf)//2]
    else:
        acf /= np.take(acf,[0],axis=axis)
        acf = acf[:,:acf.shape[1]//2]

    if return_power:
        return acf,S
    else:
        return acf

def _acf(x,maxlag,axis=-1):
    """
    Calculating autocorrelation function in slow way.
    2017-01-20
    """
    acf=np.ones((maxlag+1))
    for i in range(1,maxlag+1):
        acf[i]=np.corrcoef(x[:-i],x[i:])[0,1]
    return acf

def ccf(x,y,length=20):
    """
    Compute cross correlation function as a function of lag between two vectors.
    2016-12-08

    Params:
    -------
    x,y (vectors)
        Y will be dragged behind X.
    length (int=20 or list of ints)
    """
    from numpy import corrcoef

    if type(length) is int:
        c = np.zeros((length+1))
        c[0] = corrcoef(x,y)[0,1]
        for i in range(1,length+1):
            c[i] = corrcoef(x[:-i],y[i:])[0,1]
    elif type(length) is np.ndarray or type(length) is list:
        c = np.zeros((len(length)))
        for i,t in enumerate(length):
            if t==0:
                c[i] = corrcoef(x,y)[0,1]
            else:
                c[i] = corrcoef(x[:-t],y[t:])[0,1]
    else:
        raise Exception("length must be int or array of ints.")
    return c

def vector_ccf(x,y,length=20):
    """
    Compute cross correlation function between two vectors as the time-lagged, normalized dot product.
    2016-12-08

    Params:
    -------
    x,y (2d array)
        Each vector is a row.
    length (int=20 or list or ints)
    """
    from numpy.linalg import norm

    if type(length) is int:
        c = np.zeros((length+1))
        c[0] = ( (x*y).sum(1)/(norm(x,axis=1)*norm(y,axis=1)) ).mean()
        for i in range(1,length+1):
            c[i] = ( (x[:-i]*y[i:]).sum(1)/(norm(x[:-i],axis=1)*norm(y[i:],axis=1)) ).mean()
    elif type(length) is np.ndarray or type(length) is list:
        c = np.zeros((len(length)))
        for i,t in enumerate(length):
            if t==0:
                c[i] = ( (x*y).sum(1)/(norm(x,axis=1)*norm(y,axis=1)) ).mean()
            else:
                c[i] = ( (x[:-i]*y[i:]).sum(1)/(norm(x[:-i],axis=1)*norm(y[i:],axis=1)) ).mean()
    else:
        raise Exception("length must be int or array of ints.")
    return c

def max_likelihood_discrete_power_law(X,initial_guess=2.,lower_bound=1,minimize_kw={}):
    """
    Find the best fit power law exponent for a discrete power law distribution. Use full expression
    for finding the exponent alpha where X=X^-alpha that involves solving a transcendental equation.

    Parameters
    ----------
    X : ndarray
    initial_guess : float,2.
        Guess for power law exponent alpha.
    lower_bounds : int or list,1
        If list, then list of solns will be returned for each lower bound.
    minimize_kw : dict,{}

    Returns
    -------
    soln : scipy.optimize.minimize or list thereof
    """
    from scipy.special import zeta
    from scipy.optimize import minimize
    
    if type(lower_bound) is int:
        def f(alpha):
            if alpha<=1: return 1e30
            return (-alpha*zeta(alpha+1,lower_bound)/zeta(alpha,lower_bound)+np.log(X).mean())**2

        return minimize(f,initial_guess)
    
    soln=[]
    for lower_bound_ in lower_bound:
        def f(alpha):
            if alpha<=1: return 1e30
            return ( -alpha*zeta(alpha+1,lower_bound_)/zeta(alpha,lower_bound_) +
                     np.log(X[X>=lower_bound_]).mean() )**2
        soln.append( minimize(f,initial_guess,**minimize_kw) )
    return soln

def log_likelihood_discrete_power_law(X,alpha,lower_bound=1):
    """Log likelihood of the discrete power law with exponent X^-alpha.
    Parameters
    ----------
    X : ndarray
    alpha : float
    lower_bounds : int,1

    Returns
    -------
    log_likelihood : ndarray
    """
    from scipy.special import zeta
    return -alpha*np.log(X) - np.log(zeta(alpha,lower_bound))

