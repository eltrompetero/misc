# Module for helper functions with statistical analysis of data.
# 2017-01-17

from __future__ import division
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
    for i in xrange(1,maxlag+1):
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
        for i in xrange(1,length+1):
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
        for i in xrange(1,length+1):
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

