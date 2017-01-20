# Module for helper functions with statistical analysis of data.
# 2017-01-17

from __future__ import division
import numpy as np
from numpy import fft

def acf(x,axis=-1):
    """
    Compute the autocorrelation function of a given time series. According to the Wiener-Khintchine theorem,
    the autocorrelation function and power spectrum are Fourier transform duals.
    2017-01-17
    """
    w=fft.fft(x-x.mean(axis=axis),axis=axis)
    S=np.abs(w)**2
    acf=fft.ifft(S,axis=axis).real

    if x.ndim==1 or axis==0:
        if x.ndim>1:
            acf /= acf[0][None,:]
        else:
            acf /= acf[0]

        return acf[:len(acf)//2]
    else:
        acf = acf/acf[:,0][:,None]
        return acf[:,:acf.shape[1]//2]

def _acf(x,maxlag,axis=-1):
    """
    Calculating autocorrelation function in slow way.
    2017-01-20
    """
    acf=np.ones((maxlag+1))
    for i in xrange(1,maxlag+1):
        acf[i]=np.corrcoef(x[:-i],x[i:])[0,1]
    return acf
