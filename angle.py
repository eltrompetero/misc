# Module for manipulating angle/phase data.
# Author: Edward D. Lee
# Email: edl56@cornell.edu
# 2017-03-29
import numpy as np

def mod_angle(angle):
    """
    Modulus into (-pi,pi).
    
    Params:
    -------
    angle (ndarray)
    """
    return np.mod(angle+np.pi,2*np.pi)-np.pi

def phase_dist(phi1,phi2=None):
    """
    Phase error in each moment of time this is a maximum of pi at each moment
    in time because distance is measured on a wrapped domain from [0,2*pi].
    
    Parameters
    ----------
    phi1 : ndarray
    phi2 : ndarray,None
        n_time,n_dim

    Returns
    -------
    dist : ndarray
    """
    shape = phi1.shape
    
    if phi2 is None:
        dist = np.abs(phi1).ravel()
    else:
        dist = np.abs(phi1-phi2).ravel()
    dist[dist>np.pi] = np.pi - dist[dist>np.pi]%np.pi
    return dist.reshape(shape)
