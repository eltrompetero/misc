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

def phase_dist(phi1,phi2):
    """
    Phase error in each moment of time this is a maximum of pi at each moment in time.
    
    Params:
    -------
    phi1,phi2 (ndarrays)
        n_time,n_dim
    """
    shape = phi1.shape
    
    dist = np.abs(phi1-phi2).ravel()
    dist[dist>np.pi] = np.pi - dist[dist>np.pi]%np.pi
    return dist.reshape(shape)
