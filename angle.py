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


class Quaternion():
    """Basic quaternion class.
    """
    def __init__(self,a,b,c,d):
        self.real=a
        self.vec=np.array([b,c,d])
        
    def inv(self):
        negvec=-self.vec
        return Quaternion(self.real,*negvec)
    
    def hprod(self,q):
        """Right side Hamiltonian product."""
        p=[self.real]+self.vec.tolist()
        q=[q.real]+q.vec.tolist()
        """Hamiltonian product between two quaternions."""
        return Quaternion( p[0]*q[0] -p[1]*q[1] -p[2]*q[2] -p[3]*q[3],
                           p[0]*q[1] +p[1]*q[0] +p[2]*q[3] -p[3]*q[2],
                           p[0]*q[2] -p[1]*q[3] +p[2]*q[0] +p[3]*q[1],
                           p[0]*q[3] +p[1]*q[2] -p[2]*q[1] +p[3]*q[0] )
    
    def rotmat(self):
        qr=self.real
        qi,qj,qk=self.vec
        return np.array([[1-2*(qj**2+qk**2), 2*(qi*qj-qk*qr), 2*(qi*qk+qj*qr)],
                         [2*(qi*qj+qk*qr), 1-2*(qi**2+qk**2), 2*(qj*qk-qi*qr)],
                         [2*(qi*qk-qj*qr), 2*(qj*qk+qi*qr), 1-2*(qi**2+qj**2)]])

    def rotate(self,r):
        """Rotate this quaternion by the rotation specified in the given quaternion. The rotation
        quaternion must be of form cos(theta/2) + (a i, b j, c k)*sin(theta/2)

        Parameters
        ----------
        r : Quaternion
        """
        return r.hprod( self.hprod( r.inv() ) )

    def __str__(self):
        return "Quaternion: [%1.3f,%1.3f,%1.3f,%1.3f]"%(self.real,self.vec[0],self.vec[1],self.vec[2])
    
    def __repr__(self):
        return "Quaternion: [%1.3f,%1.3f,%1.3f,%1.3f]"%(self.real,self.vec[0],self.vec[1],self.vec[2])
#end Quaternion
