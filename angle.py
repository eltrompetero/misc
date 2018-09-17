# Module for manipulating angle/phase data.
# Author: Edward D. Lee
# Email: edl56@cornell.edu
# 2017-03-29
import numpy as np
from numpy import sin,cos,arcsin,arccos,arctan2,pi

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


class SphereCoordinate():
    """Coordinate on a spherical surface. Contains methods for easy manipulation and movement 
    of points. Sphere is normalized to unit sphere.
    """
    def __init__(self,*args):
        """
        Parameters
        ----------
        (x,y,z) or vector or (phi,theta)
        """
        self.update_xy(*args)
            
    def update_xy(self,*args):
        if len(args)==2:
            phi,theta=args
            self.vec=np.array([cos(phi)*sin(theta),sin(phi)*sin(theta),cos(theta)])
            self.phi,self.theta=phi,theta
        else:
            assert len(args)==3 or len(args[0])==3
            if len(args)==3:
                args=np.array(args)
            else:
                self.vec=args[0]
                
            self.vec=self.vec/np.linalg.norm(self.vec)
            self.phi,self.theta=self._vec_to_angle(*self.vec)
    
    @classmethod
    def _angle_to_vec(cls,phi,theta):
        return np.array([cos(phi)*sin(theta),sin(phi)*sin(theta),cos(theta)])
    
    @classmethod
    def _vec_to_angle(cls,x,y,z):
        return arctan2(y,x)+pi/2, arccos(z)
            
    def random_shift(self,return_angle=False,bds=[0,1]):
        """
        Return a vector that is randomly shifted away from this coordinate. This is done by
        imagining that hte north pole is aligned along this vector and then adding a random angle
        and then rotating the north pole to align with this vector.

        Parameters
        ----------
        return_angle : bool,False
            If True, return random vector in form of a (phi,theta) pair.
        bds : tuple,[0,1]
            Bounds on uniform number generator to only sample between fixed limits of theta. This
            can be calculated using the formula
                (1+cos(theta)) / 2 = X
            where 0<=x<=1

        Returns
        -------
        randvec : ndarray
        """
        # setup rotation operation
        rotvec=np.cross(self.vec, np.array([0,0,1]))
        a,b=cos(self.theta/2),sin(self.theta/2)
        rotq=Quaternion(a, b*rotvec[0], b*rotvec[1], b*rotvec[2])
                
        # Add random shift while to north pole
        dphi, dtheta = (np.random.uniform(0, 2*np.pi),
                        np.arccos(2*np.random.uniform(*bds)-1))
        dvec=self._angle_to_vec(dphi, dtheta)
        randq=Quaternion(0, *dvec)
        
        # Rotate north pole to this vector's orientation
        if return_angle:
            return self._vec_to_angle( *randq.rotate(rotq.inv()).vec )
        return randq.rotate(rotq.inv()).vec
#end SphereCoordinate
