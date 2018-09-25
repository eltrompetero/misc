# Module for useful functions on the 2D sphere.
import numpy as np
from numpy import cos,sin,arctan2,arccos,arcsin,pi
from .angle import Quaternion
from numba import jitclass, float64, njit, jit


def rand(n=1, degree=True):
    """Sample points from the surface of a sphere.

    Parameters
    ----------
    n : int,1
    degree : bool,True

    Returns
    -------
    randlon : float
    randlat : float
    """
    if degree:
        randlat=arccos(2*np.random.rand(n)-1)/pi*180-90
        randlon=np.random.uniform(-180,180,size=n)
        return randlon,randlat
    randlat=arccos(2*np.random.rand(n)-1)-pi/2
    randlon=np.random.uniform(-pi,pi,size=n)
    return randlon,randlat


class PoissonDiscSphere():
    """A class for generating two-dimensional Possion (blue) noise) modified from: 
    # For mathematical details of this algorithm, please see the blog
    # article at https://scipython.com/blog/poisson-disc-sampling-in-python/
    # Christian Hill, March 2017.
    """

    def __init__(self, r,
                 width_bds=(0,2*pi),
                 height_bds=(-pi/2,pi/2),
                 fast_sample_size=30,
                 k=30,
                 rng=None):
        assert r>0,r
        assert 0<=width_bds[0]<=width_bds[1]<=2*pi
        assert -pi/2<=height_bds[0]<=height_bds[1]<=pi/2

        self.width, self.height = width_bds, height_bds
        self.r = r
        self.k = k
        self.unif_theta_bounds=(1+np.cos(r))/2,(1+np.cos(2*r))/2
        self.fastSampleSize=fast_sample_size
        if rng is None:
            self.rng=np.random.RandomState()
        else:
            self.rng=rng

    def get_neighbours(self, xy, top_n=None):
        """Return top_n neighbors according to the fast Euclidean distance calculation.

        Parameters
        ----------
        xy : ndarray 
        top_n : int,None

        Returns
        -------
        neighbor_ix : list
        """

        top_n=top_n or self.fastSampleSize

        if len(self.samples)>1:
            d=self.fast_dist(xy, self.samples)
            return np.argsort(d)[:top_n]
        return []

    def _get_closest_neighbor(self, pt, ignore_zero=1e-9):
        """
        Parameters
        ----------
        pt : tuple
        ignore_zero : float,1e-9
            Distances smaller than this are ignored for returning the min distance.

        Returns
        -------
        ix : int 
        """

        distance=np.zeros(self.fastSampleSize)
        neighborix=self.get_neighbours(pt)

        for i,idx in enumerate(neighborix):
            nearby_pt = self.samples[idx]
            distance[i] = self.dist(nearby_pt,pt)
        if ignore_zero and len(neighborix)>0:
            keepix=distance>ignore_zero
            distance=distance[keepix]
            neighborix=neighborix[keepix]
        elif len(neighborix)==0:
            return []
        return neighborix[np.argmin(distance)]

    def get_closest_neighbor(self, pt, ignore_zero=1e-9):
        """
        Parameters
        ----------
        pt : tuple
        ignore_zero : float,1e-9
            Distances smaller than this are ignored for returning the min distance.

        Returns
        -------
        ix : int 
        """

        if pt.ndim==1:
            pt=pt[None,:]
        
        return [self._get_closest_neighbor(row, ignore_zero) for row in pt]

    def get_closest_neighbor_dist(self, pt, ignore_zero=1e-9):
        """
        Parameters
        ----------
        pt : tuple
        ignore_zero : float,1e-9
            Distances smaller than this are ignored for returning the min distance.

        Returns
        -------
        mindist : float
            On a unit sphere. Multiply by the radius to get it in the desired units.
        """

        distance=np.zeros(self.fastSampleSize)
        for i,idx in enumerate(self.get_neighbours(pt)):
            nearby_pt = self.samples[idx]
            # Squared distance between candidate point, pt, and this nearby_pt.
            distance[i] = self.dist(nearby_pt,pt)
        if ignore_zero:
            return distance[distance>ignore_zero].min()
        return distance.min()

    def point_valid(self, pt):
        """Is pt a valid point to emit as a sample?
        It must be no closer than r from any other point: check the cells in its immediate
        neighbourhood.
        """

        for idx in self.get_neighbours(pt):
            nearby_pt = self.samples[idx]
            # Squared distance between candidate point, pt, and this nearby_pt.
            distance = self.dist(nearby_pt,pt)
            if distance < self.r:
                # The points are too close, so pt is not a candidate.
                return False
        # All points tested: if we're here, pt is valid
        return True
    
    def get_point(self, refpt, max_iter=10):
        """Try to find a candidate point near refpt to emit in the sample.  We draw up to k points
        from the annulus of inner radius r, outer radius 2r around the reference point, refpt. If
        none of them are suitable (because they're too close to existing points in the sample),
        return False. Otherwise, return the pt in a list.
        """
        sphereRefpt=SphereCoordinate(refpt[0],refpt[1]+pi/2)
        i = 0
        while i < self.k:
            pt=sphereRefpt.random_shift(bds=self.unif_theta_bounds)
            # put back into same range as this code
            pt=np.array([pt[0],pt[1]-pi/2])
            if not (self.width[0] < pt[0] < self.width[1] and 
                    self.height[0] < pt[1] < self.height[1]):
                # This point falls outside the domain, so try again.
                continue
            if self.point_valid(pt):
                return [pt]
            i += 1
        # We failed to find a suitable point in the vicinity of refpt.
        return False

    def sample(self):
        """Poisson disc random sampling in 2D.
        Draw random samples on the domain width x height such that no two samples are closer than r
        apart. The parameter k determines the maximum number of candidate points to be chosen around
        each reference point before removing it from the "active" list.
        """
        # Pick a random point to start with.
        pt = np.array([self.rng.uniform(*self.width),
                       self.rng.uniform(*self.height)])
        self.samples = [pt]

        # and it is active, in the sense that we're going to look for more
        # points in its neighbourhood.
        active = [0]

        # As long as there are points in the active list, keep looking for
        # samples.
        while active:
            # choose a random "reference" point from the active list.
            idx = self.rng.choice(active)
            refpt = self.samples[idx]
            # Try to pick a new point relative to the reference point.
            pt = self.get_point(refpt)
            if pt:
                # Point pt is valid: add it to samples list and mark as active
                self.samples.append(pt[0])
                nsamples = len(self.samples) - 1
                active.append(nsamples)
            else:
                # We had to give up looking for valid points near refpt, so
                # remove it from the list of "active" points.
                active.remove(idx)
                
            #print("active samples left: %d"%len(active))
        
        self.samples=np.vstack(self.samples)
        if len(self.samples)<self.fastSampleSize:
            self.fastSampleSize=len(self.samples)
        return self.samples
    
    @classmethod
    def dist(cls,x,y):
        """Great circle distance"""

        from numpy import sin,cos

        if x.ndim==2 and y.ndim==1:
            return 2*arcsin( np.sqrt(sin((x[:,1]-y[:,1])/2)**2 +
                             cos(x[:,1])*cos(y[:,1])*sin((x[:,0]-y[:,0])/2)**2) )
        elif x.ndim==1 and y.ndim==2:
            return 2*arcsin( np.sqrt(sin((x[1]-y[:,1])/2)**2 +
                             cos(x[1])*cos(y[:,1])*sin((x[0]-y[:,0])/2)**2) )
        elif x.ndim==2 and y.ndim==2:
            return 2*arcsin( np.sqrt(sin((x[:,1]-y[1])/2)**2 +
                             cos(x[:,1])*cos(y[1])*sin((x[:,0]-y[0])/2)**2) )
        return 2*arcsin( np.sqrt(sin((x[1]-y[1])/2)**2+cos(x[1])*cos(y[1])*sin((x[0]-y[0])/2)**2) )
    
    @staticmethod
    def fast_dist(x,y):
        """Fast inaccurate Euclidean distance calculation accounting for periodic boundary
        conditions in phi.

        Parameters
        ----------
        x : ndarray
        y : ndarray

        Returns
        -------
        dvec : ndarray
        """
        # Account for discontinuity at phi=0 and phi=2*pi
        d=np.abs(x-y)
        ix=d[:,0]>pi
        d[ix,0]=pi-d[ix,0]%pi
        return ( d**2 ).sum(1)
#end PoissonDiscSphere


class SphereCoordinate():
    """Coordinate on a spherical surface. Contains methods for easy manipulation and movement 
    of points. Sphere is normalized to unit sphere.
    """

    def __init__(self, *args, rng=None):
        """
        Parameters
        ----------
        (x,y,z) or vector or (phi,theta)
        rng : np.random.RandomState,None
        """

        self.update_xy(*args)
        if rng is None:
            self.rng=np.random.RandomState()
        else:
            self.rng=rng
            
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
        return np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)])
    
    @classmethod
    def _vec_to_angle(cls,x,y,z):
        return arctan2(y,x)%(2*pi), arccos(z)
           
    def random_shift(self,return_angle=True,bds=[0,1]):
        """
        Return a vector that is randomly shifted away from this coordinate. This is done by
        imagining that hte north pole is aligned along this vector and then adding a random angle
        and then rotating the north pole to align with this vector.

        Angles are given relative to the north pole; that is, theta in [0,pi] and phi in [0,2*pi].

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
        if self.vec[-1]<-.5:
            # when vector is near south pole, numerical erros are dominant for the rotation and so we
            # move it to the northern hemisphere before doing any calculation
            vec=self.vec.copy()
            # move vector to the north pole
            vec[-1]*=-1
            theta=pi-self.theta
            inSouthPole=True
        else:
            vec=self.vec.copy()
            theta=self.theta
            inSouthPole=False
        
        # rotation axis given by cross product with (0,0,1)
        rotvec=np.array([vec[1], -vec[0], 0])
        rotvec/=np.sqrt(rotvec[0]**2 + rotvec[1]**2)
        a, b=cos(theta/2), sin(theta/2)
        rotq=jitQuaternion(a, b*rotvec[0], b*rotvec[1], b*rotvec[2])

        # Add random shift to north pole
        dphi, dtheta = (self.rng.uniform(0, 2*pi),
                        arccos(2*self.rng.uniform(*bds)-1))
        dvec=self._angle_to_vec(dphi, dtheta)
        randq=jitQuaternion(0, dvec[0], dvec[1], dvec[2])
        
        # Rotate north pole to this vector's orientation
        if return_angle:
            newphi, newtheta=self._vec_to_angle( *randq.rotate(rotq.inv()).vec )
            if inSouthPole:
                # move back to south pole
                newtheta=pi-newtheta
            return newphi, newtheta
        newvec=randq.rotate(rotq.inv()).vec
        if inSouthPole:
            # move back to south pole
            newvec[-1]*=-1
        return newvec
#end SphereCoordinate


spec=[
       ('vec',float64[:]),
       ('phi',float64),
       ('theta',float64)
     ]
@jitclass(spec)
class jitSphereCoordinate():
    """Coordinate on a spherical surface. Contains methods for easy manipulation and movement 
    of points. Sphere is normalized to unit sphere.
    """

    def __init__(self, phi, theta):
        """
        Parameters
        ----------
        phi : float
        theta : float
        """

        self.update_xy(phi, theta)
            
    def update_xy(self, phi, theta):
        self.vec=np.array([cos(phi)*sin(theta),sin(phi)*sin(theta),cos(theta)])
        self.phi,self.theta=phi,theta
    
    def _angle_to_vec(self,phi,theta):
        return np.array([cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)])
    
    def _vec_to_angle(self,x,y,z):
        return arctan2(y,x)%(2*pi), arccos(z)
           
    def random_shift(self, bds):
        """
        Return a vector that is randomly shifted away from this coordinate. This is done by
        imagining that hte north pole is aligned along this vector and then adding a random angle
        and then rotating the north pole to align with this vector.

        Angles are given relative to the north pole; that is, theta in [0,pi] and phi in [0,2*pi].

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
        if self.vec[-1]<-.5:
            # when vector is near south pole, numerical erros are dominant for the rotation and so we
            # move it to the northern hemisphere before doing any calculation
            vec=self.vec.copy()
            # move vector to the north pole
            vec[-1]*=-1
            theta=pi-self.theta
            inSouthPole=True
        else:
            vec=self.vec.copy()
            theta=self.theta
            inSouthPole=False
        
        # rotation axis given by cross product with (0,0,1)
        rotvec=np.array([vec[1], -vec[0], 0])
        rotvec/=np.sqrt(rotvec[0]**2 + rotvec[1]**2)
        a, b=cos(theta/2), sin(theta/2)
        rotq=jitQuaternion(a, b*rotvec[0], b*rotvec[1], b*rotvec[2])

        # Add random shift to north pole
        dphi=np.random.uniform(0, 2*pi)
        dtheta=arccos(2*np.random.uniform(bds[0], bds[1])-1)
        dvec=self._angle_to_vec(dphi, dtheta)
        randq=jitQuaternion(0, dvec[0], dvec[1], dvec[2])
        
        # Rotate north pole to this vector's orientation
        vec=randq.rotate(rotq.inv()).vec
        newphi, newtheta=self._vec_to_angle( vec[0], vec[1], vec[2] )
        if inSouthPole:
            # move back to south pole
            newtheta=pi-newtheta
        return newphi, newtheta
#end jitSphereCoordinate


spec=[
        ('real',float64),
        ('vec',float64[:])
    ]

@jitclass(spec)
class jitQuaternion():
    """Faster quaternion class with limited functionality.
    """
    def __init__(self,a,b,c,d):
        a*=1.
        b*=1.
        c*=1.
        d*=1.
        self.real=a
        self.vec=np.array([b,c,d])
        
    def inv(self):
        negvec=-self.vec
        return jitQuaternion(self.real, negvec[0], negvec[1], negvec[2])
    
    def hprod(self,t):
        """Right side Hamiltonian product."""
        p=[self.real, self.vec[0], self.vec[1], self.vec[2]]
        t=[t.real, t.vec[0], t.vec[1], t.vec[2]]
        """Hamiltonian product between two quaternions."""
        return jitQuaternion( p[0]*t[0] -p[1]*t[1] -p[2]*t[2] -p[3]*t[3],
                              p[0]*t[1] +p[1]*t[0] +p[2]*t[3] -p[3]*t[2],
                              p[0]*t[2] -p[1]*t[3] +p[2]*t[0] +p[3]*t[1],
                              p[0]*t[3] +p[1]*t[2] -p[2]*t[1] +p[3]*t[0] )
    
    def rotmat(self):
        qr=self.real
        qi, qj, qk=self.vec
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
#end jitQuaternion


class Quaternion():
    """Basic quaternion class.
    """
    def __init__(self,a,b,c,d):
        self.real=a
        self.vec=np.array([b,c,d])
        
    def inv(self):
        negvec=-self.vec
        return Quaternion(self.real,*negvec)
    
    def hprod(self,t):
        """Right side Hamiltonian product."""
        p=[self.real]+self.vec.tolist()
        t=[t.real]+t.vec.tolist()
        """Hamiltonian product between two quaternions."""
        return Quaternion( p[0]*t[0] -p[1]*t[1] -p[2]*t[2] -p[3]*t[3],
                           p[0]*t[1] +p[1]*t[0] +p[2]*t[3] -p[3]*t[2],
                           p[0]*t[2] -p[1]*t[3] +p[2]*t[0] +p[3]*t[1],
                           p[0]*t[3] +p[1]*t[2] -p[2]*t[1] +p[3]*t[0] )
    
    def rotmat(self):
        qr=self.real
        qi, qj, qk=self.vec
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

    def __eq__(self,y):
        if not type(y) is Quaternion:
            raise NotImplementedError

        if y.real==self.real and np.array_equal(y.vec,self.vec):
            return True
        return False
#end Quaternion
