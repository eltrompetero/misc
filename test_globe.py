from .globe import PoissonDiscSphere
from numpy import pi
import numpy as np


def test_PoissonDiscSphere():
    poissd=PoissonDiscSphere(pi/50,
                             fast_sample_size=5,
                             width_bds=(0,.5),
                             height_bds=(0,.5))
    poissd.sample()

    # make sure that closest neighbor is the closest one in the entire sample
    pt=np.array([.2,.3])
    nearestix=poissd.get_closest_neighbor(pt)
    d=poissd.dist(pt,poissd.samples)
    assert nearestix==np.argmin(d)

def test_quaternion():
    from numpy import sin,cos,pi
    
    # Check that rotations combine approriately
    theta=pi/4
    r=Quaternion(cos(theta/2),0,0,sin(theta/2))
    r2=r.hprod(r)
    assert np.isclose(r2.real,cos(pi/4)) and np.isclose(r2.vec[-1],sin(pi/4)),r2
    r3=r2.hprod(r)
    assert np.isclose(r3.real,cos(3*pi/8)) and np.isclose(r3.vec[-1],sin(3*pi/8)),r3
    r4=r3.hprod(r)
    assert np.isclose(r4.real,cos(pi/2)) and np.isclose(r4.vec[-1],sin(pi/2)),r4
    
    # compare with rotation matrix
    theta=pi/2
    p=Quaternion(0,1.,0,0)
    q=Quaternion(cos(theta/2),0,sin(theta/2),0)
    assert np.isclose( q.hprod(p.hprod(q.inv())).vec, q.rotmat().dot(p.vec) ).all()

    # check that inverse rotation on rotation return original vector
    theta=pi/2
    p=Quaternion(0,1.,0,0)
    q=Quaternion(cos(theta/2),0,sin(theta/2),0)
    assert np.isclose( p.vec, p.rotate(q).rotate(q.inv()).vec ).all()


    # Check that rotations combine approriately
    theta=pi/4
    r=jitQuaternion(cos(theta/2),0,0,sin(theta/2))
    r2=r.hprod(r)
    assert np.isclose(r2.real,cos(pi/4)) and np.isclose(r2.vec[-1],sin(pi/4)),r2
    r3=r2.hprod(r)
    assert np.isclose(r3.real,cos(3*pi/8)) and np.isclose(r3.vec[-1],sin(3*pi/8)),r3
    r4=r3.hprod(r)
    assert np.isclose(r4.real,cos(pi/2)) and np.isclose(r4.vec[-1],sin(pi/2)),r4
    
    # compare with rotation matrix
    theta=pi/2
    p=jitQuaternion(0,1.,0,0)
    q=jitQuaternion(cos(theta/2),0,sin(theta/2),0)
    assert np.isclose( q.hprod(p.hprod(q.inv())).vec, q.rotmat().dot(p.vec) ).all()

    # check that inverse rotation on rotation return original vector
    theta=pi/2
    p=jitQuaternion(0,1.,0,0)
    q=jitQuaternion(cos(theta/2),0,sin(theta/2),0)
    assert np.isclose( p.vec, p.rotate(q).rotate(q.inv()).vec ).all()

