from .globe import *
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

def test_SphereCoordinate():
    from numpy import pi
    np.random.seed(0)
    rotvec=np.random.rand(3)*2-1 
    rotvec/=np.linalg.norm(rotvec) 

    # Check that a full rotation returns you to the same point
    coord=SphereCoordinate(0,1)
    newcoord=coord.rotate(rotvec,2*pi)
    assert np.isclose([0,1], (newcoord.phi,newcoord.theta)).all(),(newcoord.phi,newcoord.theta)
    coord=SphereCoordinate(pi/7,2*pi/3)
    newcoord=coord.rotate(rotvec,2*pi)
    assert np.isclose([pi/7,2*pi/3], (newcoord.phi,newcoord.theta)).all(), (newcoord.phi,newcoord.theta)

    # Check that a full rotation (broken into two parts) returns you to the same point
    coord=SphereCoordinate(0,1)
    newcoord=coord.rotate(rotvec,np.pi).rotate(rotvec,np.pi)
    assert np.isclose([0,1], (newcoord.phi,newcoord.theta)).all(), (newcoord.phi,newcoord.theta)
    coord=SphereCoordinate(pi/7,2*pi/3)
    newcoord=coord.rotate(rotvec,np.pi).rotate(rotvec,np.pi)
    assert np.isclose([pi/7,2*pi/3], (newcoord.phi,newcoord.theta)).all(), (newcoord.phi,newcoord.theta)

    # Check that a rotation and its inverse return you to same point
    coord=SphereCoordinate(0,1)
    newcoord=coord.rotate(rotvec,np.pi).rotate(rotvec,-np.pi)
    assert np.isclose([0,1], (newcoord.phi,newcoord.theta)).all(), (newcoord.phi,newcoord.theta)
    
    coord=SphereCoordinate(0, pi/2)
    newcoord=coord.rotate(rotvec, 3*pi/2).rotate(rotvec, pi/2)
    assert np.isclose((coord.phi,coord.theta), (newcoord.phi,newcoord.theta), atol=1e-10).all(), (newcoord.phi,newcoord.theta)

def test_jitSphereCoordinate():
    from numpy import pi
    np.random.seed(0)
    rotvec=np.random.rand(3)*2-1 
    rotvec/=np.linalg.norm(rotvec) 

    # Check that a full rotation returns you to the same point
    coord=jitSphereCoordinate(0,1)
    newcoord=coord.rotate(rotvec,2*pi)
    assert np.isclose([0,1], (newcoord.phi,newcoord.theta)).all(),(newcoord.phi,newcoord.theta)
    coord=jitSphereCoordinate(pi/7,2*pi/3)
    newcoord=coord.rotate(rotvec,2*pi)
    assert np.isclose([pi/7,2*pi/3], (newcoord.phi,newcoord.theta)).all(), (newcoord.phi,newcoord.theta)

    # Check that a full rotation (broken into two parts) returns you to the same point
    coord=jitSphereCoordinate(0,1)
    newcoord=coord.rotate(rotvec,np.pi).rotate(rotvec,np.pi)
    assert np.isclose([0,1], (newcoord.phi,newcoord.theta)).all(), (newcoord.phi,newcoord.theta)
    coord=jitSphereCoordinate(pi/7,2*pi/3)
    newcoord=coord.rotate(rotvec,np.pi).rotate(rotvec,np.pi)
    assert np.isclose([pi/7,2*pi/3], (newcoord.phi,newcoord.theta)).all(), (newcoord.phi,newcoord.theta)

    # Check that a rotation and its inverse return you to same point
    coord=jitSphereCoordinate(0,1)
    newcoord=coord.rotate(rotvec,np.pi).rotate(rotvec,-np.pi)
    assert np.isclose([0,1], (newcoord.phi,newcoord.theta)).all(), (newcoord.phi,newcoord.theta)
    
    coord=jitSphereCoordinate(0, pi/2)
    newcoord=coord.rotate(rotvec, 3*pi/2).rotate(rotvec, pi/2)
    assert np.isclose((coord.phi,coord.theta), (newcoord.phi,newcoord.theta), atol=1e-10).all(), (newcoord.phi,newcoord.theta)

if __name__=='__main__':
    test_jitSphereCoordinate()
