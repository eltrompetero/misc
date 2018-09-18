from .angle import *

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

def plot_test():
    """Make sure that the red x is inside the halo of blue dots."""
    import matplotlib.pyplot as plt
    deviationfrompole=.1

    fig,ax=plt.subplots(figsize=(10,4),ncols=2)

    sxy=SphereCoordinate(pi+.5,pi-deviationfrompole)
    xy=[sxy.random_shift(bds=((1+cos(.05))/2,(1+cos(.1))/2)) for i in range(1000)]
    xy=vstack(xy)

    ax[0].plot(xy[:,0],xy[:,1],'o')
    ax[0].plot(sxy.phi,sxy.theta,'rx',mew=3)

    sxy=SphereCoordinate(pi+.5,deviationfrompole)
    xy=[sxy.random_shift(bds=((1+cos(.05))/2,(1+cos(.1))/2)) for i in range(1000)]
    xy=vstack(xy)

    ax[1].plot(xy[:,0],xy[:,1],'o')
    ax[1].plot(sxy.phi,sxy.theta,'rx',mew=3) 
