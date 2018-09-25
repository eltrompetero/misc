from .angle import *


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
