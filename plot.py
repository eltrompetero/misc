from __future__ import division
import numpy as np
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt

def colorcycle(n,scale=lambda i,n:1):
    """
    Generator for cycling colors through viridis. Scaling function allows you to rescale the color axis.

    Params:
    -------
    n (int)
        Number of lines to plot.
    scale (lambda)
        Function that takes in current index of line and total number of lines. Examples include
        lambda i,n:exp(-i/2)*5+1
    """
    for i in xrange(n):
        yield plt.cm.viridis(i/(n-1)*scale(i,n))

def set_ticks_radian( ax,dy=1.,axis='y' ):
    """
    Set the x or y axis tick labels to be in units of pi. Limited feature does not allow fractions of pi.
    2016-10-24
    
    Params:
    -------
    ax (axis handle)
    dy (float=1.)
        Step size for the tick labels in integer units of pi.
    axis (str='y')
        'y' or 'x'
    """
    if axis=='y':
        ylim = [i//np.pi for i in ax.get_ylim()]
    else:
        ylim = [i//np.pi for i in ax.get_xlim()]
    labels =[]
    for i in np.arange(ylim[0],ylim[1]*1.1,dy):
        if i==0:
            labels.append(r'$0$')
        elif i==1:
            labels.append(r'$\pi$')
        elif i==-1:
            labels.append(r'$-\pi$')
        else:
            labels.append( r'$%d\pi$'%i )
    if axis=='y':
        ax.set( yticks=np.arange(ylim[0],ylim[1]+.1,dy)*np.pi,
                yticklabels=labels, ylim=ax.get_ylim() )
    else:
        ax.set( xticks=np.arange(ylim[0],ylim[1]+.1,dy)*np.pi,
                xticklabels=labels, xlim=ax.get_xlim() )

def add_colorbar(fig,dimensions,cmap,label='',ticklabels=None):
    cbax = fig.add_axes(dimensions)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb1 = mpl.colorbar.ColorbarBase(cbax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    if not ticklabels is None:
        cb1.set_ticklabels(ticklabels)
    cb1.set_label(label)

def histogram_int( data, bins=10, xAsGeoMean=False ):
    """
    Discrete histogram normalized by the number of integers in each bin.
    2015-07-08
    
    Params:
    -------
    data (ndarray)
    bins (int or ndarray)
        fed directly into np.histogram
    xAsGeoMean (bool,False)
        return either arithmetic means or geometric means for x values
    
    Values:
    -------
    n (ndarray)
    x (ndarray)
        locations for plotting bins (not what np.histogram returns)
    """
    from scipy.stats import gmean
    
    n,x = np.histogram( data, bins=bins )
    nInts,_ = np.histogram( np.arange(x[-1]+1),bins=x )
    
    if xAsGeoMean:
        x = np.array([gmean(np.arange( np.ceil(x[i]),np.ceil(x[i+1]) )) if (np.ceil(x[i])-np.ceil(x[i+1]))!=0 else np.ceil(x[i])
                      for i in range(len(x)-1)])
    else:
        x = (x[:-1]+x[1:]) / 2.
    
    return n/nInts, x

