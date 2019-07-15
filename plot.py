# ===================================================================================== #
# Useful plotting functions.
# Author : Eddie Lee, edlee@alumni.princeton.edu
# ===================================================================================== #
import numpy as np
import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from matplotlib import patheffects


def bds2err(x, xbds):
    """Convert bounds for values of x to error bars that can be used for pyplot.errorbar.

    Parameters
    ----------
    x : ndarray
        Vector.
    xbds : ndarray or tuple of vectors
        Either column or rows of tuples.
    
    Returns
    -------
    ndarray
        (2, n_samples) that can be directly used in errorbar.
    """
    
    if type(xbds) is tuple:
        xbds = np.vstack(xbds)

    if xbds.shape[0]>xbds.shape[1]:
        xbds = xbds.T
    elif xbds.shape==(2,2):
        assert shape!=(2,2), "Shape is ambiguous for determining error bars."
    assert x.size==xbds.shape[1]

    errbds = np.zeros_like(xbds)
    errbds[0] = x-xbds[0]
    errbds[1] = xbds[1]-x

    return errbds

def utm_from_lon(lon):
    """
    utm_from_lon - UTM zone for a longitude

    Not right for some polar regions (Norway, Svalbard, Antartica)

    From https://stackoverflow.com/q/32333870/1532180

    :param float lon: longitude
    :return: UTM zone number
    :rtype: int
    """
    return np.floor( ( lon + 180 ) / 6) + 1

def scale_bar(ax, proj, length, location=(0.5, 0.05), linewidth=3,
              units='km', m_per_unit=1000, scale_bar_y_pos_factor=1.,
              compass_x_pos_factor=1, compass_y_pos_factor=1):
    """
    http://stackoverflow.com/a/35705477/1072212
    ax is the axes to draw the scalebar on.
    proj is the projection the axes are in
    location is center of the scalebar in axis coordinates ie. 0.5 is the middle of the plot
    length is the length of the scalebar in km.
    linewidth is the thickness of the scalebar.
    units is the name of the unit
    m_per_unit is the number of meters in a unit

    From https://stackoverflow.com/q/32333870/1532180
    """

    import cartopy.crs as ccrs

    # find lat/lon center to find best UTM zone
    x0, x1, y0, y1 = ax.get_extent(proj.as_geodetic())
    # Projection in metres
    utm = ccrs.UTM(utm_from_lon((x0+x1)/2))
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(utm)
    # Turn the specified scalebar location into coordinates in metres
    sbcx, sbcy = x0 + (x1 - x0) * location[0], y0 + (y1 - y0) * location[1]
    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbcx - length * m_per_unit/2, sbcx + length * m_per_unit/2]
    # buffer for scalebar
    buffer = [patheffects.withStroke(linewidth=5, foreground="w")]
    # Plot the scalebar with buffer
    ax.plot(bar_xs, [sbcy*scale_bar_y_pos_factor, sbcy*scale_bar_y_pos_factor], transform=utm, color='k',
            linewidth=linewidth, path_effects=buffer)
    # buffer for text
    buffer = [patheffects.withStroke(linewidth=3, foreground="w")]
    # Plot the scalebar label
    t0 = ax.text(sbcx, sbcy*1.01, str(length) + ' ' + units, transform=utm,
                horizontalalignment='center', verticalalignment='bottom', fontsize=15,
                path_effects=buffer, zorder=2)
    left = x0+(x1-x0)*0.05
    # Plot the N arrow
    t1 = ax.text(left*compass_x_pos_factor, sbcy*compass_y_pos_factor, u'\u25B2\nN', transform=utm,
                horizontalalignment='center', verticalalignment='bottom',
                fontsize=15,
                path_effects=buffer, zorder=2)
    # Plot the scalebar without buffer, in case covered by text buffer
    ax.plot(bar_xs, [sbcy*scale_bar_y_pos_factor, sbcy*scale_bar_y_pos_factor], transform=utm, color='k',
            linewidth=linewidth, zorder=3)

def _interp_r(t1,t2,r1,r2):
    """
    Linear interpolation in polar coordinates between theta1 and theta2.
    
    Params:
    -------
    t1,t2
        Angle.
    r1,r2
        Radius.
    """
    a = (r1-r2)/(t1-t2+np.nextafter(0,1))
    b = r1-a*t1
    return lambda t: a*t+b

def interp_r(t,r,dt=.1):
    """
    Linear interpolation for a set of points by looping _interp_r(). 
    
    Params:
    -------
    t (ndarray)
    r (ndarray)
    dt (float=.1)
        Approximate spacing for theta.
    """
    interpt,interpr = [],[]
    for i in range(1,len(t)):
        f = _interp_r( t[i-1],t[i],r[i-1],r[i] )
        interpt.append( np.linspace(t[i-1],t[i],abs(t[i]-t[i-1])//dt+2) )
        interpr.append( f(interpt[-1]) )
    return interpt,interpr

def smooth_polar_plot(ax,T,R,fmt='k-',plot_kwargs={}):
    """
    Plot linear interpolation for a set of points.
    
    Params:
    -------
    ax (plot axis)
    T (ndarray)
        (n_samples,n_points) matrix of angles. Each row is a separate line to plot.
    R
        (n_samples,n_points) matrix of radii. Each row is a separate line to plot.
    """
    for t,r in zip(T,R):
        interpt,interpr = interp_r(t,r)
        for t,r in zip(interpt,interpr):
            ax.plot( t,r,fmt,**plot_kwargs )

def colorcycle(n,scale=lambda i,n:1,cmap=plt.cm.viridis):
    """
    Generator for cycling colors through viridis. Scaling function allows you to rescale the color axis.

    Parameters
    ----------
    n : int
        Number of lines to plot.
    scale : lambda function
        Function that takes in current index of line and total number of lines. Examples include
        lambda i,n:exp(-i/2)*5+1
    cmap : colormap,plt.cm.viridis
    """
    if n>1:
        for i in range(n):
            yield cmap(i/(n-1)*scale(i,n))
    else:
        yield cmap(0)

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

