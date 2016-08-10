from __future__ import division
import numpy as np
import numpy
import math
from multiprocess import Pool,cpu_count
from numba import jit

# ----------------------------------#
# Useful mathematical calculations. #
# ----------------------------------#
class QuadGauss(object):
    def __init__(self,order,lobatto=False):
        """
        Straightforward gaussian integration using Chebyshev polynomials with linear mapping of the bounds into [-1,1]. Most useful for a bounded interval.
        2016-08-09
        """
        from numpy.polynomial.chebyshev import chebval,chebgauss
        
        self.order = order
        self.N = order
        self.basis = [lambda x,i=i:chebval(x,[0]*i+[1]) for i in xrange(self.N+1)]
        
        # Lobatto collocation points.
        if lobatto:
            self.coX = -np.cos(np.pi*np.arange(self.N+1)/self.N)
            self.weights = np.zeros(self.order+1)+np.pi/self.N
            self.weights[0] = self.weights[-1] = np.pi/(2*self.N)
        else:
            self.coX,self.weights = chebgauss(self.N+1)
        self.basisCox = [b(self.coX) for b in self.basis]
        self.W = 1/np.sqrt(1-self.coX**2)  # weighting function we must remove
        self.W[np.isnan(self.W)] = 0.
        
        # Map bounds to given bounds or from given bounds to [-1,1].
        self.map_to_bounds = lambda x,x0,x1: (x+1)/2*(x1-x0) + x0
        self.map_from_bounds = lambda x,x0,x1: (x-x0)/(x1-x0)*2. - 1.

    def quad(self,f,x0,x1):
        """
        Params:
        -------
        f (lambda function)
            One dimensional function
        """
        return ( f(self.map_to_bounds(self.coX,x0,x1))/self.W ).dot(self.weights) * (x1-x0)/2
        
def finite_diff( mat,order,dx=1,**kwargs ):
    """
    Front end for calling different finite differencing methods. Will calculate down the first dimension.

    >5x speed up by using Cython
    2015-09-12
    
    Params:
    -------
    mat (ndarray)
    dx (float)
    order (int=1,2)
        Order of derivative approximation to use.
    """
    from calculus import finite_diff_1, finite_diff_2
    if mat.ndim==1:
        mat = mat[:,None]

    if order==1:
        return finite_diff_1(mat,dx,**kwargs)
    elif order==2:
        return finite_diff_2(mat,dx,**kwargs)
    else:
        raise Exception("Invalid order option.")

def finite_diff_1(mat,dx,axis=0,test=None):
    """
    Compute derivative using three-stencil with third order approximation to endpoints. 
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    2015-07-18
    """
    def center_stencil(x,i):
        return ( 1/4*x[i-2] -2*x[i-1] + 2*x[i+1] -1/4*x[i+2] ) / (3 * dx)

    def forward_stencil(x,i):
        #return ( x[i] -x[i-1] ) / dx
        return ( 3/2*x[i] -2*x[i-1] +1/2*x[i-2] ) / dx
        return ( 11/16*x[i] -3*x[i-1] +3/2*x[i-2] -1/3*x[i-3] ) / dx

    def backward_stencil(x,i):
        #return ( -x[i] + x[i+1] ) / dx
        return ( -3/2*x[i] +2*x[i+1] -1/2*x[i+2] ) / dx
        return ( -11/16*x[i] +3*x[i+1] -3/2*x[i+2] +1/3*x[i+3] ) / dx

    grad = np.array([ center_stencil(mat,i) for i in range(2,len(mat)-2) ])
    
    # Extrapolate endpoints to third order.
    return np.concatenate(( [ backward_stencil(mat,0), backward_stencil(mat,1) ],
                              grad,
                            [ forward_stencil(mat,len(mat)-2), forward_stencil(mat,len(mat)-1) ] ))

def _finite_diff(ax):
    """
    2015-03-17
    """
    # Testing code:
    phi = np.sin(np.linspace(0,3*np.pi,1000))
    ax.plot( finite_diff(np.tile(phi,(3,1)).T,1,axis=0) )
    ax.plot( finite_diff(np.tile(phi,(3,1)),1,axis=1).T )

def finite_diff_2(mat,dx,axis=0,test=None):
    """
    Compute second derivative using fourth order approximation with third order approximation to endpoints. 
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    2015-07-18
    """
    def center_stencil(x,i):
        return (-1/12*x[i-2] + 4/3*x[i-1] -5/2*x[i] + 4/3*x[i+1] -1/12*x[i+2]) / dx**2

    def forward_stencil(x,i):
        return (35/12*x[i] -26/3*x[i-1] +19/2*x[i-2] -14/3*x[i-3] +11/12*x[i-4]) / dx**2

    def backward_stencil(x,i):
        return (35/12*x[i] -26/3*x[i+1] +19/2*x[i+2] -14/3*x[i+3] +11/12*x[i+4]) / dx**2

    laplacian = np.array([center_stencil(mat,i) for i in xrange(2,mat.size-2)])
    
    # Extrapolate endpoints.
    return np.concatenate(( [backward_stencil(mat,0), backward_stencil(mat,1)],
                    laplacian,
                    [forward_stencil(mat,mat.size-2), forward_stencil(mat,mat.size-1)] ))

def trapz(y,dx=1):
    """
    Integration using Simpson's 2nd order (?) rule.
    2016-05-05
    """
    return ( y[0] + 4*y[1:-1:2].sum() + 2*y[2:-1:2].sum() + y[-1] ) *dx/3

def round_nearest( x, prec ):
    """
    Round x to nearest mulitples of prec.
    """
    return np.around(x/prec)*prec



# -------#
# Other  #
# -------#

def join_connected_one_pass(ix):
    """
    Take in list of components that overlap. Make one pass through list and combine connected components. You must iterate this function til no components are joined.
    2016-05-29
    """
    connected = [list(ix[0])]
    for i in ix[1:]:
        notFound = True
        j = 0
        while notFound and j<len(connected):
            if (i[0] in connected[j]) or (i[1] in connected[j]):
                connected[j] += i
                notFound = False
            j +=1
        if notFound:
            connected.append(tuple(i))

        connected = [np.unique(c).tolist() for c in connected]
    return connected

def join_connected(ix):
    """
    Take in list of lists of components and find the connected components (one's that share at least one element with another component in the list).
    2016-05-29
    
    Params:
    -------
    ix (list of lists)
        Each list is a set of element labels.
    """
    thisComponents = [i[:] for i in ix]
    ix = join_connected_one_pass(ix)
    noChange=False
    
    while not noChange:
        thatComponents = join_connected_one_pass(thisComponents)
        if len(thatComponents)==len(thisComponents):
            noChange=True
        else:
            thisComponents = [i[:] for i in thatComponents]
    return thisComponents

def parallelize( f ):
    """
    Decorator for duplicating function over several cores and return concatenated outputs.
    2016-03-01
    
    Params:
    -------
    f (lambda)
        Function that takes in a list of arguments can be duplicated.
    """
    from copy import deepcopy
    from itertools import chain
    instances = []
    nJobs = cpu_count()
    if nJobs<=1:
        raise Exception("Not enough cores to parallelize.")
        
    def parallelized(*args):
        # Make copies of args.
        instances = [args]
        for i in xrange(nJobs-1):
            instances.append( deepcopy(args) )
        
        # Wrap f so that the args can be properly expanded and handed over as an expanded list.
        def g(args):
            return f(*args)
    
        p = Pool(nJobs)
        output = zip( *p.map(g,instances) )
        p.close()
        
        combinedOutput = []
        for o in output:
            try:
                combinedOutput.append( list(chain(*o)) )
            except TypeError:
                combinedOutput.append( list(o) )
        return tuple(combinedOutput)
    return parallelized

def zip_args(*args):
    """
    Given a mixed set of list and int/float args, turn them into a set up tuples. This can be used to faciliate pipeline operations.
    2016-02-08
    """
    listOfTuples = []
    try:
        L = max([len(i) for i in args if type(i) is list])
    except ValueError:
        L = 1
    
    for i in xrange(L):
        listOfTuples.append([])
        for j in args:
            if type(j) is list:
                listOfTuples[-1].append(j[i])
            else:
                listOfTuples[-1].append(j)
    return [tuple(i) for i in listOfTuples]


def bootstrap_f(data,f,nIters,nSamples=-1):
    """
    Take given data nad compute function f on bootstrapped data.
    2016-01-18

    Params:
    -------
    data (list or ndarray)
        List of data to sample from
    f (function)
    nIters (int)   
        Number of times to bootstrap
    nSamples (int=-1)
        Number of samples of data to take per bootstrap. Default is to take the same size as the data

    Value:
    ------
    results (list)
        List of f used on bootstrapped data.
    """
    if nSamples==-1:
        nSamples = len(data)
    results = []

    if type(data) is list or data.ndim==1:
        for i in xrange(nIters):
            results.append( f(np.random.choice(data,size=nSamples)) )
    else:
        for i in xrange(nIters):
            randIx = np.random.randint(len(data),size=nSamples)
            results.append( f(data[randIx]) )
    return results

def add_colorbar(fig,dimensions,cmap):
    import matplotlib as mpl
    cbax = fig.add_axes(dimensions)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb1 = mpl.colorbar.ColorbarBase(cbax, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    #cb1.set_ticklabels(range(106,114))
    #cb1.set_label('Congressional sessions')

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

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def sort_mat(m,refIx=0,invert=False,returnindex=False):
    """
        Sort entries in a matrix such that for a selected row/col, all entries are ordered
        sequentially. Diagonal elements are not counted.
    
        invert : Sort ascending by default. From high to low if True.
        refIx : index of row with which to order
    2014-01-23
    """
    if m.shape[0]!=m.shape[1]:
        raise Exception("Matrix must be square")
    n = m.shape[0]
    sortIx = np.expand_dims(range(m.shape[0]),0)
    
    # Put refIx row and col in front to use below algorithm.
    swap_row(m,refIx,0)
    swap_col(m,refIx,0)
    swap_col(sortIx,refIx,0)
    
    # Select a row and swap to put max in front, and then perform same operation for the
    # corresponding col. Must switch between row and col to maintain symmetry. Think of
    # where the diagonal elements end up.
    for i in range(1,n-1):
        if invert:
            ix = np.argmax(m[i:,0])+i # ix of max ignoring diagonal
        else:
            ix = np.argmin(m[i:,0])+i # ix of min ignoring diagonal
        # Swap matrix row and col.
        swap_row(m,i,ix)
        swap_col(m,i,ix)
        # Swap elements of index.
        swap_col(sortIx,i,ix)
    # Put refIx row back into original location. I don't think it makes sense to put refIx back
    # into it's original location. The entire matrix has been swapped so putting it back won't
    # make it fit in. Keep it in the 0 location to show where the references are.
    #if not invert:
    #    swap_row(m,refIx,0)
    #    swap_col(m,refIx,0)
    #    swap_col(sortIx,refIx,0)

    if returnindex:
        return m, sortIx.flatten()
    else:
        return m

def swap_row(m,ix1,ix2):
   """
   2014-01-23
   """
   _row = m[ix1,:].copy()
   m[ix1,:] = m[ix2,:].copy()
   m[ix2,:] = _row.copy()
   return

def swap_col(m,ix1,ix2):
   """
   2014-01-23
   """
   _col = m[:,ix1].copy()
   m[:,ix1] = m[:,ix2].copy()
   m[:,ix2] = _col.copy()
   return

def local_max(x,ix0):
    """
    Start at ix0 and simple gradient ascent.
    2015-03-17

    Params:
    ----------
    x (ndarray)
    ix0 (int)
        starting index

    Values:
    ix (int)
        index where max is located
    """
    from warnings import warn
    
    atmax = False
    ix = ix0
    while not atmax:
        nx = x[ix]-x[ix-1]
        mx = x[ix]-x[ix+1]
        if nx>0 and mx<0:
            ix += 1
        elif nx<0 and mx>0:
            ix -= 1
        elif nx>0 and mx>0:
            atmax = True
        else:
            warn("Undefined max.")
            atmax = True
    return ix

def fit_quad(x,y,x0,params0=[-1,-1e4],eps=1e-3):
    """
    Fit quadratic given mean and no linear component.
    Args:
        x : 
        y : f(x)
        x0 (float) : location at which to center quadratic
        params0 (opt,list) : parameter start values
        eps (1e-3,opt,float) : step size for least square minimization
    2015-03-12
    """
    from scipy.optimize import leastsq

    g = lambda params: y - (params[1] + params[0] * (x-x0)**2)
    soln = leastsq( g,[-1,-1e4],epsfcn=eps )
    if soln[1] not in [1,2,3,4]:
        import warnings
        warnings.warn("Least squares did not converge (%d)." %soln[1])
    return soln[0]

def tail(f, n):
    """
    Get last n lines of the files using an exponential search. Copied from stackexchange.
    2015-03-06
    """
    assert n >= 0
    pos, lines = n+1, []
    while len(lines) <= n:
        try:
            f.seek(-pos, 2)
        except IOError:
            f.seek(0)
            break
        finally:
            lines = list(f)
        pos *= 2
    return lines[-n:]

def find_blocks(v,val=np.nan):
    """
    Find consecutive sequences of the same value.
    2015-01-25
    """
    
    if np.isnan(val):
        ix = np.argwhere(np.isnan(v)).flatten()
    else:
        ix = np.argwhere(v==val).flatten()
    ix = np.append(ix,ix[-1]+2)
    step = np.diff(ix)

    if ix.size==0:
        return
    
    l = [[ix[0]]]
    j = 0
    for i in step:
        if i>1:
            l[-1].append(ix[j])
            l.append([ix[j+1]])
        j += 1
    # Don't include last element that was inserted to ensure the last sequential block was counted.
    return np.array(l[:-1])

def find_nearest(m,val):
    """
    2014-12-19
    """
    return np.abs(m-val).argmin()

def zero_crossing(m):
    """
        Number of times a function crosses zero.
        2014-12-19
    """
    return np.sum( np.logical_or( np.logical_and(m[:-1]<0,m[1:]>0),
               np.logical_and( m[:-1]>0,m[1:]<0) ) )

def collect_sig_2side(data,nulls,p):
    """
        Collect values that are significant on a two tail test for a two dimensional array.
    2014-06-17
    """
    _data = np.reshape( data,data.shape+(1,) )
    _nulls = nulls - np.reshape( np.mean(nulls,2),data.shape+(1,) )
    
    plo = np.sum( _data>=_nulls,2 )/float(nulls.shape[2]) < p
    phi = np.sum( _data<=_nulls,2 )/float(nulls.shape[2]) < p
    return data[np.logical_or(plo,phi)]

def sub_times(tm1,tm0):
    """
        Subtract two datetime.time objects. Return difference in seconds.
    2014-05-23
    """
    import datetime
    t1 = datetime.datetime(100, 1, 1, tm1.hour, tm1.minute, tm1.second)
    t0 = datetime.datetime(100, 1, 1, tm0.hour, tm0.minute, tm0.second)
    return (t1-t0).total_seconds()

def add_secs(tm, secs):
    import datetime
    fulldate = datetime.datetime(100, 1, 1, tm.hour, tm.minute, tm.second)
    fulldate = fulldate + datetime.timedelta(seconds=secs)
    return fulldate.time()

def unravel_utri_asymm(ix,shape):
    """
        Inefficient method for taking index of flattened upper triangle array and giving back row/col index in full array.
    2014-04-15
    """
    m,n = shape
    if m>n:
        mx,mn = m,n
    elif n>=m:
        mx,mn = n,m

    matix = np.zeros((mn,mx),dtype=int)-1
    matix[triu_indices_asymm(mn,mx)] = range(mn*(mn-1)/2+(mx-mn)*mn)
    foundix = np.argwhere(ix==matix) 
   
    if m>n:
        return (foundix[:,1].flatten(),foundix[:,0].flatten())
    else:
        return foundix[:,0].flatten(),foundix[:,1].flatten()

def triu_asymm(mat):
    """
        Extract upper triangular elements from asymmetric array.
    2014-04-14
    """
    ix = triu_indices_from_array_asymm(mat)

    return mat[ix]

def triu_indices_from_array_asymm(mat):
    """
        Extract upper triangular elements from asymmetric array.
    2014-04-14
    """
    m,n = mat.shape
    return triu_indices_asymm(m,n)

def triu_indices_asymm(m,n):
    """
        Extract upper triangular elements from asymmetric array. The problem with asymmetric matrices is that if the longer dimension is along rows, then to extract every pairwise comparison you have to extract all elements below the diagonal (not above as is typically the case).
    2014-04-15
    """
    if m>n:
        mx,mn = m,n
    elif n>=m:
        mx,mn = n,m

    mask = np.zeros((mx,mx))
    mask[np.triu_indices(mx,k=1)] = 1
    ix = np.argwhere(mask[:mn,:mx]==1)
    
    if m>n:
        return [ix[:,1],ix[:,0]]
    else:
        return [ix[:,0],ix[:,1]]

def stack_dict(list,name,axis=0):
    """
        Take list of dicts and stack numpy arrays with same names from each dictionary
        into one long array.
    2014-02-23
    """
    _l = []
    for i in list:
        _l.append(i[name])

    if axis==0:
        return np.hstack(_l)
    else:
        return np.vstack(_l)

@jit(cache=True)
def unique_rows(mat,return_inverse=False):
    """
        Return unique rows indices of a numpy array.
        Args:
            mat :
            **kwargs
            return_inverse (bool) : if True, return inverse that returns back indices 
                of unique array that would return the original array 
        Value:
            idx : row indices of given mat that will give unique array
    2014-01-30
    """
    b = numpy.ascontiguousarray(mat).view(numpy.dtype((numpy.void, mat.dtype.itemsize * mat.shape[1])))
    if not return_inverse:
        _, idx = numpy.unique(b, return_index=True)
    else:
        _, idx = numpy.unique(b, return_inverse=True)

    return idx

def acf_breaks(x,y=None, length=20,iters=0):
    """
    2013-10-28
        Calculate autocorrelation while accounting for breaks in data represented by
        nan's. Account for nan's by ignoring data points whose corresponding lag is a nan.
        Originally for the macaque data. 

        Args:
            length : value of time lags -1 to consider. Can also input vector of
                values over which to iterate
            iters : number of iterations for bootstrap sampling.

        Value:
            samp : iters x length matrix of results of ACF calculated from bootstrap sampling
                while maintaining fixed nan (break) locations
    """
    x = x.copy()

    from macaq_confdyn.data_sets import aligntimes
    if np.isscalar(length):
        if length>0:
            runiters = True
            nlength = length
            length = np.arange(length)
    else:
        nlength = length.size
    acf = np.zeros((nlength))
    acf[0] = 1
    nanix = np.isnan(x)
    fightsix = np.where(nanix==0)[0] # index of non nan's

    if y==None:
        y = x
    else:
    	y = y.copy()
        if np.sum(nanix)!=np.sum(np.logical_and(nanix,np.isnan(y))):
            raise Exception("Given vectors x and y do not agree in location of nan's.")

    _ix = 0
    for i in length:
        x0, x1 = aligntimes( x.copy(),y.copy(),offset=i ) # copy!
        acf[_ix] = np.corrcoef(x0.flatten(), x1.flatten())[0,1]
        _ix += 1

    if iters>0:
        samp = np.zeros((iters,nlength))
        for i in np.arange(iters):
            # Shuffle only x values while keeping nan's in place.
            ix = np.random.randint( 0,fightsix.size,size=fightsix.size )
            _x = x.copy() # copy!
            _x[fightsix] = x[fightsix[ix]]

            _ix = 0
            for j in length:
                x0, x1 = aligntimes( _x,y,offset=j ) #copy!
                samp[i,_ix] = np.corrcoef(x0.flatten(), x1.flatten())[0,1]
                _ix += 1
        return acf, samp
    return acf

def acf(x, length=20,iters=0,nonan=True):
    """
        Autocorrelation coefficient including for masked arrays.
        Args:
            length{20,int}: time lags to do
            iters{0,int}: number of bootstrap samplestime lags to go up to
            nonan{True,bool}: ignore nans
    2014-09-30
    """
    if type(x) is not np.ma.core.MaskedArray:
        if iters==0:
            return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1] \
                for i in range(1, length)])
        else:
            samp = np.zeros((iters,length))
            for i in np.arange(iters):
                ix = np.random.randint(0,x.size,size=x.size)
                samp[i,:] = np.array([1]+[np.corrcoef(x[ix][:-j], x[ix][j:])[0,1] \
                    for j in range(1, length)])
            return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0,1] \
                for i in range(1, length)]), samp
    else:
        if iters==0:
            return np.array([1]+[np.ma.corrcoef(x[:-i], x[i:])[0,1] \
                for i in range(1, length)])
        else:
            samp = np.zeros((iters,length))
            for i in np.arange(iters):
                ix = np.random.randint(0,x.size,size=x.size)
                samp[i,:] = np.array([1]+[np.ma.corrcoef(x[ix][:-j], x[ix][j:])[0,1] \
                    for j in range(1, length)])
            return np.array([1]+[np.ma.corrcoef(x[:-i], x[i:])[0,1] \
                for i in range(1, length)]), samp

def fix_err_bars(y,yerr,ymn,ymx, dy=1e-10):
    """
    2013-10-25
        Return 2xN array of low error and high error points having removed negative
        values. Might need this when extent of error bars return negative values that are
        impossible.
        This fits right into pyplot.errobar() xerr or yerr options
    """
    yerru = yerr.copy()
    yerru[(y+yerr)>=ymx] -= ((y+yerr)-ymx)[(y+yerr)>=ymx] +dy
    yerrl = yerr.copy()
    yerrl[(y-yerr)<=ymn] += (y-yerr)[(y-yerr)<=ymn] -dy

    return np.vstack((yerrl,yerru))

def invert_f_lin_interp(f,xmin,xmax,prec=3):
    """
    2013-10-09
        Look up table inversion of given single argument function with desired precision
        and linear interpolation.
        xmin,xmax : inclusive endpoints
        prec : number of decimal points to which to round
    """
    x = np.arange(xmin,xmax+10**(-prec),10**(-prec))
    y = f(x)

    def g(y0):
        ix = np.argmin(abs(y-y0))
        if y[ix]==y.max():
            return x[ix]
        elif (y[ix]-y0)>0:
            return np.mean([x[ix],x[ix-1]])
        elif (y[ix]-y0)<0:
            return np.mean([x[ix],x[ix+1]])
        else:
            return x[ix]

    return g

def invert_f(f,xmin,xmax,prec=3):
    """
    2013-10-09
        Look up table inversion of given single argument function with desired precision.
        xmin,xmax : inclusive endpoints
        prec : number of decimal points to which to round
    """
    x = np.arange(xmin,xmax+10**(-prec),10**(-prec))
    y = f(x)
    def g(y0):
        return x[np.argmin(abs(y-y0))]

    return g

def get_ax():
    """
    2013-08-06
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=[5,4])
    ax = fig.add_subplot(111)

    return ax,fig

def hist_log(data,bins,
             density=False,
             x0=None,x1=None,base=10.):
    """
    Return log histogram of data. Usage just like regular histogram.
    2013-06-27

    Params:
    -------
    data (ndarray)
    bins (ndarray or int)
    x0,x1 (float)
        min and max x's
    base (float)
        base of histogram binning

    Values:
    -------
    n (ndarray)
        frequency count or probability density per bin
    x (ndarray)
        centered bins for plotting with n
    xedges (ndarray)
    """
    if x0==None:
        x0 = np.min(data)
    if x1==None:
        x1 = np.max(data)

    if not np.isscalar(bins):
        n,xedges = np.histogram( data,
                                 bins=bins,
                                 density=density )
    else:
        bins = np.logspace(np.log(x0)/np.log(base), np.log(x1)/np.log(base), bins+1 )
        bins[-1] = np.ceil(bins[-1])
        n,xedges = np.histogram( data, bins=bins, density=density )

    # take difference in log space to center bins
    dx = base**(np.diff(np.log(xedges)/np.log(base))/2.0)
    x = xedges[:-1]+dx

    return (n,x,xedges)

def read_text_data(fname):
    """Read simple text file with data.
    2012-08-13"""

    fid = open(fname)

    i = 0
    for line in fid:
        vals = line.split()
        if i==0:
            data = np.zeros((1,len(vals)))
            data[0,:] = vals
        else:
            data = np.append( data,np.zeros((1,len(vals))),axis=0 )
            data[i,:] = vals
        i += 1
    return data

def read_csv(fname):
    """
    2014-04-03
    """
    import csv
    with open(fname+'.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in spamreader:
            print ', '.join(row)
    return

def convert_utri_to_array(vec,diag,N):
    """
    Take a vector of the upper triangle and convert it to an array with
    diagonal elements given separately.
    2013-03-09
    """
    # Inumpyut checking.
    try:
    # structured like this to allow use of shortcut OR operator
        if isinstance(diag,(int)) or (diag.size==N):
            mat = np.zeros((N,N))
        else:
            raise
    except:
        if diag==np.nan:
            mat = np.zeros((N,N))

    # Initialize.
    mat = np.zeros((N,N))

    k=0
    for i in range(N-1):
        for j in range(i+1,N):
            mat[i,j] = vec[k]
            k+=1

    mat = mat+np.transpose(mat)
    mat[np.eye(N)==1] = diag
    return mat

def convert_sisj_to_cij(sisj,si):
    """
    2013-04-20
    """
    N = si.size
    NN = sisj.size
    Cij = np.zeros((NN))

    k=0
    for i in range(N-1):
        for j in range(i+1,N):
            Cij[k] = sisj[k]-si[i]*si[j]
            k+=1

    return Cij

def get_network(adj, widthfactor = 15):
    """
    2012"""
    import networkx as nx
    import math

    N = adj.shape[0]
    g1 = nx.Graph()
    g2 = nx.Graph()
    nodesizes = []
    nodecolors = []
    for i in range(N-1):
        g1.add_node(i+1)
        for j in range(i,N):
            if i!=j:
                if (adj[i][j])>0:
                    g1.add_edge( i+1,j+1,weight=math.fabs(adj[i][j]) )
                else:
                    g2.add_edge( i+1,j+1,weight=math.fabs(adj[i][j]) )
            else:
                nodesizes.append(math.fabs(adj[i][j]*1000))
                if adj[i][j]>0:
                    nodecolors.append('w')
                elif adj[i][j]<0:
                    nodecolors.append('k')
                else:
                    nodecolors.append('g')
            # else don't do anything because node size is not given

    edgewidth1 = []
    edgewidth2 = []
    for (u,v) in g1.edges():
        conn = g1.get_edge_data(u,v)['weight']
        edgewidth1.append (widthfactor*conn)

    for (u,v) in g2.edges():
        conn = g2.get_edge_data(u,v)['weight']
        edgewidth2.append (widthfactor*conn)

    return (g1,g2,edgewidth1,edgewidth2,nodesizes,nodecolors)

def get_network1(adj,names,h, edgewidthfactor = 15, noderadiifactor = 100, basicnodesize=1200):
    """
    2012"""
    import networkx as nx
    import math

    N = adj.shape[0]
    g1 = nx.Graph() # positive
    g2 = nx.Graph() # negative
    nodesizes = []
    nodecolors = []
    for i in range(N):
        g1.add_node(names[i])
        for j in range(i,N):
            if i!=j:
                if ~(math.fabs(adj[i][j])<.1):
                    if (adj[i][j])>0:
                        g1.add_edge( names[i],names[j],weight=math.fabs(adj[i][j]) )
                    else:
                        g2.add_edge( names[i],names[j],weight=math.fabs(adj[i][j]) )
            else:
                nodesizes.append(basicnodesize+(h[i]*noderadiifactor)**2*math.pi)
                nodecolors.append(adj[i][j])

    # Scale edge widths.
    edgewidth1 = []
    edgewidth2 = []
    for (u,v) in g1.edges():
        conn = g1.get_edge_data(u,v)['weight']
        edgewidth1.append (15*conn)
    for (u,v) in g2.edges():
        conn = g2.get_edge_data(u,v)['weight']
        edgewidth2.append (15*conn)

    return (g1,g2,edgewidth1,edgewidth2,nodesizes,nodecolors)

def pearson_corr(x,y):
    """
        Included case where sx or sy==0.
    2014-02-19
    """
    if x.size!=y.size:
        raise Exception('Vectors must be of same size.')

    mx = np.mean(x)
    my = np.mean(y)
    sx = np.std(x)
    sy = np.std(y)
    if sx==0 or sy==0:
        return 0.
    else:
        return np.sum((x-mx)*(y-my))/(sx*sy)/x.size
