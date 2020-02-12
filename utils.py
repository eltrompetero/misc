# ===================================================================================== #
# Module for useful functions.
# Author : Eddie Lee, edlee@alumni.princeton.edu
# ===================================================================================== #
import numpy as np
import math
from multiprocess import Pool,cpu_count
from numba import jit,njit
from numbers import Number
from scipy.spatial.distance import pdist
from .easy_jit import *
i0p=(9.999999999999997e-1,2.466405579426905e-1,
	1.478980363444585e-2,3.826993559940360e-4,5.395676869878828e-6,
	4.700912200921704e-8,2.733894920915608e-10,1.115830108455192e-12,
	3.301093025084127e-15,7.209167098020555e-18,1.166898488777214e-20,
	1.378948246502109e-23,1.124884061857506e-26,5.498556929587117e-30)
i0q=(4.463598170691436e-1,1.702205745042606e-3,
	2.792125684538934e-6,2.369902034785866e-9,8.965900179621208e-13)
i0pp=(1.192273748120670e-1,1.947452015979746e-1,
	7.629241821600588e-2,8.474903580801549e-3,2.023821945835647e-4)
i0qq=(2.962898424533095e-1,4.866115913196384e-1,
      1.938352806477617e-1,2.261671093400046e-2,6.450448095075585e-4,
      1.529835782400450e-6)


# ================================= #
# Useful mathematical calculations. #
# ================================= #
def maj_min_axis_ratio(xy):
    """Find major axis of points in plane, take projection onto orthogonal axis and take
    ratio of those distances.

    Parameters
    ----------
    xy : ndarray
        xy coordinates
        
    Returns
    -------
    float
        Ratio of major and minor axis.
    """

    xy = np.unique(xy, axis=0)
    majix = max_dist_pair2D(xy)
    majAxis = xy[majix[0]] - xy[majix[1]]

    if majAxis[1]!=0:
        minAxis = np.array([1, -majAxis[0]/majAxis[1]])
    else:
        minAxis = np.array([0.,1.])
    minAxis /= np.linalg.norm(minAxis)
    minProj = xy.dot(minAxis)

    return (np.linalg.norm(majAxis) /
            np.linalg.norm(xy[minProj.argmax()]-xy[minProj.argmin()]))

def ortho_plane(v):
    """Return a plane defined by two vectors orthogonal to the given vector using random
    vector and Gram-Schmidt.
    
    Parameters
    ----------
    v : ndarray
    
    Returns
    -------
    ndarray
    ndarray
    """
    
    assert v.size==3
    
    # Get a first orthogonal vector
    r1 = np.random.rand(3)
    r1 -= v*r1.dot(v)
    r1 /= np.sqrt(r1.dot(r1))
    
    # Get second othorgonal vector
    r2 = np.cross(v,r1)
    
    return r1, r2


def max_dist_pair2D(xy, force_slow=False, return_dist=False):
    """Find most distant pair of points in 2D Euclidean space.

    Maximally distant pair of points must coincide with extrema of convex hull.

    Parameters
    ----------
    xy : ndarray
        (x,y) coordinations
    force_slow : bool, False
        Use slow calculation computing entire matrix of pairwise distances.
    return_dist : bool, False

    Returns
    -------
    tuple
        Indices of two max separated points.
    """
    
    if type(xy) is list:
        xy = np.vstack(xy)

    # it is faster to do every pairwise computation when the size of the is small
    if force_slow or len(xy)<500:
        return _max_dist_pair(xy, return_dist)
    
    hull = convex_hull(xy, recursive=True)
    dist = pdist(xy[hull])
    mxix = ind_to_sub(hull.size, dist.argmax())
    if return_dist:
        return (hull[mxix[0]], hull[mxix[1]]), dist.max()
    return hull[mxix[0]], hull[mxix[1]]

def _max_dist_pair(xy, return_dist):
    """Slow way of finding maximally distant pair by checking every pair.
    """
    
    assert len(xy)>1
    dmat = pdist(xy)
    dmaxix = dmat.argmax()
    majix = ind_to_sub(len(xy), dmaxix)
    if return_dist:
        return majix, dmat[dmaxix]
    return majix

def convex_hull(xy, recursive=False, concatenate_first=False):
    """Identify convex hull of points in 2 dimensions. I think this is the same as
    Quickhull.
    
    Recursive version. Number of points to consider typically goes like sqrt(n), so
    this can handle a good number, but this could be made faster and to handle larger
    systems by making it sequential.

    This has been tested visually on a number of random examples for the armed_conflict
    project.
    
    Parameters
    ----------
    xy : ndarray
        List of coordinates.
    concatenate_first : bool, False
        If True, will append first coordinate again at end of returned list for a closed
        path.
        
    Returns
    -------
    list
        Indices of rows in xy that correspond to the convex hull. It is traversed in a
        clockwise direction.
    
    Example
    -------
    >>> xy = np.random.normal(size=(100,2))
    >>> hull = convex_hull(xy)
    >>> fig, ax = plt.subplots()
    >>> for i in range(10):
    >>>     ax.text(xy[i,0], xy[i,1], i)
    >>> ax.plot(*xy.T,'o')
    >>> ax.plot(*xy[4],'o')
    >>> ax.plot(*xy[9],'o')
    >>> ax.plot(xy[hull][:,0], xy[hull][:,1], 'k-')
    """
    
    if len(xy)<=3:
        return np.arange(len(xy), dtype=int)
    assert xy.shape[1]==2, "This only works for 2D."
    assert len(np.unique(xy,axis=0))==len(xy), "No duplicate entries allowed."
    
    # going around clockwise, get the extrema along each axis
    endptsix = [xy[:,0].argmin(), xy[:,1].argmax(),
                xy[:,0].argmax(), xy[:,1].argmin()]
    # remove duplicates
    if endptsix[0]==endptsix[1]:
        endptsix.pop(0)
    elif endptsix[1]==endptsix[2]:
        endptsix.pop(1)
    elif endptsix[2]==endptsix[3]:
        endptsix.pop(2)
    elif endptsix[3]==endptsix[0]:
        endptsix.pop(0)
    
    if recursive:
        pairsToConsider = [(endptsix[i], endptsix[(i+1)%len(endptsix)])
                           for i in range(len(endptsix))]
        
        # for each pair, assembly a list of points to check by using a cutting region determined
        # by the line passing through that pair of points
        pointsToCheck = []
        for i,j in pairsToConsider:
            ix = np.delete(range(len(xy)), endptsix)
            pointsToCheck.append( ix[_boundaries_diag_cut_out(xy[ix], xy[i], xy[j])] )
        
        # whittle 
        hull = []
        for ix, checkxy in zip(pairsToConsider, pointsToCheck):
            subhull = []
            _check_between_pair(xy, ix[0], ix[1], checkxy, subhull)
            hull.append(subhull)

        # extract loop
        hull = np.concatenate(hull).ravel()
        # monkey patch because some elements appear twice
        hull = np.append(hull[::2], hull[-1])
        _, ix = np.unique(hull, return_index=True)
        hull = hull[ix[np.argsort(ix)]]
        if concatenate_first:
            hull = np.concatenate((hull, [hull[0]]))
        return hull
    
    # sequential Graham algorithm
    # center all the points about a centroid defined as between min/max pairs of x and y axes
    midxy = (xy.max(0)+xy.min(0))/2
    xy = xy-midxy
    sortix = _sort_by_phi(xy)[::-1]
    xysorted = xy[sortix]
    # start with point with leftmost point
    startix = xysorted[:,0].argmin()
    sortix = np.roll(sortix, -startix)
    xysorted = np.roll(xysorted, -startix, axis=0)
    hull = _check_between_triplet(xysorted)

    return sortix[hull]

def _sort_by_phi(xy):
    """Sort points in play by angle in counterclockwise direction. With unique angles.
    """

    phi = np.arctan2(xy[:,1], xy[:,0])

    # if angle repeats, remove coordinate with smaller radius
    if phi.size>np.unique(phi).size:
        _, invIx = np.unique(phi, return_inverse=True)
        ixToRemove = []
        # for every element of phi that repeats
        for ix in np.where(np.bincount(invIx)>1)[0]:
            # take larger radius
            r = np.linalg.norm(xy[invIx==ix], axis=1)
            mxix = r.argmax()
            remIx = np.where(invIx==ix)[0].tolist()
            remIx.pop(mxix)
            ixToRemove += remIx
        
        # remove duplicates
        keepix = np.delete(range(phi.size), ixToRemove)

        sortix = keepix[np.argsort(phi[keepix])]
        return sortix

    return np.argsort(phi)

def _check_between_triplet(xy):
    """Used by convex_hull().

    Sequentially checks between sets of three points that have been ordered in the
    clockwise direction (Graham algorithm).
    
    Parameters
    ----------
    xy : ndarray
        List of Cartesian coordinates. First point must belong to convex hull.

    Returns
    -------
    list
        Ordered list of indices of xy that are in convex hull.
    """
    
    hull = list(range(len(xy)))
    k = 0
    # end loop once we can traverse hull without eliminating points
    allChecked = 0
    while allChecked<len(hull):
        if _boundaries_diag_cut_out(xy[hull[(k+1)%len(hull)]][None,:],
                                    xy[hull[k%len(hull)]],
                                    xy[hull[(k+2)%len(hull)]])[0]:
            k += 1
            allChecked += 1
        else:
            # remove element that forms a concave angle with next neighbors
            hull.pop((k+1)%len(hull))
            k -= 1
            allChecked = 0
    return hull

def _check_between_pair(xy, ix1, ix2, possible_xy, chain):
    """Used by convex_hull().

    Recursively check between initial set of pairs and append results into chain such that
    chain can be read sequentially to yield a clockwise path around the hull..
    
    Parameters
    ----------
    xy : ndarray
        List of coordinates.
    ix1 : int
    ix2 : int
    possible_xy: ndarray
        List of indices.
    chain : list
        Growing list of points on convex hull.
    """
    
    pointsToCheck = possible_xy[_boundaries_diag_cut_out(xy[possible_xy], xy[ix1], xy[ix2])]
    if len(pointsToCheck)==1:
        chain.append((ix1, pointsToCheck[0]))
        chain.append((pointsToCheck[0], ix2))
        return
    if len(pointsToCheck)==0:
        chain.append((ix1, ix2))
        return
    
    # take the point that's furthest from the line passing thru xy1 and xy2
    xy1 = xy[ix1]
    xy2 = xy[ix2]
    furthestix = np.abs((xy2[1]-xy1[1])*xy[pointsToCheck][:,0]-
                        (xy2[0]-xy1[0])*xy[pointsToCheck][:,1]+
                        xy2[0]*xy1[1]-xy2[1]*xy1[0]).argmax()

    _check_between_pair(xy, ix1, pointsToCheck[furthestix], pointsToCheck, chain),
    _check_between_pair(xy, pointsToCheck[furthestix], ix2, pointsToCheck, chain)

@njit
def _boundaries_diag_cut_out(xy, xy1, xy2):
    """Used by convex_hull() to find points that are above or below the line passing thru
    xy1 and xy2.
    
    Parameters
    ----------
    xy : ndarray
        Points to test.
    xy1 : ndarray
        Origin point.
    xy2 : ndarray
        Next point in clockwise direction.

    Returns
    -------
    function
    """
    
    if xy2[0]==xy1[0]:
        return np.zeros(len(xy))==1
    dydx = (xy2[1]-xy1[1])/(xy2[0]-xy1[0])
    if xy1[0]<=xy2[0] and xy1[1]<=xy2[1]:
        return xy[:,1]>(dydx*(xy[:,0]-xy1[0])+xy1[1])
    elif xy1[0]<=xy2[0] and xy1[1]>=xy2[1]:
        return xy[:,1]>(dydx*(xy[:,0]-xy1[0])+xy1[1])
    elif xy1[0]>=xy2[0] and xy1[1]>=xy2[1]:
        return xy[:,1]<(dydx*(xy[:,0]-xy1[0])+xy1[1])
    #elif xy1[0]>=xy2[0] and xy1[1]<=xy2[1]:
    else:
        return xy[:,1]<(dydx*(xy[:,0]-xy1[0])+xy1[1])
    #else: raise Exception

def weighted_corrcoef(x,y,w):
    """
    Params:
    --------
    x (ndarray)
    y (ndarray)
    w (ndarray)
        Weights.
    """
    w /= w.sum()
    
    mx = x.dot(w)
    my = y.dot(w)
    
    covxx = ((x-mx)*(x-mx)).dot(w)
    covyy = ((y-my)*(y-my)).dot(w)
    covxy = ((x-mx)*(y-my)).dot(w)
    
    return covxy/np.sqrt(covxx*covyy)

@jit(nopython=True,cache=True)
def poly(c,x):
    """
    Smart way of calculating polynomial.
    2016-08-14
    """
    y = c[-1]
    for i in c[:-1][::-1]:
        y = y*x + i
    return y

@jit(nopython=True,cache=True)
def iv(x,v=0):
    """
    Calculate of Bessel function. Only implemented for order v=0.
    2016-08-14
    """
    ax = np.abs(x)
    y = np.zeros_like(x)
    
    for i,ix in enumerate(x):
        if ax[i]<15.:
            iy = ix*ix
            y[i] = poly(i0p[:14],iy) / poly(i0q[:5],225-iy)
        else: 
            z = 1.-15./ax[i]
            y[i] = np.exp(ax[i])*poly(i0pp[:5],z) / ( poly(i0qq[:6],z)*np.sqrt(ax[i]) )
    return y



# ===== #
# Other #
# ===== #
def merge(sets):
    """Merge a list of sets such that any two sets with any intersection are merged."""
    merged = 1  # has a merge occurred?
    while merged:
        merged = 0
        results = []
        while sets:
            common, rest = sets[0], sets[1:]
            sets = []
            for x in rest:
                if x.isdisjoint(common):
                    sets.append(x)
                else:
                    merged = 1
                    common |= x
            results.append(common)
        sets = results
    return sets

@jit(nopython=True)
def fast_histogram(x,bins):
    """
    Fast, simple version of histogram that takes in a unsorted list and histograms into the given 
    bins. For efficiency, the list is sorted in place.
    
    Parameters
    ----------
    x : list
        Elements will be sorted in place.
    bins : list
    
    Returns
    -------
    y : list
        Of integers.
    """
    x.sort()
    y=[]
    
    # Skip elements of x that are less than the min
    counter=0
    while x[counter]<bins[0]:
        counter+=1
    
    # Count elements in bins except for last bin
    binIx=0
    while counter<len(x) and binIx<(len(bins)-2):
        y.append(0)
        while x[counter]<bins[binIx+1]:
            y[-1]+=1
            counter+=1
        binIx+=1
        
    # Count last bin
    y.append(0)
    while counter<len(x) and x[counter]<bins[binIx+1]:
        y[-1]+=1
        counter+=1
        
    return y

def unravel_multi_utri_index(ix,n,d):
    """
    Generalization of unravel_index to a d-dimensional upper-triangular array with dimension size n.

    This is a slow way of doing it where I iterate through all possible index combinations.

    Parameters
    ----------
    ix : ndarray
    n : int
        Number of dimensions, range(n) to choose from.
    d : int
        Extent along each dimension, d-sized subsets  to choose.

    Returns
    -------
    subix : list
        Indices converted into subindices.
    """
    from itertools import combinations
    if isinstance(ix,Number):
        ix = np.array([ix])
    elif type(ix) is list:
        ix = np.array(ix)

    # d-dimensional sub index 
    subix = np.zeros((len(ix),d),dtype=int)
    # Iterate through ix in order.
    sortix = np.argsort(ix)
    counter = 0  # index for the next entry to find in sorted ix

    # Run through ix and when we get to its value, record it.
    for i,ijk in enumerate(combinations(list(range(n)),d)):
        while counter<len(ix) and  i==ix[sortix[counter]]:
            subix[sortix[counter],:] = ijk
            counter += 1
        if counter==len(ix):
            break
    return subix

def reinsert(x,insertix,value,check_sort=False):
    """
    Re-insert values into array after they have been deleted. np.insert cannot
    deal with cases where we want to insert values back into the array but the
    trimmed array is shorter. This is just a looped version of np.insert.
    
    Params:
    -------
    x (ndarray)
    insertix (ndarray)
        Assuming that this is sorted.
    value (float)
    
    Returns:
    --------
    x (ndarray)
        Now sorted.
    """
    for i in insertix:
        x = np.insert(x,i,value)
    return x

@jit(nopython=True,nogil=True,cache=True)
def sub_to_ind(n,i,j):
    """
    Convert pair of coordinates of a symmetric square array into consecutive
    index of flattened upper triangle. This is slimmed down so it won't throw
    errors like if i>n or j>n or if they're negative. Only checking for if the
    returned index is negative which could be problematic with wrapped indices.
    2016-08-16
    
    Params:
    -------
    n (int)
        Dimension of square array
    i,j (int)
        coordinates
    """
    if i<j:
        k = 0
        for l in range(1,i+2):
            k += n-l
        assert k>=0
        return k-n+j
    elif i>j:
        k = 0
        for l in range(1,j+2):
            k += n-l
        assert k>=0
        return k-n+i
    else:
        raise Exception("Indices cannot be the same.")

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
        for i in range(nJobs-1):
            instances.append( deepcopy(args) )
        
        # Wrap f so that the args can be properly expanded and handed over as an expanded list.
        def g(args):
            return f(*args)
    
        p = Pool(nJobs)
        output = list(zip( *p.map(g,instances) ))
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
    Given a mixed set of list and int/float args, turn them into a set up tuples. This can be used
    to facilitate pipeline operations. This only takes lists!
    """
    listOfTuples = []
    try:
        L = max([len(i) for i in args if type(i) is list])
    except ValueError:
        L = 1
    
    for i in range(L):
        listOfTuples.append([])
        for j in args:
            if type(j) is list:
                listOfTuples[-1].append(j[i])
            else:
                listOfTuples[-1].append(j)
    return [tuple(i) for i in listOfTuples]

def bootstrap_f(data,f,nIters,nSamples=-1):
    """
    Take given data nad compute function f on bootstrapped data using a parallel for loop.
    2016-08-26

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
    from multiprocess import cpu_count,Pool

    if nSamples==-1:
        nSamples = len(data)
    results = []

    if type(data) is list or data.ndim==1:
        def g(i):
            np.random.seed()
            return f(np.random.choice(data,size=nSamples))
    else:
        def g(i):
            np.random.seed()
            randIx = np.random.randint(len(data),size=nSamples)
            return f(data[randIx])

    p = Pool(cpu_count()) 
    output = p.map(g,range(nIters))
    p.close()
    return output

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
    Sort entries in a symmetric matrix such that all cols and rows are sorted
    in the sequential order for a selected row/col.  Diagonal elements are not
    considered.
    2014-01-23
    
    Params:
    -------
    m (ndarray)
        square matrix
    invert : Sort ascending by default. From high to low if True.
    refIx : index of row with which to order
    """
    if m.shape[0]!=m.shape[1]:
        raise Exception("Matrix must be square")
    n = m.shape[0]
    sortIx = np.expand_dims(list(range(m.shape[0])),0)
    
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

@njit
def _unravel_utri(ix,n):
    """
    Convert the index ix from the flattened utri array to the dimension in an nxn matrix.

    Parameters
    ----------
    ix : int
    n : int
    
    Returns
    -------
    i,j : int
    """
    i=0
    counter=0
    while counter<=ix:
        counter+=n-i-1
        i+=1
    i-=1
    if i==0:
        j=ix+1
    elif counter>ix:
        j=ix%(counter-(n-i-1))+i+1
    else:
        j=ix-counter+i+1
    return i,j
unravel_utri=np.vectorize(_unravel_utri)

@njit
def _ravel_utri(i,j,n):
    """
    Convert the index ix from the flattened utri array to the dimension in an nxn matrix.

    Parameters
    ----------
    i : int
    j : int
    
    Returns
    -------
    ix : int
    """
    assert i!=j
    if i>j:
        tmp=i
        i=j
        j=tmp

    ix=0
    for i_ in range(i):
        ix+=n-1-i_
    ix+=j-i-1
    return ix
ravel_utri=np.vectorize(_ravel_utri)

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
    matix[triu_indices_asymm(mn,mx)] = list(range(mn*(mn-1)/2+(mx-mn)*mn))
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
    Extract upper triangular elements from asymmetric array. The problem with asymmetric matrices is
    that if the longer dimension is along rows, then to extract every pairwise comparison you have
    to extract all elements below the diagonal (not above as is typically the case).
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

def unique_rows(mat, return_inverse=False):
    """
    Return unique rows indices of a numpy array.

    Parameters
    ----------
    mat : ndarray
    **kwargs
    return_inverse : bool
        If True, return inverse that returns back indices of unique array that
        would return the original array 

    Returns
    -------
    u : ndarray
        Unique elements of matrix.
    idx : ndarray
        Row indices of given mat that will give unique array.
    """

    b = np.ascontiguousarray(mat).view(np.dtype((np.void, mat.dtype.itemsize * mat.shape[1])))
    if not return_inverse:
        _, idx = np.unique(b, return_index=True)
    else:
        _, idx = np.unique(b, return_inverse=True)
    
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
    Autocorrelation coefficient including for masked arrays using the slow way of calculating with numpy.corrcoef.
    2014-09-30

    Args:
    -----
    length{20,int}: time lags to do
    iters{0,int}: number of bootstrap samplestime lags to go up to
    nonan{True,bool}: ignore nans
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

def fix_err_bars(y, yerr, ymn, ymx, dy=1e-10):
    """Return 2xN array of low error and high error points having removed negative
    values. Might need this when extent of error bars return negative values that are
    impossible.

    This fits right into pyplot.errorbar() xerr or yerr options

    Parameters
    ----------
    y : ndarray
    yerr : ndarray
        Vector of errors, one for each value in y.
    ymn : float or ndarray
    ymx : float or ndarray
    dy : float, 1e-10
        Padding to put at lower and upper bounds.

    Returns
    -------
    ndarray
        Array that can be put into errorbar yerr or xerr options.
    """

    yerru = yerr.copy()
    yerru[(y+yerr)>=ymx] -= ((y+yerr)-ymx)[(y+yerr)>=ymx] +dy
    yerrl = yerr.copy()
    yerrl[(y-yerr)<=ymn] += (y-yerr)[(y-yerr)<=ymn] -dy

    yerr = np.vstack((yerrl,yerru))
    yerr[yerr<0] = 0
    return yerr

def hist_log(data,bins,
             density=False,
             x0=None,x1=None,base=10.):
    """
    Return log histogram of data. Usage similar to regular histogram.

    Parameters
    ----------
    data : ndarray
    bins : ndarray or int
    x0, x1 : float
        min and max x's
    base : float
        base of histogram binning

    Returns
    -------
    ndarray
        frequency count or probability density per bin
    ndarray
        centered bins for plotting with n
    ndarray
        xedges
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
        bins = np.logspace(np.log(x0)/np.log(base), np.log(x1)/np.log(base), bins+1)
        bins[-1] = np.ceil(bins[-1])
        n, xedges = np.histogram( data, bins=bins, density=density )

    # take difference in log space to center bins
    dx = base**(np.diff(np.log(xedges)/np.log(base))/2.0)
    x = xedges[:-1]+dx

    return (n,x,xedges)
