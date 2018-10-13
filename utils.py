# Module for useful functions.
import numpy as np
import numpy
import math
from multiprocess import Pool,cpu_count
from numba import jit,njit
from numbers import Number
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

# ----------------------------------#
# Useful mathematical calculations. #
# ----------------------------------#
def vincenty(point1, point2, a, f, MAX_ITERATIONS=200, CONVERGENCE_THRESHOLD=1e-12):
    """
    Vincenty's formula (inverse method) to calculate the distance 
    between two points on the surface of a spheroid

    Parameters
    ----------
    point1 : twople
        (xy-angle, polar angle). These should be given in radians.
    point2 : twople
        (xy-angle, polar angle)
    a : float
        Equatorial radius.
    f : float
        eccentricity, semi-minor polar axis b=(1-f)*a
    """
    # short-circuit coincident points
    if point1[0] == point2[0] and point1[1] == point2[1]:
        return 0.0
    b=(1-f)*a

    U1 = math.atan((1 - f) * math.tan(point1[0]))
    U2 = math.atan((1 - f) * math.tan(point2[0]))
    L = point2[1] - point1[1]
    Lambda = L

    sinU1 = math.sin(U1)
    cosU1 = math.cos(U1)
    sinU2 = math.sin(U2)
    cosU2 = math.cos(U2)

    for iteration in range(MAX_ITERATIONS):
        sinLambda = math.sin(Lambda)
        cosLambda = math.cos(Lambda)
        sinSigma = math.sqrt((cosU2 * sinLambda) ** 2 +
                             (cosU1 * sinU2 - sinU1 * cosU2 * cosLambda) ** 2)
        if sinSigma == 0:
            return 0.0  # coincident points
        cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda
        sigma = math.atan2(sinSigma, cosSigma)
        sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma
        cosSqAlpha = 1 - sinAlpha ** 2
        try:
            cos2SigmaM = cosSigma - 2 * sinU1 * sinU2 / cosSqAlpha
        except ZeroDivisionError:
            cos2SigmaM = 0
        C = f / 16 * cosSqAlpha * (4 + f * (4 - 3 * cosSqAlpha))
        LambdaPrev = Lambda
        Lambda = L + (1 - C) * f * sinAlpha * (sigma + C * sinSigma *
                                               (cos2SigmaM + C * cosSigma *
                                                (-1 + 2 * cos2SigmaM ** 2)))
        if abs(Lambda - LambdaPrev) < CONVERGENCE_THRESHOLD:
            break  # successful convergence
    else:
        return None  # failure to converge

    uSq = cosSqAlpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)))
    B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)))
    deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma *
                 (-1 + 2 * cos2SigmaM ** 2) - B / 6 * cos2SigmaM *
                 (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)))
    s = b * A * (sigma - deltaSigma)

    return round(s, 6)

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

def xmax(a):
    """
    Find max in a generator.

    Params:
    -------
    a (generator for floats)
    """
    nowmx = -np.inf
    for i in a:
        if i>nowmx:
            nowmx = i
    return nowmx

def xlogsumexp(a,b):
    """
    Generator version of scipy.misc.logsumexp().
    First scan through iterator for the max. Then implement iterative sum.
    
    Params:
    -------
    a,b (generators)
        Two copies of the generator.
    """
    mx = xmax(a)
    res = 0
    for i in b:
        res += np.exp(i-mx)
    return np.log(res) + mx

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



# -------#
# Other  #
# -------#
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

def unique_rows(mat,return_inverse=False):
    """
    Return unique rows indices of a numpy array.

    Params:
    -------
    mat (ndarray)
    **kwargs
    return_inverse (bool)
        If True, return inverse that returns back indices of unique array that
        would return the original array 

    Returns:
    --------
    u (ndarray)
        Unique elements of matrix.
    idx (ndarray)
        row indices of given mat that will give unique array
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
            print(', '.join(row))
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
