import numpy as np
import math

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
        Return log histogram of data.
        2013-06-27
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
        n,xedges = np.histogram( data,
                        bins=np.logspace(np.log(x0)/np.log(base),
                        np.log(x1)/np.log(base),bins+1 ),
                        density=density )

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
