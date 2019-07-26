# ====================================================================================== #
# Module for bias testing of power law.
# Author : Eddie Lee, edlee@alumni.princeton.edu
#
# Cached functions in this module require a cache folder in the working directory.
# ====================================================================================== #
import numpy as np
from workspace.utils import cached
from ..stats import PowerLaw, DiscretePowerLaw
from multiprocess import Pool, cpu_count
import pickle
import os
from numba import njit
from warnings import warn


@njit
def has_multiple_unique_values(x):
    """Check if given list has more than one unique value. Return True if there is more
    than one unique value."""

    for i in range(1,len(x)):
        if x[i]!=x[0]:
            return True
    return False    

@cached(iprint=True, cache_pickle='cache/discrete_powerlaw_correction_spline_cache.p')
def cache_discrete_powerlaw_correction_spline(n_iters, alphaRange, Krange, lower_bound):
    """Returns spline interpolation for landscape (over alpha and K).
    
    Parameters
    ----------
    n_iters : int
        Number of samples on which to perform max likelihood procedure.
    alphaRange : list
        Tuple type is hashable. For caching.
    Krange : list
        Tuple type is hashable. For caching.
    lower_bound : int
    
    Returns
    -------
    tuple
        args to pass into scipy.interpolate.griddata
    dict
        kwargs  to pass into scipy.interpolate.griddata
    """
    
    assert lower_bound<=25, "Just use continuous variable approximation for large lower bound."
    

    alphaRange = np.array(alphaRange)
    Krange = np.array(Krange)

    def loop(args):
        alpha, K, rng = args
        K = int(K)
        alphaML = np.zeros(n_iters)
            
        for i in range(n_iters):
            goodSample = False
            nTries = 0
            while not goodSample:
                X = DiscretePowerLaw.rvs(alpha, size=K, rng=rng, lower_bound=lower_bound)
                nTries += 1
                if has_multiple_unique_values(X):
                    goodSample = True
                if nTries>1e6:
                    raise Exception(alpha, K, lower_bound)
            alphaML[i] = DiscretePowerLaw.max_likelihood(X,
                                                         initial_guess=alpha,
                                                         n_cpus=1,
                                                         max_alpha=np.inf,
                                                         lower_bound=lower_bound)
        return alphaML

    alphaGrid, Kgrid = np.meshgrid(alphaRange, Krange)
    alphaGrid = alphaGrid.ravel()
    Kgrid = Kgrid.ravel()
    
    # parallelize
    pool = Pool(cpu_count()-1)
    alphaML = pool.map(loop, zip(alphaGrid, Kgrid, [np.random.RandomState() for a in alphaGrid]))
    pool.close()
    
    #if any([(a==alphaRange[-1]).any() for a in alphaML]):
    #    print("Some alphaML have hit upper boundary. %d"%((alphaML==alphaRange[-1]).sum()))
    
    alphaML = np.array([a.mean() for a in alphaML])
    
    args = (alphaML, Kgrid), alphaGrid-alphaML
    kwargs = {'method':'linear', 'fill_value':0}
    return args, kwargs

@cached(cache_pickle='cache/discrete_powerlaw_correction_spline_cache1.p')
def _discrete_powerlaw_correction_spline(lower_bound,
                                         alphaRange=None,
                                         Krange=None,
                                         n_iters=10_000,
                                         full_output=False,
                                         cache_file='cache/discrete_powerlaw_correction_spline_cache.p',
                                         run_check=True):
    """A wrapper for cache_discrete_powerlaw_correction_spline() that returns the
    interpolated results of sampling.
    
    Parameters
    ----------
    lower_bound : int
    alphaRange : ndarray, np.arange(1.1, 10, .25)
    Krange : ndarray, np.around(2.**np.arange(2,8.5,.5)).astype(int)
    n_iters : int, 10_000
        Number of samples on which to perform max likelihood procedure. Discrete max
        likelihood is much slower than for continuous variables.
    full_output : bool, False
    cache : str, 'r'
        If 'w', always write cache file. If 'r' only write if there is no cache.
        Otherwise, do not write any cache file.
    cache_file : str, 'cache/discrete_powerlaw_correction_spline_cache.p'
    run_check : bool, True
        Compare fitted landscape with given values.
    
    Returns
    -------
    function
        Spline fit object from scipy.interpolate.interp2d.
    """

    from scipy.interpolate import griddata
    if alphaRange is None:
        alphaRange = np.arange(1.1, 10, .25)
    if Krange is None:
        Krange = np.around(2.**np.arange(2,8.5,.5)).astype(int)

    args, kwargs = cache_discrete_powerlaw_correction_spline(n_iters,
                                                             tuple(alphaRange),
                                                             tuple(Krange),
                                                             lower_bound)

    # basically a glorified look up table given the measured alpha to find the true alpha
    # this function takes in coordinates (alpha, K)
    #f = interp2d(*args)
    #correction_fun = lambda x: f(x)
    correction_fun = lambda x,y: griddata(*args, (x,y), **kwargs)
    
    # sometimes landscape is really badly spline fit. this measures deviation over entire grid
    if run_check:
        interpval = np.array([correction_fun(a,k) for (a,k) in zip(*args[0])])
        err = np.linalg.norm(interpval-args[1])
        if err>1e-5:
            print("Large error in spline fit! %1.4f"%err)

    if full_output:
        return correction_fun, (alphaRange, Krange), args
    return correction_fun

def discrete_powerlaw_correction_spline():
    """A wrapper for _discrete_powerlaw_correction_spline() because that defines a
    function for each individual lower bound. For many applications, redefining that
    function repeated will take time so this preloads the splines for lower bounds 1-15.

    This is the nice front end wrapper that doesn't worry about any of the caching behind
    the scenes. To run that caching, you should call
    _discrete_powerlaw_correction_spline().
    
    Returns
    -------
    function
        Spline fit object from scipy.interpolate.interp2d that automatically handles lower
        bound info.
    """

    pl_correction = powerlaw_correction_spline(run_check=False)
    dpl_correction = [_discrete_powerlaw_correction_spline(i, run_check=False) for i in range(1,26)]

    def correction(alpha, K, lb=1, pl_correction=pl_correction, dpl_correction=dpl_correction):
        if lb>25:
            return pl_correction(alpha, K)
        if not type(lb) is int:
            #warn("lb is not int. Converting explicitly.")
            lb = int(lb)
        return dpl_correction[lb-1](alpha, K)
    return correction

@cached(iprint=True, cache_pickle='cache/powerlaw_correction_spline_cache.p')
def cache_powerlaw_correction_spline(n_iters, alphaRange, Krange):
    """Returns spline interpolation for landscape (over alpha and K).
    
    Parameters
    ----------
    n_iters : int
        Number of samples on which to perform max likelihood procedure.
    alphaRange : list
        Tuple type is hashable. For caching.
    Krange : list
        Tuple type is hashable. For caching.
    
    Returns
    -------
    tuple
        args to pass into scipy.interpolate.griddata
    dict
        kwargs  to pass into scipy.interpolate.griddata
    """
    
    alphaRange = np.array(alphaRange)
    Krange = np.array(Krange)
        
    def loop(args):
        alpha, K, rng = args
        K = int(K)
        alphaML = np.zeros(n_iters)

        for i in range(n_iters):
            goodSample = False
            nTries = 0
            while not goodSample:
                X = PowerLaw.rvs(alpha, size=K, rng=rng, lower_bound=1)
                nTries += 1
                if has_multiple_unique_values(X):
                    goodSample = True
                if nTries>1e6:
                    raise Exception(alpha, K, lower_bound)
            alphaML[i] = PowerLaw.max_likelihood(X, 
                                                 initial_guess=alpha,
                                                 n_cpus=1,
                                                 lower_bound=1)
        return alphaML

    alphaGrid, Kgrid = np.meshgrid(alphaRange, Krange)
    alphaGrid = alphaGrid.ravel()
    Kgrid = Kgrid.ravel()
    
    # parallelize
    pool = Pool(cpu_count()-1)
    alphaML = pool.map(loop, zip(alphaGrid, Kgrid, [np.random.RandomState() for a in alphaGrid]))
    pool.close()
    
    if any([(a==alphaRange[-1]).any() for a in alphaML]):
        print("Some alphaML have hit upper boundary. %d"%((alphaML==alphaRange[-1]).sum()))
    
    alphaML = np.array([a.mean() for a in alphaML])
    
    args = (alphaML, Kgrid), alphaGrid-alphaML
    kwargs = {'method':'linear', 'fill_value':0}
    return args, kwargs

@cached(cache_pickle='cache/powerlaw_correction_spline_cache1.p')
def powerlaw_correction_spline(alphaRange=None,
                               Krange=None,
                               n_iters=100_000,
                               full_output=False,
                               run_check=True):
    """A wrapper for cache_powerlaw_correction_spline() that returns the results of sampling.
    
    Parameters
    ----------
    alphaRange : ndarray, np.arange(1.1, 10, .1)
    Krange : ndarray, np.around(2.**np.arange(2, 10.5, .5)).astype(int)
    n_iters : int, 100_000
        Number of samples on which to perform max likelihood procedure.
    full_output : bool, False
    run_check : bool, True
        Compare fitted landscape with given values.
    
    Returns
    -------
    function
        Spline fit object from scipy.interpolate.interp2d.
    """
    
    from scipy.interpolate import griddata
    if alphaRange is None:
        alphaRange = np.arange(1.1, 10, .1)
    if Krange is None:
        Krange = np.around(2.**np.arange(2, 10.5, .5)).astype(int)

    args, kwargs = cache_powerlaw_correction_spline(n_iters, tuple(alphaRange), tuple(Krange))

    # basically a glorified look up table given the measured alpha to find the true alpha
    # this function takes in coordinates (alpha, K)
    #f = interp2d(*args)
    #correction_fun = lambda x: f(x)
    correction_fun = lambda x,y: griddata(*args, (x,y), **kwargs)
    
    # sometimes landscape is really badly spline fit. this measures deviation over entire grid
    if run_check:
        interpval = np.array([correction_fun(a,k) for (a,k) in zip(*args[0])])
        err = np.linalg.norm(interpval-args[1])
        if err>1e-5:
            print("Large error in spline fit! %1.4f"%err)

    if full_output:
        return correction_fun, (alphaRange, Krange), args
    return correction_fun

def interp2d(xy, z):
    xy = np.vstack(xy)
    def f(pt, xy=xy, z=z):
        pt = np.array(pt)
        return z[np.argmin( np.linalg.norm(pt[:,None]-xy,axis=0) )]
    return f

