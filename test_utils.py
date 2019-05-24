# ===================================================================================== #
# Module for testing utils.py.
# Author : Eddie Lee, edlee@alumni.princeton.edu
# ===================================================================================== #
from .utils import *
np.random.RandomState(0)


def test_unravel_utri(n=5):
    from scipy.spatial.distance import squareform
    
    # Iterate through an array of shape nxn and check every index.
    for ix in range(n*(n-1)//2):
        i,j=unravel_utri(ix,n)
        assert ix==squareform(np.arange(n*(n-1)//2))[i,j]
    
    # Test vectorize
    ix=list(range(n*(n-1)//2))
    i,j=unravel_utri(ix,n)
    for ix_ in range(n*(n-1)//2):
        assert ix[ix_]==squareform(np.arange(n*(n-1)//2))[i[ix_],j[ix_]]

def test_ravel_utri(n=5):
    from scipy.spatial.distance import squareform
    
    # Iterate through an array of shape nxn and check every index.
    for ix in range(n*(n-1)//2):
        i,j=unravel_utri(ix,n)
        ix_=ravel_utri(i,j,n)
        assert ix==ix_

    # Test vectorized version
    ix=list(range(n*(n-1)//2))
    i,j=unravel_utri(ix,n)
    ix_=ravel_utri(i,j,n)
    assert np.array_equal(ix,ix_)

def test_fast_histogram():
    r=np.random.rand(10000).tolist()
    bins=np.linspace(0,1,1000)
    assert np.array_equal(np.histogram(r,bins)[0],fast_histogram(r,bins))
