# ===================================================================================== #
# Module for testing utils.py.
# Author : Eddie Lee, edlee@alumni.princeton.edu
# ===================================================================================== #
from .utils import *
from .utils import _sort_by_phi
np.random.RandomState(0)


def test_sort_by_phi():
    xy = np.random.normal(size=(10,2))
    assert _sort_by_phi(xy).size==10
    xy[0,:] = xy[1,:]
    assert _sort_by_phi(xy).size==9
    xy[2,:] = xy[1,:]
    assert _sort_by_phi(xy).size==8

    xy[0] = [2,0]
    xy[1] = [1,0]
    assert not 1 in _sort_by_phi(xy), _sort_by_phi(xy)

def test_convex_hull():
    for i in range(10):
        xy = np.random.normal(size=(100,2))
        hullrec = convex_hull(xy, recursive=True)
        hullseq = convex_hull(xy, recursive=False)
        assert np.array_equal(hullrec, hullseq)
    print("Test passed: Recursive convex hull algo agrees with sequential for 10 random samples.")

def test_ortho_plane():
    r = np.random.rand(3)
    r /= np.linalg.norm(r)
    r1, r2 = ortho_plane(r)

    assert np.isclose(r1.dot(r), 0)
    assert np.isclose(r2.dot(r), 0)
    assert np.isclose(r1.dot(r2), 0)
    assert np.isclose(np.linalg.norm(r1), 1)
    assert np.isclose(np.linalg.norm(r2), 1)

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
