from .utils import *
np.random.RandomState(0)


def test_unravel_utri(n=5):
    from scipy.spatial.distance import squareform
    
    for ix in range(n*(n-1)//2):
        i,j=unravel_utri(ix,n)
        assert ix==squareform(np.arange(n*(n-1)//2))[i,j],(squareform(np.arange(n*(n-1)//2))[i,j],ix)

def test_fast_histogram():
    r=np.random.rand(10000).tolist()
    bins=np.linspace(0,1,1000)
    assert np.array_equal(np.histogram(r,bins)[0],fast_histogram(r,bins))
