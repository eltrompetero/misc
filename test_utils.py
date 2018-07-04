from .utils import *

def test_unravel_utri(n=5):
    from scipy.spatial.distance import squareform
    
    for ix in range(n*(n-1)//2):
        i,j=unravel_utri(ix,n)
        assert ix==squareform(np.arange(n*(n-1)//2))[i,j],(squareform(np.arange(n*(n-1)//2))[i,j],ix)
