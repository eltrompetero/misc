# 2016-06-30
# Library for special functions like in scipy.special.
from __future__ import division
import numpy as np


def jacobi_lobatto(n,a,b):
    """
    Weights and collocation points for Gauss-Jacobi-Lobatto quadrature, i.e. Gaussian quadrature using Jacobi polynomials with collocation points as the extrema of the polynomials.
    Based on formulas from Huang (
    2016-06-30

    Params:
    -------
    n (int)
        Collocation points of polynomial of degree n.
    a,b (floats)
        alpha and beta

    Value:
    ------
    coX (ndarray)
        Lobatto collocation points.
    weights (ndarray)
        Weights at collocation points.
    """
    assert type(n) is int
    assert a>-1 and b>-1

    from scipy.special import gamma,gammaln,jacobi,j_roots,factorial
    from scipy.special import binom as binomk
    pdn = lambda n,a,b,x: (n+a+b+1)/2 * jacobi(n-1,a+1,b+1)(x)
    log_g_tilde = lambda a,b,n: ( np.log(2)*(a+b+1) + gammaln(n+a+2) + gammaln(n+b+2) 
                                  - np.log(factorial(n+1)) - gammaln(n+a+b+2) )
    an =lambda n: -(a**2-b**2+2*(a-b))/((2*n+a+b)*(2*n+a+b+2))
    bn = lambda n: 4*(n-1)*(n+a)*(n+b)*(n+a+b+1) / (2*n+a+b)**2/(2*n+a+b+1)/(2*n+a+b-1)
    
    Jn = np.diag([an(i) for i in xrange(1,n+1)]) + np.sqrt(np.diag([bn(i) for i in xrange(2,n+1)],k=1) + 
                                                           np.diag([bn(i) for i in xrange(2,n+1)],k=-1))
    coX = np.concatenate(([-1],np.sort(np.linalg.eig(Jn)[0]).real,[1]))
    
    weights = np.zeros_like(coX)
    n += 1  # In Teng, N+1 is the otal number of points including endpoints.
    loggtilde = log_g_tilde(a+1,b+1,n-2)
    denom = ( 1-coX[1:-1]**2 )**2 * (pdn(n-1,a+1,b+1,coX[1:-1]))**2
    weights[1:-1] = np.exp( loggtilde ) / denom
    
    weights[0] = np.exp( np.log(2)*(a+b+1) + np.log(b+1) + 2*gammaln(b+1) + gammaln(n) + gammaln(n+a+1) -
                         gammaln(n+b+1) - gammaln(n+a+b+2) )
    weights[-1] = np.exp( np.log(2)*(a+b+1) + np.log(a+1) + 2*gammaln(a+1) + gammaln(n) + gammaln(n+b+1) -
                          gammaln(n+a+1) - gammaln(n+a+b+2) )
    return coX,weights

def jacobi_d(n,k,a=1,b=1):
    """
    kth derivative of Jacobi polynomial.
    2016-07-02
    """
    from scipy.special import gamma,jacobi
    return lambda x: gamma(a+b+n+1+k)/2**k/gamma(a+b+n+1) * jacobi(n-k,a+k,b+k)(x)

