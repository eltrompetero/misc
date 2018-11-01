import numpy as np
from scipy.integrate import quad
from scipy.special import binom

# ========================================================= #
# Gaussian quadrature techniques
# ========================================================= #
def legcoeffs(n,k):
    """
    Generate coefficients for all polynomial terms of the Legendre polynomials using recursive
    relations.  

    Parameters
    ----------
    n : int
    k : int
    
    Returns
    -------
    coeffs : ndarray
        Vector of coefficients descending from x^n to x^0.
    """
    
    from scipy.special import binom
    return 2**n * binom(n,k) * binom((n+k-1)/2,n)

def legjacobi(n):
    """Jacobi matrix for nth order Gauss-Legendre quadrature.
    """

    J=np.zeros(n**2)
    normalization=( legcoeffs( np.arange(n-1), np.arange(n-1) ) /
                    legcoeffs( np.arange(1, n), np.arange(1, n) ) )
    J[list(range(1,n**2,n+1))]=np.sqrt((2*np.arange(1,n)-1) / (2*np.arange(1,n)+1)) * normalization
    J[list(range(n,n**2,n+1))]=np.sqrt((2*np.arange(1,n)-1) / (2*np.arange(1,n)+1)) * normalization
    return J.reshape(n,n)

def leggauss(n,x0=None):
    """
    Abscissa and weights for Gauss-Legendre quadrature.

    Parameters
    ----------
    n : int

    Returns
    -------
    x : ndarray
    weights : ndarray
    """
    
    J=legjacobi(n)
    if x0 is None:
        x,vec=np.linalg.eigh(J)
        return x,vec[0]**2 * 2
    else:
        from numpy.polynomial.legendre import legval
        J[-1,-1]=x0 - J[-1,-2]**2 * legval(x0, [0]*(n-2)+[1]) / legval(x0, [0]*(n-1)+[1])
        x,vec=np.linalg.eigh(J)
        return x,vec[0]**2 * 2


class LevyGaussQuad():
    """Construct Gaussian quadrature for Levy integral used in the Bethe lattice model for mean-field dislocation avalanches.
    
    See Numerical Recipes for details about how this works.
    """
    def __init__(self, n, x0, x1, mu):
        """
        Parameters
        ----------
        n : int
            Degree of polynomial expansion.
        x0 : float
            Lower cutoff for Levy distribution.
        x1 : float
            Upper cutoff for Levy distribution.
        mu : float
            Exponent for Levy distribution x^{-mu-1}
        """

        assert x0>0 and x1>x0 and mu>0 and n>2
        
        self.x0, self.x1=x0, x1
        self.mu=mu
        self.K=lambda x, mu=self.mu, x0=self.x0, x1=self.x1: mu/2 * x**(-mu-1) / (x0**-mu - x1**-mu)
        # check that kernel integrates to 1/2
        assert np.isclose( quad(self.K, x0, x1)[0], .5 )

        self.n=n  # degree of polynomial
        
        self.construct_polynomials()
        
    def construct_polynomials(self):
        """Construct polynomials up to nth degree. Save the results to the instance.
        """
        from numpy.polynomial.polynomial import Polynomial
        
        p=[Polynomial([1])]  # polynomials
        a=[]
        b=[0]
        innerprod=[.5]  # constructed such that inner product of f(x)=1 is 1/2

        # first polynomial is special
        a.append( quad(lambda x: self.K(x) * x, self.x0, self.x1)[0]/innerprod[0] )
        p.append( Polynomial([-a[0],1])*p[0] )

        for i in range(1,self.n+1):
            innerprod.append( quad(lambda x:p[i](x)**2 * self.K(x), self.x0, self.x1)[0] )
            a.append(quad( lambda x:x * p[i](x)**2 * self.K(x), self.x0, self.x1 )[0] / innerprod[i])
            b.append(innerprod[i] / innerprod[i-1])
            p.append(Polynomial([-a[i],1]) * p[i] - b[i] * p[i-1])
            
        self.p=p
        self.a=a
        self.b=b
        self.innerprod=innerprod

    def polish_roots(self, poly, roots, n_iters):
        """
        Parameters
        ----------
        poly : numpy.polynomial.polynomial.Polynomial
        roots : ndarray
            Initial guess for roots.
        n_iters : int

        Returns
        -------
        polished_roots : ndarray
        """
        
        polishedRoots=roots[:]
        for i,r in enumerate(roots):
            for j in range(n_iters):
                x0=polishedRoots[i]
                polishedRoots[i]=x0 - poly(x0)/poly.deriv()(x0)
        return polishedRoots

    def levy_quad(self, n, polish=True, n_iters=7):
        """
        Parameters
        ----------
        n : int
            Degree of expansion.
        polish : bool,False
        n_iters : int,10

        Returns
        -------
        abscissa : ndarray
        weights : ndarray
        """

        if n>len(self.p):
            raise Exception
        assert n>2

        # find roots of polynomial
        abscissa=self.p[n-1].roots().real
        if polish:
            abscissa=self.polish_roots(self.p[n-1], abscissa, n_iters)

        # using formula given in Numerical Recipes
        weights=self.innerprod[n-2] / (self.p[n-2](abscissa) * self.p[n-1].deriv()(abscissa))

        return abscissa, weights
#end LevyGaussQuad


class QuadGauss():
    def __init__(self,order,method='legendre',lobatto=False):
        """
        Straightforward gaussian integration using Chebyshev polynomials with mapping of the
        bounds into [-1,1]. Most useful for a bounded interval.

        Parameters
        ----------
        order : int
            Order of basis expansion.
        method : str,'legendre'
        lobatto : bool,False
            If True, use Lobatto collocation points. Only works for Chebyshev polynomials.
        """
        from numpy.polynomial.chebyshev import chebval,chebgauss
        
        self.order = order
        self.N = order
        
        if method=='chebyshev':
            self.basis = [lambda x,i=i:chebval(x,[0]*i+[1]) for i in range(self.N+1)]

            # Lobatto collocation points.
            if lobatto:
                self.coX = -np.cos(np.pi*np.arange(self.N+1)/self.N)
                self.weights = np.zeros(self.order+1)+np.pi/self.N
                self.weights[0] = self.weights[-1] = np.pi/(2*self.N)
            else:
                self.coX,self.weights = chebgauss(self.N+1)
            self.basisCox = [b(self.coX) for b in self.basis]

            if ((self.coX)==1).any():
                self.W=np.zeros_like(self.coX)
                ix=np.abs(self.coX)==1
                self.W[ix==0]=1/np.sqrt(1-self.coX[ix==0]**2)
                self.W[ix]=np.inf
            else:
                self.W = 1/np.sqrt(1-self.coX**2)  # weighting function we must remove
            self.W[np.isnan(self.W)] = 0.

        elif method=='legendre':
            from numpy.polynomial.legendre import leggauss,legval
            if order>100:
                from .high_prec.calculus import leggauss
                self.coX,self.weights=leggauss(self.N+1)
                self.coX=np.array([float(f) for f in self.coX])
                self.weights=np.array([float(f) for f in self.weights])
            else:
                self.coX,self.weights=leggauss(self.N+1)
            self.basis = [lambda x,i=i:legval(x,[0]*i+[1]) for i in range(self.N+1)]
            self.basisCox = [b(self.coX) for b in self.basis]
            self.W=np.ones_like(self.coX)

        else: raise Exception("Invalid basis choice.")
        
        # Map bounds to given bounds or from given bounds to [-1,1].
        self.map_to_bounds = lambda x,x0,x1: (x+1)/2*(x1-x0) + x0
        self.map_from_bounds = lambda x,x0,x1: (x-x0)/(x1-x0)*2. - 1.

    def quad(self, f, x0, x1, weight_factors=None):
        """
        Parameters
        ----------
        f : lambda function
            One dimensional function
        x0 : float
        x1 : float
        weight_factors : ndarray,None

        Returns
        -------
        val : float
        """
        if weight_factors is None:
            weight_factors=np.ones_like(self.weights)
        return ( f(self.map_to_bounds(self.coX,x0,x1))/self.W ).dot(self.weights*weight_factors) * (x1-x0)/2
      
    def dblquad(self,f,x0,x1,y0,y1):
        """
        Uses a meshgrid to do integrals.

        Parameters
        ----------
        f : lambda function
            Two dimensional function (x,y) where integral over y is done first
        x0,x1 : float
            Bounds for outer integral
        y0,y1 : float
            Bounds for inner integral

        Returns
        -------
        val : float
        """
        xgrid,ygrid = (np.meshgrid(self.map_to_bounds(self.coX,x0,x1), 
                       self.map_to_bounds(self.coX,y0,y1)) )
        
        return ( ( ( f(xgrid,ygrid)/self.W[:,None] )*self.weights[:,None] * (y1-y0)/2 ).sum(0) / 
                 self.W * self.weights * (x1-x0)/2 ).sum()
#end QuadGauss


class QuadGaussRadau(QuadGauss):
    def __init__(self,order,method='legendre'):
        """
        Straightforward gaussian integration using Chebyshev polynomials with mapping of the
        bounds into [-1,1]. Most useful for a bounded interval.

        Parameters
        ----------
        order : int
            Order of basis expansion.
        method : str,'legendre'
        lobatto : bool,False
            If True, use Lobatto collocation points. Only works for Chebyshev polynomials.
        """
        from numpy.polynomial.chebyshev import chebval,chebgauss
        
        self.order = order
        self.N = order
        
        if method=='legendre':
            from numpy.polynomial.legendre import legval
            self.basis = [lambda x,i=i:legval(x,[0]*i+[1]) for i in range(self.N+1)]
            self.coX,self.weights=leggauss(self.N+1,x0=-1.)
            self.basisCox = [b(self.coX) for b in self.basis]
            self.W=np.ones_like(self.coX)

        else: raise Exception("Invalid basis choice.")
        
        # Map bounds to given bounds or from given bounds to [-1,1].
        self.map_to_bounds = lambda x,x0,x1: (x+1)/2*(x1-x0) + x0
        self.map_from_bounds = lambda x,x0,x1: (x-x0)/(x1-x0)*2. - 1.
#end QuadGaussRadau



def finite_diff( mat,order,dx=1,**kwargs ):
    """
    Front end for calling different finite differencing methods. Will calculate down the first dimension.

    >5x speed up by using Cython
    2015-09-11
    
    Params:
    -------
    mat (ndarray)
    dx (float)
    order (int=1,2)
        Order of derivative approximation to use.
    """
    from .calculus import finite_diff_1, finite_diff_2
    if mat.ndim==1:
        mat = mat[:,None]

    if order==1:
        return finite_diff_1(mat,dx,**kwargs)
    elif order==2:
        return finite_diff_2(mat,dx,**kwargs)
    else:
        raise Exception("Invalid order option.")

#def finite_diff_1(mat,dx,axis=0,test=None):
#    """
#    Compute derivative using three-stencil with third order approximation to endpoints. 
#    https://en.wikipedia.org/wiki/Finite_difference_coefficient
#    2015-07-18
#    """
#    def center_stencil(x,i):
#        return ( 1/4*x[i-2] -2*x[i-1] + 2*x[i+1] -1/4*x[i+2] ) / (3 * dx)
#
#    def forward_stencil(x,i):
#        #return ( x[i] -x[i-1] ) / dx
#        return ( 3/2*x[i] -2*x[i-1] +1/2*x[i-2] ) / dx
#        return ( 11/16*x[i] -3*x[i-1] +3/2*x[i-2] -1/3*x[i-3] ) / dx
#
#    def backward_stencil(x,i):
#        #return ( -x[i] + x[i+1] ) / dx
#        return ( -3/2*x[i] +2*x[i+1] -1/2*x[i+2] ) / dx
#        return ( -11/16*x[i] +3*x[i+1] -3/2*x[i+2] +1/3*x[i+3] ) / dx
#
#    grad = np.array([ center_stencil(mat,i) for i in range(2,len(mat)-2) ])
#    
#    # Extrapolate endpoints to third order.
#    return np.concatenate(( [ backward_stencil(mat,0), backward_stencil(mat,1) ],
#                              grad,
#                            [ forward_stencil(mat,len(mat)-2), forward_stencil(mat,len(mat)-1) ] ))

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

    laplacian = np.array([center_stencil(mat,i) for i in range(2,mat.size-2)])
    
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



