from .polynomial import *


class QuadGauss(object):
    def __init__(self, order, method='legendre'):
        """
        Straightforward gaussian integration using orthogonal polynomials with mapping of the
        bounds into [-1,1]. Most useful for a bounded interval.

        Parameters
        ----------
        order : int
            Order of basis expansion.
        method : str,'legendre'
        lobatto : bool,False
            If True, use Lobatto collocation points. Only works for Chebyshev polynomials.
        """
        self.order = order
        self.N = order
        
        if method=='chebyshev':
            raise NotImplementedError
            #self.basis = [lambda x,i=i:chebval(x,[0]*i+[1]) for i in range(self.N+1)]

            ## Lobatto collocation points.
            #if lobatto:
            #    self.coX = -np.cos(np.pi*np.arange(self.N+1)/self.N)
            #    self.weights = np.zeros(self.order+1)+np.pi/self.N
            #    self.weights[0] = self.weights[-1] = np.pi/(2*self.N)
            #else:
            #    self.coX,self.weights = chebgauss(self.N+1)
            #self.basisCox = [b(self.coX) for b in self.basis]

            #if ((self.coX)==1).any():
            #    self.W=np.zeros_like(self.coX)
            #    ix=np.abs(self.coX)==1
            #    self.W[ix==0]=1/np.sqrt(1-self.coX[ix==0]**2)
            #    self.W[ix]=np.inf
            #else:
            #    self.W = 1/np.sqrt(1-self.coX**2)  # weighting function we must remove
            #self.W[np.isnan(self.W)] = 0.

        elif method=='legendre':
            self.basis = [lambda x,i=i:np.array([legendre(i,x_) for x_ in x]) for i in range(self.N+1)]
            self.coX,self.weights=leggauss(self.N+1)
            self.coX=np.array(self.coX)
            self.weights=np.array(self.weights)
            self.basisCox = [b(self.coX) for b in self.basis]
            self.W=np.ones_like(self.coX)

        else: raise Exception("Invalid basis choice.")
        
        # Map bounds to given bounds or from given bounds to [-1,1].
        self.map_to_bounds = lambda x,x0,x1: (x+1)/2*(x1-x0) + x0
        self.map_from_bounds = lambda x,x0,x1: (x-x0)/(x1-x0)*2. - 1.

    def quad(self,f,x0,x1):
        """
        Parameters
        ----------
        f : lambda function
            One dimensional function
        x0 : float
        x1 : float

        Returns
        -------
        val : float
        """
        return ( ( f(self.map_to_bounds(self.coX,x0,x1))/self.W )*(self.weights) ).sum() * (x1-x0)/2
#end QuadGauss

