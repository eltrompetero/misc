from .polynomial import *
import dill
import pickle
import os
TMP_DR=os.path.expanduser('~')+'/tmp/eddie'


class QuadGauss():
    def __init__(self, order, method='legendre', recache=False):
        """
        Straightforward gaussian integration using orthogonal polynomials with mapping of the
        bounds into [-1,1]. Most useful for a bounded interval.

        Parameters
        ----------
        order : int
            Order of basis expansion.
        method : str,'legendre'
        recache : bool,False
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
            # Quicklaod from cache if possible. Identify cache by degree and decimal precision
            cacheFile='%s/%s'%(TMP_DR, 'high_prec_gauss_quad_legendre_%d_%d.p'%(order, mp.dps))
            if not recache and os.path.isfile(cacheFile):
                try:
                    # Just in case multiple processes are trying to read from the same file, create your own
                    # separate copy of the file first
                    from shutil import copy
                    from uuid import uuid4
                    copyCacheFile=cacheFile + str(uuid4())
                    copy(cacheFile, copyCacheFile)

                    self.__setstate__(pickle.load(open(copyCacheFile, 'rb')))
                    os.remove(copyCacheFile)
                    run_setup=False
                except (AttributeError, EOFError):
                    run_setup=True
            else:
                run_setup=True

            if run_setup:
                self.basis = [lambda x,i=i:np.array([legendre(i,x_) for x_ in x]) for i in range(self.N+1)]
                self.coX, self.weights=leggauss(self.N+1)
                self.coX=np.array(self.coX)
                self.weights=np.array(self.weights)
                self.basisCox = [b(self.coX) for b in self.basis]
                self.W=np.ones_like(self.coX)
                
                print("QuadGauss is caching %s."%cacheFile)
                dill.dump(self.__dict__, open(cacheFile, 'wb'))

        else: raise Exception("Invalid basis choice.")

        self.define_domain_maps()
        
    def define_domain_maps(self):
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
            weight_factors=np.ones(len(self.weights))

        return ( ( f(self.map_to_bounds(self.coX,x0,x1))/self.W )*(self.weights*weight_factors) ).sum() * (x1-x0)/2

    def __setstate__(self, unpickled):
        self.basis=unpickled['basis']
        self.coX=unpickled['coX']
        self.weights=unpickled['weights']
        self.basisCox=unpickled['basisCox']
        self.W=unpickled['W']
        self.define_domain_maps()
#end QuadGauss

