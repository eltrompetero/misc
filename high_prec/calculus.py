from .polynomial import *
import dill
import pickle
import os
from mpmath import quad,polyroots
from numpy.polynomial.polynomial import Polynomial
TMP_DR=os.path.expanduser('~')+'/tmp/eddie'


class LevyGaussQuad():
    """Construct Gaussian quadrature for Levy integral used in the Bethe lattice model for mean-field
    dislocation avalanches.
    
    See Numerical Recipes for details about how this works.
    """
    def __init__(self, n, x0, x1, mu, dps=15, manual_root_finding_ix=17):
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
        dps : int,15
        manual_root_finding_ix : int,17
            Last index at which numpy root finding will be used as the starting point
        """
        
        # check args
        assert x0>0 and x1>x0 and mu>0 and n>2
        if not type(x1) is mpf:
            x1=mpf(x1)
        if not type(x0) is mpf:
            x0=mpf(x0)
        if not type(mu) is mpf:
            mu=mpf(mu)

        mp.dps=dps
        
        self.x0, self.x1=x0, x1
        self.mu=mu
        self.K=lambda x, mu=self.mu, x0=self.x0, x1=self.x1 : mu/2 * x**(-mu-1) / (x0**-mu - x1**-mu)
        # check that kernel integrates to 1/2
        assert np.isclose( float(quad(self.K, [x0,x1])), .5 )

        self.n=n  # degree of polynomial
        self.manualRootFindingIx=manual_root_finding_ix
        
        self.construct_polynomials()
        
    def construct_polynomials(self):
        """Construct polynomials up to nth degree. Save the results to the instance.
        """
        
        p=[Polynomial([mp.mpf(1)])]  # polynomials
        a=[]
        b=[mp.mpf(0)]
        innerprod=[mp.mpf(1/2)]  # constructed such that inner product of f(x)=1 is 1/2

        # first polynomial is special
        a.append( quad(lambda x: self.K(x) * x, [self.x0,self.x1])/innerprod[0] )
        p.append( Polynomial([-a[0],mp.mpf(1)])*p[0] )

        for i in range(1,self.n+1):
            innerprod.append( quad(lambda x:p[i](x)**2 * self.K(x), [self.x0,self.x1]) )
            a.append( quad( lambda x:x * p[i](x)**2 * self.K(x), [self.x0,self.x1] ) / innerprod[i] )
            b.append(innerprod[i] / innerprod[i-1])
            p.append(Polynomial([-a[i],mp.mpf(1)]) * p[i] - b[i] * p[i-1])
            
        self.p=p
        self.a=a
        self.b=b
        self.innerprod=innerprod

    def polish_roots(self, coeffs, roots, n_iters):
        """
        Wrapper for using self.polish_one_root on many roots.

        Parameters
        ----------
        coeffs: list of mpf
            Coefficient to use in mp.polyval. NOTE that these are in the reverse order of numpy.polyval!
        roots : ndarray
            Initial guess for roots.
        n_iters : int

        Returns
        -------
        polished_roots : ndarray
        """
        
        polishedRoots=roots[:]
        for i,r in enumerate(roots):
            polishedRoots[i]=self.polish_one_root(coeffs, polishedRoots[i], n_iters) 
        return polishedRoots

    def polish_one_root(self, coeffs, root, n_iters):
        """
        Use Newton-Raphson method to polish root.
        TODO: add termination condition based on dx

        Parameters
        ----------
        coeffs: list of mpf
            Coefficient to use in mp.polyval. NOTE that these are in the reverse order of numpy.polyval!
        root : mp.mpf
            Initial guess for roots.
        n_iters : int

        Returns
        -------
        polished_root : mp.mpf
        """
        
        prevdx=np.inf
        for i in range(n_iters):
            p, pp=mp.polyval(coeffs, root, derivative=1)
            dx=p/pp
            if abs(dx)>abs(prevdx) and abs(dx)>1e-2:
                raise Exception(prevdx,dx)
            root -= dx
            prevdx=dx
        return root

    def bisection(self, coeffs, a, b, tol=1e-6, n_iters=10):
        """Use bisection to find roots of function. Then polish off with Newton-Raphson method.

        Parameters
        ----------
        coeffs : list of mp.mpf
            This will be passed to mpmath.polyval. Remember that the order of coeffs go from highest to lowest
            degree polynomial!
        a : mp.mpf
            Lower bound on bracket.
        b : mp.mpf
            Upper bound on bracket.
        tol : float,1e-10
            Difference b-a when the Newton-Raphson method is called.

        Returns
        -------
        root : mp.mpf
            Estimate of root.
        """

        signa=np.sign(mp.polyval(coeffs, a))
        signb=np.sign(mp.polyval(coeffs, b))
        assert signa!=signb, "Bisection will fail to find root."
        assert a<b

        found=False
        # keep bisecting the interval
        while not found:
            # using the fact that the sign of the function must change when crossing the root, we can
            # repeatedly bisect and know which side the root must be on
            signa=np.sign(mp.polyval(coeffs, a))
            signmid=np.sign(mp.polyval(coeffs, (a+b)/2))
            if signa==signmid:
                a=(a+b)/2
            else:
                b=(a+b)/2
            if (b-a)<tol:
                found=True
                root=(a+b)/2
        return self.polish_one_root(coeffs, root, n_iters)

    def levy_quad(self, n, n_iters=10, eps=1e-10, iprint=False):
        """
        Abscissa and weights for quadrature.

        Parameters
        ----------
        n : int
            Degree of expansion.
        n_iters : int,10
            Number of Newton-Raphson iterations to take at end to polish found roots.
        eps : float,1e-10
            Amount to move away from brackets to find interleaved roots.
        iprint : bool,False

        Returns
        -------
        abscissa : ndarray
        weights : ndarray
        """

        if n>len(self.p):
            raise Exception
        assert n>1
        
        # numpy works fine for small polynomials
        if n<(self.manualRootFindingIx+1):
            # find roots of polynomial
            abscissa=np.array([mp.mpf(i) for i in Polynomial(self.p[n].coef.astype(float)).roots().real])
            abscissa=self.polish_roots(self.p[n].coef[::-1].tolist(), abscissa, n_iters)
        # otherwise must find the roots slow way by using bisection
        else:
            if iprint:
                print("Starting bisection algorithm for finding roots.")
            if not '_roots' in self.__dict__.keys():
                self._roots=[]

            # since the roots are interleaved, we can build them up
            # start with base root found by using numpy's root finding
            if len(self._roots)==0:
                n_=self.manualRootFindingIx
                brackets=Polynomial(self.p[n_].coef.astype(float)).roots().real
                #brackets=self.polish_roots(self.p[n_].coef[::-1].tolist(), brackets, n_iters)
            else:
                n_=self.manualRootFindingIx+len(self._roots)
                brackets=self._roots[-1]
            brackets=np.insert(brackets, 0, self.x0)
            brackets=np.append(brackets, self.x1)

            while n_<n:
                newroots=[]
                for i in range(len(brackets)-1):
                    newroots.append( self.bisection(self.p[n_+1].coef[::-1].tolist(),
                                                    brackets[i]+eps,
                                                    brackets[i+1]-eps) )
                self._roots.append(np.array( newroots ))
                brackets=self._roots[-1]
                brackets=np.insert(brackets, 0, self.x0)
                brackets=np.append(brackets, self.x1)
                n_+=1

            abscissa=self._roots[n-self.manualRootFindingIx-1]
            abscissa=self.polish_roots(self.p[n].coef[::-1].tolist(), abscissa, n_iters)

        # using formula given in Numerical Recipes
        weights=self.innerprod[n-1] / (self.p[n-1](abscissa) * self.p[n].deriv()(abscissa))

        return abscissa, weights
#end LevyGaussQuad


class QuadGauss():
    def __init__(self, order, method='legendre', recache=False, iprint=False):
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
                    if iprint:
                        print("Loaded from cache.")
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
                
                if iprint:
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

