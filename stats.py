# Module for helper functions with statistical analysis of data.
# Author: Eddie Lee
# 2017-01-17
import numpy as np
from numpy import fft
from scipy.optimize import minimize


def acf(x,axis=-1,return_power=False):
    """
    Compute the autocorrelation function of a given time series. According to the Wiener-Khintchine theorem,
    the autocorrelation function and power spectrum are Fourier transform duals. The mean is subtracted
    <f(t)f(t+dt)>-<f(t)>^2

    Parameters
    ----------
    x : ndarray
    axis : int,-1
    return_power: bool,False
        If True, return power spectrum.

    Returns
    -------
    acf : ndarray
    S : ndarray
        Power.
    """
    w = fft.fft(x-np.expand_dims(x.mean(axis=axis),axis),axis=axis)
    S = np.abs(w)**2
    # We know this must be real because the input signal is all real.
    acf = fft.ifft(S,axis=axis).real

    if x.ndim==1 or axis==0:
        if x.ndim>1:
            acf /= np.take(acf,[0],axis=axis)
        else:
            acf /= acf[0]

        acf = acf[:len(acf)//2]
    else:
        acf /= np.take(acf,[0],axis=axis)
        acf = acf[:,:acf.shape[1]//2]

    if return_power:
        return acf,S
    else:
        return acf

def _acf(x,maxlag,axis=-1):
    """
    Calculating autocorrelation function in slow way.
    2017-01-20
    """
    acf=np.ones((maxlag+1))
    for i in range(1,maxlag+1):
        acf[i]=np.corrcoef(x[:-i],x[i:])[0,1]
    return acf

def ccf(x,y,length=20):
    """
    Compute cross correlation function as a function of lag between two vectors.
    2016-12-08

    Params:
    -------
    x,y (vectors)
        Y will be dragged behind X.
    length (int=20 or list of ints)
    """
    from numpy import corrcoef

    if type(length) is int:
        c = np.zeros((length+1))
        c[0] = corrcoef(x,y)[0,1]
        for i in range(1,length+1):
            c[i] = corrcoef(x[:-i],y[i:])[0,1]
    elif type(length) is np.ndarray or type(length) is list:
        c = np.zeros((len(length)))
        for i,t in enumerate(length):
            if t==0:
                c[i] = corrcoef(x,y)[0,1]
            else:
                c[i] = corrcoef(x[:-t],y[t:])[0,1]
    else:
        raise Exception("length must be int or array of ints.")
    return c

def vector_ccf(x,y,length=20):
    """
    Compute cross correlation function between two vectors as the time-lagged, normalized dot product.
    2016-12-08

    Params:
    -------
    x,y (2d array)
        Each vector is a row.
    length (int=20 or list or ints)
    """
    from numpy.linalg import norm

    if type(length) is int:
        c = np.zeros((length+1))
        c[0] = ( (x*y).sum(1)/(norm(x,axis=1)*norm(y,axis=1)) ).mean()
        for i in range(1,length+1):
            c[i] = ( (x[:-i]*y[i:]).sum(1)/(norm(x[:-i],axis=1)*norm(y[i:],axis=1)) ).mean()
    elif type(length) is np.ndarray or type(length) is list:
        c = np.zeros((len(length)))
        for i,t in enumerate(length):
            if t==0:
                c[i] = ( (x*y).sum(1)/(norm(x,axis=1)*norm(y,axis=1)) ).mean()
            else:
                c[i] = ( (x[:-i]*y[i:]).sum(1)/(norm(x[:-i],axis=1)*norm(y[i:],axis=1)) ).mean()
    else:
        raise Exception("length must be int or array of ints.")
    return c


from scipy.special import zeta
class DiscretePowerLaw():
    _default_lower_bound=1
    _default_upper_bound=np.inf

    def __init__(self,alpha,lower_bound=1,upper_bound=np.inf):
        self.alpha=alpha
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound

    @classmethod
    def pdf(cls,alpha,lower_bound=None,upper_bound=None):
        """Return CDF function."""
        x0=lower_bound or cls._default_lower_bound
        x1=upper_bound or cls._default_upper_bound

        return lambda x: x**(1.*-alpha) / (zeta(alpha,x0)-zeta(alpha,x1+1))

    @classmethod
    def rvs(cls, alpha, size=(1,), lower_bound=None, upper_bound=None):
        x0=lower_bound or cls._default_lower_bound
        x1=upper_bound or cls._default_upper_bound
        assert x1<np.inf,"Must define upper bound."
        
        return np.random.choice(range(x0,x1+1),
                    size=size,
                    p=cls.pdf(alpha,x0,x1)(np.arange(x0,x1+1)))

    @classmethod
    def max_likelihood(cls, X,
                       initial_guess=2.,
                       lower_bound=1,
                       upper_bound=np.inf,
                       minimize_kw={},
                       full_output=False):
        """
        Find the best fit power law exponent for a discrete power law distribution. 

        Parameters
        ----------
        X : ndarray
        initial_guess : float,2.
            Guess for power law exponent alpha.
        lower_bound : int,1
        upper_bound : float,np.inf
        minimize_kw : dict,{}

        Returns
        -------
        soln : scipy.optimize.minimize or list thereof
        """

        from scipy.optimize import minimize
        if type(X) is list:
            X=np.array(X)
        assert ((X>=lower_bound)&(X<=upper_bound)).all(),"All elements must be within bounds."

        def f(alpha):
            if alpha<=1: return 1e30
            return -cls.log_likelihood(X, alpha, lower_bound, upper_bound, normalize=True)

        soln=minimize(f, initial_guess, **minimize_kw)
        if full_output:
            return soln['x'], soln
        return soln['x']
       
    @classmethod
    def pipeline_max_likelihood_alpha(cls,X,
                                      initial_guess=2.,
                                      lower_bound=1,
                                      upper_bound=np.inf,
                                      minimize_kw={}):
        """Find optimal exponential alpha for a set of different lower and upper bounds.

        Parameters
        ----------
        X : ndarray
        initial_guess : float,2.
        lower_bound : int or list,1
            If list, then list of solns will be returned for each lower bound.
        upper_bound : int or list,np.inf
            If list, then list of solns will be returned for each lower bound.
        minimize_kw : dict,{}
        
        Returns
        -------
        logL : list
            Log likelihood per data point for the given upper and lower bound combinations. First
            tuple in each element is the lower and upper bound for that solution. Then the log
            likelihood value is given.
        soln : list 
            Returns what came from scipy.optimize.minimize.
        """
        # If only one of the bounds is a list, make the other a list of the same length.
        if type(lower_bound) is list and not type(upper_bound) is list:
            upper_bound=[upper_bound]*len(lower_bound)
        elif type(lower_bound) is int and type(upper_bound) is list:
            lower_bound=[lower_bound]*len(upper_bound)
        elif not (type(lower_bound) is list and type(upper_bound) is list):
            upper_bound=[upper_bound]
            lower_bound=[lower_bound]

        soln=[]
        logL=[]
        for lower_bound_,upper_bound_ in zip(lower_bound,upper_bound):
            withinbdsIx=(X>=lower_bound_)&(X<=upper_bound_)
            soln.append( cls.max_likelihood_alpha( X[withinbdsIx],
                                                   initial_guess,
                                                   lower_bound_,
                                                   upper_bound_,
                                                   minimize_kw ) )
            logL.append( ((lower_bound_,upper_bound_),-soln[-1]['fun']/withinbdsIx.sum()) )
        return logL,soln
    
    @classmethod
    def max_likelihood_lower_bound(cls,X,alpha,
                                   initial_guess=2.,
                                   upper_bound=np.inf,
                                   minimize_kw={}):
        """
        This doesn't work. 

        Find the best fit power law exponent for a discrete power law distribution. Use full expression
        for finding the exponent alpha where X=X^-alpha that involves solving a transcendental equation.

        Parameters
        ----------
        X : ndarray
        alpha : float
            Value for power law exponent.
        initial_guess : float,2.
            Guess for lower cutoff.
        upper_bound : int,np.inf
        minimize_kw : dict,{}

        Returns
        -------
        soln : dict from scipy.optimize.minimize
        """
        from scipy.special import zeta
        from scipy.optimize import minimize
        
        def f(lower_bound):
            withinbdsIx=(X>=lower_bound)&(X<=upper_bound)
            if not 1<=lower_bound<=upper_bound: return 1e30
            return ( alpha*(zeta(alpha+1,lower_bound)-zeta(alpha+1,upper_bound+1)) /
                     (zeta(alpha,lower_bound)-zeta(alpha,upper_bound+1))+np.log(X[withinbdsIx]).mean() )**2
        return minimize(f,initial_guess)
    
    @classmethod
    def log_likelihood(cls, X, alpha, 
                       lower_bound=1, upper_bound=np.inf, 
                       normalize=False):
        """Log likelihood of the discrete power law with exponent X^-alpha.

        Parameters
        ----------
        X : ndarray
        alpha : float
        lower_bound : int,1
        upper_bound : int,np.inf
        normalize : bool,False

        Returns
        -------
        log_likelihood : ndarray
        """
        from scipy.special import zeta
        assert ((X>=lower_bound) & (X<=upper_bound)).all()
        if not normalize:
            return -alpha*np.log(X).sum()
        return ( -alpha*np.log(X) - np.log(zeta(alpha, lower_bound)-zeta(alpha, upper_bound+1))).sum()

    @classmethod
    def alpha_range(cls, x, alpha, dL, lower_bound=None, upper_bound=np.inf):
        """
        Upper and lower values for alpha that correspond to a likelihood increase/drop of dL. You must be at
        a peak of likelihood otherwise the results will be nonsensical.

        Parameters
        ----------
        x : ndarray
        alpha : float
        dL : float
        lower_bound : float,None
        upper_bound : float,np.inf

        Returns
        -------
        alphabds : twople
            Lower and upper bounds on alpha.
        """
        from scipy.optimize import minimize

        if lower_bound is None:
            lower_bound=cls._default_lower_bound
        assert (x>=lower_bound).all()
        alphabds=[0,0]

        mxlik=DiscretePowerLaw.log_likelihood(x, alpha, lower_bound, upper_bound)

        # Lower dL
        f=lambda a: (DiscretePowerLaw.log_likelihood(x, a, lower_bound, upper_bound) - (mxlik+dL))**2
        alphabds[0]=minimize( f, alpha+.1, method='nelder-mead' )['x']

        # Upper dL
        f=lambda a: (DiscretePowerLaw.log_likelihood(x, a, lower_bound, upper_bound) - (mxlik-dL))**2
        alphabds[1]=minimize( f, alpha-.1, method='nelder-mead' )['x']

        return alphabds
#end DiscretePowerLaw



class PowerLaw():
    """With upper and lower bounds."""
    _default_alpha=2.
    _default_lower_bound=1.
    _default_upper_bound=np.inf

    def __init__(self,alpha,lower_bound,upper_bound=np.inf):
        self.alpha=alpha
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound

    @classmethod
    def rvs(cls,alpha=None,lower_bound=None,upper_bound=None,size=(1,)):
        """
        Parameters
        ----------
        alpha : float,None
        lower_bound : float,None
        upper_bound : float,None
        size : tuple,(1,)

        Returns
        -------
        X : ndarray
            Sample of dimensions size.
        """
        # Input checking.
        if alpha is None:
            alpha=cls._default_alpha
        else:
            assert alpha>0
        assert type(size) is int or type(size) is tuple, "Size must be an int or tuple."
        if type(size) is int:
            size=(size,)
        assert all([type(s) is int for s in size])
        alpha*=1.

        if upper_bound is None:
            if 'self.upper_bound' in vars():
                upper_bound=self.upper_bound
            else:
                upper_bound=cls._default_upper_bound
        if lower_bound is None:
            if 'self.lower_bound' in vars():
                lower_bound=self.lower_bound
            else:
                lower_bound=cls._default_lower_bound


        return ( lower_bound**(1-alpha)-(lower_bound**(1-alpha) -
                 upper_bound**(1-alpha))*np.random.rand(*size) )**(1/(1-alpha))
    
    @classmethod
    def cdf(cls, alpha=None, lower_bound=None, upper_bound=None):
        alpha=alpha or cls._default_alpha
        lower_bound=lower_bound or cls._default_lower_bound
        
        if upper_bound is None:
            return lambda x: -(x**(1-alpha) - lower_bound**(1-alpha)) / lower_bound**(1-alpha)
        return lambda x: ( -(x**(1-alpha) - lower_bound**(1-alpha)) /
                           (lower_bound**(1-alpha) - upper_bound**(1-alpha)) )

    @classmethod
    def max_likelihood(cls, x, lower_bound=None, upper_bound=None, full_output=False):
        """
        Parameters
        ----------
        x : ndarray
        lower_bound : float,None
        upper_bound : float,None
        full_output : bool,False

        Returns
        -------
        alpha : float
        """

        if lower_bound is None:
            lower_bound=cls._default_lower_bound
        assert (x>=lower_bound).all()
        n=len(x)

        if upper_bound is None:
            return 1+n/np.log(x/lower_bound).sum()
        
        assert (x<=upper_bound).all()
        def cost(alpha):
            return -cls.log_likelihood(x, alpha, lower_bound, upper_bound, True)

        soln=minimize(cost, cls._default_alpha, bounds=[(1+1e-10,np.inf)])
        if full_output:
            return soln['x'], soln
        return soln['x']

    @classmethod
    def log_likelihood(cls, x, alpha, lower_bound, upper_bound=np.inf, normalized=False):
        assert alpha>1
        if normalized:
            Z=( lower_bound**(1-alpha)-upper_bound**(1-alpha) )/(alpha-1)
            return -alpha*np.log(x).sum() - len(x) * np.log(Z)
        return -alpha*np.log(x).sum()

    @classmethod
    def alpha_range(cls, x, alpha, dL, lower_bound=None):
        """
        Upper and lower values for alpha that correspond to a likelihood drop of dL. You must be at
        a peak of likelihood otherwise the results will be nonsensical.

        Parameters
        ----------
        x : ndarray
        alpha : float
        dL : float
        lower_bound : float,None

        Returns
        -------
        alphabds : twople
            Lower and upper bounds on alpha.
        """
        from scipy.optimize import minimize

        if lower_bound is None:
            lower_bound=cls._default_lower_bound
        assert (x>=lower_bound).all()
        alphabds=[0,0]

        mxlik=PowerLaw.log_likelihood(x, alpha, lower_bound)

        # Lower dL
        f=lambda a: (PowerLaw.log_likelihood(x, a, lower_bound) - (mxlik+dL))**2
        alphabds[0]=minimize( f, alpha+.1, method='nelder-mead' )['x']

        # Upper dL
        f=lambda a: (PowerLaw.log_likelihood(x, a, lower_bound) - (mxlik-dL))**2
        alphabds[1]=minimize( f, alpha-.1, method='nelder-mead' )['x']

        return alphabds
#end PowerLaw


class ExpTruncPowerLaw():
    """With upper and lower bounds."""
    _default_alpha=2.
    _default_lower_bound=1.
    _default_el=.1

    def __init__(self, alpha, el, lower_bound):
        self.alpha=alpha
        self.el=el
        self.lower_bound=lower_bound

        # setup inverse tranform sampling
        self.setup_inverse_cdf()

    def setup_inverse_cdf(self, cdf_res=10000):
        """Uses inverse transform sampling. CDF is approximated using cubic interpolation."""
        from scipy.interpolate import InterpolatedUnivariateSpline
        x=np.logspace(np.log10(self.lower_bound), np.log10(10/self.el), cdf_res)
        cdf=self.cdf(self.alpha, self.el, self.lower_bound)(x)
        assert (cdf<=1).all()
        
        # define inverse transform
        invcdf=InterpolatedUnivariateSpline(cdf, x, ext='const')
        self.rvs=lambda size=1: invcdf(np.random.rand(size))

    @classmethod
    def cdf(cls, alpha=None, el=None, lower_bound=None):
        from scipy.special import gamma
        from mpmath import gammainc as _gammainc

        if alpha is None:
            alpha=cls._default_alpha
        if el is None:
            el=cls._default_el
        if lower_bound is None:
            lower_bound=cls._default_lower_bound

        gammainc=np.vectorize(lambda x:float(_gammainc(1-alpha,x)))
        return lambda x: ( 1-gammainc(x*el)/gammainc(lower_bound*el) )

    @classmethod
    def max_likelihood(cls,
                       x,
                       lower_bound=None,
                       initial_guess=[2,1],
                       full_output=False):
        """
        Parameters
        ----------
        x : ndarray
        lower_bound : float,None
        initial_guess : twople,[2,1000]
        full_output : bool,False

        Returns
        -------
        params : ndarray
        """
        from scipy.optimize import minimize

        if lower_bound is None:
            lower_bound=cls._default_lower_bound
        assert (x>=lower_bound).all()
        assert initial_guess[0]>1 and initial_guess[1]>0
        
        def cost(params):
            alpha, el=params
            return -cls.log_likelihood(x, alpha, el, lower_bound, True)

        soln=minimize(cost, initial_guess, bounds=[(1+1e-10,np.inf),(1e-10,np.inf)])
        if full_output: 
            return soln['x'], soln
        return soln['x']

    @classmethod
    def log_likelihood(cls, x, alpha, el, lower_bound, normalized=False):
        """
        Parameters
        ----------
        """

        from mpmath import gammainc as _gammainc
        gammainc=lambda *args:float(_gammainc(*args))

        if normalized:
            Z=el**(alpha-1.) * gammainc(1-alpha, lower_bound*el)
            return -alpha*np.log(x).sum() - el*x.sum() - len(x) * np.log(Z)
        return -alpha*np.log(x).sum() -el*x.sum()
#end ExpTruncPowerLaw
