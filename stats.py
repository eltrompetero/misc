# Module for helper functions with statistical analysis of data.
# Author: Eddie Lee
# 2017-01-17



import numpy as np
from numpy import fft




def acf(x,axis=-1,return_power=False):
    """
    Compute the autocorrelation function of a given time series. According to the Wiener-Khintchine theorem,
    the autocorrelation function and power spectrum are Fourier transform duals. The mean is subtracted
    <f(t)f(t+dt)>-<f(t)>^2
    2017-04-05

    Parameters
    ----------
    x : ndarray
    axis : int,-1
    return_power: bool,False
        If True, return power spectrum.
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
    def rvs(cls,alpha,size=(1,),lower_bound=None,upper_bound=None):
        x0=lower_bound or cls._default_lower_bound
        x1=upper_bound or cls._default_upper_bound
        assert x1<np.inf,"Must define upper bound."
        
        return np.random.choice(range(x0,x1+1),
                    size=size,
                    p=cls.pdf(alpha,x0,x1)(np.arange(x0,x1+1)))

    #@classmethod
    #def rvs(cls,alpha,size=(1,),lower_bound=None,upper_bound=None):
    #    x0=lower_bound or cls._default_lower_bound
    #    x1=upper_bound or cls._default_upper_bound
    #    
    #    return (x0**(1-alpha)-(x0**(1-alpha)-x1**(1-alpha))*np.random.rand(*size))**(1/(1-alpha))

    @classmethod
    def max_likelihood_alpha(cls,X,
                             initial_guess=2.,
                             lower_bound=1,
                             upper_bound=np.inf,
                             minimize_kw={}):
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
        from scipy.special import zeta
        from scipy.optimize import minimize
        assert ((X>=lower_bound)&(X<=upper_bound)).all() 

        def f(alpha):
            if alpha<=1: return 1e30
            return -cls.log_likelihood(X,alpha,lower_bound,upper_bound).sum()
        return minimize(f,initial_guess,**minimize_kw)
       
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
        soln : scipy.optimize.minimize
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
    def log_likelihood(cls,X,alpha,
                       lower_bound=1,upper_bound=np.inf):
        """Log likelihood of the discrete power law with exponent X^-alpha.

        Parameters
        ----------
        X : ndarray
        alpha : float
        lower_bound : int,1
        upper_bound : int,np.inf

        Returns
        -------
        log_likelihood : ndarray
        """
        from scipy.special import zeta
        assert ((X>=lower_bound)&(X<=upper_bound)).all()
        return -alpha*np.log(X) - np.log(zeta(alpha,lower_bound)-zeta(alpha,upper_bound+1))
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
    def cdf(cls,alpha=None,lower_bound=None):
        return lambda x: x**1-alpha/lower_bound**(1-alpha)
#end PowerLaw
