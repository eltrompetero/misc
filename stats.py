# Module for helper functions with statistical analysis of data.
# Author: Eddie Lee
# 2017-01-17
import numpy as np
from numpy import fft
from scipy.optimize import minimize
from scipy.special import zeta
from mpmath import zeta as mpzeta
from multiprocess import Pool,cpu_count


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


# =============================================================================================== #
# Statistical distributions
# =============================================================================================== #
class DiscretePowerLaw():
    _default_lower_bound=1
    _default_upper_bound=np.inf
    _default_alpha=2.

    def __init__(self, alpha, lower_bound=1, upper_bound=np.inf, data=None, rng=None):
        self.alpha = alpha
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.rng = rng

    @classmethod
    def pdf(cls, alpha, lower_bound=None, upper_bound=None, normalize=True):
        """Return PDF function."""
        x0=lower_bound or cls._default_lower_bound
        x1=upper_bound or cls._default_upper_bound
        
        if normalize:
            if x1==np.inf:
                return lambda x,x0=x0,x1=x1,alpha=alpha: x**(1.*-alpha) / zeta(alpha,x0)
            elif (x1-x0)<1e6:
                Z = ( np.arange(x0, x1+1)**(1.*-alpha) ).sum()
            else:
                Z = zeta(alpha, x0) - zeta(alpha, x1+1)

            return lambda x,alpha=alpha: x**(1.*-alpha)/Z

        return lambda x,x0=x0,x1=x1,alpha=alpha: x**(1.*-alpha)

    @classmethod
    def Z(cls, alpha, lower_bound, upper_bound):
        """Return normalization."""
        x0=lower_bound
        x1=upper_bound

        if x1==np.inf:
            return zeta(alpha,x0)-zeta(alpha,x1+1)
        elif (x1-x0)<1e6:
            return ( np.arange(x0, x1+1)**(1.*-alpha) ).sum()
        else:
            return zeta(alpha, x0) - zeta(alpha, x1+1)

    @classmethod
    def pdf_as_generator(cls, alpha, lower_bound=None, upper_bound=None, normalize=True):
        """Return PDF generator."""
        x0=lower_bound or cls._default_lower_bound
        x1=upper_bound or cls._default_upper_bound
        
        if normalize:
            Z = zeta(alpha, x0) - zeta(alpha, x1+1)
            
            x = x0
            while x<=x1:
                yield x**(1.*-alpha)/Z
                x += 1
        else:
            x = x0 
            while x<=x1:
                yield x**(1.*-alpha)
                x += 1

        while True:
            yield 0.

    @classmethod
    def cdf_as_generator(cls, alpha, lower_bound=None, upper_bound=None):
        """Return CDF generator."""
        x0=lower_bound or cls._default_lower_bound
        x1=upper_bound or cls._default_upper_bound
        
        pdf = cls.pdf_as_generator(alpha, x0, x1)
        cump = 0
        while True:
            cump += next(pdf)
            yield cump
        
    @classmethod
    def cdf(cls, alpha, lower_bound=None, upper_bound=None):
        """Return CDF function."""
        x0 = lower_bound or cls._default_lower_bound
        x1 = upper_bound or cls._default_upper_bound
        
        if x1==np.inf:
            Z = zeta(alpha, x0)
        else:
            Z = zeta(alpha, x0) - zeta(alpha, x1+1)
        
        z0 = zeta(alpha, x0)
        def cdf(x, x1=x1, alpha=alpha, z0=z0, Z=Z):
            if not all(x<=x1):
                raise Exception
            return (z0 - zeta(alpha,x+1))/Z
        return cdf

    @classmethod
    def rvs(cls, alpha, size=(1,), lower_bound=None, upper_bound=None, rng=None):
        """Sample from distribution."""
        x0=lower_bound or cls._default_lower_bound
        x1=upper_bound or cls._default_upper_bound
        assert type(size) is int or type(size) is tuple, "Size must be an int or tuple."
        if not type(size) is tuple:
            size = (size,)
        if rng is None:
            rng = np.random

        if x1<=1e6:
            return rng.choice(range(x0,int(x1)+1),
                              size=size,
                              p=cls.pdf(alpha,x0,x1)(np.arange(x0,x1+1)))
        
        # when upper bound is large, use continuum approximation for tail
        xRange = np.arange(x0, 1_000_001)
        p = cls.pdf(alpha, x0, x1)(xRange)
        # only consider discretely p larger than some threshold
        p = p[:len(p)-np.searchsorted(p[::-1], 1e-6)]
        xRange = xRange[:len(p)]
        assert p[-1]>1e-6, "Sampling is impossible for very heavy-tails."
        ptail = 1-p.sum()

        X = rng.choice(xRange, p=p/p.sum(), size=size)  # random sample
        tailix = rng.rand(*size)<ptail
        if tailix.any():
            X[tailix] = PowerLaw.rvs(alpha=alpha, lower_bound=xRange[-1], upper_bound=x1, size=int(tailix.sum()))

        if (X<0).any():
            print("Some samples exceeded range for int.")
        return X

    @classmethod
    def max_likelihood(cls, X,
                       initial_guess=2.,
                       lower_bound_range=None,
                       lower_bound=1,
                       upper_bound=np.inf,
                       minimize_kw={},
                       full_output=False,
                       n_cpus=None):
        """
        Find the best fit power law exponent and min threshold for a discrete power law distribution. Lower
        bound is the one that gives the highest likelihood over the range specified.

        Parameters
        ----------
        X : ndarray
            Data sample.
        initial_guess : float, 2.
            Guess for power law exponent alpha
        lower_bound_range : duple, None
            If not None, then select the lower bound with max likelihood over the given range (inclusive).
        lower_bound : int, 1
            Lower cutoff inclusive.
        upper_bound : float, np.inf
            Upper cutoff inclusive.
        minimize_kw : dict, {}
        full_output : bool, False
        n_cpus : int, None

        Returns
        -------
        float
            alpha
        int, optional
            xmin, Only returned if the lower_bound_range is given.
        scipy.optimize.minimize or list thereof
        """

        from scipy.optimize import minimize

        if lower_bound_range is None:
            if type(X) is list:
                X=np.array(X)
            msg = "All elements must be within bounds. Given array includes range (%d, %d)."%(X.min(),X.max())
            assert ((X>=lower_bound)&(X<=upper_bound)).all(), msg

            logXsum = np.log(X).sum()
            def f(alpha):
                return alpha*logXsum + len(X)*np.log(zeta(alpha, lower_bound) - zeta(alpha, upper_bound+1))
                #return -cls._log_likelihood(X, alpha, lower_bound, upper_bound)
            
            soln = minimize(f, initial_guess, bounds=[(1+1e-10,7)], tol=1e-3, **minimize_kw)
            if full_output:
                return soln['x'], soln
            return soln['x']

        else:
            assert lower_bound_range[0]>0
            assert lower_bound_range[0]<(upper_bound-1)
            # lower bound cannot exceed the values of the elements of X, here's a not-very-well constrained
            # range
            lower_bound_range = np.arange(lower_bound_range[0], min(lower_bound_range[1]+1, X.max()+1),
                                          dtype=int)

            # set up pool to evaluate likelihood for entire range of lower bounds
            # calls cls.max_likelihood to find best alpha for the given lower bound
            def solve_one_lower_bound(lower_bound):
                if not (X>=lower_bound).any():
                    raise Exception("Lower bound is too large.")
                alpha, soln = cls.max_likelihood(X[X>=lower_bound],
                                                  initial_guess=initial_guess,
                                                  lower_bound=lower_bound,
                                                  upper_bound=upper_bound,
                                                  minimize_kw=minimize_kw,
                                                  full_output=True)
                # return normalized log likelihood
                return alpha, soln['fun']/(X>=lower_bound).sum()
            
            if n_cpus is None or n_cpus>1:
                pool = Pool(cpu_count()-1)
                alpha, negloglik = zip(*pool.map(solve_one_lower_bound, lower_bound_range))
                pool.close()
            else:
                alpha = np.zeros(len(lower_bound_range))
                negloglik = np.zeros(len(lower_bound_range))
                for i,lb in enumerate(lower_bound_range):
                    alpha[i], negloglik[i] = solve_one_lower_bound(lb)
            
            if full_output:
                return (alpha[np.argmin(negloglik)],
                        lower_bound_range[np.argmin(negloglik)],
                        negloglik[np.argmin(negloglik)])
            return alpha[np.argmin(negloglik)], lower_bound_range[np.argmin(negloglik)]
       
    @classmethod
    def log_likelihood(cls, X, alpha, 
                       lower_bound=1, upper_bound=np.inf, 
                       normalize=False,
                       return_sum=True):
        """Log likelihood of the discrete power law with exponent X^-alpha.

        Parameters
        ----------
        X : ndarray
        alpha : float
        lower_bound : int, 1
        upper_bound : int, np.inf
        normalize : bool, False

        Returns
        -------
        log_likelihood : ndarray
        """

        assert lower_bound<upper_bound
        assert ((X>=lower_bound) & (X<=upper_bound)).all()

        if not normalize:
            if return_sum:
                return -alpha*np.log(X).sum()
            return -alpha*np.log(X).sum()

        if return_sum:
            return ( -alpha*np.log(X).sum() -
                      np.log(zeta(alpha, lower_bound) - zeta(alpha, upper_bound+1))*len(X) )
        return -alpha*np.log(X) - np.log(zeta(alpha, lower_bound) - zeta(alpha, upper_bound+1))

    @staticmethod
    def _log_likelihood(X, alpha, lower_bound, upper_bound):
        """Faster log likelihood.

        Parameters
        ----------
        X : ndarray
        alpha : float
        lower_bound : int
        upper_bound : int

        Returns
        -------
        float
            Log likelihood.
        """

        return -alpha * np.log(X).sum() - np.log(zeta(alpha, lower_bound) - zeta(alpha, upper_bound+1))*len(X)

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

    @classmethod
    def mean_scaling(cls, X, upper_cutoff_range, full_output=False):
        """Use scaling of the mean with the cutoff to estimate the exponent.

        Parameters
        ----------
        X : ndarray
        upper_cutoff_range : ndarray
        full_output : bool, False

        Returns
        -------
        float
            alpha
        duple of ndarrays
            Set of x and y used to find the exponent.
        """
        
        m = np.zeros(len(upper_cutoff_range))
        for i,cut in enumerate(upper_cutoff_range):
            ix = X<=cut
            if ix.any():
                m[i] = X[ix].mean()
            else:
                m[i] = np.nan

        if np.isnan(m).any():
            if full_output:
                return np.nan, (upper_cutoff_range, m)
            return np.nan

        alpha = -loglog_fit(upper_cutoff_range, m)[0] + 2
        if full_output:
            return alpha, (upper_cutoff_range, m)
        return alpha

    def clauset_test(self, X, ksstat, lower_bound_range,
                     bootstrap_samples=1000,
                     samples_below_cutoff=None,
                     n_cpus=None):
        """
        Run bootstrapped test for significance of the max deviation from a power law fit to the
        sample distribution X. If there is a non-power law region part of the distribution, you need
        to define the sample_below_cutoff kwarg to draw samples from that part of the distribution.

        Parameters
        ----------
        X : ndarray
            Samples from the distribution.
        ksstat : float
            The max deviation from the empirical cdf of X given the model specified.
        lower_bound_range : duple
        bootstrap_samples : int, 1000
            Number of times to bootstrap to calculate p-value.
        samples_below_cutoff : ndarray, None
            Pass integer number of samples n and return n samples.
        n_cpus : int, None
            For multiprocessing.

        Returns
        -------
        float
            p-value
        ndarray
            Distribution of KS statistics used to measure p-value.
        """
        
        if n_cpus is None:
            n_cpus = cpu_count()-1
        
        if n_cpus<=1:
            self.rng = np.random.RandomState()
            ksdistribution = np.zeros(bootstrap_samples)
            for i in range(bootstrap_samples):
                ksdistribution[i] = self.ks_resample(len(X),
                                                     lower_bound_range,
                                                     samples_below_cutoff)
        else:
            if not samples_below_cutoff is None:
                assert (samples_below_cutoff<X.min()).all()
            def f(args):
                self.rng = np.random.RandomState()
                return self.ks_resample(*args)

            pool = Pool(n_cpus)
            ksdistribution = np.array(pool.map( f,
                                [(len(X),lower_bound_range,samples_below_cutoff)]*bootstrap_samples ))
            pool.close()

        return (ksstat<=ksdistribution).mean(), ksdistribution

    def ks_resample(self, K, lower_bound_range, samples_below_cutoff=None):
        """Generate a random sample from and fit to random distribution  given by specified power
        law model. This is used to generate a KS statistic.
        
        Parameters
        ----------
        K : int
            Sample size.
        lower_bound_range : duple
        samples_below_cutoff : ndarray, None
            If provided, these are included as part of the random cdf (by bootstrap sampling) and in the model
            as specified in Clauset 2007.

        Returns
        -------
        float
            KS statistic
        """

        if samples_below_cutoff is None:
            # generate random samples from best fit power law
            X = self.rvs(alpha=self.alpha,
                         size=K,
                         lower_bound=self.lower_bound,
                         upper_bound=self.upper_bound,
                         rng=self.rng)

            # fit each random sample to a power law
            alpha, lb = self.max_likelihood(X,
                                            lower_bound_range=lower_bound_range,
                                            upper_bound=self.upper_bound,
                                            initial_guess=self.alpha,
                                            n_cpus=1)
            
            # calculate ks stat from each fit
            return self.ksval(X, alpha, lb, self.upper_bound)
            
        fraction_below_cutoff = len(samples_below_cutoff)/(len(samples_below_cutoff)+K)
        K1 = self.rng.binomial(K, fraction_below_cutoff)
        K2 = K-K1
        
        if K1==0:
            return self.ks_resample(K, lower_bound_range)

        # generate random samples from best fit power law and include samples below cutoff to repeat
        # entire sampling process
        X = np.concatenate((self.rng.choice(samples_below_cutoff, size=K1),
                            self.rvs(alpha=self.alpha,
                                     size=K2,
                                     lower_bound=self.lower_bound,
                                     upper_bound=self.upper_bound,
                                     rng=self.rng)))

        # fit random sample to a power law
        alpha, lb = self.max_likelihood(X,
                                        lower_bound_range=(X.min(),lower_bound_range[1]),
                                        upper_bound=self.upper_bound,
                                        initial_guess=self.alpha,
                                        n_cpus=1)

        # calculate ks stat from each fit
        return self.ksval(X[X>=lb], alpha, lb, self.upper_bound)

    def ksval(self, X, alpha=None, lower_bound=None, upper_bound=None):
        """Build CDF from given data and compare with model. Return largest distance between the empirical and
        model CDFs (the Kolmogorov-Smirnov statistic for discrete data).

        Parameters
        ----------
        X : ndarray
        alpha : float, None
        lower_bound : int, None
        upper_bound : int, None

        Returns
        -------
        float
            KS statistic for a discrete distribution.
        """

        alpha = alpha or self.alpha
        lower_bound = lower_bound or self.lower_bound
        upper_bound = upper_bound or self.upper_bound
        
        ecdf = np.cumsum(np.bincount(X, minlength=X.max()+1)[X.min():])
        ecdf = ecdf/ecdf[-1]
        cdf = self.cdf(alpha=alpha,
                       lower_bound=lower_bound,
                       upper_bound=upper_bound)(np.arange(X.min(), X.max()+1))
        assert len(ecdf)==len(cdf)
        return np.abs(ecdf-cdf).max()
#end DiscretePowerLaw


class ExpTruncDiscretePowerLaw(DiscretePowerLaw):
    """Analogous to DiscretePowerLaw but with exponentially truncated tail.
    """
    _default_el=1

    def __init__(self, alpha, el, lower_bound=1, upper_bound=np.inf):
        self.alpha=alpha
        self.el=el
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound

    @classmethod
    def pdf(cls,
            alpha=None,
            el=None,
            lower_bound=1,
            upper_bound=np.inf):
        from mpmath import polylog
        assert upper_bound<np.inf, "Upper bound must be some finite value."

        alpha=alpha or cls._default_alpha
        el=el or cls._default_el
        
        p=( np.arange(int(lower_bound), int(upper_bound)+1)**-float(alpha) *
            np.exp(-el*np.arange(int(lower_bound), int(upper_bound)+1)) )
        p/=p.sum()
        
        def pdf(x, p=p):
            if type(x) is int:
                return p[x-int(lower_bound)]
            if type(x) is list:
                return p[np.array(x, dtype=int)-int(lower_bound)]
            return p[x.astype(int)-int(lower_bound)]

        return pdf

    def rvs(self,
            size=(1,),
            alpha=None,
            el=None,
            lower_bound=None,
            upper_bound=None):
        alpha=alpha or self.alpha
        el=el or self.el
        lower_bound=lower_bound or self.lower_bound
        upper_bound=upper_bound or self.upper_bound
        assert upper_bound<np.inf, "Upper bound must be some finite value."

        if not '_pdf' in self.__dict__:
            self._pdf=self.pdf(alpha, el, lower_bound, upper_bound)(np.arange(lower_bound,
                                                                              upper_bound+1))
        
        return np.random.choice(np.arange(lower_bound, upper_bound+1), size=size, p=self._pdf)

    @classmethod
    def cdf(cls,
            alpha=None,
            el=None,
            lower_bound=1,
            upper_bound=np.inf):
        from mpmath import polylog

        alpha=alpha or cls._default_alpha
        el=el or cls._default_el

        if upper_bound==np.inf:
            Z=( float(polylog(alpha, np.exp(-el))) - (np.arange(1,lower_bound)**-float(alpha) *
                np.exp(-el*np.arange(1,lower_bound))).sum() )
            return np.vectorize(lambda x: ( np.arange(lower_bound, int(x)+1)**-float(alpha) *
                                            np.exp(-el*np.arange(lower_bound, int(x)+1)) ).sum()/Z)
        raise NotImplementedError

    @classmethod
    def max_likelihood(cls, X,
                       initial_guess=(2.,1.),
                       lower_bound=1,
                       upper_bound=np.inf,
                       minimize_kw={},
                       full_output=False):
        """
        Find the best fit power law exponent for a discrete power law distribution. 

        Parameters
        ----------
        X : ndarray
        initial_guess : twople, (2.,1.)
            Guess for power law exponent alpha and cutoff el.
        lower_bound : int, 1
        upper_bound : float, np.inf
        minimize_kw : dict, {}

        Returns
        -------
        dict
            scipy.optimize.minimize or list thereof.
        """

        if type(X) is list:
            X=np.array(X)
        assert ((X>=lower_bound)&(X<=upper_bound)).all(),"All elements must be within bounds."

        def f(params):
            alpha, el=params
            return -cls.log_likelihood(X, alpha, el, lower_bound, upper_bound, normalize=True)

        soln = minimize(f, initial_guess, **minimize_kw, bounds=[(1+1e-10,np.inf), (1e-6,np.inf)])
        if full_output:
            return soln['x'], soln
        return soln['x']
       
    @classmethod
    def log_likelihood(cls, X, alpha, el,
                       lower_bound=1,
                       upper_bound=np.inf, 
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

        from mpmath import polylog
        assert ((X>=lower_bound) & (X<=upper_bound)).all()
        assert el>1e-8, "Precision errors occur when el is too small."

        if not normalize:
            return -alpha*np.log(X).sum() -el*X.sum()

        # simply calculate Z by summing up to infinity and then subtracting all terms up to the
        # lower_bound that should be ignored
        Z=( float(polylog(alpha, np.exp(-el))) - (np.arange(1,lower_bound)**-float(alpha) *
            np.exp(-el*np.arange(1, lower_bound))).sum() )
        return ( -alpha*np.log(X) - el*X - np.log(Z) ).sum()
#end ExpTruncDiscretePowerLaw


class PowerLaw(DiscretePowerLaw):
    """With upper and lower bounds."""
    _default_alpha=2.
    _default_lower_bound=1.
    _default_upper_bound=np.inf

    def __init__(self,alpha,lower_bound,upper_bound=np.inf):
        self.alpha=alpha
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound

    @classmethod
    def rvs(cls, alpha=None,
            lower_bound=None,
            upper_bound=None,
            size=(1,),
            rng=None):
        """
        Parameters
        ----------
        alpha : float,None
        lower_bound : float,None
        upper_bound : float,None
        size : tuple,(1,)
        rng : numpy.random.RandomState

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
        rng = rng or np.random

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
                 upper_bound**(1-alpha))*rng.rand(*size) )**(1/(1-alpha))
    
    @classmethod
    def cdf(cls, alpha=None, lower_bound=None, upper_bound=None):
        alpha=alpha or cls._default_alpha
        lower_bound=lower_bound or cls._default_lower_bound
        
        if upper_bound is None:
            return lambda x: -(x**(1-alpha) - lower_bound**(1-alpha)) / lower_bound**(1-alpha)
        return lambda x: ( -(x**(1-alpha) - lower_bound**(1-alpha)) /
                           (lower_bound**(1-alpha) - upper_bound**(1-alpha)) )

    @classmethod
    def max_likelihood(cls, x,
                       lower_bound=None,
                       upper_bound=None,
                       lower_bound_range=None,
                       initial_guess=None,
                       full_output=False,
                       n_cpus=None):
        """
        Parameters
        ----------
        x : ndarray
        lower_bound : float, None
        upper_bound : float, None
        lower_bound_range : duple, None
        initial_guess : tuple, None
        full_output : bool, False
        n_cpus : None
            Dummy argument to standardize input across classes.

        Returns
        -------
        float
            alpha
        dict
            Solution returned from scipy.optimize.minimize.
        """
        
        if lower_bound_range is None:
            if lower_bound is None:
                lower_bound=x.min()
            assert (x>=lower_bound).all()
            n=len(x)
            
            # analytic solution if lower bound is given and upper bound is at inf
            if upper_bound is None:
                return 1+n/np.log(x/lower_bound).sum()
            
            assert (x<=upper_bound).all()
            def cost(alpha):
                return -cls.log_likelihood(x, alpha, lower_bound, upper_bound, True)

            soln=minimize(cost, cls._default_alpha, bounds=[(1+1e-10,7)])
            if full_output:
                return soln['x'], soln
            return soln['x']
        
        # if lower_bound_range is given
        if upper_bound is None:
            upper_bound = np.inf
        else:
            assert (x<=upper_bound).all()
        if initial_guess is None:
            initial_guess = (cls._default_alpha, cls._default_lower_bound)
        if hasattr(initial_guess, '__len__'): 
            assert len(initial_guess)==2
        else:
            initial_guess = (initial_guess, x.min()*2)
        assert initial_guess[-1]<upper_bound
        
        def cost(args):
            alpha, lower_bound = args
            return -cls.log_likelihood(x[x>=lower_bound],
                                       alpha,
                                       lower_bound,
                                       upper_bound,
                                       True)/(x>=lower_bound).sum()

        soln = minimize(cost, initial_guess,
                        bounds=[(1+1e-10,7),lower_bound_range])
        if full_output:
            return soln['x'], soln
        return soln['x']

    @classmethod
    def log_likelihood(cls, x, alpha, lower_bound, upper_bound=np.inf, normalize=False):
        assert alpha>1
        if normalize:
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

    @classmethod
    def mean_scaling(cls, X, upper_cutoff_range, full_output=False):
        """Use scaling of the mean with the cutoff to estimate the exponent.

        Parameters
        ----------
        X : ndarray
        upper_cutoff_range : ndarray
        full_output : bool, False

        Returns
        -------
        float
            alpha
        duple of ndarrays
            Set of x and y used to find the exponent.
        """
        
        m = np.zeros(len(upper_cutoff_range))
        for i,cut in enumerate(upper_cutoff_range):
            m[i] = X[X<=cut].mean()

        alpha = -loglog_fit(upper_cutoff_range, m)[0] + 2
        if full_output:
            return alpha, (upper_cutoff_range, m)
        return alpha

    def ksval(self, X, alpha=None, lower_bound=None, upper_bound=None):
        """Build CDF from given data and compare with model. Return largest distance between the empirical and
        model CDFs (the Kolmogorov-Smirnov statistic for discrete data).

        Parameters
        ----------
        X : ndarray
        alpha : float, None
        lower_bound : int, None
        upper_bound : int, None

        Returns
        -------
        float
            KS statistic for a discrete distribution.
        """

        alpha = alpha or self.alpha
        lower_bound = lower_bound or self.lower_bound
        upper_bound = upper_bound or self.upper_bound
        
        uniqX, ecdf = np.unique(X, return_counts=True)
        ecdf = np.cumsum(ecdf)/len(X)
        cdf = self.cdf(alpha=alpha,
                       lower_bound=lower_bound,
                       upper_bound=upper_bound)(uniqX)
        assert len(ecdf)==len(cdf)
        return np.abs(ecdf-cdf).max()
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
        return lambda x: ( 1-gammainc(x*float(el))/gammainc(lower_bound*float(el)) )

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


def loglog_fit(x, y, p=2, iprint=False, full_output=False, symmetric=True):
    """Symmetric log-log fit.
    
    Parameters
    ----------
    x : ndarray
        Independent variable.
    y : ndarray
        Dependent variable.
    p : float, 2
        Exponent on cost function.
    iprint : bool, False
        If True, also print helpful messages.
    full_output : bool, False
        If True, also return output from scipy.optimize.minimize.
    symmetric : bool, True
        If True, use symmetrized cost function, otherwise use regular linear regression on log
        scale.

    Returns
    -------
    duple
        Exponent and offset (slope and intercept on log scale).
    dict
        Result from scipy.optimize.minimize.
    """

    from scipy.optimize import minimize

    if symmetric:
        def cost(params):
            a,b=params
            return (np.abs(a*np.log(x)+b-np.log(y))**p).sum()+(np.abs(np.log(x)+b/a-np.log(y)/a)**p).sum()
    else:
        def cost(params):
            a,b=params
            return (np.abs(a*np.log(x)+b-np.log(y))**p).sum()

    soln = minimize(cost, [1,0])
    if iprint and not soln['success']:
        print("loglog_fit did not converge on a solution.")
        print(soln['message'])
    if full_output:
        return soln['x'], soln
    return soln['x']
