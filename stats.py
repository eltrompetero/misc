# ========================================================================================================= #
# Module for helper functions with statistical analysis of data.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# NOTES:
# 2018-12-05 : only DiscretePowerLaw and PowerLaw are updated most recently
# ========================================================================================================= #
import numpy as np
from numpy import fft
from scipy.optimize import minimize
from scipy.special import zeta
from multiprocess import Pool,cpu_count
from numba import njit
from warnings import warn
import numpy.distutils.system_info as sysinfo
assert sysinfo.platform_bits==64


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

@njit
def has_multiple_unique_values(x):
    """Check if given list has more than one unique value. Return True if there is more
    than one unique value."""

    for i in range(1,len(x)):
        if x[i]!=x[0]:
            return True
    return False    


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
        self.rng = rng or np.random

    @classmethod
    def pdf(cls, alpha=None, lower_bound=None, upper_bound=None, normalize=True):
        """Return PDF function."""
        upper_bound = upper_bound or cls._default_upper_bound
        
        if normalize:
            Z = cls.Z(alpha, lower_bound, upper_bound)
            return lambda x, alpha=alpha, Z=Z: x**(-alpha*1.)/Z

        return lambda x,alpha=alpha: x**(-alpha*1.)

    @classmethod
    def Z(cls, alpha, lower_bound, upper_bound):
        """Return normalization."""

        assert alpha>1.000001, "alpha cannot be too close to 1."
        if upper_bound==np.inf:
            return zeta(alpha, lower_bound)
        else:
            return zeta(alpha, lower_bound) - zeta(alpha, upper_bound+1)

    @classmethod
    def pdf_as_generator(cls, alpha, lower_bound=None, upper_bound=None, normalize=True):
        """Return PDF generator."""
        
        if normalize:
            Z = cls.Z(alpha, lower_bound, upper_bound)
            
            x = lower_bound
            while x<=upper_bound:
                yield x**(1.*-alpha)/Z
                x += 1
        else:
            x = lower_bound
            while x<=upper_bound:
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
        
        Z = cls.Z(alpha, x0, x1) 
        z0 = zeta(alpha, x0)
        def cdf(x, x0=x0, x1=x1, alpha=alpha, z0=z0, Z=Z):
            assert all(x>=x0) and all(x<=x1)
            return (z0 - zeta(alpha, x+1)) / Z
        return cdf

    @classmethod
    def rvs(cls, alpha, size=(1,), lower_bound=None, upper_bound=None, rng=None):
        """Sample from discrete power law distribution and use continuous approximation for tail.
        
        Parameters
        ----------
        alpha : float
        size : tuple, (1,)
        lower_bound : int, None
        upper_bound : int, None
        rng : numpy.random.RandomState, None

        Returns
        -------
        ndarray
            Of dimensions size.
        """

        x0=lower_bound or cls._default_lower_bound
        x1=upper_bound or cls._default_upper_bound
        assert x0<=x1
        assert type(size) is int or type(size) is tuple, "Size must be an int or tuple."
        if not type(size) is tuple:
            size = (size,)
        if rng is None:
            rng = np.random

        if x1<10_001:
            try:
                return rng.choice(range(x0,int(x1)+1),
                                  size=size,
                                  p=cls.pdf(alpha,x0,x1)(np.arange(x0,x1+1)))
            except ValueError:
                print(cls.pdf(alpha,x0,x1)(np.arange(x0,x1+1)).sum(), alpha, x0, x1)
                raise Exception("Probabilities do not sum to 1.")
        
        # when upper bound is large, use continuum approximation for tail
        if x0<10_001:
            xRange = np.arange(x0, 10_001)
            p = cls.pdf(alpha, x0, x1)(xRange)
            ptail = 1-p.sum()
            X = rng.choice(xRange, p=p/p.sum(), size=size).astype(int)  # random sample
            tailix = rng.rand(*size)<ptail
            if tailix.any():
                X[tailix] = np.around( PowerLaw.rvs(alpha=alpha,
                                                    lower_bound=xRange[-1],
                                                    upper_bound=x1,
                                                    size=int(tailix.sum())) ).astype(int)
        else:
            X = np.around( PowerLaw.rvs(alpha=alpha,
                                        lower_bound=x0,
                                        upper_bound=x1,
                                        size=size ).astype(int) )
        
        if (X<0).any():
            print("Some samples exceeded numerical precision range for int. Bounding them to 2^62.")
            X[X<0] = 2**62
        return X

    @classmethod
    def max_likelihood(cls, X,
                       initial_guess=2.,
                       lower_bound_range=None,
                       lower_bound=1,
                       upper_bound=np.inf,
                       minimize_kw={},
                       full_output=False,
                       n_cpus=None,
                       max_alpha=20.,
                       decimal_resolution=None,
                       run_check=True):
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
        max_alpha : float, 20.
            max value allowed for alpha.
        decimal_resolution : int, None
        run_check : bool, True
            If True, run checks. Disable for max speed.

        Returns
        -------
        float
            alpha
        int, optional
            xmin, Only returned if the lower_bound_range is given.
        scipy.optimize.minimize or list thereof
        """

        # if only a single data is given, fitting procedure is not well defined
        if not has_multiple_unique_values(X):
            if lower_bound_range:
                if full_output:
                    return (np.nan, np.nan), {}
                return np.nan, np.nan
            if full_output:
                return np.nan, {}
            return np.nan
 
        if lower_bound_range is None:
            if run_check:
                if type(X) is list:
                    X=np.array(X)
                msg = "All elements must be within bounds. Given array includes range (%d, %d)."%(X.min(),X.max())
                assert ((X>=lower_bound)&(X<=upper_bound)).all(), msg

            logXsum = np.log(X).sum()
            if upper_bound==np.inf:
                # don't waste time computing upper bound term of 0.
                def f(alpha): 
                    # faster to eval log likelihood here
                    return alpha*logXsum + X.size*np.log(zeta(alpha, lower_bound))
            else:
                def f(alpha): 
                    # faster to eval log likelihood here
                    return alpha*logXsum + X.size*np.log(zeta(alpha, lower_bound) - zeta(alpha, upper_bound+1))

            soln = minimize(f, initial_guess, bounds=[(1.0001,max_alpha)], tol=1e-3, **minimize_kw)
            if full_output:
                return soln['x'][0], soln
            return soln['x'][0]

        # setup
        decimal_resolution = decimal_resolution or int(np.log10(X[0])+1)

        # lower bound cannot exceed the values of the elements of X, here's a very generous range
        lower_bound_range = max(lower_bound_range[0],X.min()), min(lower_bound_range[1],X.max())
        assert lower_bound_range[0]>0
        if lower_bound_range[0]>=lower_bound_range[1]:
            raise AssertionError("Impossible lower bound range.")
        assert lower_bound_range[0]<(upper_bound-1)
        assert X.min()<=lower_bound_range[1]

        boundsIx = (X>=lower_bound_range[0])&(X<=lower_bound_range[1])
        uniqLowerBounds = np.unique(np.around(X[boundsIx], decimal_resolution)).astype(int)
        if uniqLowerBounds[-1]>=X.max():
            uniqLowerBounds = uniqLowerBounds[:-1]
        if uniqLowerBounds[0]==0:
            uniqLowerBounds = uniqLowerBounds[1:]
        if uniqLowerBounds.size==0:
            if full_output:
                return (np.nan, np.nan), {}
            return np.nan, np.nan

        # set up pool to evaluate likelihood for entire range of lower bounds
        # calls cls.max_likelihood to find best alpha for the given lower bound
        def solve_one_lower_bound(lower_bound):
            alpha, soln = cls.max_likelihood(X[X>=lower_bound],
                                             initial_guess=initial_guess,
                                             lower_bound=lower_bound,
                                             upper_bound=upper_bound,
                                             minimize_kw=minimize_kw,
                                             full_output=True)
            # return CSM approach of using KS statistic
            # print("In max lik lower bound range", alpha, lower_bound)
            return alpha, cls.ksvalclass(X[X>=lower_bound], alpha, lower_bound, upper_bound), soln
        
        if n_cpus is None or n_cpus>1:
            # parallelized
            pool = Pool(cpu_count()-1)
            alpha, ksstat, soln = zip(*pool.map(solve_one_lower_bound, uniqLowerBounds))
            pool.close()
        else:
            # sequential
            alpha = np.zeros(uniqLowerBounds.size)
            ksstat = np.zeros(uniqLowerBounds.size)
            soln = []
            for i,lb in enumerate(uniqLowerBounds):
                alpha[i], ksstat[i], s = solve_one_lower_bound(lb)
                soln.append(s)
        
        bestFitIx = np.nanargmin(ksstat)
        if full_output:
            return ((alpha[bestFitIx], uniqLowerBounds[bestFitIx]),
                    (uniqLowerBounds, ksstat, soln))
        return alpha[bestFitIx], uniqLowerBounds[bestFitIx]
       
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
            return -alpha*np.log(X)

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

    def clauset_test(self, X, ksstat,
                     lower_bound_range=None,
                     bootstrap_samples=1000,
                     samples_below_cutoff=None,
                     return_all=False,
                     correction=None,
                     decimal_resolution=None,
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
        lower_bound_range : duple, None
        bootstrap_samples : int, 1000
            Number of times to bootstrap to calculate p-value.
        samples_below_cutoff : ndarray, None
            Pass integer number of samples n and return n samples.
        return_all : bool, True
        correction : function, None
        decimal_resolution : int, None
        n_cpus : int, None
            For multiprocessing.

        Returns
        -------
        float
            Fraction of random samples with deviations larger than the distribution of X.
        ndarray
            Sample of KS statistics used to measure p-value.
        tuple of (ndarray, ndarray), optional
            (alpha, lb) : the found parameters for each random sample 
        """
        
        n_cpus = n_cpus or (cpu_count()-1)
        
        if n_cpus<=1:
            self.rng = np.random.RandomState()
            ksdistribution = np.zeros(bootstrap_samples)
            alpha = np.zeros(bootstrap_samples)
            lb = np.zeros(bootstrap_samples)
            for i in range(bootstrap_samples):
                ksdistribution[i], (alpha[i],lb[i]) = self.ks_resample(len(X),
                                                                       lower_bound_range,
                                                                       samples_below_cutoff,
                                                                       return_all=True,
                                                                       correction=correction,
                                                                       decimal_resolution=decimal_resolution)
        else:
            if not samples_below_cutoff is None:
                assert (samples_below_cutoff<X.min()).all()
            def f(args):
                # scramble rng for each process
                self.rng = np.random.RandomState()
                return self.ks_resample(*args, return_all=True, correction=correction)

            pool = Pool(n_cpus)
            ksdistribution, alphalb = list(zip(*pool.map( f,
                                      [(len(X),lower_bound_range,samples_below_cutoff)]*bootstrap_samples )))
            pool.close()

            ksdistribution = np.array(ksdistribution)
            alphalb = np.array(alphalb)
            alpha = alphalb[:,0]
            lb = alphalb[:,1]
        
        assert (ksdistribution<=1).all() and (ksdistribution>=0).all()

        if return_all:
            return (ksstat<=ksdistribution).mean(), ksdistribution, (alpha,lb)
        return (ksstat<=ksdistribution).mean(), ksdistribution

    def ks_resample(self, K,
                    lower_bound_range=None,
                    samples_below_cutoff=None,
                    return_all=False,
                    correction=None,
                    decimal_resolution=None):
        """Generate a random sample from and fit to random distribution  given by specified power
        law model. This is used to generate a KS statistic.
        
        Parameters
        ----------
        K : int
            Sample size.
        lower_bound_range : duple, None
            (lb0, lb1)
        samples_below_cutoff : ndarray, None
            If provided, these are included as part of the random cdf (by bootstrap sampling) and in the model
            as specified in Clauset 2007.
        return_all : bool, False
        correction : function, None
        decimal_resolution : int, None

        Returns
        -------
        float
            KS statistic
        tuple, optional
            (alpha, lb)
        """

        if samples_below_cutoff is None or len(samples_below_cutoff)==0:
            # generate random samples from best fit power law
            # we do not consider samples with less than two unique values
            sampled = False
            while not sampled:
                X = self.rvs(alpha=self.alpha,
                             size=int(K),
                             lower_bound=self.lower_bound,
                             upper_bound=self.upper_bound,
                             rng=self.rng)
                if has_multiple_unique_values(X):
                    sampled = True

            # fit each random sample to a power law
            if lower_bound_range is None:
                alpha = self.max_likelihood(X,
                                            lower_bound=self.lower_bound,
                                            upper_bound=self.upper_bound,
                                            initial_guess=self.alpha,
                                            n_cpus=1)
                lb = self.lower_bound
            else:
                alpha, lb = self.max_likelihood(X,
                                                lower_bound_range=lower_bound_range,
                                                upper_bound=self.upper_bound,
                                                initial_guess=self.alpha,
                                                decimal_resolution=decimal_resolution,
                                                n_cpus=1)
            
            if correction:
                alpha += correction(alpha, K, lb)

            # calculate ks stat from each fit
            if return_all:
                return self.ksval(X[X>=lb], alpha, lb, self.upper_bound), (alpha, lb)
            return self.ksval(X[X>=lb], alpha, lb, self.upper_bound)
            
        fraction_below_cutoff = len(samples_below_cutoff)/(len(samples_below_cutoff)+K)
        K1 = int(self.rng.binomial(K, fraction_below_cutoff))
        K2 = int(K-K1)
        
        if K1==0:
            return self.ks_resample(K, lower_bound_range, return_all=return_all)

        # generate random samples from best fit power law and include samples below cutoff to repeat
        # entire sampling process
        # we do not consider samples with less than two unique values
        sampled = False
        while not sampled:
            X = np.concatenate((self.rng.choice(samples_below_cutoff, size=K1),
                                self.rvs(alpha=self.alpha,
                                         size=K2,
                                         lower_bound=self.lower_bound,
                                         upper_bound=self.upper_bound,
                                         rng=self.rng)))
            if has_multiple_unique_values(X):
                sampled = True

        # fit random sample to a power law
        # must set n_cpus=1 because cannot spawn processes within process
        if lower_bound_range is None:
            alpha = self.max_likelihood(X,
                                        upper_bound=self.upper_bound,
                                        initial_guess=self.alpha,
                                        decimal_resolution=decimal_resolution,
                                        n_cpus=1)
            lb = self.lower_bound
        else:
            alpha, lb = self.max_likelihood(X,
                                            lower_bound_range=(X.min(),lower_bound_range[1]),
                                            upper_bound=self.upper_bound,
                                            initial_guess=self.alpha,
                                            decimal_resolution=decimal_resolution,
                                            n_cpus=1)
        if correction:
            alpha += correction(alpha, (X>=lb).sum(), lb)

        # calculate ks stat from fit
        if return_all:
            return self.ksval(X[X>=lb], alpha, lb, self.upper_bound), (alpha, lb)
        return self.ksval(X[X>=lb], alpha, lb, self.upper_bound)

    def ksval(self, X, alpha=None, lower_bound=None, upper_bound=None, iprint=False):
        """Build CDF from given data and compare with model. Return largest distance
        between the empirical and model CDFs (the Kolmogorov-Smirnov statistic for
        discrete data).

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
        
        if iprint:
            print("In self.ksval", alpha, lower_bound)
        Xuniq, ecdf = np.unique(X, return_counts=True)
        ecdf = np.cumsum(ecdf)/len(X)
        cdf = self.cdf(alpha=alpha,
                       lower_bound=lower_bound,
                       upper_bound=upper_bound)(Xuniq)
        return np.abs(ecdf-cdf).max()
    
    @classmethod
    def ksvalclass(cls, X, alpha, lower_bound, upper_bound, iprint=False):
        """Build CDF from given data and compare with model. Return largest distance
        between the empirical and model CDFs (the Kolmogorov-Smirnov statistic for
        discrete data).

        Parameters
        ----------
        X : ndarray
        alpha : float, None
        lower_bound : int, None
        upper_bound : int, None
        iprint : bool, False

        Returns
        -------
        float
            KS statistic for a discrete distribution.
        """

        if iprint:
            print("In ksvalclass", alpha, lower_bound)
        Xuniq, ecdf = np.unique(X, return_counts=True)
        ecdf = np.cumsum(ecdf)/len(X)
        cdf = cls.cdf(alpha=alpha,
                      lower_bound=lower_bound,
                      upper_bound=upper_bound)(Xuniq)
        return np.abs(ecdf-cdf).max()
#end DiscretePowerLaw


class PowerLaw(DiscretePowerLaw):
    """With upper and lower bounds."""
    _default_alpha=2.
    _default_lower_bound=1.
    _default_upper_bound=np.inf

    def __init__(self, alpha, lower_bound=1, upper_bound=np.inf, rng=None):
        self.alpha=alpha
        self.lower_bound=lower_bound
        self.upper_bound=upper_bound
        self.rng = rng or np.random

    @classmethod
    def rvs(cls, alpha=None,
            lower_bound=None,
            upper_bound=None,
            size=(1,),
            rng=None):
        """
        Parameters
        ----------
        alpha : float, None
        lower_bound : float, None
        upper_bound : float, None
        size : tuple, (1,)
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
                upper_bound = self.upper_bound
            else:
                upper_bound = cls._default_upper_bound
        if lower_bound is None:
            if 'self.lower_bound' in vars():
                lower_bound = self.lower_bound
            else:
                lower_bound = cls._default_lower_bound

        return lower_bound * ( 1 - (1-(upper_bound/lower_bound)**(1.-alpha))*rng.rand(*size) )**(1./(1-alpha))
    
    @classmethod
    def cdf(cls, alpha=None, lower_bound=None, upper_bound=None):
        alpha=alpha or cls._default_alpha
        lower_bound=lower_bound or cls._default_lower_bound
        upper_bound=upper_bound or cls._default_upper_bound
        
        if upper_bound is None:
            def cdf(x, alpha=alpha, lower_bound=lower_bound):
                assert all(x>=lower_bound) and all(x<=upper_bound)
                return -(x**(1-alpha) - lower_bound**(1-alpha)) / lower_bound**(1-alpha)
            return cdf

        def cdf(x, alpha=alpha, lower_bound=lower_bound, upper_bound=upper_bound):
            assert (x>=lower_bound).all() and (x<=upper_bound).all(), (x.min(), x.max(),
                                                                       lower_bound, upper_bound)
            # numerical accuracy becomes a problem
            if alpha>50.:
                warn("PowerLaw.cdf uses alpha>50.")
                return np.nan
            return -(x**(1-alpha) - lower_bound**(1-alpha)) / (lower_bound**(1-alpha) - upper_bound**(1-alpha))
        return cdf

    @classmethod
    def max_likelihood(cls, X,
                       lower_bound=None,
                       upper_bound=None,
                       lower_bound_range=None,
                       decimal_resolution=None,
                       initial_guess=None,
                       full_output=False,
                       n_cpus=None,
                       max_alpha=20.,
                       minimize_kw={}):
        """
        Conventional max likelihood fit to the power law when no search for the lower
        bound is specified. When a lower bound is sought, then the max likelihood per data
        point is maximized. This can be hard to minimize correctly if the number of small
        data points is sparse (and thus the likelihood function hard to approximate as
        continuous)

        Parameters
        ----------
        X : ndarray
        lower_bound : float, None
            If no lower_bound_range is specified, then X.min() is set to lower bound.
            NOTE: Overestimation of the lower bound (which is likely when using this
            approach) can lead to serious overestimation of the true exponent.
            Underestimation is likewise an important source of bias.
        upper_bound : float, None
            Default is inf.
        lower_bound_range : duple, None
            If given, then lower bound is solved for.
        decimal_resolution : int, None
            Decimals to which to round the unique values the sample that could be lower
            bound. If unspecified, inferred by using accuracy of the lower limit of 
            lower_bound_range.
        initial_guess : float, None
            Only a guess for alpha is allowed.
        full_output : bool, False
        n_cpus : None
            Dummy argument to standardize input across classes.
        max_alpha : float, 20.
            Only used if upper_bound is specified.
        minimize_kw : dict, {}

        Returns
        -------
        float
            alpha
        dict
            Solution returned from scipy.optimize.minimize.
        """
        
        # if only a single data is given, fitting procedure is not well defined
        if not has_multiple_unique_values(X):
            if lower_bound_range:
                if full_output:
                    return (np.nan, np.nan), {}
                return np.nan, np.nan
            if full_output:
                return np.nan, {}
            return np.nan
        
        # no lower bound range is specified
        if lower_bound_range is None:
            if lower_bound is None:
                lower_bound=X.min()
            else:
                assert (X>=lower_bound).all(), "Lower bound violated."
            
            # analytic solution if lower bound is given and upper bound is at inf
            if upper_bound is None or upper_bound==np.inf:
                return 1 + 1/np.log(X/lower_bound).mean()
            
            assert (X<=upper_bound).all(), "Upper bound violated."
            def cost(alpha):
                return -cls.log_likelihood(X, alpha, lower_bound, upper_bound, True)

            soln = minimize(cost, cls._default_alpha, bounds=[(1.0001,max_alpha)])
            if full_output:
                return soln['x'], soln
            return soln['x']
        
        # if lower_bound_range is specified
        # setup
        upper_bound = upper_bound or np.inf
        if upper_bound<np.inf:
            assert (X<=upper_bound).all(), "Upper bound is violated."
        lower_bound_range = max(lower_bound_range[0], X.min()), min(lower_bound_range[-1], X.max())
        assert lower_bound_range[0]<lower_bound_range[1], "Impossible lower bound range."
        assert X.min()<=lower_bound_range[1]
        decimal_resolution = decimal_resolution or int(np.log10(lower_bound_range[0])+1)
        initial_guess = initial_guess or cls._default_alpha
        n_cpus = n_cpus or (cpu_count()-1)
      
        # try all possible lower bounds in the given range (with some coarse-grained resolution for speed)
        boundix = (X>=lower_bound_range[0])&(X<=lower_bound_range[1])
        uniqLowerBounds = np.unique(np.around(X, decimal_resolution)[boundix])
        if uniqLowerBounds[-1]>=X.max():
            uniqLowerBounds = uniqLowerBounds[:-1]
        if uniqLowerBounds[0]==0:
            uniqLowerBounds = uniqLowerBounds[1:]
        if uniqLowerBounds.size==0:
            if full_output:
                return (np.nan, np.nan), {}
            return (np.nan, np.nan)

        def parallel_wrapper(lower_bound):
            """Wrap minimization for each lower bound to try."""
            # analytic solution if lower bound is given and upper bound is at inf
            if upper_bound is None or upper_bound==np.inf:
                alphaML = 1 + 1/np.log(X[X>=lower_bound]/lower_bound).mean()
                return (alphaML,
                        cls.ksvalclass(X[X>=lower_bound], alphaML, lower_bound, upper_bound),
                        {})

            def cost(alpha, x_=X[X>=lower_bound]):
                # if only a single data point, fitting procedure is not well defined
                if not has_multiple_unique_values(x_):
                    return np.nan
                
                #assert (X<=upper_bound).all(), "Upper bound violated."
                return -cls.log_likelihood(x_, alpha, lower_bound, upper_bound, True)

            soln = minimize(lambda alpha: cost(alpha), initial_guess,
                            bounds=[(1.0001,max_alpha)],
                            **minimize_kw)
            return (soln['x'],
                    cls.ksvalclass(X[X>=lower_bound], soln['x'], lower_bound, upper_bound),
                    soln)
        
        # run max likelihood procedure over all lower bounds
        if n_cpus>1 and uniqLowerBounds.size>1:
            pool = Pool(n_cpus or cpu_count())
            alpha, ksval, soln = list(zip(*pool.map(parallel_wrapper, uniqLowerBounds)))
            pool.close()
        else:
            alpha = np.zeros(uniqLowerBounds.size)
            ksval = np.zeros(uniqLowerBounds.size)
            soln = []
            for i,ulb in enumerate(uniqLowerBounds):
                alpha[i], ksval[i], s = parallel_wrapper(ulb)
                soln.append(s)
        
        # select lower bound that minimizes the cost function
        minCostIx = np.nanargmin(ksval)
        if full_output:
            return (alpha[minCostIx], uniqLowerBounds[minCostIx]), soln[minCostIx]
        return alpha[minCostIx], uniqLowerBounds[minCostIx]

    @classmethod
    def log_likelihood(cls, X, alpha, lower_bound, upper_bound=np.inf, normalize=False):
        assert alpha>1, alpha
        if normalize:
            Z=( lower_bound**(1-alpha)-upper_bound**(1-alpha) )/(alpha-1)
            return -alpha*np.log(X).sum() - len(X) * np.log(Z)
        return -alpha*np.log(X).sum()

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

    def _posterior(self, X, density=50, sample_size=100_000):
        """Calculate confidence interval for parameters using joint posterior probability of alpha and lower
        bound. Uses discrete approximation to the posterior to generate random samples from it.

        Parameters
        ----------
        X : ndarray
            Data samples.
        alpha : float
        lb : float
        density : int
            Number of points per unit interval. Should add support to do different density along different
            axes. Lower bound should be spaced logarithmitcally as well....
        sample_size : int, 100_000
            Number of random samples to take from posterior to approximate confidence intervals.
        """
        
        from scipy.special import logsumexp

        alpha, lb = self.alpha, self.lower_bound
        logLfun = np.vectorize(lambda alpha,lb,X=X: self.log_likelihood(X[X>=lb], alpha, lb,
                                                                      normalize=True)/(X>=lb).sum())

        # mesh grid approx of logLfun
        udalpha, ldalpha = .1, min(.1,alpha-1.0001)
        udlb, ldlb = .1, min(.1,lb-1e-4)
        pThresholdRatio = 1e-1

        changeMade = True
        lbAlphaReached = False
        lbReached = False
        counter = 0
        maxIter = 8
        while changeMade and counter<maxIter:
            # just use a discrete sum to approximate the likelihood integral...should be roughly correct
            alphaRange = np.linspace(alpha-ldalpha, alpha+udalpha, int((udalpha+ldalpha)*density))
            lbRange = np.linspace(lb-ldlb, lb+udlb, int((udlb+ldlb)*density))
            alphaGrid, lbGrid = np.meshgrid(alphaRange, lbRange)

            logLGrid = logLfun(alphaGrid, lbGrid)
            
            if counter==0:
                # make sure that we're at the max likelihood peak
                mxix = np.argmax(logLGrid)
                assert (alphaGrid.ravel()[mxix]-alpha)<(2/density) and (lbGrid.ravel()[mxix]-lb)<(2/density)
            
            # identify if grid is sufficiently large
            changeMade = False
            if (not lbAlphaReached) and (logLGrid[:,0].max()-logLGrid.max())>np.log(pThresholdRatio):
                ldalpha = min(2*ldalpha, alpha-1.0001)
                changeMade = True
                lbAlphaReached = True
            if (logLGrid[:,-1].max()-logLGrid.max())>np.log(pThresholdRatio):
                udalpha *= 2
                changeMade = True
                
            if (not lbReached) and (logLGrid[0].max()-logLGrid.max())>np.log(pThresholdRatio):
                ldlb = min(2*ldlb, lb-1e-4)
                changeMade = True
                lbReached = True
            if (logLGrid[-1].max()-logLGrid.max())>np.log(pThresholdRatio):
                udlb *= 2
                changeMade = True
            print("New interval:", ldalpha, udalpha, ldlb, udlb)
            counter += 1
            
        logZ = logsumexp( logLGrid )
        p = np.exp(logLGrid-logZ)

        assert abs(np.diff(p,0)).max()<1e-3 and abs(np.diff(p,1)).max()<1e-3, "Bad approximation of probability landscape."
        randix = np.random.choice(range(alphaGrid.size), size=sample_size, p=p.ravel())
        alphaConfInterval = np.percentile(alphaGrid.ravel()[randix],5), np.percentile(alphaGrid.ravel()[randix],95)
        lbConfInterval = np.percentile(lbGrid.ravel()[randix],5), np.percentile(lbGrid.ravel()[randix],95)
        return alphaConfInterval, lbConfInterval
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

        soln = minimize(f, initial_guess, **minimize_kw, bounds=[(1.0001,7), (1e-6,np.inf)])
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
