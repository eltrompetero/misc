# ====================================================================================== #
# Testing module for helper functions with statistical analysis of data.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# ====================================================================================== #
from .stats import *
from scipy.integrate import quad
ALPHA=1.5


def test_ECDF():
    x = np.arange(10)
    print(ECDF(x, conf_interval=(5,95)))

def test_call_format():
    pl = PowerLaw(alpha=1.5, lower_bound=1.111, upper_bound=10.111)
    assert pl.alpha==1.5 and pl.lower_bound==1.111 and pl.upper_bound==10.111
    dpl = DiscretePowerLaw(alpha=1.5, lower_bound=1.111, upper_bound=10.111)
    assert dpl.alpha==1.5 and dpl.lower_bound==1.111 and dpl.upper_bound==10.111

def test_normalization():
    # check normalization of the log likelihood
    X=np.arange(1,1001)
    np.isclose(np.exp(DiscretePowerLaw.log_likelihood(X, 2,
                                                   lower_bound=1,
                                                   upper_bound=1000,
                                                   return_sum=False,
                                                   normalize=True)).sum(), 1)

    X=np.arange(10,1001)
    np.isclose(np.exp(DiscretePowerLaw.log_likelihood(X, 2,
                                                   lower_bound=10,
                                                   upper_bound=1000,
                                                   return_sum=False,
                                                   normalize=True)).sum(), 1)
    
    # alpha=3/2
    X=np.arange(1,1001)
    totalp=np.exp(DiscretePowerLaw.log_likelihood(X, 1.5,
                                                   lower_bound=1,
                                                   upper_bound=1000,
                                                   return_sum=False,
                                                   normalize=True)).sum()
    assert np.isclose(totalp, 1), totalp
    # change lower bound
    X=np.arange(10,1001)
    totalp=np.exp(DiscretePowerLaw.log_likelihood(X, 1.5,
                                                   lower_bound=10,
                                                   upper_bound=1000,
                                                   return_sum=False,
                                                   normalize=True)).sum()
    assert np.isclose(totalp, 1), totalp

    # check normalization of pdf
    # don't specify lower bound (None gets passed to zeta function equivalent to passing 1)
    X=np.arange(1,1001)
    totalp=DiscretePowerLaw.pdf(1.5,
                                upper_bound=1000)(X).sum()
    assert np.isclose(totalp, 1), totalp

    X=np.arange(10,1001)
    totalp=DiscretePowerLaw.pdf(1.5,
                               lower_bound=10,
                               upper_bound=1000)(X).sum()
    assert np.isclose(totalp, 1), totalp
    
    X=np.arange(10,100001)
    totalp=DiscretePowerLaw.pdf(1.5,
                               lower_bound=10,
                               upper_bound=X.max())(X).sum()
    assert np.isclose(totalp, 1), totalp

    # log likelihood: compare pdf calculation with log_likelihood and _log_likelihood
    logp=np.log( DiscretePowerLaw.pdf(1.5,
                                      lower_bound=10,
                                      upper_bound=X.max())(X) )
    logL=DiscretePowerLaw.log_likelihood(X, 1.5,
                                         lower_bound=10,
                                         upper_bound=X.max(),
                                         normalize=True,
                                         return_sum=False)
    logLquick=DiscretePowerLaw._log_likelihood(X, 1.5,
                                         10,
                                         X.max())
    assert np.isclose(logp, logL).all()
    assert np.isclose(logLquick, logL.sum())

def test_cdf_with_pdf():
    x = np.array(list(range(1,501)))
    cdfFromPdf = np.cumsum(DiscretePowerLaw.pdf(alpha=ALPHA, lower_bound=1, upper_bound=500)(x))
    cdf = DiscretePowerLaw.cdf(alpha=ALPHA, lower_bound=1, upper_bound=500)(x)
    gen = DiscretePowerLaw.cdf_as_generator(alpha=ALPHA, lower_bound=1, upper_bound=500)
    cdfFromGen = [next(gen) for i in x]

    assert np.isclose(cdfFromPdf, cdf).all()
    assert np.isclose(cdfFromPdf, cdfFromGen).all()

def test_pdf():
    # check that pdf decays correctly (relative probabilities)
    for a in np.arange(1.5,5.5,.5):
        p = DiscretePowerLaw.pdf(a, 1)(2**np.arange(30))
        assert np.isclose( np.diff(np.log(p)), np.log(2**-a) ).all()

def test_max_likelihood_flow():
    """Make sure returned estimates are what I've found before."""
    # continuous
    X = PowerLaw.rvs(size=1000, rng=np.random.RandomState(0))
    alphaML = PowerLaw.max_likelihood(X)
    assert alphaML==1.997014843072137

    alphaML, lb = PowerLaw.max_likelihood(X, lower_bound_range=(1,10), initial_guess=1.76)
    assert alphaML==1.9582463655267062 and lb==1.7
   
#    d = ExpTruncPowerLaw(2, 1e-3, rng=np.random.RandomState(0))
#    X = d.rvs(size=1000)
#    alphaML, elML = d.max_likelihood(X)
#    assert abs(2-alphaML)<1e-2 and abs(1e-3-elML)<2e-3, (alphaML, elML)
#
#    d = ExpTruncPowerLaw(1.75, 1e-3, rng=np.random.RandomState(0))
#    X = d.rvs(size=1000)
#    alphaML, elML = d.max_likelihood(X)
#    assert abs(1.75-alphaML)<1e-2 and abs(1e-3-elML)<2e-3, (alphaML, elML)

    # discrete
    X = DiscretePowerLaw.rvs(2., size=1000, rng=np.random.RandomState(0))
    alphaML = DiscretePowerLaw.max_likelihood(X)
    assert alphaML==1.9939545133636511

    alphaML, lb = DiscretePowerLaw.max_likelihood(X, lower_bound_range=(1,10), initial_guess=1.76)
    assert alphaML==1.9939545644486907 and lb==1

def test_ExpTruncPowerLaw():
    d = ExpTruncPowerLaw(2, 1e-3, rng=np.random.RandomState(0))
    assert np.isclose(quad(d.pdf(), 1, np.inf)[0], 1)

    X = d.rvs(size=1000)
    assert np.isclose([2,1e-3], d.max_likelihood(X), atol=2e-3, rtol=0).all()
    print("Test passed: probability distribution is normalized.")

    dpl = ExpTruncPowerLaw(2., el=.001, rng=np.random.RandomState(0))
    y = dpl.rvs(100_000)
    soln = dpl.max_likelihood(y)
    assert abs(soln[0] - 2)<=1e-2 and abs(soln[1] - 1e-3)<=5e-4

    dpl = ExpTruncPowerLaw(1.5,
                           el=.001,
                           rng=np.random.RandomState(0),
                           lower_bound=10)
    y = dpl.rvs(100_000)
    soln = dpl.max_likelihood(y)
    assert abs(soln[0] - 1.5)<=1e-2 and abs(soln[1] - 1e-3)<=5e-4
    print("Test passed: Max likelihood recovers correct parameters.")

def test_ExpTruncDiscretePowerLaw():
    dpl = ExpTruncDiscretePowerLaw(2., el=.001, rng=np.random.RandomState(0))
    y = dpl.rvs(100_000)
    soln = dpl.max_likelihood(y)
    assert abs(soln[0] - 2)<=1e-2 and abs(soln[1] - 1e-3)<=5e-4
    print("Test passed: Max likelihood recovers correct parameters.")

    dpl = ExpTruncDiscretePowerLaw(1.5,
                                   el=.001,
                                   rng=np.random.RandomState(0),
                                   lower_bound=10)
    y = dpl.rvs(100_000)
    soln = dpl.max_likelihood(y)
    assert abs(soln[0] - 1.5)<=1e-2 and abs(soln[1] - 1e-3)<=5e-4
    print("Test passed: Max likelihood recovers correct parameters.")
