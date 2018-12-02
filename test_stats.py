from .stats import *
ALPHA=1.5


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

    X=np.arange(1,1001)
    totalp=np.exp(DiscretePowerLaw.log_likelihood(X, 1.5,
                                                   lower_bound=1,
                                                   upper_bound=1000,
                                                   return_sum=False,
                                                   normalize=True)).sum()
    assert np.isclose(totalp, 1), totalp

    X=np.arange(10,1001)
    totalp=np.exp(DiscretePowerLaw.log_likelihood(X, 1.5,
                                                   lower_bound=10,
                                                   upper_bound=1000,
                                                   return_sum=False,
                                                   normalize=True)).sum()
    assert np.isclose(totalp, 1), totalp

    # check normalization of pdf
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

def test_cdf_with_pdf():
    x = np.array(list(range(1,501)))
    cdfFromPdf = np.cumsum(DiscretePowerLaw.pdf(alpha=ALPHA, lower_bound=1, upper_bound=500)(x))
    cdf = DiscretePowerLaw.cdf(alpha=ALPHA, lower_bound=1, upper_bound=500)(x)
    cdfFromGen = [i for i in DiscretePowerLaw.cdf_as_generator(alpha=ALPHA, lower_bound=1, upper_bound=500)]
    assert np.isclose(cdfFromPdf, cdf).all()
    assert np.isclose(cdfFromPdf, cdfFromGen).all()

