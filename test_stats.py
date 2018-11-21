from .stats import *


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
