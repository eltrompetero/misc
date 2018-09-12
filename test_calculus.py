from calculus import *


def test_quad():
    from scipy.integrate import quad
    mxPolyOrder=6

    # Legendre quadrature
    qg=QuadGauss(10,lobatto=True)
    for po in range(1,mxPolyOrder+1):
        f=lambda x:x**po

        if (po%2)==1:
            assert np.isclose(qg.quad(f,-1,1),0)
        else:
            trueVal=2./(po+1)
            assert np.isclose(qg.quad(f,-1,1),trueVal)

    # Chebyshev quadrature
    qg=QuadGauss(10,method='chebyshev')
    for po in range(1,mxPolyOrder+1):
        f=lambda x:x**po/np.sqrt(1-x**2)
        assert np.isclose(qg.quad(f,-1,1),quad(f,-1,1)[0])
