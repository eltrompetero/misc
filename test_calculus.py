from .calculus import *


def test_leggauss():
    """Check up to to deg=10."""

    import numpy.polynomial.legendre as npleg
    from scipy.integrate import quad
    
    # Check that weights and abscissa line up with numpy module.
    for d in range(2,10):
        myx,myw=leggauss(d)
        x,w=npleg.leggauss(d)
        assert np.isclose(myx,x).all() and np.isclose(myw,w).all()

    # Check that quadrature with these work for both regular and Radau quadrature.
    mxPolyOrder=6

    # Legendre quadrature
    x,w=leggauss(mxPolyOrder+1)
    for po in range(1,mxPolyOrder+1):
        f=lambda x:x**po

        if (po%2)==1:
            assert np.isclose(f(x).dot(w),0)
        else:
            trueVal=2./(po+1)
            print(f(x).dot(w),trueVal)
            assert np.isclose(f(x).dot(w),trueVal)

    x,w=leggauss(mxPolyOrder+1,x0=-1.)
    for po in range(1,mxPolyOrder+1):
        f=lambda x:x**po
        
        if (po%2)==0:
            trueVal=2./(po+1)
            print(f(x).dot(w),trueVal)
            assert np.isclose(f(x).dot(w),trueVal), (po, f(x).dot(w), trueVal)

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
    
    # Radau-Gauss
    # Legendre quadrature
    qg=QuadGaussRadau(10)
    for po in range(1,mxPolyOrder+1):
        f=lambda x:x**po

        if (po%2)==1:
            assert np.isclose(qg.quad(f,-1,1),0)
        else:
            trueVal=2./(po+1)
            assert np.isclose(qg.quad(f,-1,1),trueVal)

def test_LevyQuadGauss():
    x0, x1=1, 3.6
    mu=1.5
    lgq=LevyGaussQuad(30, x0, x1, mu)

    for i in range(3,15,5):
        """Test that integration matches up with integration with quad."""
        abscissa, weights=lgq.levy_quad(i)
        
        # check that weights add up to 1/2 which is same as integrating f(x)=1
        assert np.isclose(weights.sum(), .5)
        
        # check simple polynomial integrals
        f=lambda x:x
        assert np.isclose(quad(lambda x:f(x)*lgq.K(x), x0, x1)[0], weights.dot(f(abscissa)))
        f=lambda x:x**2
        assert np.isclose(quad(lambda x:f(x)*lgq.K(x), x0, x1)[0], weights.dot(f(abscissa)))
        # this is the highest order that should be possible to integrate
        f=lambda x:x**3
        assert np.isclose(quad(lambda x:f(x)*lgq.K(x), x0, x1)[0], weights.dot(f(abscissa)))
