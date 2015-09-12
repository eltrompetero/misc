from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

def finite_diff_1( np.ndarray[dtype=np.float_t,ndim=1] mat,
                   float dx,
                   int axis=0 ):
    """
    Compute derivative using three-stencil with third order approximation to endpoints. 
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    2015-07-18
    """
    def center_stencil(x,i):
        return ( 1/4*x[i-2] -2*x[i-1] + 2*x[i+1] -1/4*x[i+2] ) / (3 * dx)

    def forward_stencil(x,i):
        #return ( x[i] -x[i-1] ) / dx
        return ( 3/2*x[i] -2*x[i-1] +1/2*x[i-2] ) / dx
        return ( 11/16*x[i] -3*x[i-1] +3/2*x[i-2] -1/3*x[i-3] ) / dx

    def backward_stencil(x,i):
        #return ( -x[i] + x[i+1] ) / dx
        return ( -3/2*x[i] +2*x[i+1] -1/2*x[i+2] ) / dx
        return ( -11/16*x[i] +3*x[i+1] -3/2*x[i+2] +1/3*x[i+3] ) / dx

    grad = np.array([ center_stencil(mat,i) for i in xrange(2,len(mat)-2) ])
    
    # Extrapolate endpoints to third order.
    return np.concatenate(( [ backward_stencil(mat,0), backward_stencil(mat,1) ],
                              grad,
                            [ forward_stencil(mat,len(mat)-2), forward_stencil(mat,len(mat)-1) ] ))

def finite_diff_2( np.ndarray[dtype=np.float_t,ndim=1] mat,
                   float dx,
                   int axis=0 ):
    """
    Compute second derivative using fourth order approximation with third order approximation to endpoints. 
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    2015-07-18
    """
    def center_stencil( np.ndarray[dtype=np.float_t,ndim=1] x, int i):
        return (-1/12*x[i-2] + 4/3*x[i-1] -5/2*x[i] + 4/3*x[i+1] -1/12*x[i+2]) / dx**2

    def forward_stencil( np.ndarray[dtype=np.float_t,ndim=1] x, int i):
        return (35/12*x[i] -26/3*x[i-1] +19/2*x[i-2] -14/3*x[i-3] +11/12*x[i-4]) / dx**2

    def backward_stencil( np.ndarray[dtype=np.float_t,ndim=1] x, int i):
        return (35/12*x[i] -26/3*x[i+1] +19/2*x[i+2] -14/3*x[i+3] +11/12*x[i+4]) / dx**2

    laplacian = np.array([center_stencil(mat,i) for i in xrange(2,mat.size-2)])
    
    # Extrapolate endpoints.
    return np.concatenate(( [backward_stencil(mat,0), backward_stencil(mat,1)],
                            laplacian,
                            [forward_stencil(mat,mat.size-2), forward_stencil(mat,mat.size-1)] ))

