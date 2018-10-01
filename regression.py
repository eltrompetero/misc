# For fitting.
import numpy as np
from scipy.optimize import minimize


def quad_fit(x, y, x0, c, initial_guess=0.):
    """Quadratic fit but fixed linear and constant terms.

    Parameters
    ----------
    x : ndarray
    y
    x0 : float
        linear constant
    c : float
        constant offset

    Returns
    -------
    a,b,c
        Can go directly into numpy.polyval
    """
    f=lambda a:( (y-a*(x-x0)**2+c)**2 ).sum()
    a=minimize(f, initial_guess)['x']
    return np.concatenate((a, -a*2*x0, c+a*x0**2))

def _quad_fit(x, y, b, c, initial_guess=0.):
    """Quadratic fit but fixed linear and constant terms.

    Parameters
    ----------
    x : ndarray
    y
    b : float
        linear constant
    c : float
        constant offset

    Returns
    -------
    a,b,c
        Can go directly into numpy.polyval
    """
    f=lambda a:( (y-a*x**2+b*x+c)**2 ).sum()

    return minimize(f, initial_guess)['x'], b, c
