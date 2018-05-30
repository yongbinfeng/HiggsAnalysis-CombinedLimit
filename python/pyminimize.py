"""
Unified interfaces to minimization algorithms.

Functions
---------
- minimize : minimization of a function of several variables.
- minimize_scalar : minimization of a function of one variable.
"""
from __future__ import division, print_function, absolute_import


#__all__ = ['minimize', 'minimize_scalar']

__all__ = ['minimize']


from warnings import warn

import numpy as np

from scipy._lib.six import callable

# unconstrained minimization
from .pyoptimize import _minimize_bfgs, MemoizeJac


def minimize(fun, x0, args=(), jac=None, invhess=None,
             hessp=None, bounds=None, constraints=(), tol=None,
             callback=None, options=None):
    """Minimization of scalar function of one or more variables.
    based on scipy minimization
    """
    x0 = np.asarray(x0)
    if x0.dtype.kind in np.typecodes["AllInteger"]:
        x0 = np.asarray(x0, dtype=float)

    if not isinstance(args, tuple):
        args = (args,)

    if options is None:
        options = {}
        
   # fun also returns the jacobian
    if not callable(jac):
        if bool(jac):
            fun = MemoizeJac(fun)
            jac = fun.derivative
        else:
            jac = None
 
    return _minimize_bfgs(fun, x0, args, jac, invhess, callback, **options)
    
