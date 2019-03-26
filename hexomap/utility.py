#!/usr/bin/env python

"""
Utility module for hexomap package.
"""

import numpy as np

from functools import singledispatch
from functools import update_wrapper


def load_kernel_code(filename):
    """return the cuda source code"""
    with open(filename, 'r') as f:
        kernel_code = f.read()
    return kernel_code


def methdispatch(func):
    """class method overload decorator"""
    # ref:
    #   https://stackoverflow.com/questions/24601722/how-can-i-use-functools-singledispatch-with-instance-methods
    dispatcher = singledispatch(func)
    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


def isone(a):
    """Work around with float precision issues"""
    return np.isclose(a, 1.0, atol=1.0e-8, rtol=0.0)


def iszero(a):
    """Work around with float precision issues"""
    return np.isclose(a, 0.0, atol=1.0e-12, rtol=0.0)


if __name__ == "__main__":
    pass
