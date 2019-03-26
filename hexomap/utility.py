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


def isone(a: float) -> bool:
    """Work around with float precision issues"""
    return np.isclose(a, 1.0, atol=1.0e-8, rtol=0.0)


def iszero(a: float) -> bool:
    """Work around with float precision issues"""
    return np.isclose(a, 0.0, atol=1.0e-12, rtol=0.0)


def standarize_euler(euler: np.ndarray, in_radian=True) -> np.ndarray:
    """
    Force Euler angle to be betewen 
        [0~2pi, 0~pi, 0~2pi]
    """
    if not in_radian:
        euler = np.radians(euler)
    return np.where(
        euler<0, 
        (euler+2.0*np.pi)%np.array([2.0*np.pi,np.pi,2.0*np.pi]),
        euler%(2*np.pi)
        )


if __name__ == "__main__":
    pass
