#!/usr/bin/env python

"""
Utility module for hexomap package.
"""


def load_kernel_code(filename):
    """return the cuda source code"""
    with open(filename, 'r') as f:
        kernel_code = f.read()
    return kernel_code


if __name__ == "__main__":
    pass