#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

"""
Useful math functions for processing HEDM data implemented in numpy.
"""

import numpy as np
from numpy.linalg import norm


def normalize(vec, axis=None):
    """
    return a normalized vector/matrix
    axis=None : normalize entire vector/matrix
    axis=0    : normalize by column
    axis=1    : normalize by row
    """
    vec = np.array(vec, dtype=np.float64)
    if axis is None:
        return vec/norm(vec)
    else:
        return np.divide(vec,
                         np.tile(norm(vec, axis=axis),
                                 vec.shape[axis],
                                 ).reshape(vec.shape,
                                           order='F' if axis == 1 else 'C',
                                           )
                         )


def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical
    distribution Algo from
    http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    """
    phi = np.random.uniform(0, np.pi*2)
    costheta = np.random.uniform(-1, 1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def safe_dotprod(vec1, vec2):
    """
    Return the dot product that is forced to be between -1.0 and 1.0.  Both
    vectors are normalized to prevent error.
    """
    return min(1.0, max(-1.0, np.dot(normalize(vec1), normalize(vec2))))


def ang_between(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """return the angle (radians) bewteen vec1 and vec2"""
    return np.arccos(np.dot(normalize(vec1), normalize(vec2)))


if __name__ == "__main__":
    pass