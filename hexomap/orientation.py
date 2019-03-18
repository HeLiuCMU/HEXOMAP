#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

"""
General math module for crystal orientation related calculation.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Eulers:
    """Euler angles representation of orientation."""
    phi1: float
    phi:  float
    phi2: float
    in_radians: bool=True
    order: str='zxz'

    @property
    def as_array(self):
        return np.array([self.phi1, self.phi, self.phi2])


@dataclass
class Quaternion:
    """
    Quaternion representation of orientation.
            q = w + Xi + Yj + Zk
    
    reference:
        http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/
    """
    w: float  # cos(theta/2)
    x: float  # sin(theta/2) * rotation_axis_x
    y: float  # sin(theta/2) * rotation_axis_y
    z: float  # sin(theta/2) * rotation_axis_z
    normalized: bool=False

    def __post_init__(self):
        # quaternion need to be a unit vector to represent orientation
        if not self.normalized:
            norm = np.linalg.norm([self.w, self.x, self.y, self.z])
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm
            self.normalized = True

    @property
    def as_array(self):
        return np.array([self.w, self.x, self.y, self.z])

    @property
    def magnitude(self):
        return np.linalg.norm(self.as_array)
    
    @property
    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z, normalized=True)

    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        # NOTE:
        # Adding quaternions has no physical meaning unless the results is
        # averaged to apprixmate the intermedia statem, provided that the
        # two rotations are infinitely small.
        return Quaternion(*(self.as_array + other.as_array))
    
    def __sub__(self, other: 'Quaternion') -> 'Quaternion':
        # NOTE:
        # Adding quaternions has no physical meaning unless the results is
        # averaged to apprixmate the intermedia statem, provided that the
        # two rotations are infinitely small.
        return Quaternion(*(self.as_array - other.as_array))
    

if __name__ == "__main__":
    q1 = Quaternion(1, 0, 0, 0)
    q2 = Quaternion(1, 0, 1, 1)
    print(q1)
    print(q2)
    q2 = q2 - q1
    print(q2)