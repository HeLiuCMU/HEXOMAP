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
    normalized: bool=True

    def __post__init__(self):
        # quaternion need to be a unit vector to represent orientation
        if not self.normalized:
            self.normalize()

    @property
    def magnitude(self):
        return np.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        self.w /= self.magnitude
        self.x /= self.magnitude
        self.y /= self.magnitude
        self.z /= self.magnitude

if __name__ == "__main__":
    q = Quaternion(1, 0, 0, 0)
    print(q)