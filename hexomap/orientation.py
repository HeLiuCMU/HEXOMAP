#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

"""
General math module for crystal orientation related calculation.

Most of the conventions used in this module is based on:
    D Rowenhorst et al. 
    Consistent representations of and conversions between 3D rotations
    10.1088/0965-0393/23/8/083501

with the exceptions:
    1. An orientation is always attached to a frame, and all calculations
       between orientations can only be done when all of them are converted
       to the same frame.
    2. Always prefer SI units.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Eulers:
    """
    Euler angles representation of orientation.

    Euler angle definitions:
        'Bunge' :  z -> x -> z     // prefered
        'Tayt–Briant' x -> y -> z  // roll-pitch-yaw
    """
    phi1: float  # [0, 2pi)
    phi:  float  # [0,  pi]
    phi2: float  # [0, 2pi)
    in_radians: bool=True
    order: str='zxz'
    convention: str='Bunge'

    def __post_init__(self):
        self.phi = self.phi%np.pi 

    @property
    def as_array(self):
        return np.array([self.phi1, self.phi, self.phi2])


@dataclass
class Quaternion:
    """
    Unitary quaternion representation of rotation.
            q = w + Xi + Yj + Zk
    
    reference:
        http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/
    
    Note:
        No conversion methods to other representations is provided in this
        class as the conversion requires the knowledge of reference frame,
        whereas quaternion itself does not have a frame (an abstract concept).
    """
    w: float  # cos(theta/2)
    x: float  # sin(theta/2) * rotation_axis_x
    y: float  # sin(theta/2) * rotation_axis_y
    z: float  # sin(theta/2) * rotation_axis_z
    normalized: bool=False

    def __post_init__(self) -> None:
        # quaternion need to be a unit vector to represent orientation
        if not self.normalized:
            norm = np.linalg.norm([self.w, self.x, self.y, self.z])
            self.w /= norm
            self.x /= norm
            self.y /= norm
            self.z /= norm
            self.normalized = True

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z])

    @property
    def norm(self) -> float:
        return np.linalg.norm(self.as_array)
    
    @property
    def conjugate(self) -> 'Quaternion':
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
    
    def __neg__(self) -> 'Quaternion':
        return Quaternion(*(-self.as_array))

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        # For two unit quaternions (q1, q2) that represent rotation, the
        # multiplication defined here denotes the combined rotation, namely
        #           q3 = q1 * q2
        # where q3 is a single rotation action that is equivalent to rotate
        # an object by q1, then by q2.
        Aw = self.w
        Ax = self.x
        Ay = self.y
        Az = self.z
        Bw = other.w
        Bx = other.x
        By = other.y
        Bz = other.z
        return Quaternion(
            - Ax * Bx - Ay * By - Az * Bz + Aw * Bw,
            + Ax * Bw + Ay * Bz - Az * By + Aw * Bx,
            - Ax * Bz + Ay * Bw + Az * Bx + Aw * By,
            + Ax * By - Ay * Bx + Az * Bw + Aw * Bz,
        )

    @staticmethod
    def average_quaternions(cls, qs: list) -> 'Quaternion':
        """
        Description
        -----------
        Return the average quaternion based on algorithm published in
            F. Landis Markley et.al.
            Averaging Quaternions,
            doi: 10.2514/1.28949

        Parameters
        ----------
        qs: list
            list of quaternions for average
        
        Returns
        -------
        Quaternion
            average quaternion of the given list
        """
        _sum = np.sum([np.outer(q.as_array, q.as_array) for q in qs], axis=0)
        _eigval, _eigvec = np.linalg.eig(_sum/len(qs))
        return Quaternion(*np.real(_eigvec.T[_eigval.argmax()]))


@dataclass
class Frame:
    """
    Reference frame represented as three base vectors
    
    NOTE: in most cases, frames are represented as orthorgonal bases.
    """
    e1: np.ndarray = np.array([1, 0, 0])
    e2: np.ndarray = np.array([0, 1, 0])
    e3: np.ndarray = np.array([0, 0, 1])
    name: str = "lab"


@dataclass
class Orientation:
    """
    Orientation is used to described a given object relative position to the
    given reference frame, more specifically
    
        the orientation of the crystal is described as a passive
        rotation of the sample reference frame to coincide with the 
        crystal’s standard reference frame
    
    """
    _q: Quaternion
    _f: Frame

    @property
    def frame(self) -> 'Frame':
        return self._f
    
    @frame.setter
    def frame(self, new_frame: Frame) -> None:
        pass

    @property
    def as_quaternion(self) -> 'Quaternion':
        return self._q

    @property
    def as_eulers(self) -> 'Eulers':
        pass

    @property
    def as_angleaxis(self) -> tuple:
        pass

    @staticmethod
    def random_orientations(cls, n: int, frame: Frame) -> list:
        """Return n random orientations represented in the given frame"""
        return []


def rotate_point(rotation: Quaternion, point: np.ndarray) -> np.ndarray:
    pass
    

if __name__ == "__main__":
    q1 = Quaternion(1, 0, 0, 0)
    q2 = Quaternion(1, 0, 1, 1)
    print(-q1)
    print(q2)