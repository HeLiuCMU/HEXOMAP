#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

"""
Unit test for the orientation module from hexomap package.
"""

import unittest
import numpy as np
from functools import reduce
from hexomap.orientation import Quaternion
from hexomap.orientation import Eulers
from hexomap.orientation import Frame
from hexomap.orientation import Orientation
from hexomap.npmath import ang_between
from hexomap.npmath import random_three_vector
from hexomap.npmath import normalize

class TestQuaternion(unittest.TestCase):

    def setUp(self):
        self.n_cases = 1000
        self.angs = np.random.random(self.n_cases) * np.pi
        self.axis = (np.random.random(3) - 0.5)
        self.qs = [Quaternion.from_angle_axis(ang, self.axis) 
                        for ang in self.angs
                  ]

    def test_reduce(self):
        # the cumulative rotations around the same axis should be the same 
        # as the one single rotation with the total rotation angle
        q_reduced = reduce(Quaternion.combine_two, self.qs)
        q_target = Quaternion.from_angle_axis(sum(self.angs), self.axis)
        np.testing.assert_allclose(q_reduced.as_array, q_target.as_array)

    def test_average_fixaxis(self):
        q_avg = Quaternion.average_quaternions(self.qs)
        q_target = Quaternion.from_angle_axis(np.average(self.angs), self.axis)
        np.testing.assert_allclose(q_avg.as_array, 
                                   q_target.as_array,
                                   rtol=1e-01,
                                  )
    
    def test_rotate_vec(self):
        # the testing rotation axis need to be perpendicular to the vector,
        # otherwise the angle is not the same as the input step
        ang_step = np.radians(10)
        vec = np.array([1,0,0])
        axis = np.array([0,0,1])
        for _ in range(5):
            new_vec = Quaternion.quatrotate(Quaternion.from_angle_axis(ang_step, axis), vec)
            np.testing.assert_allclose(ang_step, ang_between(vec, new_vec))
            vec = new_vec

    def test_conversion_quaternion_eulers(self):
        for _ in range(self.n_cases):
            euler = Eulers(*((np.random.random(3)-0.5)*2*np.pi))
            q = Quaternion.from_eulers(euler)
            np.testing.assert_allclose(euler.as_array, q.as_eulers.as_array)


class TestFrame(unittest.TestCase):

    def setUp(self):
        # ref
        # http://www.continuummechanics.org/techforms/Tensor.html
        self.f1 = Frame(np.array([1, 0, 0]),
                        np.array([0, 1, 0]),
                        np.array([0, 0, 1]),
                        np.array([0, 0, 0]),
                        'old',
                       )
        # simple 45 degree rotation around z
        sqr2 = np.sqrt(2)
        self.f2 = Frame(np.array([ 1/sqr2, 1/sqr2, 0]),
                        np.array([-1/sqr2, 1/sqr2, 0]),
                        np.array([0, 0, 1]),
                        np.array([0, 0, 0]),
                        'r_z_45',
                       )
        # random non-orthogonal bases
        q = Quaternion.from_random()
        self.f3 = Frame(Quaternion.quatrotate(q, self.f1.e1),
                        Quaternion.quatrotate(q, self.f1.e2),
                        Quaternion.quatrotate(q, self.f1.e3),
                        np.array([0, 0, 0]), 'randomlyRotated')
        
        # tensor in f2 frame
        self.tnsr_f2 = np.array([[3,  0, 0],
                                 [0, -1, 0],
                                 [0,  0, 0],
                                ])  # bi-axial along x,y in f2
        self.tnsr_f1 = np.array([[ 1, 2, 0],
                                 [ 2, 1, 0],
                                 [ 0, 0, 0],
                                ])  # same tensor in f1
    
    def test_transformation(self):
        # f2 bases are described in f1, therefore if we transfer f2.base|f1
        # from f1 to f2, we should get identity matrix, which is the natural
        # trievel outcome.
        _base = np.array(self.f2.base).T  # to column space
        np.testing.assert_allclose(np.eye(3), 
                                   Frame.transform_vector(_base, self.f1, self.f2)
                                )
        
        # randomly rotated frame should also work
        _base = np.array(self.f3.base).T
        np.testing.assert_allclose(np.eye(3), 
                                   Frame.transform_vector(_base, self.f1, self.f3),
                                   atol=1e-8,
                                )
        
        # tensor transform
        # NOTE:
        # need better testing case here
        np.testing.assert_allclose(self.tnsr_f2, 
                                   Frame.transform_tensor(self.tnsr_f1, self.f1, self.f2),
                                )
        np.testing.assert_allclose(self.tnsr_f1, 
                                   Frame.transform_tensor(self.tnsr_f2, self.f2, self.f1),
                                )

class TestOrientation(unittest.TestCase):

    def setUp(self):
        self.quat = Quaternion(1/np.sqrt(2), 0, 0, -1/np.sqrt(2))
        self.matx = np.array([[ 0, 1, 0],
                              [-1, 0, 0],
                              [ 0, 0, 1],
                             ])
        self.ang = np.radians(90)
        self.axis = np.array([0, 0, -1])
        self.rodrigues = np.array(0, 0, -1)
        self.frame = Frame()
        # make the Orientation
        self.ori = Orientation(self.quat, self.frame)


if __name__ == "__main__":
    unittest.main()
