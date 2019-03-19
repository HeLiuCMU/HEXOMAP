#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

"""
Unit test for the orientation module from hexomap package.
"""

import unittest
import numpy as np
from functools import reduce
from hexomap.orientation import Quaternion
from hexomap.npmath import ang_between

class TestQuaternion(unittest.TestCase):

    def setUp(self):
        n_cases = 1000
        self.angs = np.random.random(n_cases) * np.pi
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
                                   rtol=1e-02,
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


if __name__ == "__main__":
    unittest.main()
