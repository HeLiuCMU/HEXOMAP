#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

"""
Unit test for the orientation module from hexomap package.
"""

import unittest
import numpy as np
from functools import reduce
from hexomap.orientation import Quaternion


class TestQuaternion(unittest.TestCase):

    def setUp(self):
        pass

    def test_reduce(self):
        angs = np.random.random(10) * np.pi
        axis = (np.random.random(3) - 0.5)
        qs = [Quaternion.from_angle_axis(ang, axis) 
                for ang in angs
        ]
        q_total = reduce(Quaternion.reduce_two, qs)
        q_target = Quaternion.from_angle_axis(sum(angs), axis)

        np.testing.assert_almost_equal(q_total.as_array, q_target.as_array)


if __name__ == "__main__":
    unittest.main()
