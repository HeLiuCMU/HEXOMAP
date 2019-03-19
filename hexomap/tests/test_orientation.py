#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

"""
Unit test for the orientation module from hexomap package.
"""

import unittest
import numpy as np


if __name__ == "__main__":
    # Ref:
    # D Rowenhorst et al. 
    # Consistent representations of and conversions between 3D rotations
    # 10.1088/0965-0393/23/8/083501
    eulers = np.array([2.721670, 0.148401, 0.148886])  # radians
