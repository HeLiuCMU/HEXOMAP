#!/usr/bin/env python

"""
Provide backward compatibity by provide aliase to the essential functions

NOTE:
    Simply add 
        from hexomap.past import *
    to the script to continue use deprecated functions from previous version.
"""

from hexomap.orientation import Eulers
from hexomap.orientation import Rodrigues

# Backward compatibility for RotRep
# -- Euler -> Rotation matrix
EulerZXZ2Mat           = lambda e: Eulers(*e).as_matrix
EulerZXZ2MatVectorized = Eulers.eulers_to_matrices
# -- Rotation matrix -> EulerZXZ
Mat2EulerZXZ           = lambda m: Eulers.from_matrix(m).as_array
Mat2EulerZXZVectorized = Eulers.matrices_to_eulers 
# -- rod_from_quaternion
# NOTE:
#   the original function use COLUMN (axis=1) stacked quaternions,
#   the new function use ROW (axis=0) stacked to be consistent with 
#   the other methods in the same module.
rod_from_quaternion = lambda qs: Rodrigues.rodrigues_from_quaternions(qs.T).T

# Backward compatibility for FZfile

# Backward compatibility for MicFileTool

# Backward compatibility for reconstruction
