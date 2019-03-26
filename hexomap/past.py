#!/usr/bin/env python

"""
Provide backward compatibity by provide aliase to the essential functions

NOTE:
    Simply add 
        from hexomap.past import *
    to the script to continue use deprecated functions from previous version.
"""

from hexomap.orientation import Eulers

# Backward compatibility for RotRep
# -- Euler -> Rotation matrix
EulerZXZ2Mat           = lambda e: Eulers(*e).as_matrix
EulerZXZ2MatVectorized = Eulers.eulers_to_matrices
# -- Rotation matrix -> EulerZXZ
Mat2EulerZXZ           = lambda m: Eulers.from_matrix(m).as_array
Mat2EulerZXZVectorized = Eulers.matrices_to_eulers 
# Backward compatibility for FZfile

# Backward compatibility for MicFileTool

# Backward compatibility for reconstruction
