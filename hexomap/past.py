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
EulerZXZ2Mat = lambda e: Eulers(*e).as_matrix

# Backward compatibility for FZfile

# Backward compatibility for MicFileTool

# Backward compatibility for reconstruction
