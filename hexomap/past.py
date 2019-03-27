#!/usr/bin/env python

"""
Provide backward compatibity by provide aliase to the essential functions

NOTE:
    Simply add 
        from hexomap.past import *
    to the script to continue use deprecated functions from previous version.
"""

import numpy as np

from hexomap.orientation import Eulers
from hexomap.orientation import Rodrigues
from hexomap.orientation import Orientation
from hexomap.orientation import Quaternion
from hexomap.orientation import Frame
from hexomap.orientation import sym_operator

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
# -- Misorien2FZ1
def Misorien2FZ1(m1, m2, symtype='Cubic'):
    _f = Frame()
    o1 = Orientation(Quaternion.from_matrix(m1), _f)
    o2 = Orientation(Quaternion.from_matrix(m2), _f)
    
    ang, axis = o1.misorientation(o2, symtype)
    return Quaternion.from_angle_axis(ang, axis).as_matrix, ang
# -- GetSymRotMat
GetSymRotMat = lambda s='Cubic': np.array([q.as_matrix for q in sym_operator(s)])

# Backward compatibility for FZfile
# --generate_random_rot_mat
def  generate_random_rot_mat(n, method='new'):
    if method.lower() == 'new':
        return [Quaternion.from_random().as_matrix for _ in range(n)]
    else:
        nEuler = int(n)
        alpha = np.random.uniform(-np.pi,np.pi,nEuler)
        gamma = np.random.uniform(-np.pi,np.pi,nEuler)
        z = np.random.uniform(-1,1,nEuler)
        beta = np.arccos(z)
        result = np.empty([nEuler,3,3])
        for i in range(nEuler):
            matTmp = EulerZXZ2Mat(np.array([alpha[i],beta[i],gamma[i]]))
            result[i,:,:] = matTmp
        return result

# Backward compatibility for MicFileTool

# Backward compatibility for reconstruction
