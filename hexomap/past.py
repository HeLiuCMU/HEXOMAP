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
quaternion_from_matrix = lambda m: Quaternion.from_matrix(m).as_array.ravel()
rod_from_quaternion = lambda qs: Rodrigues.rodrigues_from_quaternions(qs.T).T.ravel()
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
def Orien2FZ(m, symtype='Cubic'):
    """
    Reduce orientation to fundamental zone, input and output are both active matrices
    Careful, it is m*op not op*m

    Parameters
    -----------
    m:      ndarray
            Matrix representation of orientation
    symtype:string
            The crystal symmetry

    Returns
    -----------
    oRes:   ndarray
            The rotation matrix after reduced. Note that this function doesn't actually
            reduce the orientation to fundamental zone, only make sure the angle is the
            smallest one, so there are multiple orientations have the same angle but
            different directions. oRes is only one of them.
    angle:  scalar
            The reduced angle.
    """
    ops = GetSymRotMat(symtype)
    angle = 6.3
    for op in ops:
        #print(op)
        tmp = m.dot(op)
        cosangle = 0.5 * (tmp.trace() - 1)
        cosangle = min(0.9999999, cosangle)
        cosangle = max(-0.9999999, cosangle)
        newangle = np.arccos(cosangle)
        if newangle < angle:
            angle = newangle
            oRes = tmp
    return oRes, angle
# --misorien
# NOTE:
#    The original misorien is implememnted in Cuda.  The equivalent one
#    is available in Orientation class with Python native multi-threading.

# Backward compatibility for MicFileTool

# Backward compatibility for reconstruction
