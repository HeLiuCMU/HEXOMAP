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
    dqcore = Quaternion.from_matrix(m1.T@m2)
    dqs  = [dqcore*op for op in sym_operator(symtype)]
    angs = [q.rot_angle for q in dqs]
    idx  = np.argmin(angs)
    return dqs[idx].as_matrix, angs[idx]
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


def generarte_random_eulerZXZ(eulerIn, range, NAngle=10):
    '''
    generate random euler angles, for detector geometry optimization

    :param eulerIn: in degree!!!, in shape[:,3]
    :param range:
    :return:
        np.array,[NAngle,3], the first one is the same as input
    '''
    eulerIn = eulerIn.reshape([-1, 3])
    shape = eulerIn.shape
    eulerIn = eulerIn * np.pi / 180.0
    eulerOut = np.repeat(eulerIn, NAngle, axis=0)
    range = range * np.pi / 180.0
    # randomAngle = np.random.rand(eulerOut.shape[0], eulerOut.shape[1])
    randomAngle = np.random.normal(0.5, 0.2, eulerOut.shape).reshape(eulerOut.shape)
    # print(randomAngle)
    eulerOut[:, 0] = eulerOut[:, 0] + range * (randomAngle[:, 0] * 2 - 1)
    eulerOut[:, 2] = eulerOut[:, 2] + range * (randomAngle[:, 2] * 2 - 1)
    z = np.cos(eulerOut[:, 1]) + range * (randomAngle[:, 1] * 2 - 1) * np.sin(eulerOut[:, 1])
    z[z > 1] = 1
    z[z < -1] = -1
    eulerOut[:, 1] = np.arccos(z)
    eulerOut[0, :] = eulerIn[0, :]
    eulerOut = eulerOut * 180.0 / np.pi
    return eulerOut

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
