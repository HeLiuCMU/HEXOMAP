#!/usr/bin/env python

'''
Supporting modules for crystal orientation related calculations.
'''

import os
import hexomap
import numpy           as     np
from   hexomap.utility import load_kernel_code
from   pycuda.compiler import SourceModule
import pycuda.gpuarray as     gpuarray

# load&compile GPU code
_kernel_code = os.path.join(os.path.dirname(hexomap.__file__), 
                            "kernel_cuda/device_code.cu",
                           ) 
cuda_kernel = SourceModule(load_kernel_code(_kernel_code))
misoren_gpu = cuda_kernel.get_function("misoren")


def misorien(m0, m1,symMat):
    '''
    Description
    -----------
    calculate misorientation for given list

    Parameters
    ----------
    m0: ndarray, [n,3,3]
        input orientation list
    m1: ndarray, [n,3,3]
        input orientation list 2
    symMat: ndarray
        symmetry matrix

    Returns
    -------
    ndarray
        misorientation
    '''
    m0 = m0.reshape([-1,3,3])
    m1 = m1.reshape([-1,3,3])
    symMat = symMat.reshape([-1,3,3])
    if m0.shape != m1.shape:
        raise ValueError('!m0 and m1 must have the same shape!')
    NM = m0.shape[0]
    NSymM = symMat.shape[0]
    afMisOrienD = gpuarray.empty([NM,NSymM], np.float32)
    afM0D = gpuarray.to_gpu(m0.astype(np.float32))
    afM1D = gpuarray.to_gpu(m1.astype(np.float32))
    afSymMD = gpuarray.to_gpu(symMat.astype(np.float32))
    misoren_gpu(afMisOrienD, afM0D, afM1D, afSymMD,
                block=(NSymM,1,1),
                grid=(NM,1),
               )
    return np.amin(afMisOrienD.get(), axis=1)



if __name__ == "__main__":
    # testing
    m0 = RotRep.EulerZXZ2Mat(np.array([89.5003, 80.7666, 266.397])/180.0*np.pi)
    m1 = RotRep.EulerZXZ2Mat(np.array([89.5003, 2.7666, 266.397])/180.0*np.pi)
    m0 = m0[np.newaxis,:,:].repeat(3,axis=0)
    m1 = m1[np.newaxis, :, :].repeat(3, axis=0)
    m1[0,:,:] = RotRep.EulerZXZ2Mat(np.array([89.5003, 80.7666, 266.397])/180.0*np.pi)
    symMat = RotRep.GetSymRotMat('Hexagonal')
    print(misorien(m0,m1,symMat))
    print(RotRep.Misorien2FZ1(m0, m1, 'Hexagonal'))