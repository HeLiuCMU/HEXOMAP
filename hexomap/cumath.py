#!/usr/bin/env python

"""
Auxilary module for math functions realized through Cuda.
"""
import os
import hexomap
import numpy           as     np
from   hexomap.utility import load_kernel_code
from   pycuda.compiler import SourceModule
import pycuda.gpuarray as     gpuarray


def curandom(N: int, nvalues: int):
    """
    Description
    -----------

    Parameters
    ----------
    N: int

    nvalues: int

    Returns
    -------

    """
    # load&compile GPU code
    _kernel_code = os.path.join(os.path.dirname(hexomap.__file__), 
                                "kernel_cuda/random_cuda.cu",
                            )
    cuda_kernel = SourceModule(load_kernel_code(_kernel_code).replace('%(NGENERATORS)', N))

    # get device code
    init_func = cuda_kernel.get_function("_Z10initkerneli")
    fill_func = cuda_kernel.get_function("_Z14randfillkernelPfi")

    # get the random numbers
    seed = np.int32(123456789)  # why use fix seed here?
    init_func(seed, block=(N,1,1), grid=(1,1,1))
    gdata = gpuarray.zeros(nvalues, dtype=np.float32)
    fill_func(gdata, np.int32(nvalues), block=(N,1,1), grid=(1,1,1))

    return gdata

if __name__ == "__main__":
    pass