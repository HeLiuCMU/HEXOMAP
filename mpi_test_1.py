import numpy as np
from mpi4py import MPI
import atexit
atexit.register(MPI.Finalize)
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import pycuda.driver as cuda
import numpy as np
import matplotlib.pyplot as plt
import pickle
# customized module

import reconstruction  # g-force caller
import MicFileTool     # io for reconstruction rst
import IntBin          # io for binary image (reduced data)

def gen_mpi_masks(imgsize, n_node, mask=None, mode='square'):
    '''
    generate mask used for mpi
    imgsize: [200,200]
    n_node: 1,2, or 4
    mask: original mask
    mode: 'horizontal', 'vertical', 'square'
    '''
    lMask = []
    if mask is None:
        mask = np.ones(imgsize)
    if n_node==4:
        nx = 2
        ny = 2
    elif n_node==2:
        nx = 2
        ny = 1
    else:
        raise ValueError('not implemented, choose n_node is 2 or 4')
    for i in range(nx):
        for j in range(ny):
            new_mask = np.zeros(imgsize)
            new_mask[i*imgsize[0]//nx: (i+1)*imgsize[0]//nx,j*imgsize[0]//ny: (j+1)*imgsize[0]//ny] = mask[i*imgsize[0]//nx: (i+1)*imgsize[0]//nx,j*imgsize[0]//ny: (j+1)*imgsize[0]//ny]
            lMask.append(new_mask.astype(np.int32))
    return lMask
    

# check result of parameter get from blind search:
with open('data/johnson_aug18_demo/geometry_au_johnson_aug18_twiddle_1.p', 'rb') as input:
    centerL, centerJ, centerK, centerRot  = pickle.load(input)
cuda.init()
ctx = cuda.Device(rank).make_context()
S = reconstruction.Reconstructor_GPU(ctx=ctx)
S.set_det_param(centerL, centerJ, centerK, centerRot) # set parameter
S.set_Q(7)
print(S.maxQ)
S.FZFile = 'data/johnson_aug18_demo/CubicFZ.dat'        # fundamental zone file
S.set_sample('iron_bcc')
S.energy = 65.351
S.expDataInitial = 'data/johnson_aug18_demo/SB1_postheat_restart_V1_1degree/SB1_V1_1degree_z0_'     # reduced binary data
S.expdataNDigit = 6                                                       # number of digit in the binary file name

imgsize = [50, 50]
voxelSize = 0.03
shift = [0.0, 0.0, 0.0]
mask = None
lMask = gen_mpi_masks(imgsize, size, mask=mask)
S.create_square_mic(imgsize,
                    voxelsize=voxelSize,
                    shift=shift,
                    mask=lMask[rank]
                   )# resolution of reconstruction and voxel size
S.squareMicOutFile = f'output/mpi_test_part{rank}_Fe_johnson_aug17_' \
                    + f'{"x".join(map(str,imgsize))}_{voxelSize}' \
                    + f'_shift_{"_".join(map(str, shift))}.npy' # output file name
S.searchBatchSize = 6000                                                 # number of orientations search at each iteration, larger number will take longer time.
S.recon_prepare(reverseRot=True)  # at 1ID, left hand rotation needs reverseRot=True
S.serial_recon_multi_stage(enablePostProcess=False)

comm.Barrier()

# if rank > 0:
#     req = comm.isend(S.squareMicData, dest=0, tag=11)
#     req.wait()
# elif rank == 0:
#     for i in range(1, size):
#         req = comm.irecv(source=i, tag=11)
#         data = req.wait()
#         S.squareMicData[lMask[i]] = data[lMask[i]]
# sendbuf = S.squareMicData.astype(np.float32).ravel()

# recvbuf = None
# if rank == 0:
#     recvbuf = np.empty([size, sendbuf.shape[0]],  dtype=np.float32)
# comm.Gather(sendbuf, recvbuf, root=0)
data = S.squareMicData
data = comm.gather(data, root=0)
comm.Barrier()

if rank == 0:
    for i in range(size):
        S.squareMicData[np.repeat(lMask[i][:,:,np.newaxis], S.squareMicData.shape[2], axis=2)] = data[i][np.repeat(lMask[i][:,:,np.newaxis],  S.squareMicData.shape[2], axis=2)] 
    if mask is None:
        mask = np.ones(imgsize)
    S.squareMicData[:,:,7] = mask
    S.load_square_mic(S.squareMicData)
    S.squareMicOutFile = 'output/mpi_test_whole_Fe_johnson_aug17_' \
                        + f'{"x".join(map(str,imgsize))}_{voxelSize}' \
                        + f'_shift_{"_".join(map(str, shift))}.npy' # output file name
    S.serial_recon_multi_stage(enablePostProcess=True)

ctx.pop()
    #388seconds with verbose
