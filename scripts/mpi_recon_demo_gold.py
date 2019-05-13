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
import sys
sys.path.insert(0, '..')
from hexomap import reconstruction, MicFileTool, IntBin, config   
################# configuration #########################
Au_Config={
    'micsize' : np.array([20, 20]),
    'micVoxelSize' : 0.01,
    'micShift' : np.array([0.0, 0.0, 0.0]),
    'micMask' : None,
    'expdataNDigit' : 6,
    'energy' : 65.351,      #55.587 # in kev
    'sample' : 'gold',
    'maxQ' : 9,
    'etalimit' : 81 / 180.0 * np.pi,
    'NRot' : 180,
    'NDet' : 2,
    'searchBatchSize' : 6000,
    'reverseRot' : True,          # for aero, is True, for rams: False
    'detL' : np.array([[4.53571404, 6.53571404]]),
    'detJ' : np.array([[1010.79405782, 1027.43844558]]),
    'detK' : np.array([[2015.95118521, 2014.30163539]]),
    'detRot' : np.array([[[89.48560133, 89.53313565, -0.50680978],
  [89.42516322, 89.22570012, -0.45511278]]]),
    'fileBin' : '../examples/johnson_aug18_demo/Au_reduced_1degree/Au_int_1degree_suter_aug18_z',
    'fileBinDigit' : 6,
    'fileBinDetIdx' : np.array([0, 1]),
    'fileBinLayerIdx' : 0,
    '_initialString' : 'demo_gold_'}
    
c = config.Config(**Au_Config)
initialString = c._initialString
############################### reconstruction #################################################
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
            lMask.append(new_mask.astype(np.bool_))
    return lMask

cuda.init()
ctx = cuda.Device(rank).make_context()
S = reconstruction.Reconstructor_GPU(ctx=ctx)

mask = None  # overall mask
lMask = gen_mpi_masks(c.micsize, size, mask=mask)
c.micMask = lMask[rank]
c._initialString = f'part_demo_gold{rank}'
S.load_config(c)
S.serial_recon_multi_stage(enablePostProcess=False)
comm.Barrier()
data = S.squareMicData
data = comm.gather(data, root=0)
comm.Barrier()
################################ post process #########################################################
if rank == 0:
    mic = np.zeros(S.squareMicData.shape)
    for i in range(size):
        mic[np.repeat(lMask[i][:,:,np.newaxis], mic.shape[2], axis=2)] = data[i][np.repeat(lMask[i][:,:,np.newaxis],  mic.shape[2], axis=2)] 
    c._initialString = initialString
    c.micMask = mask
    S.load_config(c, reloadData=False)
    S.load_square_mic(mic)
    S.voxelIdxStage0 = []
    S.serial_recon_multi_stage(enablePostProcess=True)
ctx.pop()
