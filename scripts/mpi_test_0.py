import numpy as np
from mpi4py import MPI
import atexit
atexit.register(MPI.Finalize)
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
import pycuda.driver as cuda
if rank==0:
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    # customized module

    import reconstruction  # g-force caller
    import MicFileTool     # io for reconstruction rst
    import IntBin          # io for binary image (reduced data)
    # check result of parameter get from blind search:
    with open('data/johnson_aug18_demo/geometry_au_johnson_aug18_twiddle_1.p', 'rb') as input:
        centerL, centerJ, centerK, centerRot  = pickle.load(input)
    cuda.init()
    ctx = cuda.Device(rank).make_context()
    S = reconstruction.Reconstructor_GPU(ctx=ctx)
    S.set_det_param(centerL, centerJ, centerK, centerRot) # set parameter
    S.set_Q(7)
    print(S.maxQ)
    S.FZFile = 'data/johnson_aug18_demo/CubicFZ.dat'         # fundamental zone file
    S.set_sample('gold')
    S.energy = 65.351
    S.expDataInitial = 'data/johnson_aug18_demo/Au_reduced_1degree/Au_int_1degree_suter_aug18_z0_'  # reduced binary data
    S.expdataNDigit = 6                                                       # number of digit in the binary file name
    imgsize = [100, 200]
    voxelSize = 0.0005
    shift = [-0.025, 0.0, 0.0]
    S.create_square_mic(imgsize,
                        voxelsize=voxelSize,
                        shift=shift,
                       )# resolution of reconstruction and voxel size
    S.squareMicOutFile = 'output/mpi_test_part0_Au_johnson_aug17_' \
                        + f'{"x".join(map(str,imgsize))}_{voxelSize}' \
                        + f'_shift_{"_".join(map(str, shift))}.npy' # output file name
    S.searchBatchSize = 6000                                                 # number of orientations search at each iteration, larger number will take longer time.
    S.recon_prepare(reverseRot=True)  # at 1ID, left hand rotation needs reverseRot=True
    S.serial_recon_multi_stage(enablePostProcess=False)
    ctx.pop()
    
if rank==1:
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    # customized module

    import reconstruction  # g-force caller
    import MicFileTool     # io for reconstruction rst
    import IntBin          # io for binary image (reduced data)
    # check result of parameter get from blind search:
    with open('data/johnson_aug18_demo/geometry_au_johnson_aug18_twiddle_1.p', 'rb') as input:
        centerL, centerJ, centerK, centerRot  = pickle.load(input)
    cuda.init()
    ctx = cuda.Device(rank).make_context()
    S = reconstruction.Reconstructor_GPU(ctx=ctx)
    S.set_det_param(centerL, centerJ, centerK, centerRot) # set parameter
    S.set_Q(7)
    print(S.maxQ)
    S.FZFile = 'data/johnson_aug18_demo/CubicFZ.dat'         # fundamental zone file
    S.set_sample('gold')
    S.energy = 65.351
    S.expDataInitial = 'data/johnson_aug18_demo/Au_reduced_1degree/Au_int_1degree_suter_aug18_z0_'      # reduced binary data
    S.expdataNDigit = 6                                                       # number of digit in the binary file name
    imgsize = [100, 200]
    voxelSize = 0.0005
    shift = [0.025, 0.0, 0.0]
    S.create_square_mic(imgsize,
                        voxelsize=voxelSize,
                        shift=shift,
                       )# resolution of reconstruction and voxel size
    S.squareMicOutFile = 'output/mpi_test_part1_Au_johnson_aug17_' \
                        + f'{"x".join(map(str,imgsize))}_{voxelSize}' \
                        + f'_shift_{"_".join(map(str, shift))}.npy' # output file name
    S.searchBatchSize = 6000                                                 # number of orientations search at each iteration, larger number will take longer time.
    S.recon_prepare(reverseRot=True)  # at 1ID, left hand rotation needs reverseRot=True
    S.serial_recon_multi_stage(enablePostProcess=False)
    ctx.pop()
    #388seconds with verbose
