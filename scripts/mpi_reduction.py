import sys
sys.path.insert(0, '/home/heliu/work/dev/v0.2/HEXOMAP')
from hexomap import reduction
import numpy as np
from mpi4py import MPI
import atexit
import matplotlib.pyplot as plt
import time
from hexomap.reduction import segmentation_numba
from hexomap import IntBin
import time
from hexomap import mpi_log
import os

atexit.register(MPI.Finalize)
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

############################# Example useage: #######################################
'''
mpirun -n 1 python mpi_reduction_Au2_NF_LineFocus.py
To monitor progress
    tail -f output_path/*.log
'''
################################ Input Session #######################################
startIdx = 0
NRot = 360
NDet = 5
NLayer = 1
idxLayer = [3] # binary name is f'{binInitial}z{idxLayer[lIdxLayer[i]]}_{str(lIdxRot[i]).zfill(digitLength)}.bin{lIdxDet[i]}'
aIdxImg = None # must be 3d array [i][j][k] is the ith layer, jth detector, kth rotation
extention = '.tif'
initial = f'/home/heliu/work/Au_calibrate/Raw/Integrated-fullRotation/Au_volume2_NSUM10_bsf_fullrotation_'
digitLength = 6
outputDirectory = '/home/heliu/work/Au_calibrate/Reduction/reduced_z3_new/'
identifier = 'Au_calibrate'
generateBkg = True
generateBin = True
baseline = 10
minNPixel = 4
####################################################################################
bkgInitial = os.path.join(outputDirectory, f'{identifier}_bkg')
binInitial = os.path.join(outputDirectory, f'{identifier}_bin')
logFileName = os.path.join(outputDirectory, f'{identifier}_reduction.log')

if rank==0:
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)
comm.Barrier()

logfile = mpi_log.MPILogFile(
    comm, logFileName, 
    MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_APPEND
)
logfile.write(f"rank: {rank} : hello\n")
for layer in range(NLayer):
    NTotal = NDet * NRot
    if aIdxImg is None:
        lIdxImg = np.arange(NTotal) + startIdx
    else:
        assert aIdxImg.shape == (NLayer, NDet, NRot)
        lIdxImg = aIdxImg[layer,:,:].ravel()
    
    lIdxLayer = np.array([layer]).repeat(NDet * NRot)
    lIdxDet = np.arange(NDet).repeat(NRot)
    lIdxRot = np.tile(np.arange(NRot), NDet)
    lBkgIdx = np.arange(NDet).repeat(NRot)
    NPerCore = int(np.ceil(float(NTotal)/size))
    lIdxImg = lIdxImg[rank::size]
    lIdxLayer = lIdxLayer[rank::size]
    lIdxDet = lIdxDet[rank::size]
    lIdxRot = lIdxRot[rank::size]
    lBkgIdx = lBkgIdx[rank::size]
    print(lIdxLayer, lIdxDet, lIdxRot, lBkgIdx,lIdxImg)
    # generate background:
    if rank==0:
        logfile.write('start generating bkg \n')
        start =  time.time()
    if generateBkg:
        if rank==0:
            lBkg = reduction.median_background(initial, startIdx, bkgInitial,NRot=NRot, NDet=NDet, NLayer=1,layerIdx=[layer], end=extention,logfile=logfile)
        else:
            lBkg = None
        lBkg = comm.bcast(lBkg, root=0)

    else:
        lBkg = []
        for det in range(NDet):
            lBkg.append(np.load(f'{bkgInitial}_z{layer}_det_{det}.npy'))
    comm.Barrier()
    if rank==0:
        logfile.write('end generating bkg \n')
        end = time.time()
        logfile.write(f'time take generating bkg: {end-start} \n')
        start = time.time()
    if generateBin:
        for i in range(NPerCore):
            bkg = lBkg[lBkgIdx[i]]
            fName = f'{initial}{str(lIdxImg[i]).zfill(digitLength)}{extention}'
            logfile.write(f"generate binary: rank: {rank} : layer: {layer}, det: {lIdxDet[i]}, rot: {lIdxRot[i]}, {os.path.basename(fName)}\n")
            img = plt.imread(fName)
            binFileName = f'{binInitial}z{idxLayer[lIdxLayer[i]]}_{str(lIdxRot[i]).zfill(digitLength)}.bin{lIdxDet[i]}'
            snp = segmentation_numba(img, bkg, baseline=baseline, minNPixel=minNPixel)
            IntBin.WritePeakBinaryFile(snp, binFileName) 
    comm.Barrier()

    if rank==0:
        end = time.time()
        logfile.write(f'time taken generating binary: {end - start} seconds \n')
    startIdx += NTotal
logfile.close() 
    
