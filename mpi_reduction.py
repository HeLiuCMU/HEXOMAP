import sys
sys.path.insert(0, '/home/heliu/work/dev/HEXOMAP/')
import reduction
import numpy as np
from mpi4py import MPI
import atexit
import matplotlib.pyplot as plt
import time
atexit.register(MPI.Finalize)
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
from reduction import segmentation_numba
# reduce a layer of images together:
import IntBin
import time
import mpi_log



################################ Input Session #######################################
startIdx = 0
NRot = 900
NDet = 5
NLayer = 1
idxLayer = [2]
lIdxImg  = [np.arange(x*1800,x*1800+NRot) for x in range(5)]
lIdxImg = np.hstack(lIdxImg).astype(np.int32)
identifier = 'Au_volume2_NSUM2_bsf_fullrotation'
extention = '.tif'
initial = f'/home/heliu/work/AuCal/Raw/Integrated-fullRotation-z2-quarterDegree/Au_volume2_NSUM2_bsf_fullrotation_z2_'
digitLength = 6
bkgInitial = f'{identifier}_bkg'
binInitial = f'/home/heliu/work/AuCal/reduced_new/{identifier}_'
generateBkg = False
generateBin = True
baseline = 10
minNPixel = 4
logFileName = f'{identifier}_reduction.log'
####################################################################################

logfile = mpi_log.MPILogFile(
    comm, logFileName, 
    MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_APPEND
)
logfile.write("rank: {rank} : hello\n")

NTotal = NLayer * NDet * NRot
if lIdxImg is None:
    lIdxImg = np.arange(NTotal) + startIdx
lIdxLayer = np.arange(NLayer).repeat(NDet * NRot)
lIdxDet = np.tile(np.arange(NDet).repeat(NRot), NLayer)
lIdxRot = np.tile(np.arange(NRot), NDet * NLayer)
lBkgIdx = np.arange(NLayer * NDet).repeat(NRot)
NPerCore = int(np.ceil(float(NTotal)/size))

lIdxImg = lIdxImg[rank*NPerCore:(rank+1)*NPerCore]
lIdxLayer = lIdxLayer[rank*NPerCore:(rank+1)*NPerCore]
lIdxDet = lIdxDet[rank*NPerCore:(rank+1)*NPerCore]
lIdxRot = lIdxRot[rank*NPerCore:(rank+1)*NPerCore]
lBkgIdx = lBkgIdx[rank*NPerCore:(rank+1)*NPerCore]
#print(lIdxImg)
# generate background:
if rank==0:
    logfile.write('start generating bkg \n')
    start =  time.time()
if generateBkg:
    if rank==0:
        lBkg = reduction.median_background(initial, startIdx, bkgInitial,NRot=NRot, NDet=NDet, NLayer=NLayer,end=extention)
    else:
        lBkg = None
    lBkg = comm.bcast(lBkg, root=0)
    
else:
    lBkg = []
    for layer in range(NLayer):
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
        logfile.write(f"rank: {rank} : {fName}\n")
        img = plt.imread(fName)
        binFileName = f'{binInitial}z{idxLayer[lIdxLayer[i]]}_{str(lIdxRot[i]).zfill(digitLength)}.bin{lIdxDet[i]}'
        snp = segmentation_numba(img, bkg, baseline=baseline, minNPixel=minNPixel)
        IntBin.WritePeakBinaryFile(snp, binFileName) 
    comm.Barrier()

if rank==0:
    end = time.time()
    logfile.write(f'time taken generating binary: {end - start} seconds')
logfile.close() 
