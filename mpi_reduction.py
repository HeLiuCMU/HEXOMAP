import sys
sys.path.insert(0, '/home/heliu/work/dev/HEXOMAP/')
import reduction
import numpy as np
from mpi4py import MPI
import atexit
import matplotlib.pyplot as plt
import time
from reduction import segmentation_numba
import IntBin
import time
import mpi_log
import os

atexit.register(MPI.Finalize)
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

############################# EXample useage: #######################################
'''
mpirun -n 1 python mpi_reduction_Au2_NF_LineFocus.py
'''
################################ Input Session #######################################
startIdx = 198218
NRot = 720
NDet = 2
NLayer = 26
idxLayer = np.arange(26)
lIdxImg = None
#lIdxImg  = [np.arange(x*1800,x*1800+NRot) for x in range(5)]
#lIdxImg = np.hstack(lIdxImg).astype(np.int32)
extention = '.tif'
initial = f'/media/heliu/feb2019/shahani_feb19/nf/dummy_2_rt_815_2_nf/dummy_2_rt_815_2_nf_'
digitLength = 6
outputDirectory = '/home/heliu/work/shahani_feb19/reduction/dummy_2_rt_815_2_nf/'
identifier = 'dummy_2_rt_815_2_nf'
bkgInitial = os.path.join(outputDirectory, f'{identifier}_bkg')
binInitial = os.path.join(outputDirectory, f'{identifier}_bin')
logFileName = os.path.join(outputDirectory, f'{identifier}_reduction.log')
generateBkg = False
generateBin = True
baseline = 10
minNPixel = 4
####################################################################################
if rank==0:
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)
comm.Barrier()

logfile = mpi_log.MPILogFile(
    comm, logFileName, 
    MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_APPEND
)
logfile.write(f"rank: {rank} : hello\n")

NTotal = NLayer * NDet * NRot
if lIdxImg is None:
    lIdxImg = np.arange(NTotal) + startIdx
lIdxLayer = np.arange(NLayer).repeat(NDet * NRot)
lIdxDet = np.tile(np.arange(NDet).repeat(NRot), NLayer)
lIdxRot = np.tile(np.arange(NRot), NDet * NLayer)
lBkgIdx = np.arange(NLayer * NDet).repeat(NRot)
NPerCore = int(np.ceil(float(NTotal)/size))


lIdxImg = lIdxImg[rank::size]
lIdxLayer = lIdxLayer[rank::size]
lIdxDet = lIdxDet[rank::size]
lIdxRot = lIdxRot[rank::size]
lBkgIdx = lBkgIdx[rank::size]
# lIdxImg = lIdxImg[rank*NPerCore:(rank+1)*NPerCore]
# lIdxLayer = lIdxLayer[rank*NPerCore:(rank+1)*NPerCore]
# lIdxDet = lIdxDet[rank*NPerCore:(rank+1)*NPerCore]
# lIdxRot = lIdxRot[rank*NPerCore:(rank+1)*NPerCore]
# lBkgIdx = lBkgIdx[rank*NPerCore:(rank+1)*NPerCore]
#print(lIdxImg)
# generate background:
if rank==0:
    logfile.write('start generating bkg \n')
    start =  time.time()
if generateBkg:
    if rank==0:
        lBkg = reduction.median_background(initial, startIdx, bkgInitial,NRot=NRot, NDet=NDet, NLayer=NLayer,end=extention,logfile=logfile)
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
    logfile.write(f'time taken generating binary: {end - start} seconds \n')
logfile.close() 
