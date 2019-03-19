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

################################ Input Session #######################################
startIdx = 180904
NRot = 720
NDet = 2
NLayer = 1
idxLayer = [0]
end = '.tif'
initial = '/home/heliu/work/shahani_feb19_part/nf_part/dummy_2_rt_before_heat_nf/dummy_2_rt_before_heat_nf_'
digitLength = 6
bkgInitial = 'test_output_bkg'
binInitial = '/home/heliu/work/dev/reduction/mpi_output_test/dummy_2_rt_before_heat_nf_test'
generateBkg = True
generateBin = False
baseline = 10
minNPixel = 4
####################################################################################

NTotal = NLayer * NDet * NRot
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
# generate background:
if rank==0:
    print('start generating bkg')
    start =  time.time()
if generateBkg:
    if rank==0:
        lBkg = reduction.median_background(initial, startIdx, bkgInitial,NRot=NRot, NDet=NDet, NLayer=NLayer,end=end)
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
    print('end generating bkg')
    end = time.time()
    print(f'time take generating bkg: {end-start}')
if generateBin:
	if rank==0:
	    start = time.time()
	    
	for i in range(NPerCore):
	    bkg = lBkg[lBkgIdx[i]]
	    fName = f'{initial}{str(lIdxImg[i]).zfill(digitLength)}{end}'
	    img = plt.imread(fName)
	    binFileName = f'{binInitial}z{idxLayer[lIdxLayer[i]]}_{str(lIdxRot[i]).zfill(digitLength)}.bin{lIdxDet[i]}'
	    #print(binFileName)
	    snp = segmentation_numba(img, bkg, baseline=baseline, minNPixel=minNPixel)
	    IntBin.WritePeakBinaryFile(snp, binFileName) 
	comm.Barrier()

	if rank==0:
	    end = time.time()
	    print(f'time taken generating binary: {end - start} seconds')
