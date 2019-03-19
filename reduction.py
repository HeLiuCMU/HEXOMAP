import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import glob
import time
from numba import jit
def median_background(initial,startIdx,outInitial, NRot=720, NDet=2,NLayer=1,layerIdx=[0],end='.tif', imgshape=[2048,2048]):
    '''
    take median over omega as background
    initial: finle name initial. e.g.:'/home/heliu/work/shahani_feb19_part/nf_part/dummy_2_rt_before_heat_nf/dummy_2_rt_before_heat_nf_'
    startIdx: idex of first imagej.
    outInitial: output initial
    NRot: number of omega interval.
    NDet: number of detector.
    NLayer: number of layer.
    layerIdx: the index used for layers
    end: file end format.
    imgshape: image resolution
    '''
    lBkg = []
    imgStack = np.empty([imgshape[0], imgshape[1], NRot],dtype=np.int32)
    start = time.time()
    for layer in range(NLayer):
        print(f'layer: {layer}')
        for det in range(NDet):
            print(f'det: {det}')
            for rot in range(NRot):
                print(f'rot: {rot}')
                idx = layer * NDet * NRot + det * NRot + rot + startIdx
                fName = f'{initial}{idx:06d}{end}'
                print(fName)
                imgStack[:,:,rot] = plt.imread(fName)
                print('img loaded')
                #sys.stdout.write(f'\r {rot}')
                #sys.stdout.flush()
            bkg = np.median(imgStack, axis=2)
            np.save(f'{outInitial}_z{layer}_det_{det}.npy', bkg)
            lBkg.append(bkg)
    end = time.time()
    print('\r')
    print(end - start)
    return lBkg

@jit(nopython=True,parallel=True)
def extract_peak(label,N, imgSubMed,imgSub, minNPixel, baseline):
    '''
    0.0079seconds
    '''
    lXOut = []
    lYOut = []
    lIDOut = []
    lIntensityOut = []
    lXTmp = []
    lYTmp = []
    lIDTmp = []
    lSubMedTmp = []
    lSubTmp = []
    lIdxStart = []
    visited = label==0
    lXOut.append(1)
    lYOut.append(1)
    lIDOut.append(1)
    lIntensityOut.append(1)
    NX = label.shape[0]
    NY = label.shape[1]
    idx = 0
    for x in range(label.shape[0]):
        for y in range(label.shape[1]):
            if not visited[x,y]:
                lXTmp.append(x)
                lYTmp.append(y)
                lIDTmp.append(label[x,y])
                lSubMedTmp.append(imgSubMed[x,y])
                lSubTmp.append(imgSub[x,y])
                queX = [x]
                queY = [y]
                lIdxStart.append(idx)
                idx += 1
                while queX:
                    for dx in [-1,0,1]:
                        for dy in [-1,0,1]:
                            newX = queX[0]+dx
                            newY = queY[0]+dy
                            if newX>=0 and newX<NX and newY>=0 and newY<NY:
                                if not visited[newX, newY] and label[newX, newY]==label[queX[0],queY[0]]:
                                    queX.append(newX)
                                    queY.append(newY)
                                    lXTmp.append(newX)
                                    lYTmp.append(newY)
                                    lIDTmp.append(label[newX, newY])
                                    lSubMedTmp.append(imgSubMed[newX,newY])
                                    lSubTmp.append(imgSub[newX,newY])
                                    visited[newX, newY] = 1
                                    idx +=1
                    queX.pop(0)
                    queY.pop(0)
                lIdxStart.append(idx)
    #print(len(lIdxStart), N)
    
    for i in range(N):
        start = lIdxStart[2*i]
        end = lIdxStart[2*i + 1]
        vMax = np.max(np.array(lSubMedTmp[start:end]))
        if vMax > baseline:
            lXX = []
            lYY = []
            lVV = []
            lIDTmp = []
            for j in range(start, end):
                if lSubTmp[j]>(max(vMax*0.1,1)):
                    lXX.append(lXTmp[j])
                    lYY.append(lYTmp[j])
                    lVV.append(lSubTmp[j])
                    lIDTmp.append(i)
            if len(lXX)>minNPixel:
                lXOut.extend(lXX)
                lYOut.extend(lYY)
                lIntensityOut.extend(lVV)
                lIDOut.extend(lIDTmp)
    return lXOut, lYOut, lIDOut, lIntensityOut

def segmentation_numba(img, bkg, baseline=10, minNPixel=4):
    '''
    
    '''
    #start = time.time()
    imgSub = img- bkg
    imgSubMed = ndi.median_filter(imgSub,size=2)
    imgBase = imgSubMed - baseline
    imgBase[imgBase<0] = 0
    imgBaseMedian = ndi.median_filter(imgBase, size=2)
    log = ndi.gaussian_laplace(imgBaseMedian,sigma=1.5)
    label,N = ndi.label(log<0)
    label = ndi.grey_dilation(label,size=(3,3))
    #start = time.time()
    lX, lY, lID, lIntensity = extract_peak(label,N, imgSubMed,imgSub, minNPixel, baseline)
    #end = time.time()
    #print(f'time taken:{end- start}')
    return (img.shape[1]- 1 - np.array(lY)).astype(np.int32), np.array(lX).astype(np.int32), np.array(lID).astype(np.int32), np.array(lIntensity).astype(np.int32)


def segmentation(img, bkg, baseline=10, minNPixel=4):
    '''
    
    '''
    start = time.time()
    imgSub = img- bkg
    imgSubMed = ndi.median_filter(imgSub,size=2)
    imgBase = imgSubMed - baseline
    imgBase[imgBase<0] = 0
    imgBaseMedian = ndi.median_filter(imgBase, size=2)
    log = ndi.gaussian_laplace(imgBaseMedian,sigma=1.5)
    label,N = ndi.label(log<0)
    label = ndi.grey_dilation(label,size=(3,3))
    lX = []
    lY = []
    lID = []
    lIntensity = []
    #start = time.time()
# fill hole??? probably not a good idea in some cases.
    for i in range(N):
        mask = (label==i)
        # fill hole??? probably not a good idea in some cases.
        #mask = ndi.binary_fill_holes(mask)
        #mask = ndi.binary_dilation(mask,iterations=2)
        vMax = np.max(imgSubMed[mask].ravel())
        if vMax > baseline:
            x, y = np.where(mask*(imgSub>max(vMax*0.1,1)))
            if x.size>minNPixel:
                for i,xx in enumerate(x):
                    lX.append(xx)
                    yy = y[i]
                    lY.append(yy)
                    lIntensity.append(imgSub[xx,yy])
                    lID.append(label[xx,yy])
    end = time.time()
    print(f'time taken:{end- start}')
    return (img.shape[1]- 1 - np.array(lY)).astype(np.int32), np.array(lX).astype(np.int32), np.array(lID).astype(np.int32), np.array(lIntensity).astype(np.int32)

def reduce_image(initial,startIdx,bkgInitial,binInitial, NRot=720, NDet=2,NLayer=1,idxLayer=[0],end='.tif', imgshape=[2048,2048],
                baseline=10, minNPixel=4):
    '''
    example usage:
        # reduce a layer of images together:
        import IntBin
        import time
        startIdx = 180904
        NRot = 720
        NDet = 1
        NLayer = 1
        end = '.tif'
        initial = '/home/heliu/work/shahani_feb19_part/nf_part/dummy_2_rt_before_heat_nf/dummy_2_rt_before_heat_nf_'
        bkgInitial = 'test_output_bkg'
        binInitial = 'dummy_2_rt_before_heat_nf_test_'
        reduce_image(initial,startIdx,bkgInitial,binInitial, NRot=NRot, NDet=Det,NLayer=1,idxLayer=[0],digitLength=6,end='.tif', imgshape=[2048,2048], baseline=10, minNPixel=4)        
    '''
    lBkg = median_background(initial, startIdx, bkgInitial,NRot=NRot, NDet=NDet, NLayer=NLayer,end=end)
    print('bkg images created')
    idxBkg = 0
    idxLayer = [0]
    for layer in range(NLayer):
        for det in range(NDet):
            bkg = lBkg[idxBkg]
            idxBkg += 1
            for rot in range(NRot):
                idx = layer * NDet * NRot + det * NRot + rot + startIdx
                fName = f'{initial}{str(idx).zfill(digitLength)}{end}'
                img = plt.imread(fName)
                binFileName = f'{binInitial}z{idxLayer[layer]}_{rot.zfill(digitLength)}.bin{det}'
                print(binFileName)
                snp = segmentation(img, bkg, baseline=baseline, minNPixel=minNPixel)
                IntBin.WritePeakBinaryFile(snp, binFileName)
 
if  __name__ == '__main__':
    
    import sys
    sys.path.insert(0, '/home/heliu/work/dev/HEXOMAP/')
    import IntBin
    plt.rcParams["figure.figsize"] = (10,10)

    # images
    startIdx = 180904
    NRot = 720
    NDet = 1
    NLayer = 1
    end = '.tif'
    initial = '/home/heliu/work/shahani_feb19_part/nf_part/dummy_2_rt_before_heat_nf/dummy_2_rt_before_heat_nf_'
    fName = f'{initial}{startIdx:06d}{end}'
    img = plt.imread(fName)
    bkg = np.load('test_output_bkg_z0_det_0.npy')
    lX, lY, lIntensity, lID = segmentation(img, bkg)
    lX, lY, lIntensity, lID = segmentation_numba(img, bkg)
    lX, lY, lIntensity, lID = segmentation_numba(img, bkg)
    lX, lY, lIntensity, lID = segmentation_numba(img, bkg)
    lX, lY, lIntensity, lID = segmentation_numba(img, bkg)