import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import glob
import time
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

def segmentation(img, bkg, baseline=10, minNPixel=4):
    '''
    
    '''
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
    start = time.time()
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