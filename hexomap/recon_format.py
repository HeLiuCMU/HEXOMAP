'''
transform file format
.npy:
     numpy array, load with np.load()
.ang: 
    EBSD file format, referene: https://www.material.ntnu.no/ebsd/EBSD/OIM%20DC%207.2%20Manual.pdf, page 240.
    # The fields of each line in the body of the file are as follows:
    # j1 F j2 x y IQ CI Phase ID Detector Intensity Fit
    # where:
    # j1,F,j2: Euler angles (in radians) in Bunge's notation for describing the lattice orientations and are given in radians.
    # x,y: The horizontal and vertical coordinates of the points in the scan, in microns. The origin (0,0) is defined as the top-left corner of the scan.
    # IQ: The image quality parameter that characterizes the contrast of the EBSP associated with each measurement point.
    # CI: The confidence index that describes how confident the software is that it has correctly indexed the EBSP, i.e., confidence that the angles are correct.
    # Phase ID: The material phase identifier. This field is 0 for single phase OIM scans or 1,2,3... for multi-phase scans.
    # Detector Intensity: An integer describing the intensity from whichever detector was hooked up to the OIM system at the time of data collection, typically a forward scatter detector.
    # Fit: The fit metric that describes how the indexing solution matches the bands detected by the Hough transform or manually by the user.m
.h5:
    
'''

import numpy as np
from glob import glob
import os
import sys
import tifffile
import h5py
from hexomap.past import *

def h5printR(item, leading = ''):
    for key in item:
        if isinstance(item[key], h5py.Dataset):
            print(leading + key + ': ' + str(item[key].shape))
        else:
            print(leading + key)
            h5printR(item[key], leading + '  ')

# Print structure of a `.h5` file            
def h5print(filename):
    with h5py.File(filename, 'r') as h:
        print(filename)
        h5printR(h, '  ')
        
def npy2h5(lFName, h5Name, material,lLayerIdx, q=11):
    '''
    example usage:
        lFName = [f'dummy_2_single_crystal_furnace_nf_copperBCC_q11_rot720_z{layer}_150x150_0.007_shift_0.0_0.0_0.0.npy' for layer in range(3)]
        print(lFName)
        h5Name = f'dummy_2_single_crystal_furnace_nf_copperBCC_q11_rot720_3layers_150x150_0.007_shift_0.0_0.0_0.0.h5'
        print(h5Name)
        lLayerIdx = np.arange(3)
        npy2h5(lFName, h5Name, 'copperBCC', lLayerIdx,q=11)
        lFName: list of filenames
        lLayerIdx: list of layer index
    '''
    with h5py.File(h5Name,'w') as fout:
        md=fout.create_group('meta_data')
        sls=fout.create_group('slices')
        md.create_dataset('material',data=np.string_(material))
        md.create_dataset('maxQ',data=np.int_(q))
        for i,f in enumerate(lFName):
            print(i,f)
            grp=sls.create_group('z{:d}'.format(lLayerIdx[i]))
            a=np.load(f)
            ds=grp.create_dataset('x',data=a[:,:,0]*1000,dtype='float32')
            ds.attrs['info']='X coordinate (micron meter).'
            ds=grp.create_dataset('y',data=a[:,:,1]*1000,dtype='float32')
            ds.attrs['info']='Y coordinate (micron meter).'
            ds=grp.create_dataset('EulerAngles',data=a[:,:,3:6]*np.pi/180,dtype='float32')
            ds.attrs['info']='active ZXZ Euler angles (radian)'
            ds=grp.create_dataset('phase',data=a[:,:,7],dtype='uint16')
            ds.attrs['info']='material phases'
            ds=grp.create_dataset('Confidence',data=a[:,:,6],dtype='float32')
            ds.attrs['info']='hit ratio of simulated peaks and experimental peaks'
    print('=== saved format:')
    h5print(h5Name)

def npy_2_tiffstack(lNpyFile,stack_initial,minHitRatio=0.6, startIdx=0):
    '''
    npy into tiff format.
    example usage: 
        npy_2_tiffstack(['/home/heliu/work/krause_jul19/recon/s1400poly1/s1400poly1_q9_rot180_z0_500x500_0.002_shift_0.0_0.0_0.0.npy',
            '/home/heliu/work/krause_jul19/recon/s1400poly1/s1400poly1_q9_rot180_z1_500x500_0.002_shift_0.0_0.0_0.0.npy',
            '/home/heliu/work/krause_jul19/recon/s1400poly1/s1400poly1_q9_rot180_z2_500x500_0.002_shift_0.0_0.0_0.0.npy',],
            'tiffstack/test',
            minHitRatio=0.2)
    :params: lNpyFile:
    :params: stack_initials:
    :params: minHitRatio: hitratio threshold to show.
    '''
    outputDirectory = os.path.dirname(stack_initial)
    if not os.path.exists(outputDirectory):
        print(f'creating directory: {outputDirectory}')
        os.makedirs(outputDirectory)
    for npy in lNpyFile:
        if not os.path.exists(npy):
            raise FileNotFoundError(f'file {npy} is not found !!!')
    for ii,npy in enumerate(lNpyFile):
        print(f'processing layer {ii}: {npy}')
        squareMicData = np.load(npy)
        eulers = squareMicData[:,:, 3:6].reshape([-1, 3]) / 180.0 * np.pi
        quats = Quaternion.quaternions_from_eulers(eulers)
        rods = Rodrigues.rodrigues_from_quaternions(quats)
        hitRatioMask = (squareMicData[:,:,6]>minHitRatio)[:,:,np.newaxis].repeat(3,axis=2)
        img = ((rods + np.array([1, 1, 1])) / 2).reshape([squareMicData.shape[0],squareMicData.shape[1],3]) * hitRatioMask * 255
        img = np.swapaxes(img,0,1).astype(np.int8)
        save_path = f'{stack_initial}_{ii+startIdx}.TIFF'
        print(save_path)
        tifffile.imwrite(save_path, img)

def npy_2_ang(lNpyFile = ['/home/heliu/work/krause_jul19/recon/s1400poly1/s1400poly1_q9_rot180_z0_500x500_0.002_shift_0.0_0.0_0.0.npy',
         '/home/heliu/work/krause_jul19/recon/s1400poly1/s1400poly1_q9_rot180_z1_500x500_0.002_shift_0.0_0.0_0.0.npy',
         '/home/heliu/work/krause_jul19/recon/s1400poly1/s1400poly1_q9_rot180_z2_500x500_0.002_shift_0.0_0.0_0.0.npy',],
            stack_initial = 'angstack/test',
            sample='Sample',
            startIdx=0
             ):
    '''
    npy into ang format.
    example usage: 
        lNpyFile = []
        for i in range(50):
            lNpyFile.append(f'/home/heliu/work/krause_jul19/recon/s1400poly1/s1400poly1_q9_rot180_z{i}_500x500_0.002_shift_0.0_0.0_0.0.npy')
        npy_2_ang(lNpyFile,'s1400poly1_ang/s1400poly1','SrTiO3')
    :params: lNpyFile:
    :params: stack_initials:
    :params: sample: sample name, string.
    :params: startIdx: output starting index.
    '''
    # check output folder:
    outputDirectory = os.path.dirname(stack_initial)
    if not os.path.exists(outputDirectory):
        print(f'creating directory: {outputDirectory}')
        os.makedirs(outputDirectory)
    # check input exist:
    for npy in lNpyFile:
        if not os.path.exists(npy):
            raise FileNotFoundError(f'file {npy} is not found !!!')
    mic = np.load(lNpyFile[0])
    
    AngHeader="""
 TEM_PIXperUM          1.000000
 x-star                0.476443
 y-star                0.885339
 z-star                0.659644
 WorkingDistance       10.000000

 Phase 1
 MaterialName      {3:s}
 Formula           Sr
 Info
 Symmetry              43
 LatticeConstants      3.9053 3.9053 3.9053  90.000  90.000  90.000
 NumberFamilies        1
 hklFamilies        1 -1 -1 1 8.469246 1
 Categories16992652 0 16992652 16992536 2009385186

 GRID: SqrGrid
 XSTEP: {2:f}
 YSTEP: {2:f}
 NCOLS_ODD: {1:d}
 NCOLS_EVEN: {1:d}
 NROWS: {0:d}

 OPERATOR: 

 SAMPLEID:

 SCANID:
""".format(mic.shape[0],mic.shape[1], 1000*np.abs(mic[0,0,0]-mic[1,0,0]),sample)

    for ii in range(len(lNpyFile)):
        sys.stdout.write(f'\r processing: {ii}')
        sys.stdout.flush()
        ANG=f'{stack_initial}_z{ii+startIdx}.ang' #output

        a=np.load(lNpyFile[ii])
        b=a.reshape((-1,10),order='C')
        #print(b.shape)
        c=np.empty(b.shape,dtype=np.float32)
        c[:,0:3]=b[:,3:6]*np.pi/180 #Euler ZXZ
        c[:,3:5]=b[:,0:2]*1000 #x,y coordinate in um
        c[:,5]=1000 #image quality
        c[:,6]=b[:,6] #confidence index
        c[:,7]=b[:,7] #phase
        c[:,8]=1 #SEM signal
        c[:,9]=10 #fit

        np.savetxt(ANG,c,fmt=['%.6f']*3+['%.3f']*2+['%d']+['%.3f']+['%d']*3,delimiter='\t',header=AngHeader)



if __name__ == '__main__':
    lNpyFile = []
    for i in range(3):
        lNpyFile.append(f'/home/heliu/work/krause_jul19/recon/s1400poly1/s1400poly1_q9_rot180_z{i}_500x500_0.002_shift_0.0_0.0_0.0.npy')
    npy_2_ang(lNpyFile,'s1400poly1_ang/s1400poly1','SrTiO3')
    h5Name = f'test_save.h5'
    lLayerIdx = np.arange(len(lNpyFile))
    npy2h5(lNpyFile, h5Name, 'SrTiO3', lLayerIdx,q=11)
    stack_initial = 'test_tiffstack/test_stack_'
    npy_2_tiffstack(lNpyFile,stack_initial,minHitRatio=0.001)

