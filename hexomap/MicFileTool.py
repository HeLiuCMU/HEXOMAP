'''
Writen by He Liu
Wed Apr 26 2017
This script will contains the basic tool for reading mic file and plot them.
'''
import numpy as np
import matplotlib
#matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.collections import PolyCollection
from hexomap.past import *
from hexomap import IntBin
import sys
from hexomap.orientation import Quaternion
from hexomap.orientation import Eulers
from hexomap.orientation import Rodrigues
import os
#import bokeh

def dist_to_line(point,line):
    '''

    :param point: array,size=[1,2]
    :param line: array, size = [2,2]
    :return:
    '''
    point = np.array(point)
    line = np.array(line)
    r_1 = point-line[0,:]
    r_2 = line[0,:]-line[1,:]
    #print r_1,r_2
    dist = np.linalg.norm(r_1)*np.sqrt(1-(np.abs(np.sum(r_1*r_2))/(np.linalg.norm(r_1)*np.linalg.norm(r_2)))**2)
    return dist

def select_line_mic(snp):
    '''
    select mic along a line
    :param snp:
    :return:
    '''
    line = np.array([[0,0.24],[0.22,0.13]])
    d = 0.02
    N = snp.shape[0]
    bool_lst = [False]*N
    for i in range(N):
        dist = dist_to_line(snp[i,0:2],line)
        if dist < d:
            bool_lst[i] = True
    new_snp = snp[bool_lst,:]
    plt.plot(line[:,0],line[:,1])
    return new_snp

def save_mic_file(fname,snp,sw):
    '''
    save to mic file
    :param fname:
    :param snp:
    :param sw:
    :return:
    '''
    # target = open(fname, 'w')
    # target.write(str(sw))
    # target.write('\n')
    # target.close()
    np.savetxt(fname,snp,delimiter=' ',fmt='%f',header=str(sw),comments='')


def read_mic_file(fname):
    '''
    this will read the mic file
      %%
      %% Legacy File Format:
      %% Col 0-2 x, y, z
      %% Col 3   1 = triangle pointing up, 2 = triangle pointing down
      %% Col 4 generation number; triangle size = sidewidth /(2^generation number )
      %% Col 5 Phase - 1 = exist, 0 = not fitted
      %% Col 6-8 orientation
      %% Col 9  Confidence
      %%
    :param fname:
    :return:
        sw: float, the side width
        snp: [n_voxel,n_feature] numpy array

    '''
    with open(fname) as f:
        content = f.readlines()
    print(content[1])
    print(type(content[1]))
    sw = float(content[0])
    try:
        snp = np.array([[float(i) for i in s.split(' ')] for s in content[1:]])
    except ValueError:
        try:
            snp = np.array([[float(i) for i in s.split('\t')] for s in content[1:]])
        except ValueError:
            print('unknown deliminater')

    print('sw is {0} \n'.format(sw))
    print('shape of snp is {0}'.format(snp.shape))
    return sw,snp

    # snp = pd.read_csv(filename,delim_whitespace=True,skiprows=0,header=0).values
    # sw =  pd.read_csv(filename,delim_whitespace=True,nrows=1,header=0).values
    # print snp
    # print(snp.shape)
    # print sw


def plot_mic(snp,sw,plotType,minConfidence,scattersize=2):
    '''
    plot the mic file
    :param snp:
    :param sw:
    :param plotType:
    :param minConfidence:
    :return:
    '''
    snp = snp[snp[:,9]>=minConfidence,:]
    N = snp.shape[0]
    mat = np.empty([N,3,3])
    quat = np.empty([N,4])
    rod = np.empty([N,3])
    if plotType==2:
        fig, ax = plt.subplots()
        sc = ax.scatter(snp[:,0],snp[:,1],c=snp[:,9],cmap='cool')
        plt.colorbar(sc)
        plt.show()
    if plotType==3:
        print('h')
        for i in range(N):
            mat[i,:,:] = EulerZXZ2Mat(snp[i,6:9]/180*np.pi)
            #print mat[i,:,:]
            quat[i,:] = quaternion_from_matrix(mat[i,:,:])
            #print quat[i,:]
            rod[i,:] = rod_from_quaternion(quat[i,:])
        print(rod)
        fig, ax = plt.subplots()
        ax.scatter(snp[:,0],snp[:,1],s=scattersize,facecolors=(rod+np.array([1,1,1]))/2)
        ax.axis('scaled')
        plt.show()


def segment_grain(mic, symType='Hexagonal', threshold=0.01,show=True, save=True,outFile='default_segment_grain.npy',mask=None):
    '''
    m0: symmetry matrix.
    :return: image of grain ID
    '''
    #self.recon_prepare()

    # timing tools:
    NX = mic.shape[0]
    NY = mic.shape[1]
    m0 = EulerZXZ2MatVectorized(mic[:,:,3:6].reshape([-1,3])/180.0*np.pi)

    visited = np.zeros(NX * NY)
    result = np.empty(NX * NY)
    #m0 = m0.reshape([-1,9])
    id = 0
    if mask is None:
        mask = np.ones(NX*NY)
    else:
        mask = mask.ravel()
    visited[mask==0] = 1
    while(np.sum(visited.ravel()==0)>0):
        idxs,  = np.where(visited.ravel()==0)
        startIdx = idxs[0]
        q = [startIdx]
        visited[startIdx] = 1
        result[startIdx] = id
        while q:
            n = q.pop(0)
            for x in [min(n + 1, n-n%NY + NY-1), max(n-n%NY, n - 1), min(n//NY+1, NX-1)*NY + n%NY, max(0,n//NY-1)*NY + n%NY]:
                _, misorientation = Misorien2FZ1(m0[n,:].reshape([3,3]), m0[x,:].reshape([3,3]), symType)
                #print(misorientation)
                if misorientation<threshold and visited[x] == 0:
                    q.append(x)
                    visited[x] = 1
                    result[x] = id
        id +=1
        #print(id)
        sys.stdout.write(f'\r {id}')
        sys.stdout.flush()
    result = result.reshape([NX,NY])
    if save:
        np.save(outFile, result)
    if show:
        plt.imshow(result.T, origin='lower')
        plt.show()
    return result
def grain_boundary(mic, symType):
    pass
        
def misorien_between(mic0, mic1, symType, angleRange=None,colorbar=True, saveName=None,outUnit='degree', mask=None):
    '''
    misorientation between two mics
    '''
    if mic0.shape!=mic1.shape:
        raise ValueError(f'shape does not match : {mic0.shape}, {mic1.shape}')
    NX = mic0.shape[0]
    NY = mic0.shape[1]
    if mask is None:
        mask = np.ones([NX, NY])
    eulers0 = mic0[:, :, 3:6]
    mats0 = EulerZXZ2MatVectorized(eulers0.reshape([-1, 3]) / 180.0 * np.pi).reshape([NX, NY, 3, 3])
    confs0 = mic0[:, :, 6]
    eulers1 = mic1[:, :, 3:6]
    mats1 = EulerZXZ2MatVectorized(eulers1.reshape([-1, 3]) / 180.0 * np.pi).reshape([NX, NY, 3, 3])
    confs1 = mic1[:, :, 6]
    misorien = np.empty([NX, NY])
    for x in range(NX):
        for y in range(NY):
            _, misOrien = Misorien2FZ1(mats0[x, y, :, :], mats1[x, y, :, :], symtype=symType)
            if outUnit=='degree':
                misorien[x,y] = misOrien/np.pi*180.0
            elif outUnit=='radian':
                misorien[x,y] = misOrien
            else:
                raise ValueError('must be radian or degree')
    misorien[mask==0] = 0 
    plt.imshow(misorien.T, origin='lower',vmin=0, vmax=angleRange)
    plt.title(f'miorientation map in {outUnit}')
    if colorbar:
        plt.colorbar()
    if saveName is not None:
        plt.savefig(saveName)
    plt.show()
    return misorien
def plot_misorien_square_mic(squareMicData, eulerIn,symType, angleRange=None,colorbar=True, saveName=None,outUnit='degree', mask=None, ax=None):
    '''
    plot the misorientation compared to certain euler angle.
    Input:
        squareMicData:
        eulerIn: input euler angle, in degree
        symType:'Cubic' or 'Hexagonal'
        angleRAnge: colorbar range
        colorbar:
        saveName:
        outUnit:'degree' or 'raidan', the unit of output misorientation map
    Output:
        misorientation map,shape=[squareMicData.shape[0], squareMicData.shape[1]]
    '''
    NX = squareMicData.shape[0]
    NY = squareMicData.shape[1]
    if mask is None:
        mask = np.ones([NX, NY])
    eulers = squareMicData[:, :, 3:6]
    mats = EulerZXZ2MatVectorized(eulers.reshape([-1, 3]) / 180.0 * np.pi).reshape([NX, NY, 3, 3])
    confs = squareMicData[:, :, 6]
    mat0 = EulerZXZ2Mat(eulerIn/ 180.0 * np.pi)
    misorien = np.empty([NX, NY])
    for x in range(NX):
        for y in range(NY):
            _, misOrien = Misorien2FZ1(mat0, mats[x, y, :, :], symtype=symType)
            if outUnit=='degree':
                misorien[x,y] = misOrien/np.pi*180.0
            elif outUnit=='radian':
                misorien[x,y] = misOrien
            else:
                raise ValueError('must be radian or degree')
    plt.imshow(misorien.T, origin='lower',vmin=0, vmax=angleRange)
    plt.title(f'miorientation map in {outUnit}')
    if colorbar:
        plt.colorbar()
    if saveName is not None:
        plt.savefig(saveName)
    plt.show()
    return misorien
    
def plot_conf_square_mic(squareMicData, colorbar=True,saveName=None):
    
    plt.imshow(squareMicData[:,:,6].T,origin='lower',extent=[squareMicData[0,0,0],squareMicData[-1,0,0],squareMicData[0,0,1],squareMicData[0,-1,1]])
    if colorbar:
        plt.colorbar()
    if saveName is not None:
        plt.savefig(saveName)
    plt.show()
    
def plot_square_mic_bokeh(squareMicData,minHitRatio,saveName=None):
    '''
    not implemented
    plot the square mic data
    image already inverted, x-horizontal, y-vertical, x dow to up, y: left to right
    :param squareMicData: [NVoxelX,NVoxelY,10], each Voxel conatains 10 columns:
            0-2: voxelpos [x,y,z]
            3-5: euler angle
            6: hitratio
            7: maskvalue. 0: no need for recon, 1: active recon region
            8: voxelsize
            9: additional information
    :return:
    '''
    mat = EulerZXZ2MatVectorized(squareMicData[:,:,3:6].reshape([-1,3])/180.0 *np.pi )
    quat = np.empty([mat.shape[0],4])
    rod = np.empty([mat.shape[0],3])
    for i in range(mat.shape[0]):
        quat[i, :] = quaternion_from_matrix(mat[i, :, :])
        rod[i, :] = rod_from_quaternion(quat[i, :])
    hitRatioMask = (squareMicData[:,:,6]>minHitRatio)[:,:,np.newaxis].repeat(3,axis=2)
    img = ((rod + np.array([1, 1, 1])) / 2).reshape([squareMicData.shape[0],squareMicData.shape[1],3]) * hitRatioMask
    # make sure display correctly
    #img[:,:,:] = img[::-1,:,:]
    img = np.swapaxes(img,0,1)
    
def plot_binary(rawInitial, NRot=180, NDet=2, idxRot=0,idxLayer=0):
    '''
    visualize binary files, first column is single frame, second column is integrated frames
    '''
    figure, ax = plt.subplots(2, NDet)
    figure.set_size_inches(10, 10)
    idxRotSingleFrame = idxRot
    for idxDet in range(NDet):
        # single frame
        #idxRot = 0  # index of rotation (0~719)
        #idxLayer = 0
        b=IntBin.ReadI9BinaryFiles(f'{rawInitial}{idxLayer}_{0:06d}.bin{idxDet}'.format(int(idxRotSingleFrame),idxDet))
        ax[0,idxDet].plot(2047-b[0],2047-b[1],'b.')
        ax[0,idxDet].axis('scaled')
        ax[0,idxDet].set_xlim((0,2048))
        ax[0,idxDet].set_ylim((0,2048))
        ax[0,idxDet].set_title(f'single frame layer:{idxLayer}, det:{idxDet}, rot:{idxRotSingleFrame}')

        # integrated frame:
        lX = []
        lY = []
        for idxRot in range(NRot):
            #print(b)
            b = IntBin.ReadI9BinaryFiles(f'{rawInitial}{idxLayer}_{idxRot:06d}.bin{idxDet}')
            lX.append(b[0])
            lY.append(b[1])
        aX = np.concatenate(lX)
        aY = np.concatenate(lY)
        print(aX)
        print(aY)
        ax[1,idxDet].plot(2047-aX,2047-aY,'b.')  
        ax[1,idxDet].axis('scaled')
        ax[1,idxDet].set_xlim((0,2048))
        ax[1,idxDet].set_ylim((0,2048))
        ax[1,idxDet].set_title(f'integrated frame layer:{idxLayer}, det:{idxDet}')

    plt.show()

def plot_binary_with_tiff(fBin, img,alpha=0.5):
    '''
    plot binary file together with raw image.
    example usage:
        # overlay binary and tiff
        import tifffile
        import matplotlib.pyplot as plt
        import matplotlib
        %matplotlib notebook
        from hexomap import MicFileTool
        import scipy.ndimage as ndi
        import os
        plt.rcParams["figure.figsize"] = (10,10)

        layer = 0
        rot = 0
        det = 0
        idx = layer * 360 + det * 180 + rot

        tiff = tifffile.imread(f'/media/heliu/Seagate Backup Plus Drive/krause_jul19/nf/s1350_110_1_nf/s1350_110_1_nf_int4_{idx:06d}.tif')
        bkg = tifffile.imread(f'/home/heliu/work/krause_jul19/s1350_110_1/Reduced/s1350_110_1_from_int_tiff_baseline5_bkg_z{layer}_det_{det}.tiff')

        sub = tiff-bkg
        sub = ndi.median_filter(sub, size=3)

        fBin = f'/home/heliu/work/krause_jul19/s1350_110_1_nf_reduced/s1350_110_1_nf_int4_z{layer}_{rot:06d}.bin{det}'
        MicFileTool.plot_binary_with_tiff(fBin, sub>3)
    : alpha:
        transparency of binary, (0,1), 0: total transparent. 1: not transparent.
    '''
    b=IntBin.ReadI9BinaryFiles(fBin)
    plt.imshow(img[::-1,:]) #,origin='lower')
    print(f'shape of b[0]: {b[0].shape}')
    plt.scatter(2047-b[0],2047-b[1],s=0.1,alpha=alpha)
    plt.axis('scaled')
    plt.xlim((0,2048))
    plt.ylim((0,2048))
    plt.title(f'bin: {os.path.basename(fBin)}')
    plt.show()
def plot_mic_and_conf(squareMicData,minHitRatio,saveName=None,figSizeX=10,figSizeY=10):
    '''
    plot the square mic data
    image already inverted, x-horizontal, y-vertical, x dow to up, y: left to right
    :param squareMicData: [NVoxelX,NVoxelY,10], each Voxel conatains 10 columns:
            0-2: voxelpos [x,y,z]
            3-5: euler angle
            6: hitratio
            7: maskvalue. 0: no need for recon, 1: active recon region
            8: voxelsize
            9: additional information
    :return:
    '''
    eulers = squareMicData[:,:, 3:6].reshape([-1, 3]) / 180.0 * np.pi
    quats = Quaternion.quaternions_from_eulers(eulers)
    rods = Rodrigues.rodrigues_from_quaternions(quats)
    hitRatioMask = (squareMicData[:,:,6]>minHitRatio)[:,:,np.newaxis].repeat(3,axis=2)
    img = ((rods + np.array([1, 1, 1])) / 2).reshape([squareMicData.shape[0],squareMicData.shape[1],3]) * hitRatioMask
    img = np.swapaxes(img,0,1)
    fig, axes = plt.subplots(1,2)
    axes[0].imshow(img,origin='lower',extent=[squareMicData[0,0,0],squareMicData[-1,0,0],squareMicData[0,0,1],squareMicData[0,-1,1]])
    confMap = axes[1].imshow(squareMicData[:,:,6].T,origin='lower',extent=[squareMicData[0,0,0],squareMicData[-1,0,0],squareMicData[0,0,1],squareMicData[0,-1,1]])
    fig.colorbar(confMap, ax=axes[1],fraction=0.046, pad=0.04)
    fig.set_size_inches(figSizeX, figSizeY)
    if saveName is not None:
        plt.savefig(saveName)
    plt.show()
def plot_square_mic(squareMicData,minHitRatio,saveName=None):
    '''
    plot the square mic data
    image already inverted, x-horizontal, y-vertical, x dow to up, y: left to right
    :param squareMicData: [NVoxelX,NVoxelY,10], each Voxel conatains 10 columns:
            0-2: voxelpos [x,y,z]
            3-5: euler angle
            6: hitratio
            7: maskvalue. 0: no need for recon, 1: active recon region
            8: voxelsize
            9: additional information
    :return:
    '''
    eulers = squareMicData[:,:, 3:6].reshape([-1, 3]) / 180.0 * np.pi
    quats = Quaternion.quaternions_from_eulers(eulers)
    rods = Rodrigues.rodrigues_from_quaternions(quats)
    hitRatioMask = (squareMicData[:,:,6]>minHitRatio)[:,:,np.newaxis].repeat(3,axis=2)
    img = ((rods + np.array([1, 1, 1])) / 2).reshape([squareMicData.shape[0],squareMicData.shape[1],3]) * hitRatioMask
    img = np.swapaxes(img,0,1)
    plt.imshow(img,origin='lower',extent=[squareMicData[0,0,0],squareMicData[-1,0,0],squareMicData[0,0,1],squareMicData[0,-1,1]])
    if saveName is not None:
        plt.savefig(saveName)
    plt.show()


def plot_square_mic_backup(squareMicData,minHitRatio,saveName=None):
    '''
    plot the square mic data
    image already inverted, x-horizontal, y-vertical, x dow to up, y: left to right
    :param squareMicData: [NVoxelX,NVoxelY,10], each Voxel conatains 10 columns:
            0-2: voxelpos [x,y,z]
            3-5: euler angle
            6: hitratio
            7: maskvalue. 0: no need for recon, 1: active recon region
            8: voxelsize
            9: additional information
    :return:
    '''
    mat = EulerZXZ2MatVectorized(squareMicData[:,:,3:6].reshape([-1,3])/180.0 *np.pi )
    quat = np.empty([mat.shape[0],4])
    rod = np.empty([mat.shape[0],3])
    for i in range(mat.shape[0]):
        quat[i, :] = quaternion_from_matrix(mat[i, :, :])
        rod[i, :] = rod_from_quaternion(quat[i, :])
    hitRatioMask = (squareMicData[:,:,6]>minHitRatio)[:,:,np.newaxis].repeat(3,axis=2)
    img = ((rod + np.array([1, 1, 1])) / 2).reshape([squareMicData.shape[0],squareMicData.shape[1],3]) * hitRatioMask
    # make sure display correctly
    #img[:,:,:] = img[::-1,:,:]
    img = np.swapaxes(img,0,1)
    # plt.imshow(img,origin='lower',extent=[squareMicData[0,0,0],squareMicData[-1,0,0],squareMicData[0,0,1],squareMicData[0,-1,1]])
    # if saveName is not None:
    #     plt.savefig(saveName)
    # plt.show()
class MicFile():
    def __init__(self,fname):
        self.sw, self.snp=self.read_mic_file(fname)
        self.color2=self.snp[:,9]
        self.bpatches=False
        self.bcolor1=False

    def read_mic_file(self,fname):
        '''
        this will read the mic file
          %%
          %% Legacy File Format:
          %% Col 0-2 x, y, z
          %% Col 3   1 = triangle pointing up, 2 = triangle pointing down
          %% Col 4 generation number; triangle size = sidewidth /(2^generation number )
          %% Col 5 Phase - 1 = exist, 0 = not fitted
          %% Col 6-8 orientation
          %% Col 9  Confidence
          %%
        :param fname:
        :return:
            sw: float, the side width
            snp: [n_voxel,n_feature] numpy array

        '''
        with open(fname) as f:
            content = f.readlines()
        print(content[1])
        print(type(content[1]))
        sw = float(content[0])
        try:
            snp = np.array([[float(i) for i in s.split(' ')] for s in content[1:]])
        except ValueError:
            try:
                snp = np.array([[float(i) for i in s.split('\t')] for s in content[1:]])
            except ValueError:
                print('unknown deliminater')

        print('sw is {0} \n'.format(sw))
        print('shape of snp is {0}'.format(snp.shape))
        return sw,snp

    def plot_mic_patches(self,plotType,minConfidence):
        indx=self.snp[:,9]>=minConfidence
        minsw=self.sw/float(2**self.snp[0,4])
        tsw1=minsw*0.5
        tsw2=-minsw*0.5*3**0.5
        ntri=len(self.snp)
        if plotType==2:
            fig, ax = plt.subplots()
            if self.bpatches==False:
                xy=self.snp[:,:2]
                tmp=np.asarray([[tsw1]*ntri,(-1)**self.snp[:,3]*tsw2]).transpose()
                tris=np.asarray([[[0,0]]*ntri,[[minsw,0]]*ntri,tmp])
                self.patches=np.swapaxes(tris+xy,0,1)
                self.bpatches=True
            p=PolyCollection(self.patches[indx],cmap='viridis')
            p.set_array(self.color2[indx])
            p.set_edgecolor('face')
            ax.add_collection(p)
            ax.set_xlim([-0.6,0.6])
            ax.set_ylim([-0.6,0.6])
            fig.colorbar(p,ax=ax)
            plt.show()
        if plotType==1:
            fig, ax = plt.subplots()
            N=len(self.snp)
            mat = np.empty([N,3,3])
            quat = np.empty([N,4])
            rod = np.empty([N,3])
            if self.bcolor1==False:
                for i in range(N):
                    mat[i,:,:] = EulerZXZ2Mat(self.snp[i,6:9]/180.0*np.pi)
                    quat[i,:] = quaternion_from_matrix(mat[i,:,:])
                    rod[i,:] = rod_from_quaternion(quat[i,:])
                self.color1=(rod+np.array([1,1,1]))/2
                self.bcolor1=True
            if self.bpatches==False:
                xy=self.snp[:,:2]
                tmp=np.asarray([[tsw1]*ntri,(-1)**self.snp[:,3]*tsw2]).transpose()
                tris=np.asarray([[[0,0]]*ntri,[[minsw,0]]*ntri,tmp])
                self.patches=np.swapaxes(tris+xy,0,1)
                self.bpatches=True
            p=PolyCollection(self.patches[indx],cmap='viridis')
            p.set_color(self.color1[indx])
            ax.add_collection(p)
            ax.set_xlim([-0.6,0.6])
            ax.set_ylim([-0.6,0.6])
            plt.show()

def simple_plot(snp,sw,plotType,minConfidence):
    '''
    just plot the location, without orientation information
    :param snp:
    :param sw:
    :return:
    '''
    snp = snp[snp[:,9]>minConfidence,:]
    plt.plot(snp[:,0],snp[:,1],'*-')
    plt.show()

################# test session ###################
def test_for_dist():
    point = np.array([0.2,0.2])
    line = np.array([[0,0.24],[0.22,0.13]])
    dist = dist_to_line(point,line)
    print('dist should be',dist)
    plt.plot(point[0],point[1])
    plt.plot(line[:,0],line[:,1])
    plt.show()

def test_euler2mat():
    pass
def test_plot_mic():
    sw,snp = read_mic_file('Ti7_SYF_.mic.LBFS')
    #snp = snp[:100,:]
    plot_mic(snp,sw,3,0.35)

def combine_mic():
    sw_82,snp_82 = read_mic_file('Cu_.mic.opt')
    sw_81,snp_81 = read_mic_file('Cu_.mic_opt_81')
    sw_77, snp_77 = read_mic_file('Cu_.mic_opt_77')
    sw_89, snp_89 = read_mic_file('Cu_.mic.opt_89')
    snp = np.concatenate((snp_81,snp_82,snp_77,snp_89), axis=0)
    plot_mic(snp,sw_77,3,0)
    save_mic_file('eulerangles',snp[:,6:9],1)

def test_plot_square_mic():
    sMic = np.load('/home/heliu/work/krause_jul19/recon/s1400_100_1/s1400_100_1_q9_rot180_z1_500x500_0.002_shift_0.0_0.0_0.0.npy')
    plot_square_mic(sMic, 0.5)
if __name__ == '__main__':
    test_plot_square_mic()
    # sw,snp = read_mic_file('1000micron9GenSquare0.5.mic')
    #simple_plot(snp,sw,0,0.5)

    # new_snp = select_line_mic(snp)
    # plt.plot(new_snp[:,0],new_snp[:,1],'*')
    # plt.show()
    # save_mic_file('Cu_line.mic', new_snp, sw)
    #save_mic_file('Cu_combine.mic',snp,sw_82)
    #test_for_dist()
    #test_euler2mat()
    #test_plot_mic()
    #combine_mic()

