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
import RotRep

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
            print 'unknown deliminater'

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
        print 'h'
        for i in range(N):
            mat[i,:,:] = RotRep.EulerZXZ2Mat(snp[i,6:9]/180*np.pi)
            #print mat[i,:,:]
            quat[i,:] = RotRep.quaternion_from_matrix(mat[i,:,:])
            #print quat[i,:]
            rod[i,:] = RotRep.rod_from_quaternion(quat[i,:])
        print rod
        fig, ax = plt.subplots()
        ax.scatter(snp[:,0],snp[:,1],s=scattersize,facecolors=(rod+np.array([1,1,1]))/2)
        ax.axis('scaled')
        plt.show()

def plot_square_mic(squareMicData,minHitRatio):
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
    mat = RotRep.EulerZXZ2MatVectorized(squareMicData[:,:,3:6].reshape([-1,3])/180.0 *np.pi )
    quat = np.empty([mat.shape[0],4])
    rod = np.empty([mat.shape[0],3])
    for i in range(mat.shape[0]):
        quat[i, :] = RotRep.quaternion_from_matrix(mat[i, :, :])
        rod[i, :] = RotRep.rod_from_quaternion(quat[i, :])
    hitRatioMask = (squareMicData[:,:,6]>minHitRatio)[:,:,np.newaxis].repeat(3,axis=2)
    img = ((rod + np.array([1, 1, 1])) / 2).reshape([squareMicData.shape[0],squareMicData.shape[1],3]) * hitRatioMask
    # make sure display correctly
    #img[:,:,:] = img[::-1,:,:]
    img = np.swapaxes(img,0,1)
    plt.imshow(img,origin='lower')
    plt.show()

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
                print 'unknown deliminater'

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
                    mat[i,:,:] = RotRep.EulerZXZ2Mat(self.snp[i,6:9]/180.0*np.pi)
                    quat[i,:] = RotRep.quaternion_from_matrix(mat[i,:,:])
                    rod[i,:] = RotRep.rod_from_quaternion(quat[i,:])
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
    print 'dist should be',dist
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
    sMic = np.load('SquareMicTest1.npy')
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

