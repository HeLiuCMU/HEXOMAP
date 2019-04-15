# Generate finer FZ file
# He Liu
# CMU
# 20180122
# see reference paper: http://refbase.cvc.uab.es/files/PIE2012.pdf
import cProfile, pstats
from io import StringIO
import pycuda.gpuarray as gpuarray
from pycuda.autoinit import context
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
from pycuda.curandom import MRG32k3aRandomNumberGenerator
import sim_utilities
import RotRep
import IntBin
import FZfile
import time
import random

mod = SourceModule("""
#include <stdio.h>
const float PI = 3.14159265359;
const float HALFPI = 0.5*PI;

__device__ void d_euler_zxz_to_mat(float* afEuler, float* afMat){
        float s1 = sin(afEuler[0]);
        float s2 = sin(afEuler[1]);
        float s3 = sin(afEuler[2]);
        float c1 = cos(afEuler[0]);
        float c2 = cos(afEuler[1]);
        float c3 = cos(afEuler[2]);
        afMat[0] = c1 * c3 - c2 * s1 * s3;
        afMat[1] = -c1 * s3 - c3 * c2 * s1;
        afMat[2] = s1 * s2;
        afMat[3] = s1 * c3 + c2 * c1 * s3;
        afMat[4] = c1 * c2 * c3 - s1 * s3;
        afMat[5] = -c1 * s2;
        afMat[6] = s3 * s2;
        afMat[7] = s2 * c3;
        afMat[8] = c2;
}

__global__ void mat_to_euler_ZXZ(float* afMatIn, float* afEulerOut, int iNAngle){
    /*
    * transform active rotation matrix to euler angles in ZXZ convention, not right(seems right now)
    * afMatIn: iNAngle * 9
    * afEulerOut: iNAngle* 3
    * TEST PASSED
    */
    float threshold = 0.9999999;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<iNAngle){
        if(afMatIn[i * 9 + 8] > threshold){
            afEulerOut[i * 3 + 0] = 0;
            afEulerOut[i * 3 + 1] = 0;
            afEulerOut[i * 3 + 2] = atan2(afMatIn[i*9 + 3], afMatIn[i*9 + 0]);           //  atan2(m[1, 0], m[0, 0])
        }
        else if(afMatIn[i * 9 + 8] < - threshold){
            afEulerOut[i * 3 + 0] = 0;
            afEulerOut[i * 3 + 1] = PI;
            afEulerOut[i * 3 + 2] = atan2(afMatIn[i*9 + 1], afMatIn[i*9 + 0]);           //  atan2(m[0, 1], m[0, 0])
        }
        else{
            afEulerOut[i * 3 + 0] = atan2(afMatIn[i*9 + 2], - afMatIn[i*9 + 5]);          //  atan2(m[0, 2], -m[1, 2])
            afEulerOut[i * 3 + 1] = atan2( sqrt(afMatIn[i*9 + 6] * afMatIn[i*9 + 6]
                                                + afMatIn[i*9 + 7] * afMatIn[i*9 + 7]),
                                            afMatIn[i*9 + 8]);                             //     atan2(np.sqrt(m[2, 0] ** 2 + m[2, 1] ** 2), m[2, 2])
            afEulerOut[i * 3 + 2] = atan2( afMatIn[i*9 + 6], afMatIn[i*9 + 7]);           //   atan2(m[2, 0], m[2, 1])
            if(afEulerOut[i * 3 + 0] < 0){
                afEulerOut[i * 3 + 0] += 2 * PI;
            }
            if(afEulerOut[i * 3 + 1] < 0){
                afEulerOut[i * 3 + 1] += 2 * PI;
            }
            if(afEulerOut[i * 3 + 2] < 0){
                afEulerOut[i * 3 + 2] += 2 * PI;
            }
        }
    }
}

__global__ void rand_mat_neighb_from_euler(float* afEulerIn, float* afMatOut, float* afRand, float fBound){
    /* generate random matrix according to the input EulerAngle
    * afEulerIn: iNEulerIn * 3, !!!!!!!!!! in radian  !!!!!!!!
    * afMatOut: iNNeighbour * iNEulerIn * 9
    * afRand:   iNNeighbour * iNEulerIn * 3
    * fBound: the range for random angle [-fBound,+fBound]
    * iNEulerIn: number of Input Euler angles
    * iNNeighbour: number of random angle generated for EACH input
    * call:: <<(iNNeighbour,1),(iNEulerIn,1,1)>>
    * TEST PASSED
    */
    //printf("%f||",fBound);
    // keep the original input
        float afEulerTmp[3];

        afEulerTmp[0] = afEulerIn[threadIdx.x * 3 + 0] + (2 * afRand[blockIdx.x * blockDim.x * 3 + threadIdx.x * 3 + 0] - 1) * fBound;
        afEulerTmp[2] = afEulerIn[threadIdx.x * 3 + 2] + (2 * afRand[blockIdx.x * blockDim.x * 3 + threadIdx.x * 3 + 2] - 1) * fBound;
        float z = cos(afEulerIn[threadIdx.x * 3 + 1]) +
                        (afRand[blockIdx.x * blockDim.x * 3 + threadIdx.x * 3 + 1] * 2 - 1) * sin(afEulerIn[threadIdx.x * 3 + 1] * fBound);
        if(z>1){
            z = 1;
        }
        else if(z<-1){
            z = -1;
        }
        afEulerTmp[1] = acosf(z);

        if(blockIdx.x>0){
            d_euler_zxz_to_mat(afEulerTmp, afMatOut + blockIdx.x * blockDim.x * 9 + threadIdx.x * 9);
        }
        else{
            // keep the original input
            d_euler_zxz_to_mat(afEulerIn + threadIdx.x * 3, afMatOut + blockIdx.x * blockDim.x * 9 + threadIdx.x * 9);
        }
}


__device__ void mat3_transpose(float* afOut, float* afIn){
    /*
    * transpose 3x3 matrix
    */
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            afOut[i * 3 + j] = afIn[j * 3 + i];
        }
    }
}
__device__ void mat3_dot(float* afResult, float* afM0, float* afM1){
    /*
    * dot product of two 3x3 matrix
    */
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            afResult[i * 3 + j] = 0;
            for(int k=0;k<3;k++){
                afResult[i * 3 + j] += afM0[i * 3 + k] * afM1[k * 3 + j];
            }
        }
    }
}

__global__ void misoren(float* afMisOrien, float* afM0, float* afM1, float* afSymM){
    /*
    * calculate the misorientation betwen afM0 and afM1
    * afMisOrien: iNM * iNSymM
    * afM0: iNM * 9
    * afM1: iNM * 9
    * afSymM: symmetry matrix, iNSymM * 9
    * NSymM: number of symmetry matrix
    * call method: <<<(iNM,1),(iNSymM,1,1)>>>
    */
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float afTmp0[9];
    float afTmp1[9];
    float afM1Transpose[9];
    float fCosAngle;
    mat3_transpose(afM1Transpose, afM1 + blockIdx.x * 9);
    mat3_dot(afTmp0, afSymM + threadIdx.x * 9, afM1Transpose);
    mat3_dot(afTmp1, afM0 + blockIdx.x * 9, afTmp0);
    //for(int i=0;i<9;i++){
    //    printf("%d, %f ",i, afSymM[i]);
    //}
    fCosAngle = 0.5 * (afTmp1[0] + afTmp1[4] + afTmp1[8] - 1);
    fCosAngle = min(0.9999999999, fCosAngle);
    fCosAngle = max(-0.99999999999, fCosAngle);
    printf("fCosAngle: %f " ,fCosAngle);
    afMisOrien[i] = acosf(fCosAngle);
}
""")
misoren_gpu = mod.get_function("misoren")
def misorien(m0, m1,symMat):
    '''
    calculate misorientation
    :param m0: [n,3,3]
    :param m1: [n,3,3]
    :param symMat: symmetry matrix,
    :return:
    '''
    m0 = m0.reshape([-1,3,3])
    m1 = m1.reshape([-1,3,3])
    symMat = symMat.reshape([-1,3,3])
    if m0.shape != m1.shape:
        raise ValueError(' m0 and m1 should in the same shape')
    NM = m0.shape[0]
    NSymM = symMat.shape[0]
    afMisOrienD = gpuarray.empty([NM,NSymM], np.float32)
    afM0D = gpuarray.to_gpu(m0.astype(np.float32))
    afM1D = gpuarray.to_gpu(m1.astype(np.float32))
    afSymMD = gpuarray.to_gpu(symMat.astype(np.float32))
    misoren_gpu(afMisOrienD, afM0D, afM1D, afSymMD,block=(NSymM,1,1),grid=(NM,1))
    #print(symMat[0])
    #print(symMat[0].dot(np.matrix(m1)))
    print(afMisOrienD.shape)
    return np.amin(afMisOrienD.get(), axis=1)

def test_miscorien():
    m0 = RotRep.EulerZXZ2Mat(np.array([89.5003, 80.7666, 266.397])/180.0*np.pi)
    m1 = RotRep.EulerZXZ2Mat(np.array([89.5003, 2.7666, 266.397])/180.0*np.pi)
    m0 = m0[np.newaxis,:,:].repeat(3,axis=0)
    m1 = m1[np.newaxis, :, :].repeat(3, axis=0)
    m1[0,:,:] = RotRep.EulerZXZ2Mat(np.array([89.5003, 80.7666, 266.397])/180.0*np.pi)
    symMat = RotRep.GetSymRotMat('Hexagonal')
    print(misorien(m0,m1,symMat))
    print(RotRep.Misorien2FZ1(m0, m1, 'Hexagonal'))

def  generate_random_rot_mat(nEuler):
    '''
    generate random euler angles, and bring back to FZ
    :param nEuler: number of Euler angles
    :param symtype: 'Cubic' or 'Hexagonal'
    :return: array, nEulerx3
    '''
    # ############### method 1 ###############
    # not vectorized
    nEuler = int(nEuler)
    alpha = np.random.uniform(-np.pi,np.pi,nEuler)
    gamma = np.random.uniform(-np.pi,np.pi,nEuler)
    z = np.random.uniform(-1,1,nEuler)
    beta = np.arccos(z)
    result = np.empty([nEuler,3,3])
    for i in range(nEuler):
        matTmp = RotRep.EulerZXZ2Mat(np.array([alpha[i],beta[i],gamma[i]]))
        result[i,:,:] = matTmp
    return result
    ########## vectorized version ##############
def write_mat_to_file(mat,fName):
    '''
    write rotation matrix to file
    :param mat:
    :param fName:
    :return:
    '''
    np.savetxt(fName,mat)
def test_mat_to_euler():
    e = np.array([10.1237, 75.4599, 340.791])/180*np.pi
    mat = RotRep.EulerZXZ2Mat(e)
    print(mat.shape)
    eResult = np.array(RotRep.Mat2EulerZXZ(mat))/np.pi*180
    print(eResult)

def random_angle_around_mat(mat,nAngle,boundBox,symtype):
    '''
    !!!!!CAUTION!!! current version ignore crystal type, does not bring back to FZ.
    The generation of Random Rotation will affect the reconstruction alot
    generate rotation angles around certain rotation matrix,
    :param mat: input rotation matrix, [n_mat,3,3]
    :param nAngle: number of angles to generate
    :param boundBox: in radian, angle range to generate [-boundBox,boundBox]
    :param symtype: 'Cubic' or 'Hexagonal'
    :return: [n_mat*nAngle,3,3] matrix
    test passed
    '''
    ################### NOT Vectorized #######################################
    # mat = mat.reshape([-1,3,3])
    # #print(mat.shape)
    # result = np.empty([mat.shape[0]*nAngle, 3,3])
    # #print(result.shape)
    # for i in range(mat.shape[0]):
    #     eulerTmp = RotRep.Mat2EulerZXZ(mat[i,:,:])
    #     alpha = np.random.uniform(eulerTmp[0]-boundBox, eulerTmp[0]+boundBox, nAngle)
    #     gamma = np.random.uniform(eulerTmp[2]-boundBox, eulerTmp[2]+boundBox, nAngle)
    #     z = np.random.uniform(np.cos(eulerTmp[1])-boundBox*np.sin(eulerTmp[1]),np.cos(eulerTmp[1])+boundBox*np.sin(eulerTmp[1]), nAngle) # cos(a+b) ~ cosa+b*sina
    #     z[z>1] = 1
    #     z[z<-1] = -1
    #     beta = np.arccos(z)
    #     for j in range(nAngle):
    #         matTmp = RotRep.EulerZXZ2Mat(np.array([alpha[j], beta[j], gamma[j]]))
    #         #result[i*mat.shape[0]+j, :,:] = np.array(RotRep.Orien2FZ(matTmp, symtype=symtype)[0])
    #         result[i * mat.shape[0] + j, :, :] = matTmp
    #     result[i*mat.shape[0],:, : ] = mat[i,:,:]
    # return result
    ################### vectorized version 0.2 ###############
    ################### NOT Vectorized #######################################
    mat = mat.reshape([-1,3,3])
    result = np.empty([mat.shape[0]*nAngle, 3,3])
    eulerTmp = RotRep.Mat2EulerZXZVectorized(mat)
    randEulerTmp = np.empty([nAngle,3])

    for i in range(mat.shape[0]):
        randEulerTmp[:,0] = np.random.uniform(eulerTmp[i,0]-boundBox, eulerTmp[i,0]+boundBox, nAngle)
        randEulerTmp[:,2] = np.random.uniform(eulerTmp[i,2]-boundBox, eulerTmp[i,2]+boundBox, nAngle)
        z = np.random.uniform(np.cos(eulerTmp[i,1])-boundBox*np.sin(eulerTmp[i,1]),np.cos(eulerTmp[i,1])+boundBox*np.sin(eulerTmp[i,1]), nAngle) # cos(a+b) ~ cosa+b*sina
        z[z>1] = 1
        z[z<-1] = -1
        randEulerTmp[:,1] = np.arccos(z)
        result[i * mat.shape[0]: (i*mat.shape[0] + nAngle),:,:] = RotRep.EulerZXZ2MatVectorized(randEulerTmp)
        result[i*mat.shape[0],:, : ] = mat[i,:,:] # keep the original input rotation matrix
    return result

if __name__ =='__main__':
    test_miscorien()
    #mat = generate_random_rot_mat(50000,'Hexagonal')
    #print(mat)
    #write_mat_to_file(mat,'FZ_MAT.txt')
    #test_mat_to_euler()
    # euler = np.array([174.956, 55.8283, 182.94])/180*np.pi
    # print(euler)
    # mat = RotRep.EulerZXZ2Mat(euler)
    # #print(mat)
    # result = random_angle_around_mat(mat,10,0.1,'Hexagonal')
    # print(result.shape)
    # #print(result)
    # for i in range(result.shape[0]):
    #     print(RotRep.Mat2EulerZXZ(result[i,:,:]))
    #     print(RotRep.Misorien2FZ1(mat,result[i,:,:],symtype='Hexagonal'))
    #
    # print (RotRep.Mat2EulerZXZ(mat))
