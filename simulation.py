# implemente simulation code with pycuda
# He Liu CMU
# 20180117
import pycuda.gpuarray as gpuarray
from pycuda.autoinit import context
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
import sim_utilities
import RotRep
import gpustruct
mod = SourceModule("""
#include <stdio.h>
const float PI = 3.14159265359;
const float HALFPI = 0.5*PI;

typedef struct {
    int iNPixelJ, iNPixelK;
    float fPixelJ, fPixelK;
	float afCoordOrigin[3];
	float afNorm[3];
	float afJVector[3];
	float afKVector[3];


} DetInfo;
__device__ float Dot3(const float *afArray1, const float *afArray2){
	// dot product of vectors with length 3
	float fResult = afArray1[0] * afArray2[0] + afArray1[1] * afArray2[1] + afArray1[1] * afArray2[1];
	return fResult;
}

__device__ bool GetScatteringOmegas( float &fOmegaRes1, float &fOmegaRes2,
		float &fTwoTheta,float &fEta,float &fChi,
        const float *aScatteringVec,const float &fBeamEnergy){
	////////////////////////test passed ////////////////////////
 //aScatterinVec: float[3];
	// need take PI as constant
  /// fOmegaRe
  ///  NOTE:  reciprical vectors are measured in angstrom
  ///  k  = 2pi/lambda  = E/ (h-bar c)
  ///
  ///aScatteringVec: float array,len=3, contains the scattering vecter x,y,z.
	//////////////////////////////////////////////////////////////
  float fScatteringVecMag = sqrt(aScatteringVec[0]*aScatteringVec[0] + aScatteringVec[1]*aScatteringVec[1] +aScatteringVec[2]*aScatteringVec[2]);

  float fSinTheta = fScatteringVecMag / ( (float)2.0 * 0.506773182 * fBeamEnergy);   // Bragg angle
  float fCosTheta = sqrt( (float)1.0 - fSinTheta * fSinTheta);
  float fCosChi = aScatteringVec[2] / fScatteringVecMag;             // Tilt angle of G relative to z-axis
  float fSinChi = sqrt( (float)1.0 - fCosChi * fCosChi );
  //float fSinChiLaue = sin( fBeamDeflectionChiLaue );     // ! Tilt angle of k_i (+ means up)
  //float fCosChiLaue = cos( fBeamDeflectionChiLaue );

  if( fabsf( fSinTheta ) <= fabsf( fSinChi) )
  {
	float fPhi = acosf(fSinTheta / fSinChi);
	float fSinPhi = sin(fPhi);
	fEta = asinf(fSinChi * fSinPhi / fCosTheta);
	// [-pi:pi]: angle to bring G to nominal position along +y-axis
	float fDeltaOmega0 = atan2f( aScatteringVec[0], aScatteringVec[1]);

	//  [0:pi/2] since arg >0: phi goes from above to Bragg angle
	float fDeltaOmega_b1 = asinf( fSinTheta/fSinChi );

	float fDeltaOmega_b2 = PI -  fDeltaOmega_b1;

	fOmegaRes1 = fDeltaOmega_b1 + fDeltaOmega0;  // oScatteringVec.m_fY > 0
	fOmegaRes2 = fDeltaOmega_b2 + fDeltaOmega0;  // oScatteringVec.m_fY < 0

	if ( fOmegaRes1 > PI )          // range really doesn't matter
	  fOmegaRes1 -=  2.f * PI;

	if ( fOmegaRes1 < -PI)
	  fOmegaRes1 +=  2.f * PI;

	if ( fOmegaRes2 > PI)
	  fOmegaRes2 -= 2.f * PI;

	if ( fOmegaRes2 < -PI)
	  fOmegaRes2 += 2.f * PI;
	fTwoTheta = 2.f * asinf(fSinTheta);
	fChi = acosf(fCosChi);
	return true;
  }
  else
  {
	fOmegaRes1 = fOmegaRes2 = 0;     // too close to rotation axis to be illumination
	fTwoTheta = fEta = fChi = 0;
	return false;
  }


}


__device__ bool GetPeak(int &iJ1,int &iJ2,int &iK1, int &iK2,float &fOmega1, float &fOmega2,bool &bHit1,bool &bHit2,
		const float &fOmegaRes1, const float &fOmegaRes2,
		const float &fTwoTheta, const float &fEta,const float &fChi,const float &fEtaLimit,
		const float *afVoxelPos,const float *afDetInfo){
	/*
	 *  TEST PASSED 20171215
	 * cDetMat:	char matrix, float[nOmega*nPixelX*nPixelY];
	 * fVoxelPos float vector, float[3] [x,y,z];
	 * afDetInfo:   	  	int iNPixelJ=0, iNPixelK=1;
    						float fPixelJ=2, fPixelK=3;
							float afCoordOrigin[3]=[4,5,6];
							float afNorm[3]=[7,8,9];
							float afJVector[3][10,11,12];
							float afKVector[3]=[13,14,15];
	 */
	if (fChi>= 0.5*PI){
		bHit1 = false;
		bHit2 = false;
		return false;
	}
	else if(fEta>fEtaLimit){
		bHit1 = false;
		bHit2 = false;
		return false;
	}
	if ((-HALFPI<=fOmegaRes1) && (fOmegaRes1<=HALFPI)){
		float fVoxelPosX = cos(fOmegaRes1)*afVoxelPos[0] - sin(fOmegaRes1)*afVoxelPos[1];
		float fVoxelPosY = cos(fOmegaRes1)*afVoxelPos[1] + sin(fOmegaRes1)*afVoxelPos[0];
		float fVoxelPosZ = afVoxelPos[2];
		float fDist;
		fDist = afDetInfo[7]*(afDetInfo[4] - fVoxelPosX)
				+ afDetInfo[8]*(afDetInfo[5] - fVoxelPosY)
				+ afDetInfo[9]*(afDetInfo[6] - fVoxelPosZ);
		float afScatterDir[3]; //scattering direction
		afScatterDir[0] = cos(fTwoTheta);
		afScatterDir[1] = sin(fTwoTheta) * sin(fEta);
		afScatterDir[2] = sin(fTwoTheta) * cos(fEta);
		float afInterPos[3];
		float fAngleNormScatter = afDetInfo[7]*afScatterDir[0]
		                          + afDetInfo[8]*afScatterDir[1]
		                          + afDetInfo[9]*afScatterDir[2];
		afInterPos[0] = fDist / fAngleNormScatter * afScatterDir[0] + fVoxelPosX;
		afInterPos[1] = fDist / fAngleNormScatter * afScatterDir[1] + fVoxelPosY;
		afInterPos[2] = fDist / fAngleNormScatter * afScatterDir[2] + fVoxelPosZ;
		float fJ,fK;
		fJ = (afDetInfo[10]*(afInterPos[0]-afDetInfo[4])
				+ afDetInfo[11]*(afInterPos[1]-afDetInfo[5])
				+ afDetInfo[12]*(afInterPos[2]-afDetInfo[6]) )/afDetInfo[2];
		fK = (afDetInfo[13]*(afInterPos[0]-afDetInfo[4])
				+ afDetInfo[14]*(afInterPos[1]-afDetInfo[5])
				+ afDetInfo[15]*(afInterPos[2]-afDetInfo[6]) )/afDetInfo[3];
		int iJ = int(fJ);
		int iK = int(fK);
		if ((0<=iJ )&&(iJ<=afDetInfo[0]) &&(0<=iK) && (iK<=afDetInfo[1])){
			iJ1 = iJ;
			iK1 = iK;
			fOmega1 = fOmegaRes1;
			bHit1 = true;
		}
		else{
			bHit1 = false;
		}
	}

	if ((-HALFPI<=fOmegaRes2) && (fOmegaRes2<=HALFPI)){
		float fVoxelPosX = cos(fOmegaRes2)*afVoxelPos[0] - sin(fOmegaRes2)*afVoxelPos[1];
		float fVoxelPosY = cos(fOmegaRes2)*afVoxelPos[1] + sin(fOmegaRes2)*afVoxelPos[0];
		float fVoxelPosZ = afVoxelPos[2];
		float fDist;
		fDist = afDetInfo[7]*(afDetInfo[4] - fVoxelPosX)
				+ afDetInfo[8]*(afDetInfo[5] - fVoxelPosY)
				+ afDetInfo[9]*(afDetInfo[6] - fVoxelPosZ);
		float afScatterDir[3]; //scattering direction
		afScatterDir[0] = cos(fTwoTheta);
		afScatterDir[1] = sin(fTwoTheta) * sin(-fEta);  // caution: -fEta!!!!!!
		afScatterDir[2] = sin(fTwoTheta) * cos(-fEta);  // caution: -fEta!!!!!!
		float afInterPos[3];
		float fAngleNormScatter = afDetInfo[7]*afScatterDir[0]
		                          + afDetInfo[8]*afScatterDir[1]
		                          + afDetInfo[9]*afScatterDir[2];
		afInterPos[0] = fDist / fAngleNormScatter * afScatterDir[0] + fVoxelPosX;
		afInterPos[1] = fDist / fAngleNormScatter * afScatterDir[1] + fVoxelPosY;
		afInterPos[2] = fDist / fAngleNormScatter * afScatterDir[2] + fVoxelPosZ;
		float fJ,fK;
		fJ = (afDetInfo[10]*(afInterPos[0]-afDetInfo[4])
				+ afDetInfo[11]*(afInterPos[1]-afDetInfo[5])
				+ afDetInfo[12]*(afInterPos[2]-afDetInfo[6]) )/afDetInfo[2];
		fK = (afDetInfo[13]*(afInterPos[0]-afDetInfo[4])
				+ afDetInfo[14]*(afInterPos[1]-afDetInfo[5])
				+ afDetInfo[15]*(afInterPos[2]-afDetInfo[6]) )/afDetInfo[3];
		int iJ = int(fJ);
		int iK = int(fK);
		if ((0<=iJ )&&(iJ<=afDetInfo[0]) &&(0<=iK) && (iK<=afDetInfo[1])){
			iJ2 = iJ;
			iK2 = iK;
			fOmega2 = fOmegaRes2;
			bHit2 = true;
		}
		else{
			bHit2 = false;
		}
	}
	return true;

}
__device__ bool print_DetInfo(DetInfo *sDetInfo){
	printf("afCoordOrigin: %f %f %f", sDetInfo->afCoordOrigin[0], sDetInfo->afCoordOrigin[1], sDetInfo->afCoordOrigin[2]);
	printf("afNorm: %f %f %f", sDetInfo->afNorm[0],sDetInfo->afNorm[1],sDetInfo->afNorm[1]);
	printf("afJVector: %f %f %f", sDetInfo->afJVector[0],sDetInfo->afJVector[1],sDetInfo->afJVector[2]);
	printf("afKVector: %f %f %f", sDetInfo->afKVector[0],sDetInfo->afKVector[1],sDetInfo->afKVector[2]);

	printf("fPixelJ: %f", sDetInfo->fPixelJ);
	printf("fPixelK: %f", sDetInfo->fPixelK);
	return true;
}
__global__ void simulation(int *aiJ, int *aiK, float *afOmega, bool *abHit,
		const int iNVoxel, const int iNOrientation, const int iNG,const float *afOrientation,const float *afG,
		const float *afVoxelPos, const float fBeamEnergy, const float fEtaLimit, const float *afDetInfo){
	/*
	 * int aiJ: output of J values,len =  iNVoxel*iNOrientation*iNG*2
	 * int aiK: len =  iNVoxel*iNOrientation*iNG*2
	 * float afOmega: len =  iNVoxel*iNOrientation*iNG*2
	 * bool abHit: len =  iNVoxel*iNOrientation*iNG*2
	 * int iNVoxel: number of voxels
	 * int iNOrientation: number of orientations on each voxel
	 * int iNG: number of reciprocal vector on each diffraction process
	 * float *afOrientation: the array of all the orientation matrices of all the voxels,len=iNVoxel*iNOrientaion*9
	 * float *afG: list of reciprical vector len=iNG*3
	 * float *afVoxelPos: location of the voxels, len=iNVoxel*3;
	 * number of
	 * the dimesion of GPU grid should be iNVoxel*iNOrientation*iNG
	 * <<< (iNVoxel,iNOrientation),iNG>>>;
	 */
	// test section
	//print_DetInfo(sDetInfo);
	// change orientation of Gs
	// simulation:
	float fOmegaRes1,fOmegaRes2,fTwoTheta,fEta,fChi;
	float afScatteringVec[3]={0,0,0};
	float afOrienMat[9];
	float _afVoxelPos[3];
	for (int i=0;i<3;i++){
	_afVoxelPos[i] = afVoxelPos[blockIdx.x*3+i];
	}
	//printf("afVoxelPos: %f %f %f",_afVoxelPos[0],_afVoxelPos[1],_afVoxelPos[2]);
	//original G vector
	//rotation matrix 3x3
	//G' = M.dot(G)
	for (int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			afScatteringVec[i] += afOrientation[blockIdx.x*gridDim.y*9+blockIdx.y*9+i*3+j]*afG[threadIdx.x*3+j];
		}
	}
	if(GetScatteringOmegas( fOmegaRes1, fOmegaRes2, fTwoTheta, fEta, fChi , afScatteringVec,fBeamEnergy)){
		int i = blockIdx.x*gridDim.y*blockDim.x*2+ blockIdx.y*blockDim.x*2 + threadIdx.x*2;
		GetPeak(aiJ[i],aiJ[i+1],aiK[i],aiK[i+1],afOmega[i],afOmega[i+1],abHit[i],abHit[i+1],fOmegaRes1, fOmegaRes2,fTwoTheta,
				fEta,fChi,fEtaLimit,_afVoxelPos,afDetInfo);
		//if(abHit[i]){
		//	printf("iJ1: %d, iK1 %d, fOmega1 %f ",aiJ[i],aiK[i],afOmega[i]);
		//}
		//if(abHit[i+1]){
		//	printf("iJ2: %d, iK2 %d, fOmega2 %f ",aiJ[i+1],aiK[i+1],afOmega[i+1]);
		//}
	}
}
""")

class DetectorGPU:
    mem_size = 4*np.intp(0).nbytes + 4*8
    def __init__(self, detector, struct_ptr):
        self.afCoordOrigin = cuda.to_device(detector.CoordOrigin.astype(np.float32))
        self.afNorm = cuda.to_device(detector.Norm.astype(np.float32))
        self.afJVector = cuda.to_device(detector.Jvector.astype(np.float32))
        self.afKVector = cuda.to_device(detector.Kvector.astype(np.float32))
        self.iNPixelJ = detector.NPixelJ
        self.iNPixelK = detector.NPixelK
        self.fPixelJ = detector.PixelJ
        self.fPixelK = detector.PixelK
        cuda.memcpy_htod(int(struct_ptr), np.getbuffer(np.intp(int(self.afCoordOrigin))))
        cuda.memcpy_htod(int(struct_ptr)+ int(1*np.intp(0).nbytes), np.getbuffer(np.intp(int(self.afNorm))))
        cuda.memcpy_htod(int(struct_ptr)+int(2*np.intp(0).nbytes), np.getbuffer(np.intp(int(self.afJVector))))
        cuda.memcpy_htod(int(struct_ptr)+int(3*np.intp(0).nbytes), np.getbuffer(np.intp(int(self.afKVector))))
        cuda.memcpy_htod(int(struct_ptr) + int(4 * np.intp(0).nbytes), np.getbuffer(np.int32(self.iNPixelJ)))
        cuda.memcpy_htod(int(struct_ptr) + int(4 * np.intp(0).nbytes + 1*8), np.getbuffer(np.int32(self.iNPixelK)))
        cuda.memcpy_htod(int(struct_ptr) + int(4 * np.intp(0).nbytes + 2*8), np.getbuffer(np.float32(self.fPixelJ)))
        cuda.memcpy_htod(int(struct_ptr) + int(4 * np.intp(0).nbytes + 3*8), np.getbuffer(np.float32(self.fPixelK)))
        #print(d.CoordOrigin)
        # self.gpuStruct = gpustruct.GPUStruct([(np.float32, '*afCoordOrigin', detector.CoordOrigin.astype(np.float32)),
        #                                       (np.float32, '*afNorm', detector.Norm.astype(np.float32)),
        #                                       (np.float32, '*afJVector', detector.Jvector.astype(np.float32)),
        #                                       (np.float32, '*afKVector', detector.Kvector.astype(np.float32)),
        #                                       (np.int32, 'iNPixelJ', detector.NPixelJ),
        #                                       (np.int32, 'iNPixelK', detector.NPixelK),
        #                                       (np.float32, 'fPixelJ', detector.PixelJ),
        #                                       (np.float32, 'fPixelK', detector.PixelK)])
class Simulator_GPU():
    def __init__(self):
        self.voxelpos = np.array([[0, 0.0974279, 0]])    # nx3 array,
        self.orientationEuler = np.array([[89.5003, 80.7666, 266.397]])
        self.orientationMat = 0 # nx3x3 array
        self.energy = 71.676 # in kev
        self.sample = sim_utilities.CrystalStr('Ti7') # one of the following options:
        self.maxQ = 10
        self.etalimit = 81 / 180.0 * np.pi
        self.detector = sim_utilities.Detector()
        self.centerJ = 935.166                              # center, horizental direction
        self.centerK = 1998.96                              # center, verticle direction
        self.detPos = np.array([4.72573,0,0])               # in mm
        self.detRot = np.array([90.6659, 89.4069,359.073])  # Euler angle
        self.detector.Move(self.centerJ, self.centerK, self.detPos, RotRep.EulerZXZ2Mat(self.detRot / 180.0 * np.pi))
        print('test point 0')
        print(self.detector.Norm)
        self.afDetInfoH = np.concatenate([np.array([self.detector.NPixelJ,self.detector.NPixelK,self.detector.PixelJ,self.detector.PixelK]),
                                    self.detector.CoordOrigin,self.detector.Norm,self.detector.Jvector,self.detector.Kvector])
        print('afDetInfoH shape is {0}'.format(self.afDetInfoH.shape))
    def run_sim(self):
        # initialize Scattering Vectors
        self.sample.getRecipVec()
        self.sample.getGs(self.maxQ)
        # initialize orientation Matrices !!! implement on GPU later
        self.orientationMat = np.zeros([self.orientationEuler.shape[0], 3, 3])
        if self.orientationEuler.ndim == 1:
            print('wrong format of input orientation, should be nx3 numpy array')
            return 0
        for i in range(self.orientationEuler.shape[0]):
            self.orientationMat[i,:,:] = RotRep.EulerZXZ2Mat(self.orientationEuler[i,:]/180.0*np.pi).reshape([3,3])
        #initialize parameters and outputs
        afOrientationMatD = gpuarray.to_gpu(self.orientationMat.astype(np.float32))
        afGD = gpuarray.to_gpu(self.sample.Gs[8:10,:].astype(np.float32))
        afVoxelPosD = gpuarray.to_gpu(self.voxelpos.astype(np.float32))
        afDetInfoD = gpuarray.to_gpu(self.afDetInfoH.astype(np.float32))
        NVoxel = self.voxelpos.shape[0]
        NOrientation = self.orientationMat.shape[0]
        NG = self.sample.Gs[8:10,:].shape[0]
        #output parameters:
        aiJD = gpuarray.empty(NVoxel*NOrientation*NG*2,np.int32)
        aiKD = gpuarray.empty(NVoxel*NOrientation*NG*2,np.int32)
        afOmegaD= gpuarray.empty(NVoxel * NOrientation * NG * 2,np.float32)
        abHitD = gpuarray.empty(NVoxel*NOrientation*NG*2,np.bool_)
        sim_func = mod.get_function("simulation")
        print('start of simulation \n')
        print('nvoxel: {0}, norientation:{1}'.format(NVoxel,NOrientation))
        sim_func(aiJD, aiKD, afOmegaD, abHitD,\
                 np.int32(NVoxel), np.int32(NOrientation), np.int32(NG), afOrientationMatD,afGD,\
                 afVoxelPosD,np.float32(self.energy),np.float32(self.etalimit), afDetInfoD,\
                 grid=(NVoxel,NOrientation), block=(NG,1,1))
        context.synchronize()
        self.aJH = aiJD.get()
        self.aKH = aiKD.get()
        self.aOmegaH = afOmegaD.get()
        self.bHitH = abHitD.get()
        print('end of simulation \n')

    def print_results(self):
        print(self.aJH)
        print(self.aKH)
        print(self.aOmegaH)
        print(self.bHitH)
        for i,hit in enumerate(self.bHitH):
            if hit:
                print('J: {0}, K: {1}, Omega: {2}'.format(self.aJH[i],self.aKH[i],self.aOmegaH[i]))
class Simulator_t():
    def __init__(self):
        self.voxels = []     # nx3 array,
        self.orientationEuler = np.array([[89.5003, 80.7666, 266.397]])
        self.orientationMat = 0 # nx3x3 array
        self.energy = 71.676 # in kev
        self.sample = sim_utilities.CrystalStr('Ti7') # one of the following options:
        self.maxQ = 10
        self.etalimit = 81 / 180.0 * np.pi
        self.detector = sim_utilities.Detector()
        self.centerJ = 935.166                              # center, horizental direction
        self.centerK = 1998.96                              # center, verticle direction
        self.detPos = np.array([4.72573,0,0])               # in mm
        self.detRot = np.array([90.6659, 89.4069,359.073])  # Euler angle
    def run_sim(self):
        self.orientationMat = np.zeros([self.orientationEuler.shape[0],3,3])
        self.grainpos = np.array([0, 0.0974279, 0])

        if self.orientationEuler.ndim==1:
            print('wrong format of input orientation, should be nx3 numpy array')
            return 0
        self.sample.getRecipVec()
        self.sample.getGs(self.maxQ)

        self.detector.Move(self.centerJ, self.centerK,self.detPos,RotRep.EulerZXZ2Mat(self.detRot/180.0*np.pi))
        #self.detector.Print()
        for i in range(self.orientationEuler.shape[0]):
            self.orientationMat[i,:,:] = RotRep.EulerZXZ2Mat(self.orientationEuler[i,:]/180.0*np.pi).reshape([3,3])
            ######################### SIMULATION PART ###################################33
        # Get the observable peaks, the 'Peaks' is a n*3 ndarray
        Peaks = []
        # CorGs=[]
        print('originalGs shape is {0}'.format(self.sample.Gs.shape))
        print('original Gs {0}\n{1}'.format(self.sample.Gs[8,:],self.sample.Gs[9,:]))

        print('orienattionmat is : {0}'.format(self.orientationMat))
        self.rotatedG = self.orientationMat.dot(self.sample.Gs.T).T
        print('rotetedG shape is {0}'.format(self.rotatedG.shape))
        self.detector.Print()

        for i,g1 in enumerate(self.rotatedG[8:10,:,:]):
            print('================================================')
            print('g1: {0} \n'.format(g1))
            res = sim_utilities.frankie_angles_from_g(g1, verbo=False, **{'energy':self.energy})
            if res['chi'] >= 90:
                pass
            elif res['eta'] > self.etalimit:
                pass
            else:
                if -90 <= res['omega_a'] <= 90:
                    omega = res['omega_a'] / 180.0 * np.pi
                    newgrainx = np.cos(omega) * self.grainpos[0] - np.sin(omega) * self.grainpos[1]
                    newgrainy = np.cos(omega) * self.grainpos[1] + np.sin(omega) * self.grainpos[0]
                    try:
                        idx = self.detector.IntersectionIdx(np.array([newgrainx, newgrainy, 0]), res['2Theta'], res['eta'])
                    except:
                        print('hohoho')
                    if idx != -1:
                        Peaks.append([idx[0], idx[1], res['omega_a']])
                        print(i)
                        #                CorGs.append(g)
                if -90 <= res['omega_b'] <= 90:
                    omega = res['omega_b'] / 180.0 * np.pi
                    newgrainx = np.cos(omega) * self.grainpos[0] - np.sin(omega) * self.grainpos[1]
                    newgrainy = np.cos(omega) * self.grainpos[1] + np.sin(omega) * self.grainpos[0]
                    idx = self.detector.IntersectionIdx(np.array([newgrainx, newgrainy, 0]), res['2Theta'], -res['eta'])
                    if idx != -1:
                        Peaks.append([idx[0], idx[1], res['omega_b']])
                        print(i)
            print('Peaks: {0},omega: {1}'.format(Peaks,Peaks[0][2]/180.0*np.pi))
        # CorGs.append(g)
        Peaks = np.array(Peaks)
        # print(Peaks)
        # print(self.rotatedG.shape)
        # print(Peaks.shape)
        # print(self.rotatedG[0,:,:])
        # print('peak0 is {0}'.format(Peaks[0,:]))
        # CorGs=np.array(CorGs)
    def test_func(self):
        #only works when there is only one voxel and one orientation
        self.orientationMat = np.zeros([self.orientationEuler.shape[0],3,3])
        if self.orientationEuler.ndim==1:
            print('wrong format of input orientation, should be nx3 numpy array')
            return 0
        self.sample.getRecipVec()
        self.sample.getGs(self.maxQ)
        self.detector.Move(self.centerJ, self.centerK,self.detPos,RotRep.EulerZXZ2Mat(self.detRot/180.0*np.pi))
        self.orientationMat = RotRep.EulerZXZ2Mat(self.orientationEuler[0,:]/180.0*np.pi).reshape([1,3,3])
        self.rotatedG=self.orientationMat.dot(self.sample.Gs.T).T
        self.detector.Print()
        for i, g1 in enumerate(self.rotatedG[0:2,:,:]):
            print(i,g1)
            res = sim_utilities.frankie_angles_from_g(g1, verbo=False, **{'energy':71.676})
            print(res['omega_a']/180.0*np.pi,res['omega_b']/180.0*np.pi,res['2Theta'],res['eta'],res['chi'])
        print(self.rotatedG.shape)
if __name__ == "__main__":
    print('test')
    S = Simulator_GPU()
    S.run_sim()
    S.print_results()
    ############## test #####3
    # S = Simulator_t()
    # S.run_sim()
