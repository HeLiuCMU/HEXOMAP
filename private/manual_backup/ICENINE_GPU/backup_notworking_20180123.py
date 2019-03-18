# implement simulation code with pycuda
# implement of Reconstruction Code with Pycuda
# He Liu CMU
# 20180117
import cProfile, pstats, StringIO
import pycuda.gpuarray as gpuarray
from pycuda.autoinit import context
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
import sim_utilities
import RotRep
import IntBin
import matplotlib.pyplot as plt
import FZfile
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
	float fNRot, fAngleStart,fAngleEnd;


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
__global__ void simulation(int *aiJ, int *aiK, float *afOmega, bool *abHit,int *aiRotN,
		const int iNVoxel, const int iNOrientation, const int iNG, const int iNDet,
		const float *afOrientation,const float *afG,const float *afVoxelPos,
		const float fBeamEnergy, const float fEtaLimit, const float *afDetInfo){
	/*
	 * int aiJ: output of J values,len =  iNVoxel*iNOrientation*iNG*2*iNDet
				basic unit iNVoxel*iNOrientation*iNG*[omega0 of det0, omega0 of det1,omega1,det0,omega1,det1]...
	 * int aiK: len =  iNVoxel*iNOrientation*iNG*2*iNDet
	 * float afOmega: len =  iNVoxel*iNOrientation*iNG*2*iNDet
	 * bool abHit: len =  iNVoxel*iNOrientation*iNG*2*iNDet
	 * int *aiRotN, the number of image that the peak is on, len=iNVoxel*iNOrientation*2*iNDet
	 * int iNVoxel: number of voxels
	 * int iNOrientation: number of orientations on each voxel
	 * int iNG: number of reciprocal vector on each diffraction process
	 * float *afOrientation: the array of all the orientation matrices of all the voxels,len=iNVoxel*iNOrientaion*9
	 * float *afG: list of reciprical vector len=iNG*3
	 * float *afVoxelPos: location of the voxels, len=iNVoxel*3;
	 * afDetInfo: [det0,det1,...], iNDet*19;
	 * number of
	 * the dimesion of GPU grid should be iNVoxel*iNOrientation*iNG
	 * <<< (iNVoxel,iNOrientation),(iNG)>>>;
	 */
	 //printf("blockIdx: %d || ",blockIdx.x);
	float fOmegaRes1,fOmegaRes2,fTwoTheta,fEta,fChi;
	float afScatteringVec[3]={0,0,0};
	//float afOrienMat[9];
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
		int i = blockIdx.x*gridDim.y*blockDim.x*2*iNDet+ blockIdx.y*blockDim.x*2*iNDet + threadIdx.x*2*iNDet;
		for(int iDetIdx=0;iDetIdx<iNDet;iDetIdx++){
			GetPeak(aiJ[i+iDetIdx],aiJ[i+iDetIdx+iNDet],aiK[i+iDetIdx],aiK[i+iDetIdx+iNDet],
					afOmega[i+iDetIdx],afOmega[i+iDetIdx+iNDet],abHit[i+iDetIdx],abHit[i+iDetIdx+iNDet],
					fOmegaRes1, fOmegaRes2,fTwoTheta,
					fEta,fChi,fEtaLimit,afVoxelPos+blockIdx.x*3,afDetInfo+19*iDetIdx);
			//printf("%f %f %f || ", (afVoxelPos+blockIdx.x*3)[0],(afVoxelPos+blockIdx.x*3)[1],(afVoxelPos+blockIdx.x*3)[2]);
			//printf("blockIdx: %d || ",blockIdx.x);
			if(abHit[i+iDetIdx]){
				////////assuming they are using the same rotation number in all the detectors!!!!!!!///////////////////
				aiRotN[i+iDetIdx] = floor((fOmegaRes1-afDetInfo[17])/(afDetInfo[18]-afDetInfo[17])*afDetInfo[16]);
				//printf("iJ1: %d, iK1 %d, fOmega1 %f, iRotN %d",aiJ[i],aiK[i],afOmega[i],aiRotN[i]);
			}
			if(abHit[i+iDetIdx+iNDet]){
				aiRotN[i+iDetIdx+iNDet] = floor((fOmegaRes2-afDetInfo[17])/(afDetInfo[18]-afDetInfo[17])*afDetInfo[16]);
				//printf("iJ2: %d, iK2 %d, fOmega2 %f, iRotN %d ",aiJ[i+1],aiK[i+1],afOmega[i+1],aiRotN[i+1]);
			}
		}
	}
}

__global__ void create_bin_expimages(char* acExpDetImages, const int* aiDetStartIdx,
		const float* afDetInfo,const int iNDet, const int iNRot,
		const int* aiDetIndex, const int* aiRotN, const int* aiJExp,const int* aiKExp, int const iNPeak){
	/*
	 * create the image matrix
	 * acExpDetImages: Sigma_i(iNDet*iNRot*iNJ[i]*iNK[i]) , i for each detector, detectors may have different size
	 * aiDetStartIdx:   index of Detctor start postition in self.acExpDetImages,
	 * 					e.g. 3 detectors with size 2048x2048, 180 rotations,
	 * 			 		aiDetStartIdx = [0,180*2048*2048,2*180*2048*2048]
	 * afDetInfo: iNDet*19, detector information
	 * iNDet: number of detectors, e.g. 2 or 3;
	 * iNRot: number of rotations, e.g. 180,720;
	 * aiDetIndex: len=iNPeak the index of detector, e.g. 0,1 or 2
	 * aiRotN: aiJExp: aiKExp: len=iNPeak
	 * iNPeak number of diffraction peaks
	 * test ?
	 */
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i<iNPeak){
		acExpDetImages[aiDetStartIdx[aiDetIndex[i]]
		                       + aiRotN[i]*int(afDetInfo[0+19*aiDetIndex[i]])*int(afDetInfo[1+19*aiDetIndex[i]])
		                        + aiKExp[i]*int(afDetInfo[0+19*aiDetIndex[i]]) + aiJExp[i]] = 1;
	}
}

__global__ void cost_val(const int iNVoxel,const int iNOrientation,const int iNG,
		const float* afDetInfo,const char* acExpDetImages,const int iNDet,const int iNRot,
		const int* aiJ, const int* aiK,const int* aiRotN, const bool* abHit,
		float* afHitRatio, int* aiPeakCnt){
	/*
	 * acExpDetImages: iNDet*iNRot*NPixelJ*NPixelK matrix, 1 for peak, 0 for no peak;
	 * aiJ: iNDet*iNVoxel*iNOrientation*iNG*2 ,2 is for omega1 and omega2
	 * afHitRatio: iNVoxel*iNOrientation: #hitpeak/#allDiffractPeak
	 * aiHitCnt: iNVoxel*iNOrientation: number of all diffraction Peaks
	 */
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<iNVoxel*iNOrientation){
	//printf("i: %d.",i);
		afHitRatio[i] = 0.0f;
		aiPeakCnt[i] = 0;
		for(int k=0;k<iNDet;k++){
			for(int j=0;j<iNG*2;j++){
				if(abHit[k*iNVoxel*iNOrientation*iNG*2+i*iNG*2+j]){
					aiPeakCnt[i]+=1;
					if(acExpDetImages[k*int(afDetInfo[0])*int(afDetInfo[1])*iNRot
					                  +aiRotN[k*iNVoxel*iNOrientation*iNG+i*iNG*2+j]*int(afDetInfo[0])*int(afDetInfo[1])
					                  +aiK[k*iNVoxel*iNOrientation*iNG+i*iNG*2+j]*int(afDetInfo[0])
					                  +aiJ[k*iNVoxel*iNOrientation*iNG+i*iNG*2+j]] == 1){
						afHitRatio[i] +=1;
					}
				}
			}
		}
		afHitRatio[i] = afHitRatio[i]/float(aiPeakCnt[i]);
	}

}

__global__ void hitratio_multi_detector(const int iNVoxel,const int iNOrientation,const int iNG,
		const float* afDetInfo,const char* acExpDetImages, const int* aiDetStartIdx, const int iNDet,const int iNRot,
		const int* aiJ, const int* aiK,const int* aiRotN, const bool* abHit,
		float* afHitRatio, int* aiPeakCnt){
	/*
	 * calculate hit ratio with multiple detector input
	 * consider as hit only when the peak hit all the detectors
	 * acExpDetImages: Sigma_i(iNDet*iNRot*NPixelJ[i]*NPixelK[i]),i for different detector matrix, 1 for peak, 0 for no peak;
	 * aiDetStartIdx:   index of Detctor start postition in self.acExpDetImages,
	 * 					e.g. 3 detectors with size 2048x2048, 180 rotations,
	 * 			 		aiDetStartIdx = [0,180*2048*2048,2*180*2048*2048]
	 * aiJ: iNDet*iNVoxel*iNOrientation*iNG*2*iDet ,2 is for omega1 and omega2
	 * afHitRatio: iNVoxel*iNOrientation: #hitpeak/#allDiffractPeak
	 * aiHitCnt: iNVoxel*iNOrientation: number of all diffraction Peaks hit on detector in the simulation
	 */
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	bool allTrue0 = true; // if a simulated peak hit all detector, allTrue0 = true;
	bool allTrue1 = true; // if a simulated peak overlap with all expimages on all detector, allTrue1 = true;
	if(i<iNVoxel*iNOrientation){
	//printf("i: %d.",i);
		afHitRatio[i] = 0.0f;
		aiPeakCnt[i] = 0;
		for(int j=0;j<iNG*2;j++){
			allTrue0 = true;
			allTrue1 = true;
			for(int k=0;k<iNDet;k++){
				if(!abHit[i*iNG*2*iNDet+j*iNDet+k]){
					allTrue0 = false;
				}
				if(acExpDetImages[aiDetStartIdx[k]
				 		          + aiRotN[i*iNG*2*iNDet+j*iNDet+k]*int(afDetInfo[0+19*k])*int(afDetInfo[1+19*k])
				 		          + aiK[i*iNG*2*iNDet+j*iNDet+k]*int(afDetInfo[0+19*k])
				 		          + aiJ[i*iNG*2*iNDet+j*iNDet+k]]==0){
					allTrue1 = false;
				}
			}
			if(allTrue0){
				aiPeakCnt[i]+=1;
			}
			if(allTrue1){
				afHitRatio[i]+=1;
			}
		}
		if(aiPeakCnt[i]!=0){
		afHitRatio[i] = afHitRatio[i]/float(aiPeakCnt[i]);
		}
		else{
			afHitRatio[i]=0;
		}
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
class Reconstructor_GPU():
    def __init__(self):
        self.voxelpos = np.array([[-0.0953125, 0.00270633, 0]])    # nx3 array,
        self.orientationEuler = np.array([[174.956, 55.8283, 182.94]])
        self.orientationMat = 0 # nx3x3 array
        self.energy = 55.587 # in kev
        self.sample = sim_utilities.CrystalStr('Ti7') # one of the following options:
        self.maxQ = 7
        self.etalimit = 81 / 180.0 * np.pi
        self.detectors = [sim_utilities.Detector(),sim_utilities.Detector()]
        self.NRot = 180
        self.NDet = 2
        self.centerJ = [975.022/4,968.773/4]# center, horizental direction
        self.centerK = [2013.83/4,2010.7/4]# center, verticle direction
        self.detPos = [np.array([5.46311,0,0]),np.array([7.44617,0,0])] # in mm
        self.detRot = [np.array([91.5667, 91.2042, 359.204]),np.array([90.7728, 91.0565, 359.3])]# Euler angleZXZ
        self.detectors[0].NPixelJ = 2048/4
        self.detectors[0].NPixelK = 2048/4
        self.detectors[0].PixelJ = 0.00148*4
        self.detectors[0].PixelK = 0.00148*4
        self.detectors[1].NPixelJ = 2048/4
        self.detectors[1].NPixelK = 2048/4
        self.detectors[1].PixelJ = 0.00148*4
        self.detectors[1].PixelK = 0.00148*4

        self.detectors[0].Move(self.centerJ[0], self.centerK[0], self.detPos[0], RotRep.EulerZXZ2Mat(self.detRot[0] / 180.0 * np.pi))
        self.detectors[1].Move(self.centerJ[1], self.centerK[1], self.detPos[1], RotRep.EulerZXZ2Mat(self.detRot[1] / 180.0 * np.pi))

        #detinfor for GPU[0:NJ,1:JK,2:pixelJ, 3:pixelK, 4-6: coordOrigin, 7-9:Norm 10-12 JVector, 13-16: KVector, 17: NRot, 18: angleStart, 19: angleEnd
        lDetInfoTmp = []
        for i in range(self.NDet):
            lDetInfoTmp.append(np.concatenate([np.array([self.detectors[i].NPixelJ,self.detectors[i].NPixelK,
                                                         self.detectors[i].PixelJ,self.detectors[i].PixelK]),
                                                self.detectors[i].CoordOrigin,self.detectors[i].Norm,self.detectors[i].Jvector,
                                               self.detectors[i].Kvector,np.array([self.NRot,-np.pi/2,np.pi/2])]))
        self.afDetInfoH = np.concatenate(lDetInfoTmp)
        # initialize Scattering Vectors
        self.sample.getRecipVec()
        self.sample.getGs(self.maxQ)
    def load_fz(self,fName):
        # load FZ.dat file
        # self.FZEuler: n_Orientation*3 array
        #test passed
        self.FZEuler = np.loadtxt(fName)
        return self.FZEuler
    def load_exp_data(self,fInitials,digits):
        '''
        load experimental binary data
        :param fInitials: e.g./home/heliu/work/I9_test_data/Integrated/S18_z1_
        :param digits: number of digits in file name, usually 6,
        :return:
        '''
        lJ = []
        lK = []
        lRot = []
        lDet = []
        lIntensity = []
        lID = []
        for i in range(self.NDet):
            for j in range(self.NRot):
                print('loading det {0}, rotation {1}'.format(i,j))
                fName = fInitials+str(j).zfill(digits) + '.bin' + str(i)
                x,y,intensity,id = IntBin.ReadI9BinaryFiles(fName)
                lJ.append(x[:,np.newaxis])
                lK.append(y[:,np.newaxis])
                lDet.append(i*np.ones(x[:,np.newaxis].shape))
                lRot.append(j*np.ones(x[:,np.newaxis].shape))
                lIntensity.append(intensity[:,np.newaxis])
                lID.append(id)
        self.expData = np.concatenate([np.concatenate(lDet,axis=0),np.concatenate(lRot,axis=0),np.concatenate(lJ,axis=0),np.concatenate(lK,axis=0)],axis=1)
        print('exp data loaded, shape is: {0}.'.format(self.expData.shape))

    def cp_expdata_to_gpu(self):
        # require have defiend self.NDet,self.NRot, and Detctor informations;
        #self.expData = np.array([[0,24,324,320],[0,0,0,1]]) # n_Peak*3,[detIndex,rotIndex,J,K] !!! be_careful this could go wrong is assuming wrong number of detectors
        #self.expData = np.array([[0,24,648,640],[0,172,285,631],[1,24,720,485],[1,172,207,478]]) #[detIndex,rotIndex,J,K]
        print('=============start of copy exp data to gpu ===========')
        if self.expData.shape[1]!=4:
            raise ValueError('expdata shape should be n_peaks*4')
        if np.max(self.expData[:,0])>self.NDet-1:
            raise ValueError('expData contains detector index out of bound')
        if np.max(self.expData[:,1])>self.NRot-1:
            raise  ValueError('expData contaisn rotation number out of bound')
        self.aiDetStartIdxH = [0] # index of Detctor start postition in self.acExpDetImages, e.g. 3 detectors with size 2048x2048, 180 rotations, self.aiDetStartIdx = [0,180*2048*2048,2*180*2048*2048]
        self.iExpDetImageSize = 0
        for i in range(self.NDet):
            self.iExpDetImageSize += self.NRot*self.detectors[i].NPixelJ*self.detectors[i].NPixelK
            if i<(self.NDet-1):
                self.aiDetStartIdxH.append(self.iExpDetImageSize)
        self.aiDetStartIdxH = np.array(self.aiDetStartIdxH)
        self.acExpDetImages = gpuarray.zeros(self.iExpDetImageSize,np.int8)   # experimental image data on GPUlen=sigma_i(NDet*NRot*NPixelJ[i]*NPxielK[i])
        self.aiDetStartIdxD = gpuarray.to_gpu(self.aiDetStartIdxH.astype(np.int32))
        self.afDetInfoD = gpuarray.to_gpu(self.afDetInfoH.astype(np.float32))

        self.aiDetIndxD = gpuarray.to_gpu(self.expData[:, 0].ravel().astype(np.int32))
        self.aiRotND = gpuarray.to_gpu(self.expData[:, 1].ravel().astype(np.int32))
        self.aiJExpD = gpuarray.to_gpu(self.expData[:, 2].ravel().astype(np.int32))
        self.aiKExpD = gpuarray.to_gpu(self.expData[:, 3].ravel().astype(np.int32))
        self.iNPeak = np.int32(self.expData.shape[0])
        create_bin_expimages = mod.get_function("create_bin_expimages")
        create_bin_expimages(self.acExpDetImages, self.aiDetStartIdxD, self.afDetInfoD, np.int32(self.NDet), np.int32(self.NRot),
                             self.aiDetIndxD, self.aiRotND, self.aiJExpD, self.aiKExpD, self.iNPeak, block=(256,1,1),grid=(self.iNPeak//256+1,1))
        print('=============end of copy exp data to gpu ===========')
        # self.out_expdata = self.acExpDetImages.get()
        # for i in range(self.NDet):
        #     detImageSize = self.NRot*self.detectors[i].NPixelK*self.detectors[i].NPixelJ
        #     print(self.out_expdata[self.aiDetStartIdxH[i]:(self.aiDetStartIdxH[i]+detImageSize)].reshape([self.NRot,self.detectors[i].NPixelK,self.detectors[i].NPixelJ]))

    def sim_precheck(self):
        #check if inputs are correct
        if self.NDet!= len(self.detectors):
            raise ValueError('self.NDet does not match self.detectors')
        if self.orientationMat.shape[0]!=self.NVoxel*self.NOrientation:
            raise ValueError('self.orientationMat should have shape of NVoxel*NOrientation*9')
        if self.NOrientation==0:
            raise ValueError('self.NOrientation could not be 0')
    def run_sim(self):
        # timing tools:
        start = cuda.Event()
        end = cuda.Event()
        start.record()  # start timing
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

        #initialize device parameters and outputs
        afOrientationMatD = gpuarray.to_gpu(self.orientationMat.astype(np.float32))
        afGD = gpuarray.to_gpu(self.sample.Gs.astype(np.float32))
        afVoxelPosD = gpuarray.to_gpu(self.voxelpos.astype(np.float32))
        afDetInfoD = gpuarray.to_gpu(self.afDetInfoH.astype(np.float32))
        self.NVoxel = self.voxelpos.shape[0]
        self.NOrientation = self.orientationMat.shape[0]/self.NVoxel
        NG = self.sample.Gs.shape[0]
        #output device parameters:
        aiJD = gpuarray.empty(self.NVoxel*self.NOrientation*NG*2*self.NDet,np.int32)
        aiKD = gpuarray.empty(self.NVoxel*self.NOrientation*NG*2*self.NDet,np.int32)
        afOmegaD= gpuarray.empty(self.NVoxel*self.NOrientation*NG*2*self.NDet,np.float32)
        abHitD = gpuarray.empty(self.NVoxel*self.NOrientation*NG*2*self.NDet,np.bool_)
        aiRotND = gpuarray.empty(self.NVoxel*self.NOrientation*NG*2*self.NDet, np.int32)
        sim_func = mod.get_function("simulation")


        # start of simulation
        print('============start of simulation ============= \n')
        start.record()  # start timing
        print('nvoxel: {0}, norientation:{1}\n'.format(self.NVoxel,self.NOrientation))
        self.sim_precheck()
        sim_func(aiJD, aiKD, afOmegaD, abHitD, aiRotND,\
                 np.int32(self.NVoxel), np.int32(self.NOrientation), np.int32(NG), np.int32(self.NDet), afOrientationMatD,afGD,\
                 afVoxelPosD,np.float32(self.energy),np.float32(self.etalimit), afDetInfoD,\
                 grid=(self.NVoxel,self.NOrientation), block=(NG,1,1))
        context.synchronize()
        end.record()
        self.aJH = aiJD.get()
        self.aKH = aiKD.get()
        self.aOmegaH = afOmegaD.get()
        self.bHitH = abHitD.get()
        self.aiRotNH = aiRotND.get()

        end.synchronize()
        print('============end of simulation================ \n')
        secs = start.time_till(end) * 1e-3
        print("SourceModule time {0} seconds.".format(secs))
        #self.print_sim_results()
    def test_recon(self):
        # RECONSTRUCT SINGLE VOXEL
        #self.orientationMat = np.loadtxt('FZ_MAT.txt').reshape([-1, 3, 3])
        #self.orientationEuler = self.load_fz('/home/heliu/work/I9_test_data/FIT/DataFiles/HexFZ.dat')
        #self.orientationEuler = np.concatenate([self.orientationEuler,[[174.956, 55.8283, 182.94]]],axis=0)
        self.orientationEuler = np.array([[174.956, 55.8283, 182.94]])
        self.orientationMat = np.zeros([self.orientationEuler.shape[0], 3, 3])
        if self.orientationEuler.ndim == 1:
            print('wrong format of input orientation, should be nx3 numpy array')
            return 0
        for i in range(self.orientationEuler.shape[0]):
            self.orientationMat[i,:,:] = RotRep.EulerZXZ2Mat(self.orientationEuler[i,:]/180.0*np.pi).reshape([3,3])

        self.load_exp_data('/home/heliu/work/I9_test_data/Integrated/S18_z1_',6)
        #self.expData = np.loadtxt('/home/heliu/work/I9_test_data/expData_for_pycuda.txt')
        #np.save(self.expData, '/home/heliu/work/I9_test_data/expData_for_pycuda.npy')
        #self.expData = np.array([[1,2,3,4]])
        self.expData[:,2:4] = self.expData[:,2:4]/4 # half the detctor size, to rescale real data
        self.cp_expdata_to_gpu()

        # timing tools:
        start = cuda.Event()
        end = cuda.Event()
        start.record()  # start timing

        # initialize device parameters and outputs
        self.NOrientation = self.orientationMat.shape[0]
        self.NVoxel = self.voxelpos.shape[0]
        self.NG = self.sample.Gs.shape[0]
        afGD = cuda.mem_alloc(self.sample.Gs.astype(np.float32).nbytes)
        cuda.memcpy_htod(afGD, self.sample.Gs.astype(np.float32))
        self.voxelpos = self.voxelpos.astype(np.float32)
        afVoxelPosD = cuda.mem_alloc(self.voxelpos.nbytes)
        cuda.memcpy_htod(afVoxelPosD, self.voxelpos)
        self.afDetInfoH = self.afDetInfoH.astype(np.float32)
        afDetInfoD = cuda.mem_alloc(self.afDetInfoH.nbytes)
        cuda.memcpy_htod(afDetInfoD, self.afDetInfoH)

        # start of simulation
        print('start of simulation \n')
        aiJD = cuda.mem_alloc(self.NVoxel * self.NOrientation * self.NG * 2 * self.NDet * (np.int32(0).nbytes))
        aiKD = cuda.mem_alloc(self.NVoxel * self.NOrientation * self.NG * 2 * self.NDet * (np.int32(0).nbytes))
        afOmegaD = cuda.mem_alloc(self.NVoxel * self.NOrientation * self.NG * 2 * self.NDet * (np.float32(0).nbytes))
        abHitD = cuda.mem_alloc(self.NVoxel * self.NOrientation * self.NG * 2 * self.NDet * (np.bool_(True).nbytes))
        aiRotND = cuda.mem_alloc(self.NVoxel * self.NOrientation * self.NG * 2 * self.NDet * (np.int32(0).nbytes))
        afHitRatioD = cuda.mem_alloc(self.NVoxel * self.NOrientation * (np.float32(0).nbytes))
        aiPeakCntD = cuda.mem_alloc(self.NVoxel * self.NOrientation * (np.int32(0).nbytes))
        # print(i)
        # if i==0:
        #     self.orientationMat = self.orientationMat[:self.NOrientation,:,:].astype(np.float32)
        # else:
        #     self.orientationMat = FZfile.random_angle_around_mat(self.maxMat,self.NOrientation,0.1,'Hexagonal').astype(np.float32)
        print(self.orientationMat.shape)
        afOrientationMatD = cuda.mem_alloc(self.orientationMat.nbytes)
        cuda.memcpy_htod(afOrientationMatD, self.orientationMat)
        # output device parameters:

        self.sim_precheck()
        self.sim_func = mod.get_function("simulation")
        self.sim_func(aiJD, aiKD, afOmegaD, abHitD, aiRotND, \
                      np.int32(self.NVoxel), np.int32(self.NOrientation), np.int32(self.NG), np.int32(self.NDet),
                     afOrientationMatD, afGD,
                      afVoxelPosD, np.float32(self.energy), np.float32(self.etalimit), afDetInfoD,
                      grid=(self.NVoxel, self.NOrientation), block=(self.NG, 1, 1))
        NBlock = 256
        #context.synchronize()
        self.hitratio_func = mod.get_function("hitratio_multi_detector")
        self.hitratio_func(np.int32(self.NVoxel), np.int32(self.NOrientation), np.int32(self.NG),
                           afDetInfoD, self.acExpDetImages, self.aiDetStartIdxD, np.int32(self.NDet),
                           np.int32(self.NRot),
                           aiJD, aiKD, aiRotND, abHitD,
                           afHitRatioD, aiPeakCntD,
                           block=(NBlock, 1, 1), grid=(self.NVoxel * self.NOrientation // NBlock + 1, 1))
        context.synchronize()
        #print(afOrientationMatD.mem_size)
        #print(afHitRatioD.mem_size)

        #lOrientationD[i].free()

        self.afHitRatioH = np.empty(self.NVoxel * self.NOrientation,dtype=np.float32)
        self.aiPeakCntH = np.empty(self.NVoxel * self.NOrientation,dtype=np.int32)
        cuda.memcpy_dtoh(self.afHitRatioH,afHitRatioD)
        cuda.memcpy_dtoh(self.aiPeakCntH,aiPeakCntD)
        self.maxHitratioIdx = np.argmax(self.afHitRatioH)
        self.maxMat = self.orientationMat[self.maxHitratioIdx,:,:]

        afGD.free()
        afVoxelPosD.free()
        afDetInfoD.free()
        aiJD.free()
        aiKD.free()
        afOmegaD.free()
        abHitD.free()
        aiRotND.free()
        afHitRatioD.free()
        aiPeakCntD.free()
        end.record()  # end timing
        end.synchronize()

        print('end of simulation \n')
        print('hit ratio: {0}, hit cnt: {1}'.format(self.afHitRatioH, self.aiPeakCntH))
        secs = start.time_till(end) * 1e-3
        print("SourceModule time {0} seconds.".format(secs))
        print('max hitratio is {0},hitcnt = {1}'.format(self.afHitRatioH[self.maxHitratioIdx],self.aiPeakCntH[self.maxHitratioIdx]))
        self.maxEuler = np.array(RotRep.Mat2EulerZXZ(self.maxMat))/np.pi*180
        print('max euler angle is {0}'.format(self.maxEuler))
        #np.savetxt('hitratios',self.afHitRatioH)
    def print_sim_results(self):
        # print(self.aJH)
        # print(self.aKH)
        # print(self.aiRotNH)
        # print(self.aOmegaH)
        # print(self.bHitH)

        for i,hit in enumerate(self.bHitH):
            if hit:
                print('Detector: {0}, J: {1}, K: {2},,RotN:{3}, Omega: {4}'.format(i%self.NDet, self.aJH[i],self.aKH[i],self.aiRotNH[i], self.aOmegaH[i]))
    def test_sim_0(self):
        # timing tools:
        start = cuda.Event()
        end = cuda.Event()
        start.record()  # start timing
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

        #initialize device parameters and outputs
        afOrientationMatD = gpuarray.to_gpu(self.orientationMat.astype(np.float32))
        afGD = gpuarray.to_gpu(self.sample.Gs[8:10,:].astype(np.float32))
        afVoxelPosD = gpuarray.to_gpu(self.voxelpos.astype(np.float32))
        afDetInfoD = gpuarray.to_gpu(self.afDetInfoH.astype(np.float32))
        self.NVoxel = self.voxelpos.shape[0]
        self.NOrientation = self.orientationMat.shape[0]/self.NVoxel
        NG = self.sample.Gs[8:10,:].shape[0]
        #output device parameters:
        aiJD = gpuarray.empty(self.NVoxel*self.NOrientation*NG*2*self.NDet,np.int32)
        aiKD = gpuarray.empty(self.NVoxel*self.NOrientation*NG*2*self.NDet,np.int32)
        afOmegaD= gpuarray.empty(self.NVoxel*self.NOrientation*NG*2*self.NDet,np.float32)
        abHitD = gpuarray.empty(self.NVoxel*self.NOrientation*NG*2*self.NDet,np.bool_)
        aiRotND = gpuarray.empty(self.NVoxel*self.NOrientation*NG*2*self.NDet, np.int32)
        sim_func = mod.get_function("simulation")


        # start of simulation
        print('start of simulation \n')
        print('nvoxel: {0}, norientation:{1}'.format(self.NVoxel,self.NOrientation))
        self.sim_precheck()
        sim_func(aiJD, aiKD, afOmegaD, abHitD, aiRotND,\
                 np.int32(self.NVoxel), np.int32(self.NOrientation), np.int32(NG), np.int32(self.NDet), afOrientationMatD,afGD,\
                 afVoxelPosD,np.float32(self.energy),np.float32(self.etalimit), afDetInfoD,\
                 grid=(self.NVoxel,self.NOrientation), block=(NG,1,1))
        context.synchronize()
        self.aJH = aiJD.get()
        self.aKH = aiKD.get()
        self.aOmegaH = afOmegaD.get()
        self.bHitH = abHitD.get()
        self.aiRotNH = aiRotND.get()
        end.record()  # end timing
        end.synchronize()
        print('end of simulation \n')
        secs = start.time_till(end) * 1e-3
        print("SourceModule time {0} seconds.".format(secs))
        self.print_sim_results()
        print("the result should be ...\n \
            [ True  True  True  True]\n \
            J: 1297, K: 1280, Omega: -1.15093827248 \n\
            J: 570, K: 1262, Omega: 1.43476486206 \n\
            J: 1078, K: 1240, Omega: -0.430149376392 \n\
            J: 848, K: 1221, Omega: 1.19551122189'\n ")
'''
[ True  True  True  True  True  True  True  True]
J: 648, K: 640,,RotN:24, Omega: -1.15093827248
J: 720, K: 485,,RotN:24, Omega: -1.15093827248
J: 285, K: 631,,RotN:172, Omega: 1.43476486206
J: 207, K: 478,,RotN:172, Omega: 1.43476486206
J: 539, K: 620,,RotN:65, Omega: -0.430149376392
J: 557, K: 458,,RotN:65, Omega: -0.430149376392
J: 424, K: 610,,RotN:158, Omega: 1.19551122189
J: 400, K: 449,,RotN:158, Omega: 1.19551122189
'''

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
        self.detPos = np.array([6.72573,0,0])               # in mm
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
        #print('originalGs shape is {0}'.format(self.sample.Gs.shape))
        #print('original Gs {0}\n{1}'.format(self.sample.Gs[8,:],self.sample.Gs[9,:]))

        #print('orienattionmat is : {0}'.format(self.orientationMat))
        self.rotatedG = self.orientationMat.dot(self.sample.Gs.T).T
        print(self.sample.Gs[8:10,:])
        #print('self.rotatedG: {0}'.format(self.rotatedG[8:10,:,:]))
        #print('rotetedG shape is {0}'.format(self.rotatedG.shape))
        self.detector.Print()

        for i,g1 in enumerate(self.rotatedG[8:10,:,:]):
            print(g1)
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
                        #                CorGs.append(g)
                if -90 <= res['omega_b'] <= 90:
                    omega = res['omega_b'] / 180.0 * np.pi
                    newgrainx = np.cos(omega) * self.grainpos[0] - np.sin(omega) * self.grainpos[1]
                    newgrainy = np.cos(omega) * self.grainpos[1] + np.sin(omega) * self.grainpos[0]
                    idx = self.detector.IntersectionIdx(np.array([newgrainx, newgrainy, 0]), res['2Theta'], -res['eta'])
                    if idx != -1:
                        Peaks.append([idx[0], idx[1], res['omega_b']])
        # CorGs.append(g)
        Peaks = np.array(Peaks)
        print(Peaks)
        # print(self.rotatedG.shape)
        # print(Peaks.shape)
        # print(self.rotatedG[0,:,:])
        # print('peak0 is {0}'.format(Peaks[0,:]))
        # CorGs=np.array(CorGs)
    def test_func(self, n_iterate):
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

        for i in range(n_iterate):
            Peaks = []
            self.rotatedG = self.orientationMat.dot(self.sample.Gs.T).T

            for i,g1 in enumerate(self.rotatedG):
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
                            #                CorGs.append(g)
                    if -90 <= res['omega_b'] <= 90:
                        omega = res['omega_b'] / 180.0 * np.pi
                        newgrainx = np.cos(omega) * self.grainpos[0] - np.sin(omega) * self.grainpos[1]
                        newgrainy = np.cos(omega) * self.grainpos[1] + np.sin(omega) * self.grainpos[0]
                        idx = self.detector.IntersectionIdx(np.array([newgrainx, newgrainy, 0]), res['2Theta'], -res['eta'])
                        if idx != -1:
                            Peaks.append([idx[0], idx[1], res['omega_b']])
            # CorGs.append(g)
        Peaks = np.array(Peaks)

def profile_python():
    nIterate = 1000
    S = Simulator_t()

    pr = cProfile.Profile()
    pr.enable()
    S.test_func(nIterate)
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
def test_load_fz():
    S = Reconstructor_GPU()
    S.load_fz('/home/heliu/work/I9_test_data/FIT/DataFiles/MyFZ.dat')
    print(S.FZEuler)
    print((S.FZEuler.shape))
def test_load_expdata():
    S = Reconstructor_GPU()
    S.NDet = 2
    S.NRot = 2
    S.load_exp_data('/home/heliu/work/I9_test_data/Integrated/S18_z1_',6)
def calculate_misoren_euler_zxz(euler0,euler1):
    rotMat0 = RotRep.EulerZXZ2Mat(euler0 / 180.0 * np.pi)
    rotMat1 = RotRep.EulerZXZ2Mat(euler1 / 180.0 * np.pi)
    return RotRep.Misorien2FZ1(rotMat0,rotMat1,symtype='Hexagonal')
if __name__ == "__main__":
    S = Reconstructor_GPU()
    S.test_recon()
    #S.run_sim()
    #S.print_sim_results()
    #print(calculate_misoren_euler_zxz(np.array([10.1237, 75.4599, 340.791]),np.array([174.956, 55.8283, 182.94])))
