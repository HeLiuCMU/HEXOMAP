# implemente simulation code with pycuda
# He Liu CMU
# 20180117
import cProfile, pstats, StringIO
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
	bHit1 = false;
	bHit2 = false;
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
		int iJ = (int)fJ;
		int iK = (int)fK;
		if ((0<=iJ )&&(iJ<afDetInfo[0]) &&(0<=iK) && (iK<afDetInfo[1])){
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
		int iJ = (int)fJ;
		int iK = (int)fK;
		if ((0<=iJ )&&(iJ<afDetInfo[0]) &&(0<=iK) && (iK<afDetInfo[1])){
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
	 //printf("start sim");
	 //if(blockIdx.x==0 && threadIdx.x==0){
	 //   printf(" %f, %f, %f||",afOrientation[blockIdx.x*gridDim.y*9+blockIdx.y*9+0*3+0],
	 //   afOrientation[blockIdx.x*gridDim.y*9+blockIdx.y*9+0*3+1],
	 //   afOrientation[blockIdx.x*gridDim.y*9+blockIdx.y*9+0*3+2]);
	 // }
	//printf("blockIdx: %d || ",blockIdx.x);
	float fOmegaRes1,fOmegaRes2,fTwoTheta,fEta,fChi;
	float afScatteringVec[3]={0,0,0};
	//float afOrienMat[9];
	//original G vector
	//rotation matrix 3x3
	//G' = M.dot(G)
	//printf("originG: %f,%f,%f. ||",afG[threadIdx.x*3+0],afG[threadIdx.x*3+1],afG[threadIdx.x*3+2]);
	for (int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			afScatteringVec[i] += afOrientation[blockIdx.x*gridDim.y*9+blockIdx.y*9+i*3+j]*afG[threadIdx.x*3+j];
		    //printf("%d,%d: %f. ||",i,j,afOrientation[blockIdx.x*gridDim.y*9+blockIdx.y*9+i*3+j]);
		}
	}
	//printf("%f,%f,%f ||",afScatteringVec[0],afScatteringVec[1],afScatteringVec[1]);
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
				aiRotN[i+iDetIdx] = floor((fOmegaRes1-afDetInfo[17])/(afDetInfo[18]-afDetInfo[17])*(afDetInfo[16]-1));
				//printf("iJ1: %d, iK1 %d, fOmega1 %f, iRotN %d",aiJ[i],aiK[i],afOmega[i],aiRotN[i]);
			}
			if(abHit[i+iDetIdx+iNDet]){
				aiRotN[i+iDetIdx+iNDet] = floor((fOmegaRes2-afDetInfo[17])/(afDetInfo[18]-afDetInfo[17])*(afDetInfo[16]-1));
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
	 //printf("start hitratio ||");
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	bool allTrue0 = true; // if a simulated peak hit all detector, allTrue0 = true;
	bool allTrue1 = true; // if a simulated peak overlap with all expimages on all detector, allTrue1 = true;
	if(i<iNVoxel*iNOrientation){
	    //printf("|+|i: %d.",i);
		afHitRatio[i] = 0.0f;
		aiPeakCnt[i] = 0;
		for(int j=0;j<iNG*2;j++){
		    //printf("j: %d ||",j);
			allTrue0 = true;
			allTrue1 = true;
			for(int k=0;k<iNDet;k++){
			    //printf("abHit: %f||",abHit[i*iNG*2*iNDet+j*iNDet+k]);
				if(!abHit[i*iNG*2*iNDet+j*iNDet+k]){
					allTrue0 = false;
					allTrue1 = false;
				}
			    //printf("imagedataIdx: %d", aiDetStartIdx[k]
				// 		          + aiRotN[i*iNG*2*iNDet+j*iNDet+k]*int(afDetInfo[0+19*k])*int(afDetInfo[1+19*k])
				//		          + aiK[i*iNG*2*iNDet+j*iNDet+k]*int(afDetInfo[0+19*k])
				// 		          + aiJ[i*iNG*2*iNDet+j*iNDet+k]);
			    //if(aiDetStartIdx[k]
				// 		          + aiRotN[i*iNG*2*iNDet+j*iNDet+k]*int(afDetInfo[0+19*k])*int(afDetInfo[1+19*k])
				// 		          + aiK[i*iNG*2*iNDet+j*iNDet+k]*int(afDetInfo[0+19*k])
				// 		          + aiJ[i*iNG*2*iNDet+j*iNDet+k] >= 2*180*512*512){
			    //   printf("detIdx: %d, rotN: %d, aiK: %d, aiJ: %d ||",k,aiRotN[i*iNG*2*iNDet+j*iNDet+k],aiK[i*iNG*2*iNDet+j*iNDet+k],aiJ[i*iNG*2*iNDet+j*iNDet+k]);
			    //}
				else if(acExpDetImages[aiDetStartIdx[k]
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
		if(aiPeakCnt[i]>0){
		    afHitRatio[i] = afHitRatio[i]/float(aiPeakCnt[i]);
		}
		else{
			afHitRatio[i]=0;
		}
		//printf("afHitRatio: %f, aiPeakCnt: %d", afHitRatio[i], aiPeakCnt[i]); // so does not run to this step
	}
}

__global__ void display_rand(float* afRandom, int iNRand){
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        printf("=%d=",i);
        if (i<iNRand){
        printf(" %f ||", afRandom[i]);
        }
}

__global__ void euler_zxz_to_mat(float* afEuler, float* afMat, int iNAngle){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<iNAngle){
        float s1 = sin(afEuler[i * 3 + 0]);
        float s2 = sin(afEuler[i * 3 + 1]);
        float s3 = sin(afEuler[i * 3 + 2]);
        float c1 = cos(afEuler[i * 3 + 0]);
        float c2 = cos(afEuler[i * 3 + 1]);
        float c3 = cos(afEuler[i * 3 + 2]);
        afMat[i * 9 + 0] = c1 * c3 - c2 * s1 * s3;
        afMat[i * 9 + 1] = -c1 * s3 - c3 * c2 * s1;
        afMat[i * 9 + 2] = s1 * s2;
        afMat[i * 9 + 3] = s1 * c3 + c2 * c1 * s3;
        afMat[i * 9 + 4] = c1 * c2 * c3 - s1 * s3;
        afMat[i * 9 + 5] = -c1 * s2;
        afMat[i * 9 + 6] = s3 * s2;
        afMat[i * 9 + 7] = s2 * c3;
        afMat[i * 9 + 8] = c2;
    }
}

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

""")


def test_gpuarray_take():
    a = gpuarray.arange(0,100,1,dtype=np.int32)
    indices = gpuarray.to_gpu(np.array([1,2,4,5]).astype(np.int32))
    b = gpuarray.take(a,indices)
    print(b.get())
    del indices
    indices = gpuarray.to_gpu(np.array([7, 8, 23, 5]).astype(np.int32))
    b = gpuarray.take(a, indices)
    print(b.get())
def test_mat2eulerzxz():
    NEulerIn = 6
    euler = np.array([[111.5003, 80.7666, 266.397],[1.5003, 80.7666, 266.397]]).repeat(NEulerIn/2, axis=0) / 180 * np.pi
    print(euler)
    matIn = RotRep.EulerZXZ2MatVectorized(euler)
    matInD = gpuarray.to_gpu(matIn.astype(np.float32))
    eulerOutD = gpuarray.empty(NEulerIn*3,np.float32)
    func = mod.get_function("mat_to_euler_ZXZ")
    NBlock = 128
    func(matInD,eulerOutD, np.int32(NEulerIn),block=(NBlock,1,1), grid = (NEulerIn//NBlock+1, 1))
    eulerOutH = eulerOutD.get().reshape([-1,3])
    print(eulerOutH)
def test_rand_amt_neighb():
    NEulerIn = 2
    NEighbour = 2
    bound = 0.1
    euler = np.array([[89.5003, 80.7666, 266.397]]).repeat(NEulerIn, axis=0)/180*np.pi
    matIn = RotRep.EulerZXZ2MatVectorized(euler).repeat(NEighbour,axis=0)
    matInD = gpuarray.to_gpu(matIn.astype(np.float32))
    S = Reconstructor_GPU()
    matOutD = S.gen_random_matrix(matInD,NEulerIn,NEighbour,0.01)
    # print(matIn.shape)
    # eulerD = gpuarray.to_gpu(euler.astype(np.float32))
    # matOutD = gpuarray.empty(NEighbour*NEulerIn*9,np.float32)
    # g = MRG32k3aRandomNumberGenerator()
    # afRandD = g.gen_uniform(NEighbour*NEulerIn*3, np.float32)
    # func = mod.get_function("rand_mat_neighb_from_euler")
    # func(eulerD, matOutD, afRandD, np.float32(bound),grid = (NEighbour,1),block=(NEulerIn,1,1))
    matH = matOutD.get().reshape([-1,3,3])
    print(matH.reshape([-1,3,3]))
    print(RotRep.Mat2EulerZXZVectorized(matH)/np.pi*180)
    for i in range(matH.shape[0]):
        print(RotRep.Misorien2FZ1(matIn[i,:,:], matH[i,:,:], 'Hexagonal'))
def test_random():
    N = 100
    g = MRG32k3aRandomNumberGenerator()
    rand = g.gen_uniform(N, np.float32)
    disp_func = mod.get_function("display_rand")
    disp_func(rand, np.int32(N), block=(N,1,1))
def test_eulerzxz2mat():
    N = 10000
    euler = np.array([[89.5003, 80.7666, 266.397]]).repeat(N,axis=0)
    eulerD = gpuarray.to_gpu(euler.astype(np.float32))
    matD = gpuarray.empty(N*9, np.float32 )
    gpu_func = mod.get_function("euler_zxz_to_mat")
    gpu_func(eulerD,matD,np.int32(N),block=(N,1,1))
    print(matD.get().reshape([-1,3,3]))
    print(RotRep.EulerZXZ2MatVectorized(euler))

class Reconstructor_GPU():
    '''
    example usage:
    Todo:
        load voxelpos only once, not like now, copy every time
        implement random matrix in GPU! high priority
        flood filling problem, this kind of filling will actually affecto the shape of grain,
        voxel at boundary need to compare different grains.

    '''
    def __init__(self):
        # initialize voxel position information
        self.voxelpos = np.array([[-0.0953125, 0.00270633, 0]])    # nx3 array, voxel positions
        self.NVoxel = self.voxelpos.shape[0]                       # number of voxels
        self.voxelAcceptedMat = np.zeros([self.NVoxel,3,3])        # reconstruced rotation matrices
        self.voxelHitRatio = np.zeros(self.NVoxel)                 # reconstructed hit ratio
        self.voxelIdxStage0 = range(self.voxelpos.shape[0])       # this contains index of the voxel that have not been tried to be reconstructed, used for flood fill process
        self.voxelIdxStage1 = []                    # this contains index  the voxel that have hit ratio > threshold on reconstructed voxel, used for flood fill process
        self.micData = np.zeros([self.NVoxel,11])                  # mic data loaded from mic file, get with self.load_mic(fName), detail format see in self.load_mic()
        self.FZEuler = np.array([[89.5003, 80.7666, 266.397]])     # fundamental zone euler angles, loaded from I9 fz file.
        self.oriMatToSim = np.zeros([self.NVoxel,3,3])             # the orientation matrix for simulation, nx3x3 array, n = Nvoxel*OrientationPerVoxel
        # experimental data
        self.energy = 55.587 # in kev
        self.sample = sim_utilities.CrystalStr('Ti7') # one of the following options:
        self.maxQ = 8
        self.etalimit = 81 / 180.0 * np.pi
        self.detectors = [sim_utilities.Detector(),sim_utilities.Detector()]
        self.NRot = 180
        self.NDet = 2
        self.centerJ = [976.072/4,968.591/4]# center, horizental direction
        self.centerK = [2014.13/4,2011.68/4]# center, verticle direction
        self.detPos = [np.array([5.46569,0,0]),np.array([7.47574,0,0])] # in mm
        self.detRot = [np.array([91.6232, 91.2749, 359.274]),np.array([90.6067, 90.7298, 359.362])]# Euler angleZXZ
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
        self.NG = self.sample.Gs.shape[0]

        # reconstruction parameters:
        self.floodFillStartThreshold = 0.61 # orientation with hit ratio larger than this value is used for flood fill.
        self.floodFillSelectThreshold = 0.6 # voxels with hitratio less than this value will be reevaluated in flood fill process.
        self.floodFillAccptThreshold = 0.6  #voxel with hit ratio > floodFillTrheshold will be accepted to voxelIdxStage1
        self.floodFillRandomRange = 0.001   # voxel in voxelIdxStage1 will generate random angles in this window
        self.floodFillNumberAngle = 1000 # number of rangdom angles generated to voxel in voxelIdxStage1
        self.floodFillNumberVoxel = 20000  # number of orientations for flood fill process each time, due to GPU memory size.
        self.floodFillNIteration = 2       # number of iteration for flood fill angles
        self.searchBatchSize = 20000      # number of orientations to search per GPU call, due to GPU memory size
        self.NSelect = 100                 # number of orientations selected with maximum hitratio from last iteration
        # retrieve gpu kernel
        self.sim_func = mod.get_function("simulation")
        self.hitratio_func = mod.get_function("hitratio_multi_detector")
        self.mat_to_euler_ZXZ = mod.get_function("mat_to_euler_ZXZ")
        self.rand_mat_neighb_from_euler = mod.get_function("rand_mat_neighb_from_euler")
        # GPU random generator
        self.randomGenerator = MRG32k3aRandomNumberGenerator()
    def set_voxel_pos(self,pos):
        '''
        set voxel positions
        :param pos: shape=[n_voxel,3] , in form of [x,y,z]
        :return:
        '''
        self.voxelpos = pos.reshape([-1,3])  # nx3 array,
        self.NVoxel = self.voxelpos.shape[0]
        self.voxelAcceptedMat = np.zeros([self.NVoxel, 3, 3])
        self.voxelHitRatio = np.zeros(self.NVoxel)
        self.voxelIdxStage0 = range(self.voxelpos.shape[0])       # this contains index of the voxel that have not been tried to be reconstructed, used for flood fill process
        self.voxelIdxStage1 = []                    # this contains index  the voxel that have hit ratio > threshold on reconstructed voxel, used for flood fill process
        print("voxelpos shape is {0}".format(self.voxelpos.shape))
    def load_mic(self,fName):
        '''
        load mic file
        set voxelPos,voxelAcceptedEuler, voxelHitRatio,micEuler
        :param fNmame:
        %% Legacy File Format:
          %% Col 0-2 x, y, z
          %% Col 3   1 = triangle pointing up, 2 = triangle pointing down
          %% Col 4 generation number; triangle size = sidewidth /(2^generation number )
          %% Col 5 Phase - 1 = exist, 0 = not fitted
          %% Col 6-8 orientation
          %% Col 9  Confidence
        :return:
        '''

        # self.micData = np.loadtxt(fName,skiprows=skiprows)
        # self.micSideWith =
        # print(self.micData)
        # if self.micData.ndim==1:
        #     micData = self.micData[np.newaxis,:]
        # if self.micData.ndim==0:
        #     raise ValueError('number of dimension of mic file is wrong')
        # self.set_voxel_pos(self.micData[:,:3])
        with open(fName) as f:
            content = f.readlines()
        # print(content[1])
        # print(type(content[1]))
        sw = float(content[0])
        try:
            snp = np.array([[float(i) for i in s.split()] for s in content[1:]])
        except ValueError:
            print 'unknown deliminater'
        if snp.ndim<2:
            raise ValueError('snp dimension if not right, possible empty mic file or empty line in micfile')
        self.micSideWidth = sw
        self.micData = snp
        # set the center of triable to voxel position
        voxelpos = snp[:,:3].copy()
        voxelpos[:,0] = snp[:,0] + 0.5*sw/(2**snp[:,4])
        voxelpos[:,1] = snp[:,1] + 2*(1.5-snp[:,3]) * sw/(2**snp[:,4])/2/np.sqrt(3)
        self.set_voxel_pos(voxelpos)

    def save_mic(self,fName):
        '''
        save mic
        :param fName:
        :return:
        '''
        print('======= saved to mic file: {0} ========'.format(fName))
        np.savetxt(fName, self.micData, fmt=['%.6f'] * 2 + ['%d'] * 4 + ['%.6f'] * (self.micData.shape[1] - 6),
                   delimiter='\t', header=str(self.micSideWidth), comments='')

    def load_fz(self,fName):
        # load FZ.dat file
        # self.FZEuler: n_Orientation*3 array
        #test passed
        self.FZEuler = np.loadtxt(fName)
        # initialize orientation Matrices !!! implement on GPU later
        # self.FZMat = np.zeros([self.FZEuler.shape[0], 3, 3])
        if self.FZEuler.ndim == 1:
            print('wrong format of input orientation, should be nx3 numpy array')
        self.FZMat = RotRep.EulerZXZ2MatVectorized(self.FZEuler/ 180.0 * np.pi)
        # for i in range(self.FZEuler.shape[0]):
        #     self.FZMat[i, :, :] = RotRep.EulerZXZ2Mat(self.FZEuler[i, :] / 180.0 * np.pi).reshape(
        #         [3, 3])
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
        # check is detector size boyond the number int type could hold
        if self.iExpDetImageSize<0 or self.iExpDetImageSize>2147483647:
            raise ValueError("detector image size {0} is wrong, \n\
                             possible too large detector size\n\
                            currently use int type as detector pixel index\n\
                            future implementation use lognlong will solve this issure")

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
        if  not np.any(self.oriMatToSim):
            raise  ValueError(' oriMatToSim not set ')

    def serial_recon_precheck(self):
        pass
    def run_sim(self):
        '''
        example usage:
            S = Reconstructor_GPU()
            S.load_mic('/home/heliu/Dropbox/pycuda/test_recon_one_grain_20180124.txt')
            S.oriMatToSim = RotRep.EulerZXZ2MatVectorized(S.micData[:,6:9])[0,:,:].reshape(-1,3,3)
            S.oriMatToSim = S.oriMatToSim.repeat(S.NVoxel,axis=0)
            print('rotmatrices: {0}'.format(S.oriMatToSim))
            S.run_sim()
            S.print_sim_results()
        :return:
        '''
        # timing tools:
        start = cuda.Event()
        end = cuda.Event()
        start.record()  # start timing
        # initialize Scattering Vectors
        self.sample.getRecipVec()
        self.sample.getGs(self.maxQ)

        # initialize orientation Matrices !!! implement on GPU later

        #initialize device parameters and outputs
        afOrientationMatD = gpuarray.to_gpu(self.oriMatToSim.astype(np.float32))
        afGD = gpuarray.to_gpu(self.sample.Gs.astype(np.float32))
        afVoxelPosD = gpuarray.to_gpu(self.voxelpos.astype(np.float32))
        afDetInfoD = gpuarray.to_gpu(self.afDetInfoH.astype(np.float32))
        if self.oriMatToSim.shape[0]%self.NVoxel !=0:
            raise ValueError('dimension of oriMatToSim should be integer number  of NVoxel')
        NOriPerVoxel = self.oriMatToSim.shape[0]/self.NVoxel
        self.NG = self.sample.Gs.shape[0]
        #output device parameters:
        aiJD = gpuarray.empty(self.NVoxel*NOriPerVoxel*self.NG*2*self.NDet,np.int32)
        aiKD = gpuarray.empty(self.NVoxel*NOriPerVoxel*self.NG*2*self.NDet,np.int32)
        afOmegaD= gpuarray.empty(self.NVoxel*NOriPerVoxel*self.NG*2*self.NDet,np.float32)
        abHitD = gpuarray.empty(self.NVoxel*NOriPerVoxel*self.NG*2*self.NDet,np.bool_)
        aiRotND = gpuarray.empty(self.NVoxel*NOriPerVoxel*self.NG*2*self.NDet, np.int32)
        sim_func = mod.get_function("simulation")


        # start of simulation
        print('============start of simulation ============= \n')
        start.record()  # start timing
        print('nvoxel: {0}, norientation:{1}\n'.format(self.NVoxel,NOriPerVoxel))
        self.sim_precheck()
        sim_func(aiJD, aiKD, afOmegaD, abHitD, aiRotND,\
                 np.int32(self.NVoxel), np.int32(NOriPerVoxel), np.int32(self.NG), np.int32(self.NDet), afOrientationMatD,afGD,\
                 afVoxelPosD,np.float32(self.energy),np.float32(self.etalimit), afDetInfoD,\
                 grid=(self.NVoxel,NOriPerVoxel), block=(self.NG,1,1))
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
    def single_voxel_recon(self, voxelIdx, afFZMatD, NsearchOrien, NIteration=10, BoundStart=0.5):
        # reconstruction of single voxel
        afVoxelPosD = gpuarray.to_gpu(self.voxelpos[voxelIdx, :].astype(np.float32))
        for i in range(NIteration):
            # print(i)
            # print('nvoxel: {0}, norientation:{1}'.format(1, NSearchOrien)
            # update rotation matrix to search
            if i == 0:
                rotMatSearchD = afFZMatD.copy()
            else:
                rotMatSearchD = self.gen_random_matrix(maxMatD, self.NSelect,
                                                       NsearchOrien // self.NSelect + 1, BoundStart * (0.7 ** i))
            afHitRatioH, aiPeakCntH = self.unit_run_hitratio(afVoxelPosD, rotMatSearchD, 1, NsearchOrien)
            maxHitratioIdx = np.argsort(afHitRatioH)[
                             :-(self.NSelect + 1):-1]  # from larges hit ratio to smaller
            maxMatIdx = 9 * maxHitratioIdx.ravel().repeat(9)  # self.NSelect*9
            for jj in range(1, 9):
                maxMatIdx[jj::9] = maxMatIdx[0::9] + jj
            maxHitratioIdxD = gpuarray.to_gpu(maxMatIdx.astype(np.int32))
            maxMatD = gpuarray.take(rotMatSearchD, maxHitratioIdxD)
            # print('max hitratio: {0},maxMat: {1}'.format(afHitRatioH[maxHitratioIdx[0]], maxMat[0, :, :]))
            #print('voxelIdx: {0}, max hitratio: {1}, peakcnt: {2}'.format(voxelIdx,afHitRatioH[maxHitratioIdx[0]],aiPeakCntH[maxHitratioIdx[0]]))
            del rotMatSearchD

        maxMat = maxMatD.get().reshape([-1, 3, 3])
        print('voxelIdx: {0}, max hitratio: {1}, peakcnt: {2},reconstructed euler angle {3}'.format(voxelIdx, afHitRatioH[maxHitratioIdx[0]],
                                                                      aiPeakCntH[maxHitratioIdx[0]],np.array(RotRep.Mat2EulerZXZ(maxMat[0, :, :])) / np.pi * 180))
        self.voxelAcceptedMat[voxelIdx, :, :] = RotRep.Orien2FZ(maxMat[0, :, :], 'Hexagonal')[0]
        self.voxelHitRatio[voxelIdx] = afHitRatioH[maxHitratioIdx[0]]
        del afVoxelPosD
    def unit_run_hitratio(self,afVoxelPosD, rotMatSearchD, NVoxel, NOrientation):
        '''
        CAUTION: strange bug, if NVoxel*NOrientation is too small( < 200), memery access erro will occur.
        :param afVoxelPosD: NVoxel*3
        :param rotMatSearchD: NVoxel*NOrientation*9
        :param NVoxel:
        :param NOrientation:
        :return:
        '''
        # if not (isinstance(afVoxelPosD, pycuda.gpuarray.GPUArray) and isinstance(rotMatSearchD, pycuda.gpuarray.GPUArray)):
        #     raise TypeError('afVoxelPosD and rotMatSearchD should be gpuarray, not allocator or other.')
        # if NVoxel*NOrientation < 350:
        #     print(" number of input may be too small")
        # if NVoxel==0 or NOrientation==0:
        #     print('number of voxel {0} and orientation {1} is not in right form'.format(NVoxel,NOrientation))
        #     return 0,0
        aiJD = cuda.mem_alloc(NVoxel * NOrientation * self.NG * 2 * self.NDet * np.int32(0).nbytes)
        aiKD = cuda.mem_alloc(NVoxel * NOrientation * self.NG * 2 * self.NDet * np.int32(0).nbytes)
        afOmegaD = cuda.mem_alloc(NVoxel * NOrientation * self.NG * 2 * self.NDet * np.float32(0).nbytes)
        abHitD = cuda.mem_alloc(NVoxel * NOrientation * self.NG * 2 * self.NDet * np.bool_(0).nbytes)
        aiRotND = cuda.mem_alloc(NVoxel * NOrientation * self.NG * 2 * self.NDet * np.int32(0).nbytes)
        # kernel calls
        self.sim_func(aiJD, aiKD, afOmegaD, abHitD, aiRotND, \
                      np.int32(NVoxel), np.int32(NOrientation), np.int32(self.NG), np.int32(self.NDet),
                      rotMatSearchD, self.afGD,
                      afVoxelPosD, np.float32(self.energy), np.float32(self.etalimit), self.afDetInfoD,
                      grid=(NVoxel, NOrientation), block=(self.NG, 1, 1))
        afHitRatioD = cuda.mem_alloc(NVoxel * NOrientation * np.float32(0).nbytes)
        aiPeakCntD = cuda.mem_alloc(NVoxel * NOrientation * np.int32(0).nbytes)
        NBlock = 256
        self.hitratio_func(np.int32(NVoxel), np.int32(NOrientation), np.int32(self.NG),
                           self.afDetInfoD, self.acExpDetImages, self.aiDetStartIdxD, np.int32(self.NDet),
                           np.int32(self.NRot),
                           aiJD, aiKD, aiRotND, abHitD,
                           afHitRatioD, aiPeakCntD,
                           block=(NBlock, 1, 1), grid=((NVoxel * NOrientation - 1) // NBlock + 1, 1))
        # print('finish sim')
        # memcpy_dtoh
        context.synchronize()
        afHitRatioH = np.empty(NVoxel*NOrientation,np.float32)
        aiPeakCntH = np.empty(NVoxel*NOrientation, np.int32)
        cuda.memcpy_dtoh(afHitRatioH, afHitRatioD)
        cuda.memcpy_dtoh(aiPeakCntH, aiPeakCntD)
        aiJD.free()
        aiKD.free()
        afOmegaD.free()
        abHitD.free()
        aiRotND.free()
        afHitRatioD.free()
        aiPeakCntD.free()
        return afHitRatioH, aiPeakCntH

    def profile_recon_layer(self):
        '''
                ==================working version =========================
                Todo:
                    fix two reconstructed orientation:
                        simulate the two different reconstructed oreintation and compare peaks
                        compute their misoritentation
                serial reconstruct orientation in a layer, loaded in mic file
                example usage:
                R = Reconstructor_GPU()
                Not Implemented: Setup R experimental parameters
                R.searchBatchSize = 20000  # number of orientations to search per GPU call
                R.load_mic('/home/heliu/work/I9_test_data/FIT/DataFiles/Ti_SingleGrainFit1_.mic.LBFS')
                R.load_fz('/home/heliu/work/I9_test_data/FIT/DataFiles/HexFZ.dat')
                R.load_exp_data('/home/heliu/work/I9_test_data/Integrated/S18_z1_', 6)
                R.serial_recon_layer()
                :return:
                '''
        ############## reform for easy reading #############
        ############ added generate random in GPU #########3
        ############# search parameters ######################
        # try adding multiple stage search, first fz file, then generate random around max hitratio
        # self.load_mic('/home/heliu/work/I9_test_data/FIT/DataFiles/Ti_Fit1_.mic.LBFS')
        self.load_mic('test_recon_one_grain_20180124.txt')
        # self.load_mic('/home/heliu/work/I9_test_data/FIT/DataFiles/Ti_SingleGrainFit1_.mic.LBFS')
        # self.load_mic('/home/heliu/work/I9_test_data/FIT/test_recon.mic.LBFS')
        self.load_fz('/home/heliu/work/I9_test_data/FIT/DataFiles/HexFZ.dat')

        #self.load_exp_data('/home/heliu/work/I9_test_data/Integrated/S18_z1_', 6)
        #self.expData[:, 2:4] = self.expData[:, 2:4] / 4  # half the detctor size, to rescale real data
        self.expData = np.array([[1,2,3,4]])
        self.cp_expdata_to_gpu()

        # setup serial Reconstruction rotMatCandidate
        self.FZMatH = np.empty([self.searchBatchSize, 3, 3])
        if self.searchBatchSize > self.FZMat.shape[0]:
            self.FZMatH[:self.FZMat.shape[0], :, :] = self.FZMat
            self.FZMatH[self.FZMat.shape[0]:, :, :] = FZfile.generate_random_rot_mat(
                self.searchBatchSize - self.FZMat.shape[0])
        else:
            raise ValueError(" search batch size less than FZ file size, please increase search batch size")

        # initialize device parameters and outputs
        self.afGD = gpuarray.to_gpu(self.sample.Gs.astype(np.float32))
        self.afDetInfoD = gpuarray.to_gpu(self.afDetInfoH.astype(np.float32))
        afFZMatD = gpuarray.to_gpu(self.FZMatH.astype(np.float32))  # no need to modify during process

        # timing tools:
        start = cuda.Event()
        end = cuda.Event()

        print('==========start of reconstruction======== \n')
        start.record()  # start timing
        pr = cProfile.Profile()
        pr.enable()
        for voxelIdx in range(self.NVoxel):
            self.single_voxel_recon(voxelIdx, afFZMatD,self.searchBatchSize, NIteration=10)
        pr.disable()
        s = StringIO.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print s.getvalue()
        print('===========end of reconstruction========== \n')
        end.record()  # end timing
        end.synchronize()
        secs = start.time_till(end) * 1e-3
        print("SourceModule time {0} seconds.".format(secs))
        # save roconstruction result
        self.micData[:, 6:9] = RotRep.Mat2EulerZXZVectorized(self.voxelAcceptedMat) / np.pi * 180
        self.micData[:, 9] = self.voxelHitRatio
        self.save_mic('test_recon_one_grain_gpu_random_out.txt')
    def serial_recon_layer(self):
        '''
        ==================working version =========================
        Todo:
            fix two reconstructed orientation:
                simulate the two different reconstructed oreintation and compare peaks
                compute their misoritentation
        serial reconstruct orientation in a layer, loaded in mic file
        example usage:
        R = Reconstructor_GPU()
        Not Implemented: Setup R experimental parameters
        R.searchBatchSize = 20000  # number of orientations to search per GPU call
        R.load_mic('/home/heliu/work/I9_test_data/FIT/DataFiles/Ti_SingleGrainFit1_.mic.LBFS')
        R.load_fz('/home/heliu/work/I9_test_data/FIT/DataFiles/HexFZ.dat')
        R.load_exp_data('/home/heliu/work/I9_test_data/Integrated/S18_z1_', 6)
        R.serial_recon_layer()
        :return:
        '''
        ############## reform for easy reading #############
        ############ added generate random in GPU #########3
        ############# search parameters ######################
        # try adding multiple stage search, first fz file, then generate random around max hitratio
        #self.load_mic('/home/heliu/work/I9_test_data/FIT/DataFiles/Ti_Fit1_.mic.LBFS')
        self.load_mic('test_recon_one_grain_20180124.txt')
        #self.load_mic('/home/heliu/work/I9_test_data/FIT/DataFiles/Ti_SingleGrainFit1_.mic.LBFS')
        #self.load_mic('/home/heliu/work/I9_test_data/FIT/test_recon.mic.LBFS')
        self.load_fz('/home/heliu/work/I9_test_data/FIT/DataFiles/HexFZ.dat')

        self.load_exp_data('/home/heliu/work/I9_test_data/Integrated/S18_z1_', 6)
        self.expData[:, 2:4] = self.expData[:, 2:4] / 4  # half the detctor size, to rescale real data
        #self.expData = np.array([[1,2,3,4]])
        self.cp_expdata_to_gpu()

        # setup serial Reconstruction rotMatCandidate
        self.FZMatH = np.empty([self.searchBatchSize,3,3])
        if self.searchBatchSize > self.FZMat.shape[0]:
            self.FZMatH[:self.FZMat.shape[0], :, :] = self.FZMat
            self.FZMatH[self.FZMat.shape[0]:,:,:] = FZfile.generate_random_rot_mat(self.searchBatchSize - self.FZMat.shape[0])
        else:
            raise ValueError(" search batch size less than FZ file size, please increase search batch size")

        # initialize device parameters and outputs
        self.afGD = gpuarray.to_gpu(self.sample.Gs.astype(np.float32))
        self.afDetInfoD = gpuarray.to_gpu(self.afDetInfoH.astype(np.float32))
        afFZMatD = gpuarray.to_gpu(self.FZMatH.astype(np.float32))          # no need to modify during process

        # timing tools:
        start = cuda.Event()
        end = cuda.Event()

        print('==========start of reconstruction======== \n')
        start.record()  # start timing

        for voxelIdx in range(self.NVoxel):
            self.single_voxel_recon(voxelIdx, afFZMatD,self.searchBatchSize)
        print('===========end of reconstruction========== \n')
        end.record()  # end timing
        end.synchronize()
        secs = start.time_till(end) * 1e-3
        print("SourceModule time {0} seconds.".format(secs))
        # save roconstruction result
        self.micData[:,6:9] = RotRep.Mat2EulerZXZVectorized(self.voxelAcceptedMat)/np.pi*180
        self.micData[:,9] = self.voxelHitRatio
        self.save_mic('test_recon_one_grain_gpu_random_out.txt')
    def flood_fill(self):
        '''
        flood fill all the voxel with confidence level lower than self.floodFillAccptThreshold
        :return:
        '''
        print('====================== entering flood fill ===================================')
        # select voxels to conduct filling
        print('indexstage0 {0}'.format(len(self.voxelIdxStage0)))
        lFloodFillIdx = list(np.where(self.voxelHitRatio<self.floodFillSelectThreshold)[0])
        if not lFloodFillIdx:
            return 0
        idxToAccept = []
        print(len(lFloodFillIdx))
        # try orientation to fill on all other voxels
        for i in range((len(lFloodFillIdx)-1)//self.floodFillNumberVoxel+1):     #make sure memory is enough

            idxTmp = lFloodFillIdx[i*self.floodFillNumberVoxel: (i+1)*self.floodFillNumberVoxel]
            if len(idxTmp)==0:
                print('no voxel to reconstruct')
                return 0
            elif len(idxTmp)<350:
                idxTmp = idxTmp * (349/len(idxTmp)+1)
            print('i: {0}, idxTmp: {1}'.format(i,len(idxTmp)))
            #afVoxelPosH = self.voxelpos[idxTmp,:].astype(np.float32)
            #afVoxelPosD = cuda.mem_alloc(afVoxelPosH.nbytes)
            #cuda.memcpy_htod(afVoxelPosD, afVoxelPosH)
            afVoxelPosD = gpuarray.to_gpu(self.voxelpos[idxTmp,:].astype(np.float32))
            rotMatH = self.voxelAcceptedMat[self.voxelIdxStage0[0], :, :].reshape([-1, 3, 3]).repeat(len(idxTmp),
                                                                                                     axis=0).astype(
                np.float32)
            #rotMatSearchD = cuda.mem_alloc(rotMatH.nbytes)
            #cuda.memcpy_htod(rotMatSearchD, rotMatH)
            rotMatSearchD = gpuarray.to_gpu(rotMatH)
            # call kernel
            #print(rotMatH[0,:,:])
            #print('before call kernel')
            afFloodHitRatioH, aiFloodPeakCntH = self.unit_run_hitratio(afVoxelPosD,rotMatSearchD,len(idxTmp),1)
            #time.sleep(1)
            # afVoxelPosD.free()
            # rotMatSearchD.free()
            #print('after call kernel')
            # add voxel index with hitratio larger than threshold
            idxToAccept.append(np.array(idxTmp)[afFloodHitRatioH>self.floodFillAccptThreshold])
            del afVoxelPosD
            del rotMatSearchD
        #print('idxToAccept: {0}'.format(idxToAccept))
        idxToAccept = np.concatenate(idxToAccept).ravel()
        # local optimize each voxel
        for i, idxTmp in enumerate(idxToAccept):
            # remove from stage 0
            try:
                self.voxelIdxStage0.remove(idxTmp)
            except ValueError:
                pass
            # do one time search:
            rotMatSearchD = self.gen_random_matrix(gpuarray.to_gpu(self.voxelAcceptedMat[self.voxelIdxStage0[0], :, :].astype(np.float32)),
                                                   1, self.floodFillNumberAngle, self.floodFillRandomRange)
            self.single_voxel_recon(idxTmp,rotMatSearchD,self.floodFillNumberAngle, NIteration=self.floodFillNIteration, BoundStart=self.floodFillRandomRange)
            del rotMatSearchD
        #print('fill {0} voxels'.format(idxToAccept.shape))
        print('++++++++++++++++++ leaving flood fill +++++++++++++++++++++++')
        return 1
    def serial_recon_multi_stage(self):
        # add multiple stage in serial reconstruction:
        # Todo:
        #   add flood fill
        #   add post stage check, refill reconstructed orientations to each voxel.
        ############# search parameters ######################

        ############## reform for easy reading #############
        ############ added generate random in GPU #########3
        ############# search parameters ######################
        # try adding multiple stage search, first fz file, then generate random around max hitratio
        # self.load_mic('/home/heliu/work/I9_test_data/FIT/DataFiles/Ti_Fit1_.mic.LBFS')
        self.load_mic('Ti7_S18_whole_layer.mic')
        #self.load_mic('partial_layer_i9.mic')
        # self.load_mic('/home/heliu/work/I9_test_data/FIT/DataFiles/Ti_SingleGrainFit1_.mic.LBFS')
        # self.load_mic('/home/heliu/work/I9_test_data/FIT/test_recon.mic.LBFS')
        self.load_fz('/home/heliu/work/I9_test_data/FIT/DataFiles/HexFZ.dat')

        self.load_exp_data('/home/heliu/work/I9_test_data/Integrated/S18_z1_', 6)
        self.expData[:, 2:4] = self.expData[:, 2:4] / 4  # half the detctor size, to rescale real data
        #self.expData = np.array([[1,2,3,4]])
        self.cp_expdata_to_gpu()

        # setup serial Reconstruction rotMatCandidate
        self.FZMatH = np.empty([self.searchBatchSize, 3, 3])
        if self.searchBatchSize > self.FZMat.shape[0]:
            self.FZMatH[:self.FZMat.shape[0], :, :] = self.FZMat
            self.FZMatH[self.FZMat.shape[0]:, :, :] = FZfile.generate_random_rot_mat(
                self.searchBatchSize - self.FZMat.shape[0])
        else:
            raise ValueError(" search batch size less than FZ file size, please increase search batch size")

        # initialize device parameters and outputs
        self.afGD = gpuarray.to_gpu(self.sample.Gs.astype(np.float32))
        self.afDetInfoD = gpuarray.to_gpu(self.afDetInfoH.astype(np.float32))
        afFZMatD = gpuarray.to_gpu(self.FZMatH.astype(np.float32))  # no need to modify during process

        # timing tools:
        start = cuda.Event()
        end = cuda.Event()
        print('==========start of reconstruction======== \n')
        start.record()  # start timing
        while self.voxelIdxStage0:
            # start of simulation
            voxelIdx = self.voxelIdxStage0[0]
            self.single_voxel_recon(voxelIdx, afFZMatD,self.searchBatchSize)
            if self.voxelHitRatio[voxelIdx] > self.floodFillStartThreshold:
                self.flood_fill()
            try:
                self.voxelIdxStage0.remove(voxelIdx)
            except ValueError:
                pass

        print('===========end of reconstruction========== \n')
        end.record()  # end timing
        end.synchronize()
        secs = start.time_till(end) * 1e-3
        print("SourceModule time {0} seconds.".format(secs))
        # save roconstruction result
        self.micData[:,6:9] = RotRep.Mat2EulerZXZVectorized(self.voxelAcceptedMat)/np.pi*180
        self.micData[:,9] = self.voxelHitRatio
        self.save_mic('Ti7_S18_whole_layer_GPU_output.mic')
        #self.save_mic('partial_layer_gpu_output_reform.mic')
    def print_sim_results(self):
        # print(self.aJH)
        # print(self.aKH)
        # print(self.aiRotNH)
        # print(self.aOmegaH)
        # print(self.bHitH)
        NOriPerVoxel = (self.oriMatToSim.shape[0] / self.NVoxel)
        for i,hit in enumerate(self.bHitH):
            if hit:
                print('VoxelIdx:{5}, Detector: {0}, J: {1}, K: {2},,RotN:{3}, Omega: {4}'.format(i%self.NDet, self.aJH[i],self.aKH[i],self.aiRotNH[i], self.aOmegaH[i],i//(NOriPerVoxel*self.NG*2*self.NDet)))
    def gen_random_matrix(self, matInD, NMatIn, NNeighbour, bound):
        '''
        generate orientations around cetrain rotation matrix
        :param matInD: gpuarray
        :param NMatIn: number of input mat
        :param NNeighbour:
        :param  bound, rangdom angle range.
        :return:
        '''
        # if NMatIn<=0 or NNeighbour<=0 or bound<=0:
        #     raise ValueError('number of matin {0} or nneighbour {1} or bound {2} is not right')
        #if isinstance(matInD,pycuda.gpuarray.GPUArray) or isinstance(matInD, pycuda._driver.DeviceAllocation):
        eulerD = gpuarray.empty(NMatIn*3, np.float32)
        matOutD = gpuarray.empty(NMatIn*NNeighbour*9, np.float32)
        NBlock = 128

        self.mat_to_euler_ZXZ(matInD, eulerD, np.int32(NMatIn), block=(NBlock, 1, 1), grid=((NMatIn-1) // NBlock + 1, 1))
        afRandD = self.randomGenerator.gen_uniform(NNeighbour * NMatIn * 3, np.float32)
        self.rand_mat_neighb_from_euler(eulerD, matOutD, afRandD, np.float32(bound), grid=(NNeighbour, 1), block=(NMatIn, 1, 1))
        return matOutD
import matplotlib.pyplot as plt
class SimPlot():
    import matplotlib.pyplot as plt
    def __init__(self):
        self.data = np.zeros([2,6]) # format[voxelIdx,detectorIdx,J,K,RotN,Omega
        self.voxelID = np.unique(self.data[:,0])
    def plot_jk(self,j,k):
        '''
        plot only on j and k value
        :param
        :return:
        '''
        plt.plot(j,k,'.')
    def plot_overlap_single_voxel(self,voxelIdx):
        self.plot_jk(self.data[self.data[:,0]==voxelIdx,:][:,2],self.data[self.data[:,0]==voxelIdx,:][:,3])




def profile_recon_layer():
    S = Reconstructor_GPU()
    pr = cProfile.Profile()
    pr.enable()
    S.serial_recon_layer()
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
class Simulator_old():
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

def test_floodfill():
    S = Reconstructor_GPU()
    S.load_mic('/home/heliu/Dropbox/pycuda/test_recon_one_grain_20180124.txt')
    S.flood_fill()
if __name__ == "__main__":
    S = Reconstructor_GPU()
    #S.profile_recon_layer()
    S.serial_recon_multi_stage()
    #S.serial_recon_layer_backup()
    #test_gpuarray_take()
    #test_mat2eulerzxz()
    #test_eulerzxz2mat()
    #test_rand_amt_neighb()
    #test_random()
    #profile_recon_layer()
    #test_floodfill()
    #S = Reconstructor_GPU()
    # S.flood_fill()
    # # S.oriMatToSim = RotRep.EulerZXZ2MatVectorized(S.micData[:,6:9])
    # # S.run_sim()
    # # S.print_sim_results()
    #S.serial_recon_multi_stage()
    #
    #S.save_mic('test_save_mic.txt')
    # print('voxel pos: {0}, micEuler: {1}.'.format(S.voxelpos,S.micEuler))

    #print(calculate_misoren_euler_zxz(np.array([10.1237, 75.4599, 340.791]),np.array([174.956, 55.8283, 182.94])))
