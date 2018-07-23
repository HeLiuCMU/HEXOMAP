'''
stores devece code that runs on GPU
He Liu CMU
20180219
'''
from pycuda.compiler import SourceModule
mod = SourceModule("""
#include <stdio.h>
const float PI = 3.14159265359;
const float HALFPI = 0.5*PI;
texture<unsigned char, cudaTextureType3D, cudaReadModeElementType> tcExpData;
texture<float, cudaTextureType2D, cudaReadModeElementType> tfG;  // texture to store scattering vectors;
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
  //float fCosTheta = sqrt( 1.f - fSinTheta * fSinTheta);
  float fCosChi = aScatteringVec[2] / fScatteringVecMag;             // Tilt angle of G relative to z-axis
  float fSinChi = sqrt( 1.f - fCosChi * fCosChi );
  //float fSinChiLaue = sin( fBeamDeflectionChiLaue );     // ! Tilt angle of k_i (+ means up)
  //float fCosChiLaue = cos( fBeamDeflectionChiLaue );

  if( fabsf( fSinTheta ) <= fabsf( fSinChi) )
  {
	//float fPhi = acosf(fSinTheta / fSinChi);
	float fSinPhi = sin(acosf(fSinTheta / fSinChi));
	//float fCosTheta = sqrt( 1.f - fSinTheta * fSinTheta);
	fEta = asinf(fSinChi * fSinPhi / sqrt( 1.f - fSinTheta * fSinTheta));
	// [-pi:pi]: angle to bring G to nominal position along +y-axis
	float fDeltaOmega0 = atan2f( aScatteringVec[0], aScatteringVec[1]);

	//  [0:pi/2] since arg >0: phi goes from above to Bragg angle
	float fDeltaOmega_b1 = asinf( fSinTheta/fSinChi );

	//float fDeltaOmega_b2 = PI -  fDeltaOmega_b1;

	fOmegaRes1 = fDeltaOmega_b1 + fDeltaOmega0;  // oScatteringVec.m_fY > 0
	fOmegaRes2 = PI - fDeltaOmega_b1 + fDeltaOmega0;  // oScatteringVec.m_fY < 0
    	//fOmegaRes1 -= 2.f * PI*(trunc(fOmegaRes1/PI));
    	//fOmegaRes2 -= 2.f * PI*(trunc(fOmegaRes2/PI));
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


__device__ bool GetPeak_s(int &iJ1,int &iJ2,int &iK1, int &iK2,int &bHit1,int &bHit2,
		const float &fOmegaRes1, const float &fOmegaRes2,
		const float &fTwoTheta, const float &fEta,const float &fChi,const float &fEtaLimit,
		const float *afVoxelPos,const float *afDetInfo){
	/*
	 *  modified to use shared memory, failed to improve speed.
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
		bHit1 = 0;
		bHit2 = 0;
		return false;
	}
	else if(fEta>fEtaLimit){
		bHit1 = 0;
		bHit2 = 0;
		return false;
	}
	bHit1 = 0;
	bHit2 = 0;
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
			bHit1 = 1;
		}
		else{
			bHit1 = 0;
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
			bHit2 = 1;
		}
		else{
			bHit2 = 0;
		}
	}
	return true;

}

__device__ bool GetPeak(int &iJ1,int &iJ2,int &iK1, int &iK2,bool &bHit1,bool &bHit2,
		const float &fOmegaRes1, const float &fOmegaRes2,
		const float &fTwoTheta, const float &fEta,const float &fChi,const float &fEtaLimit,
		const float *afVoxelPos,const float* __restrict__ afDetInfo){
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
	bHit1 = false;
	bHit2 = false;
	if (fChi>= 0.5*PI || fEta>fEtaLimit){
		return false;
	}
	float fVoxelPosX,fVoxelPosY,fVoxelPosZ,fDist,fAngleNormScatter;
	int iJ,iK;
	float afScatterDir[3]; //scattering direction
	float afInterPos[3];
	if ((-HALFPI<=fOmegaRes1) && (fOmegaRes1<=HALFPI)){
		fVoxelPosX = cos(fOmegaRes1)*afVoxelPos[0] - sin(fOmegaRes1)*afVoxelPos[1];
		fVoxelPosY = cos(fOmegaRes1)*afVoxelPos[1] + sin(fOmegaRes1)*afVoxelPos[0];
		fVoxelPosZ = afVoxelPos[2];
		fDist = afDetInfo[7]*(afDetInfo[4] - fVoxelPosX)
				+ afDetInfo[8]*(afDetInfo[5] - fVoxelPosY)
				+ afDetInfo[9]*(afDetInfo[6] - fVoxelPosZ);
		afScatterDir[0] = cos(fTwoTheta);
		afScatterDir[1] = sin(fTwoTheta) * sin(fEta);
		afScatterDir[2] = sin(fTwoTheta) * cos(fEta);
		fAngleNormScatter = afDetInfo[7]*afScatterDir[0]
		                          + afDetInfo[8]*afScatterDir[1]
		                          + afDetInfo[9]*afScatterDir[2];
		afInterPos[0] = fDist / fAngleNormScatter * afScatterDir[0] + fVoxelPosX;
		afInterPos[1] = fDist / fAngleNormScatter * afScatterDir[1] + fVoxelPosY;
		afInterPos[2] = fDist / fAngleNormScatter * afScatterDir[2] + fVoxelPosZ;
		iJ = floor((afDetInfo[10]*(afInterPos[0]-afDetInfo[4])
				+ afDetInfo[11]*(afInterPos[1]-afDetInfo[5])
				+ afDetInfo[12]*(afInterPos[2]-afDetInfo[6]) )/afDetInfo[2]);
		iK = floor((afDetInfo[13]*(afInterPos[0]-afDetInfo[4])
				+ afDetInfo[14]*(afInterPos[1]-afDetInfo[5])
				+ afDetInfo[15]*(afInterPos[2]-afDetInfo[6]) )/afDetInfo[3]);
		bHit1 = (0<=iJ ) && (iJ<afDetInfo[0]) && (0<=iK) && (iK<afDetInfo[1]);
		iJ1 = iJ;
		iK1 = iK;
	}

	if ((-HALFPI<=fOmegaRes2) && (fOmegaRes2<=HALFPI)){
		fVoxelPosX = cos(fOmegaRes2)*afVoxelPos[0] - sin(fOmegaRes2)*afVoxelPos[1];
		fVoxelPosY = cos(fOmegaRes2)*afVoxelPos[1] + sin(fOmegaRes2)*afVoxelPos[0];
		fVoxelPosZ = afVoxelPos[2];
		fDist = afDetInfo[7]*(afDetInfo[4] - fVoxelPosX)
				+ afDetInfo[8]*(afDetInfo[5] - fVoxelPosY)
				+ afDetInfo[9]*(afDetInfo[6] - fVoxelPosZ);
		afScatterDir[0] = cos(fTwoTheta);
		afScatterDir[1] = sin(fTwoTheta) * sin(-fEta);  // caution: -fEta!!!!!!
		afScatterDir[2] = sin(fTwoTheta) * cos(-fEta);  // caution: -fEta!!!!!!
		fAngleNormScatter = afDetInfo[7]*afScatterDir[0]
		                          + afDetInfo[8]*afScatterDir[1]
		                          + afDetInfo[9]*afScatterDir[2];
		afInterPos[0] = fDist / fAngleNormScatter * afScatterDir[0] + fVoxelPosX;
		afInterPos[1] = fDist / fAngleNormScatter * afScatterDir[1] + fVoxelPosY;
		afInterPos[2] = fDist / fAngleNormScatter * afScatterDir[2] + fVoxelPosZ;
		iJ = floor((afDetInfo[10]*(afInterPos[0]-afDetInfo[4])
				+ afDetInfo[11]*(afInterPos[1]-afDetInfo[5])
				+ afDetInfo[12]*(afInterPos[2]-afDetInfo[6]) )/afDetInfo[2]);
		iK = floor((afDetInfo[13]*(afInterPos[0]-afDetInfo[4])
				+ afDetInfo[14]*(afInterPos[1]-afDetInfo[5])
				+ afDetInfo[15]*(afInterPos[2]-afDetInfo[6]) )/afDetInfo[3]);
		bHit2 = (0<=iJ ) && (iJ<afDetInfo[0]) && (0<=iK) && (iK<afDetInfo[1]);
		iJ2 = iJ;
		iK2 = iK;
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
		const float* afOrientation,const float* afVoxelPos,
		const float fBeamEnergy, const float fEtaLimit, const float* __restrict__ afDetInfo){
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
		    afScatteringVec[i] += afOrientation[blockIdx.x*gridDim.y*9+blockIdx.y*9+i*3+j]*tex2D(tfG,(float)j,(float)threadIdx.x);
			//afScatteringVec[i] += afOrientation[blockIdx.x*gridDim.y*9+blockIdx.y*9+i*3+j]*afG[threadIdx.x*3+j];
		    //printf("%d,%d: %f. ||",i,j,afOrientation[blockIdx.x*gridDim.y*9+blockIdx.y*9+i*3+j]);
		}
	}
	//printf("%f,%f,%f ||",afScatteringVec[0],afScatteringVec[1],afScatteringVec[1]);
	if(GetScatteringOmegas( fOmegaRes1, fOmegaRes2, fTwoTheta, fEta, fChi , afScatteringVec,fBeamEnergy)){
		int i = blockIdx.x*gridDim.y*blockDim.x*2*iNDet+ blockIdx.y*blockDim.x*2*iNDet + threadIdx.x*2*iNDet;
		for(int iDetIdx=0;iDetIdx<iNDet;iDetIdx++){
			GetPeak(aiJ[i+iDetIdx],aiJ[i+iDetIdx+iNDet],aiK[i+iDetIdx],aiK[i+iDetIdx+iNDet],
					abHit[i+iDetIdx],abHit[i+iDetIdx+iNDet],
					fOmegaRes1, fOmegaRes2,fTwoTheta,
					fEta,fChi,fEtaLimit,afVoxelPos+blockIdx.x*3,afDetInfo+19*iDetIdx);
			//printf("%f %f %f || ", (afVoxelPos+blockIdx.x*3)[0],(afVoxelPos+blockIdx.x*3)[1],(afVoxelPos+blockIdx.x*3)[2]);
			//printf("blockIdx: %d || ",blockIdx.x);
			//	////////assuming they are using the same rotation number in all the detectors!!!!!!!///////////////////
			afOmega[i+iDetIdx] = fOmegaRes1;
			aiRotN[i+iDetIdx] = floor((fOmegaRes1-afDetInfo[17])/(afDetInfo[18]-afDetInfo[17])*(afDetInfo[16]-1));
				//printf("iJ1: %d, iK1 %d, fOmega1 %f, iRotN %d",aiJ[i],aiK[i],afOmega[i],aiRotN[i]);
			//}
			//if(abHit[i+iDetIdx+iNDet]){
			afOmega[i+iDetIdx+iNDet] = fOmegaRes2;
			aiRotN[i+iDetIdx+iNDet] = floor((fOmegaRes2-afDetInfo[17])/(afDetInfo[18]-afDetInfo[17])*(afDetInfo[16]-1));
			//	//printf("iJ2: %d, iK2 %d, fOmega2 %f, iRotN %d ",aiJ[i+1],aiK[i+1],afOmega[i+1],aiRotN[i+1]);
			//}
			//if(abHit[i+iDetIdx]){
			//	////////assuming they are using the same rotation number in all the detectors!!!!!!!///////////////////
			//	afOmega[i+iDetIdx] = fOmegaRes1;
			//	aiRotN[i+iDetIdx] = floor((fOmegaRes1-afDetInfo[17])/(afDetInfo[18]-afDetInfo[17])*(afDetInfo[16]-1));
				//printf("iJ1: %d, iK1 %d, fOmega1 %f, iRotN %d",aiJ[i],aiK[i],afOmega[i],aiRotN[i]);
			//}
			//if(abHit[i+iDetIdx+iNDet]){
			 //   afOmega[i+iDetIdx+iNDet] = fOmegaRes2;
			//	aiRotN[i+iDetIdx+iNDet] = floor((fOmegaRes2-afDetInfo[17])/(afDetInfo[18]-afDetInfo[17])*(afDetInfo[16]-1));
			//	//printf("iJ2: %d, iK2 %d, fOmega2 %f, iRotN %d ",aiJ[i+1],aiK[i+1],afOmega[i+1],aiRotN[i+1]);
			//}
		}
	}
}

__global__ void simulation_backup_20180206(int *aiJ, int *aiK, float *afOmega, bool *abHit,int *aiRotN,
		const int iNVoxel, const int iNOrientation, const int iNG, const int iNDet,
		const float* __restrict__ afOrientation,const float* __restrict__ afG,const float* __restrict__ afVoxelPos,
		const float fBeamEnergy, const float fEtaLimit, const float* __restrict__ afDetInfo){
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
		    afScatteringVec[i] += afOrientation[blockIdx.x*gridDim.y*9+blockIdx.y*9+i*3+j]*tex2D(tfG,(float)j,(float)threadIdx.x);
			//afScatteringVec[i] += afOrientation[blockIdx.x*gridDim.y*9+blockIdx.y*9+i*3+j]*afG[threadIdx.x*3+j];
		    //printf("%d,%d: %f. ||",i,j,afOrientation[blockIdx.x*gridDim.y*9+blockIdx.y*9+i*3+j]);
		}
	}
	//printf("%f,%f,%f ||",afScatteringVec[0],afScatteringVec[1],afScatteringVec[1]);
	if(GetScatteringOmegas( fOmegaRes1, fOmegaRes2, fTwoTheta, fEta, fChi , afScatteringVec,fBeamEnergy)){
		int i = blockIdx.x*gridDim.y*blockDim.x*2*iNDet+ blockIdx.y*blockDim.x*2*iNDet + threadIdx.x*2*iNDet;
		for(int iDetIdx=0;iDetIdx<iNDet;iDetIdx++){
			GetPeak(aiJ[i+iDetIdx],aiJ[i+iDetIdx+iNDet],aiK[i+iDetIdx],aiK[i+iDetIdx+iNDet],
					abHit[i+iDetIdx],abHit[i+iDetIdx+iNDet],
					fOmegaRes1, fOmegaRes2,fTwoTheta,
					fEta,fChi,fEtaLimit,afVoxelPos+blockIdx.x*3,afDetInfo+19*iDetIdx);
			//printf("%f %f %f || ", (afVoxelPos+blockIdx.x*3)[0],(afVoxelPos+blockIdx.x*3)[1],(afVoxelPos+blockIdx.x*3)[2]);
			//printf("blockIdx: %d || ",blockIdx.x);
			if(abHit[i+iDetIdx]){
				////////assuming they are using the same rotation number in all the detectors!!!!!!!///////////////////
				afOmega[i+iDetIdx] = fOmegaRes1;
				aiRotN[i+iDetIdx] = floor((fOmegaRes1-afDetInfo[17])/(afDetInfo[18]-afDetInfo[17])*(afDetInfo[16]-1));
				//printf("iJ1: %d, iK1 %d, fOmega1 %f, iRotN %d",aiJ[i],aiK[i],afOmega[i],aiRotN[i]);
			}
			if(abHit[i+iDetIdx+iNDet]){
			    afOmega[i+iDetIdx+iNDet] = fOmegaRes2;
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


__global__ void hitratio_multi_detector_old_backup(const int iNVoxel,const int iNOrientation,const int iNG,
		const float* afDetInfo,const char* __restrict__ acExpDetImages, const int* aiDetStartIdx, const int iNDet,const int iNRot,
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

__global__ void hitratio_multi_detector_backup_20180206(const int iNVoxel,const int iNOrientation,const int iNG,
		const float* __restrict__ afDetInfo,const char* __restrict__ acExpDetImages, const int* __restrict__ aiDetStartIdx, const int iNDet,const int iNRot,
		const int* aiJ, const int* aiK,const int* aiRotN, const bool* abHit,
		float* afHitRatio, int* aiPeakCnt){
	/*
	 * this version runs faster than hitratio_multi_detector_old_backup, no error so far.
	 * now 100x100 takes 15s
	 * If set block=(16,1,1) 100x100 will reach 25s.
	 * 100x100 31s, original: 43s, improved!
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
	bool allTrue0; // if a simulated peak hit all detector, allTrue0 = true;
	bool allTrue1; // if a simulated peak overlap with all expimages on all detector, allTrue1 = true;
	int k;
	int iPeakCnt;
	float fHitRatio;
	int idx;
	if(i<iNVoxel*iNOrientation){
	    iPeakCnt = 0;
	    fHitRatio = 0;
		for(int j=0;j<iNG*2;j++){
		    //printf("j: %d ||",j);
			allTrue0 = true;
			allTrue1 = true;
			k = 0;
			while(allTrue0 && k<iNDet){
				allTrue0 *= abHit[i*iNG*2*iNDet+j*iNDet+k];
				k += 1;
			}
			allTrue1 = allTrue0;
            k = 0;
            while(allTrue1 && k<iNDet){
                idx = i*iNG*2*iNDet+j*iNDet+k;
                allTrue1 *= acExpDetImages[aiDetStartIdx[k]
                                 + aiRotN[idx]*int(afDetInfo[0+19*k])*int(afDetInfo[1+19*k])
                                  + aiK[idx]*int(afDetInfo[0+19*k])
                                  + aiJ[idx]];
                k += 1;
            }
            iPeakCnt += allTrue0;
			fHitRatio += allTrue1;
		}
		aiPeakCnt[i] = iPeakCnt;
		if(iPeakCnt>0){
		    afHitRatio[i] = fHitRatio/float(iPeakCnt);
		}
		else{
			afHitRatio[i]=0;
		}
	}
}

__global__ void hitratio_multi_detector(const int iNVoxel,const int iNOrientation,const int iNG,
		const float* __restrict__ afDetInfo, const int iNDet,const int iNRot,
		const int* aiJ, const int* aiK,const int* aiRotN, const bool* abHit,
		float* afHitRatio, int* aiPeakCnt){
	/*
	 * 100x100 voxel takes 13.8123515625s
	 * This version using texture memory to storage tcExpData
	 * this version optimize some detail, runs faster than hitratio_multi_detector_old_backup, no error so far.
	 * now 100x100 takes 15s
	 * If set block=(16,1,1) 100x100 will reach 25s.
	 * 100x100 31s, original: 43s, improved!
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
	bool allTrue0; // if a simulated peak hit all detector, allTrue0 = true;
	bool allTrue1; // if a simulated peak overlap with all expimages on all detector, allTrue1 = true;
	int k;
	int iPeakCnt;
	float fHitRatio;
	int idx;

	if(i<iNVoxel*iNOrientation){
	    iPeakCnt = 0;
	    fHitRatio = 0;
		for(int j=0;j<iNG*2;j++){
		    //printf("j: %d ||",j);
			allTrue0 = true;
			allTrue1 = true;
			k = 0;
			while(allTrue0 && k<iNDet){
				allTrue0 *= abHit[i*iNG*2*iNDet+j*iNDet+k];
				k += 1;
			}
			allTrue1 = allTrue0;
            k = 0;
            while(allTrue1 && k<iNDet){
                idx = i*iNG*2*iNDet+j*iNDet+k;
                allTrue1 *= tex3D(tcExpData,(float)aiJ[idx], (float)aiK[idx],(float)(k*iNRot + aiRotN[idx]) );
                //if (threadIdx.x==0 && blockIdx.x==0){
                //    printf("%d  %d ||",tex3D(tcExpData,(float)aiJ[idx], (float)aiK[idx],(float)(k*iNRot + aiRotN[idx]) ),
                //    acExpDetImages[aiDetStartIdx[k]
                //                 + aiRotN[idx]*int(afDetInfo[0+19*k])*int(afDetInfo[1+19*k])
                //                  + aiK[idx]*int(afDetInfo[0+19*k])
                //                  + aiJ[idx]]);
                //}
                k += 1;
            }
            iPeakCnt += allTrue0;
			fHitRatio += allTrue1;
		}
		aiPeakCnt[i] = iPeakCnt;
		if(iPeakCnt>0){
		    afHitRatio[i] = fHitRatio/float(iPeakCnt);
		}
		else{
			afHitRatio[i]=0;
		}
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

__global__ void misorien(float* afMisOrien, float* afM0, float* afM1, float* afSymM){
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
    fCosAngle = 0.5 * (afTmp1[0] + afTmp1[4] + afTmp1[8] - 1);
    fCosAngle = min(0.9999999999, fCosAngle);
    fCosAngle = max(-0.99999999999, fCosAngle);
    afMisOrien[i] = acosf(fCosAngle);
}

__device__ void d_misorien(float& fMisOrien, float* afM0, float* afM1, float* afSymM){
        /*
    * calculate the misorientation betwen afM0 and afM1
    * fMisOrien: 1
    * afM0: 9
    * afM1: 9

    * call method:
    */
    float afTmp0[9];
    float afTmp1[9];
    float afM1Transpose[9];
    float fCosAngle;
    mat3_transpose(afM1Transpose, afM1);
    mat3_dot(afTmp0, afSymM , afM1Transpose);
    mat3_dot(afTmp1, afM0, afTmp0);
    fCosAngle = 0.5 * (afTmp1[0] + afTmp1[4] + afTmp1[8] - 1);
    fCosAngle = min(0.9999999999, fCosAngle);
    fCosAngle = max(-0.99999999999, fCosAngle);
    fMisOrien = acosf(fCosAngle);
}

__global__ void sim_hitratio_unit_backup20180206(int *aiJ, int *aiK, float *afOmega, bool *abHit,int *aiRotN,
		const int iNVoxel, const int iNOrientation, const int iNG, const int iNDet,
		const float *afOrientation,const float *afG,const float *afVoxelPos,
		const float fBeamEnergy, const float fEtaLimit, const float *afDetInfo,
		const char* __restrict__ acExpDetImages, const int* __restrict__ aiDetStartIdx,
		float* afHitRatio, int* aiPeakCnt){
	/*
	 * a simple combination of simulation and hitratio function on GPU, attempt for speed optimization.
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
	 extern __shared__ int abAllTrue[];  //2*2*iNG [allHit0*iNG,allHit1*iNG,allHit0*iNG,allHit1*iNG] means: [simulation hit on det, overlap of sim and exp], NVoxel*NOrien*(iNG*2)*2
	 //////////////////////////// simulation part ////////////////////////////////////////
	float fOmegaRes1,fOmegaRes2,fTwoTheta,fEta,fChi;
	float afScatteringVec[3]={0,0,0};
	for (int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			afScatteringVec[i] += afOrientation[blockIdx.x*gridDim.y*9+blockIdx.y*9+i*3+j]*afG[threadIdx.x*3+j];
		    //printf("%d,%d: %f. ||",i,j,afOrientation[blockIdx.x*gridDim.y*9+blockIdx.y*9+i*3+j]);
		}
	}
	int idx = blockIdx.x*gridDim.y*blockDim.x*2*iNDet+ blockIdx.y*blockDim.x*2*iNDet + threadIdx.x*2*iNDet;
	if(GetScatteringOmegas( fOmegaRes1, fOmegaRes2, fTwoTheta, fEta, fChi , afScatteringVec,fBeamEnergy)){
		for(int iDetIdx=0;iDetIdx<iNDet;iDetIdx++){
			GetPeak(aiJ[idx+iDetIdx],aiJ[idx+iDetIdx+iNDet],aiK[idx+iDetIdx],aiK[idx+iDetIdx+iNDet],
					abHit[idx+iDetIdx],abHit[idx+iDetIdx+iNDet],
					fOmegaRes1, fOmegaRes2,fTwoTheta,
					fEta,fChi,fEtaLimit,afVoxelPos+blockIdx.x*3,afDetInfo+19*iDetIdx);
			if(abHit[idx+iDetIdx]){
				////////assuming they are using the same rotation number in all the detectors!!!!!!!///////////////////
				aiRotN[idx+iDetIdx] = floor((fOmegaRes1-afDetInfo[17])/(afDetInfo[18]-afDetInfo[17])*(afDetInfo[16]-1));
			}
			if(abHit[idx+iDetIdx+iNDet]){
				aiRotN[idx+iDetIdx+iNDet] = floor((fOmegaRes2-afDetInfo[17])/(afDetInfo[18]-afDetInfo[17])*(afDetInfo[16]-1));
			}
		}
	}
	/////////////////////////////// hit ratio part ///////////////////////////////////////

    int k;
    for(int j=0;j<2;j++){
        // j for Omega1 and Omega2
        abAllTrue[j * 2 * iNG + threadIdx.x] = 1;
        abAllTrue[j * 2 * iNG + iNG + threadIdx.x] = 1;
        k = 0;
        while(abAllTrue[j * 2 * iNG + threadIdx.x] && k<iNDet){
            //check if peak hit all the detetros
            abAllTrue[j * 2 * iNG + threadIdx.x] *= abHit[idx + j * iNDet + k];
            k += 1;
        }
        abAllTrue[j * 2 * iNG + iNG + threadIdx.x] = abAllTrue[j * 2 * iNG + threadIdx.x];
        k = 0;
        while(abAllTrue[j * 2 * iNG + iNG + threadIdx.x] && k<iNDet){
            abAllTrue[j * 2 * iNG + iNG + threadIdx.x] *= acExpDetImages[aiDetStartIdx[k]
                              + aiRotN[idx+j*iNDet+k]*int(afDetInfo[0+19*k])*int(afDetInfo[1+19*k])
                              + aiK[idx+j*iNDet+k]*int(afDetInfo[0+19*k])
                              + aiJ[idx+j*iNDet+k]];
            k += 1;
        }
    }
    __syncthreads();

    if(threadIdx.x==0){
        int iPeakCnt = 0;
        float fHitRatio = 0;
        for(int j=0;j<2;j++){
            for(int i=0;i<iNG;i++){
                iPeakCnt += abAllTrue[j * 2 * iNG + i];
                fHitRatio +=  abAllTrue[j * 2 * iNG + iNG + i];
            }
        }
        aiPeakCnt[blockIdx.x * gridDim.y + blockIdx.y] = iPeakCnt;
        if(iPeakCnt>0){
            afHitRatio[blockIdx.x * gridDim.y + blockIdx.y] = fHitRatio/((float)iPeakCnt);
        }
        else{
            afHitRatio[blockIdx.x * gridDim.y + blockIdx.y] = 0;
        }
    }
}

__global__ void sim_hitratio_unit(const int iNVoxel, const int iNOrientation, const int iNG, const int iNDet,
		const float *afOrientation,const float *afG,const float *afVoxelPos,
		const float fBeamEnergy, const float fEtaLimit, const float *afDetInfo,
		float* afHitRatio, int* aiPeakCnt){
	/*
	* performance drops...
	 *Try to use shared memory to store j,k,omega,hit, and rotN, use texture memory to access expdata
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
	 //////////////////////////// simulation part ////////////////////////////////////////
	extern __shared__ int aiSimResult[]; // iNG*2*4*iNDet, each storage is [jo0d0,ko0d0,Ro0d0,ho0d0,jo0d1,ko0d1...jo1d0,ko1d0...]
	float fOmegaRes1,fOmegaRes2,fTwoTheta,fEta,fChi;
	float afScatteringVec[3]={0,0,0};
	for (int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			afScatteringVec[i] += afOrientation[blockIdx.x*gridDim.y*9+blockIdx.y*9+i*3+j]*afG[threadIdx.x*3+j];
		    //printf("%d,%d: %f. ||",i,j,afOrientation[blockIdx.x*gridDim.y*9+blockIdx.y*9+i*3+j]);
		}
	}
	if(GetScatteringOmegas( fOmegaRes1, fOmegaRes2, fTwoTheta, fEta, fChi , afScatteringVec,fBeamEnergy)){
		for(int iDetIdx=0;iDetIdx<iNDet;iDetIdx++){
			GetPeak_s(aiSimResult[threadIdx.x*2*4*iNDet + iDetIdx*4 + 0], aiSimResult[threadIdx.x*2*4*iNDet + iNDet*4 + iDetIdx*4 + 0],
			        aiSimResult[threadIdx.x*2*4*iNDet + iDetIdx*4 + 1], aiSimResult[threadIdx.x*2*4*iNDet + iNDet*4 + iDetIdx*4 + 1],
					aiSimResult[threadIdx.x*2*4*iNDet + iDetIdx*4 + 3], aiSimResult[threadIdx.x*2*4*iNDet + iNDet*4 + iDetIdx*4 + 3],
					fOmegaRes1,fOmegaRes2,
                    fTwoTheta,fEta,fChi,fEtaLimit,afVoxelPos+blockIdx.x*3,afDetInfo+19*iDetIdx);
			if( aiSimResult[threadIdx.x*2*4*iNDet + iDetIdx*4 + 3]){
				////////assuming they are using the same rotation number in all the detectors!!!!!!!///////////////////
				aiSimResult[threadIdx.x*2*4*iNDet + iDetIdx*4 + 2] = floor((fOmegaRes1-afDetInfo[17])/(afDetInfo[18]-afDetInfo[17])*(afDetInfo[16]-1));
			}
			if( aiSimResult[threadIdx.x*2*4*iNDet + iNDet*4 + iDetIdx*4 + 3]){
				aiSimResult[threadIdx.x*2*4*iNDet + iNDet*4 + iDetIdx*4 + 2] = floor((fOmegaRes2-afDetInfo[17])/(afDetInfo[18]-afDetInfo[17])*(afDetInfo[16]-1));
			}
		}
	}
	__syncthreads();
	/////////////////////////////// hit ratio part ///////////////////////////////////////
	/*
	if(threadIdx.x==0){
	    int i = blockIdx.x * blockDim.y + blockIdx.y;
		bool allTrue0; // if a simulated peak hit all detector, allTrue0 = true;
        bool allTrue1; // if a simulated peak overlap with all expimages on all detector, allTrue1 = true;
        int k;
        int iPeakCnt = 0;
        float fHitRatio = 0;
        int iNRot = (int)afDetInfo[16];
		for(int j=0;j<iNG*2;j++){
		    //printf("j: %d ||",j);
			allTrue0 = true;
			allTrue1 = true;
			k = 0;
			while(allTrue0 && k<iNDet){
				allTrue0 *= aiSimResult[j*4*iNDet + k*4 + 3]; //abHit[i*iNG*2*iNDet+j*iNDet+k];
				k += 1;
			}
			allTrue1 = allTrue0;
            k = 0;
            while(allTrue1 && k<iNDet){
                allTrue1 *= tex3D(tcExpData,(float)aiSimResult[j*4*iNDet + k*4 + 0], (float)aiSimResult[j*4*iNDet + k*4 + 1],(float)(k*iNRot + aiSimResult[j*4*iNDet + k*4 + 2]) );
                k += 1;
            }
            iPeakCnt += allTrue0;
			fHitRatio += allTrue1;
		}
		aiPeakCnt[i] = iPeakCnt;
		if(iPeakCnt>0){
		    afHitRatio[i] = fHitRatio/float(iPeakCnt);
		}
		else{
			afHitRatio[i]=0;
		}
	}
	*/
}


""")