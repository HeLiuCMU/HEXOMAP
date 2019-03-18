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

        afMat[0] =  c1 * c3 - c2 * s1 * s3;
        afMat[1] = -c1 * s3 - c3 * c2 * s1;
        afMat[2] =  s1 * s2;
        afMat[3] =  s1 * c3 + c2 * c1 * s3;
        afMat[4] =  c1 * c2 * c3 - s1 * s3;
        afMat[5] = -c1 * s2;
        afMat[6] =  s3 * s2;
        afMat[7] =  s2 * c3;
        afMat[8] =  c2;
}

__global__ void mat_to_euler_ZXZ(float* afMatIn, 
                                 float* afEulerOut, 
                                 int iNAngle,
                                ){
    /*
    * transform active rotation matrix to euler angles in ZXZ convention, 
    * not right (seems right now)
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
    /* 
    * generate random matrix according to the input EulerAngle
    * afEulerIn:   iNEulerIn * 3, !!!!!!!!!! in radian  !!!!!!!!
    * afMatOut:    iNNeighbour * iNEulerIn * 9
    * afRand:      iNNeighbour * iNEulerIn * 3
    * fBound:      the range for random angle [-fBound,+fBound]
    * iNEulerIn:   number of Input Euler angles
    * iNNeighbour: number of random angle generated for EACH input
    * call:: <<(iNNeighbour,1),(iNEulerIn,1,1)>>
    * TEST PASSED
    */

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
    * afM0:       iNM * 9
    * afM1:       iNM * 9
    * afSymM:     symmetry matrix, iNSymM * 9
    * NSymM:      number of symmetry matrix
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