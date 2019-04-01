#include <curand_kernel.h>

const int nstates = %(NGENERATORS);
__device__ curandState_t* states[nstates];

__global__ void initkernel(int seed)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;

    if (tidx < nstates) {
        curandState_t* s = new curandState_t;
        if (s != 0) {
            curand_init(seed, tidx, 0, s);
        }

        states[tidx] = s;
    }
}

__global__ void randfillkernel(float *values, int N)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;

    if (tidx < nstates) {
        curandState_t s = *states[tidx];
        for(int i=tidx; i < N; i += blockDim.x * gridDim.x) {
            values[i] = curand_uniform(&s);
        }
        *states[tidx] = s;
    }
}