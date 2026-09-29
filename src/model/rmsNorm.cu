#include "../../include/rmsnorm.cuh"

__global__ void rmsNormKernel(__nv_bfloat16 *input, __nv_bfloat16 *output, __nv_bfloat16 *norm_weights)
{
    __shared__ float rms_vector[1024];
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    rms_vector[threadIdx.x] = (float)input[workIndex] * (float)input[workIndex] + (float)input[workIndex + 1024] * (float)input[workIndex + 1024];
    __syncthreads();
    for (int i = 1; i < 1024; i = i * 2) {
        if (threadIdx.x % (i * 2) == 0) {
            rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        rms_vector[0] = sqrt(rms_vector[0] / 2048.0 + 1.0e-5);
    }
    __syncthreads();

    output[workIndex] = (__nv_bfloat16)(((float)input[workIndex] / rms_vector[0]) * (float)norm_weights[threadIdx.x]);
    output[workIndex + 1024] = (__nv_bfloat16)(((float)input[workIndex + 1024] / rms_vector[0]) * (float)norm_weights[threadIdx.x + 1024]);
}
