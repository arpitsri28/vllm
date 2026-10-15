#include <cstddef>
#include "operation_profile.cuh"
#include "../../include/silu.cuh"

__global__ void siluKernel(__nv_bfloat16 *a, __nv_bfloat16 *b) {
    int workIndex = threadIdx.x + blockIdx.x * 8192;
    for (int i = 0; i < 8192; i += 1024) {
        a[workIndex + i] = (__nv_bfloat16)((float)a[workIndex + i] * (1 / (1 + expf(-(float)a[workIndex + i]))) * (float)b[workIndex + i]);
    }
}

void silu(__nv_bfloat16 *a, __nv_bfloat16 *b, int num_tokens) {
    OP_SCOPE("swiglu",true);
    operationKernel();
    siluKernel<<<num_tokens, 1024>>>(a, b);
}
