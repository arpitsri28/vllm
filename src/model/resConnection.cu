#include <cstddef>
#include "operation_profile.cuh"
#include "../../include/residual.cuh"
#include <iostream>

__global__ void residualKernel(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds) {
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    input[workIndex] = input[workIndex] + input_embeds[workIndex];
    input[workIndex + 1024] = input[workIndex + 1024] + input_embeds[workIndex + 1024];
}

void residualAdd(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds, int num_tokens) {
    OP_SCOPE("residual",true);
    operationKernel();
    residualKernel<<<num_tokens, 1024>>>(input, input_embeds);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}
