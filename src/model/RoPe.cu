#include "../../include/rope.cuh"
#include <iostream>

constexpr int HEAD_DIM = 64;
__global__ void ropeKernel(__nv_bfloat16 *input, int num_tokens, int proj_dim, int position_offset) {
  if (2 * threadIdx.x + 1 + blockIdx.x * proj_dim < num_tokens * proj_dim) {
    int position = position_offset + blockIdx.x;
    int double_i = 2 * (threadIdx.x % 32);
    float theta = 1.0/ powf(500000.0, ((float)double_i/HEAD_DIM));
    float angle = position * theta;
    __nv_bfloat16 x_i = input[2 * threadIdx.x + blockIdx.x * proj_dim];
    __nv_bfloat16 y_i = input[2 * threadIdx.x + 1 + blockIdx.x * proj_dim];
    input[2 * threadIdx.x + blockIdx.x * proj_dim] = (__nv_bfloat16)((float) x_i * cosf(angle) - (float) y_i * sinf(angle));
    input[2 * threadIdx.x + 1 + blockIdx.x * proj_dim] = (__nv_bfloat16)((float) x_i * sinf(angle) + (float) y_i * cosf(angle));
  }
}

void rope(__nv_bfloat16 *input, int num_tokens, int proj_dim, int position_offset) {
    int num_threads = proj_dim / 2;
    if (num_threads > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, RoPE kernel not launched";
        return;
    }

    ropeKernel<<<num_tokens, num_threads>>>(input, num_tokens, proj_dim, position_offset);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}
