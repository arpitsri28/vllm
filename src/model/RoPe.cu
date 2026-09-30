#include "../../include/rope.cuh"
#include <iostream>

constexpr int HEAD_DIM = 64;
__global__ void ropeKernel(__nv_bfloat16 *input, int num_tokens, int proj_dim, int position_offset) {
  if (2 * threadIdx.x + 1 + blockIdx.x * proj_dim < num_tokens * proj_dim) {
    int position = position_offset + blockIdx.x;
    int pair = threadIdx.x % (HEAD_DIM / 2);
    int head = threadIdx.x / (HEAD_DIM / 2);
    int x_index = blockIdx.x * proj_dim + head * HEAD_DIM + pair;
    int y_index = x_index + HEAD_DIM / 2;
    float theta = 1.0f / powf(500000.0f, (2.0f * pair) / HEAD_DIM);
    float wavelength = 6.283185307179586f / theta;
    if (wavelength > 8192.0f) {
      theta /= 32.0f;
    } else if (wavelength >= 2048.0f) {
      float smooth = (8192.0f / wavelength - 1.0f) / 3.0f;
      theta = (1.0f - smooth) * theta / 32.0f + smooth * theta;
    }
    float angle = position * theta;
    __nv_bfloat16 x_i = input[x_index];
    __nv_bfloat16 y_i = input[y_index];
    input[x_index] = (__nv_bfloat16)((float) x_i * cosf(angle) - (float) y_i * sinf(angle));
    input[y_index] = (__nv_bfloat16)((float) x_i * sinf(angle) + (float) y_i * cosf(angle));
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
