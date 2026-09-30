#include "argmax.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>

int argmaxLogits(const float* logits) {
    std::vector<float> logits_cpu(VOCAB_SIZE);
    cudaError_t status = cudaMemcpy(logits_cpu.data(), logits, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    float max_logit = logits_cpu[0];
    int max_token_idx = 0;
    for (int token_idx = 1; token_idx < VOCAB_SIZE; ++token_idx) {
      if (logits_cpu[token_idx] > max_logit) {
        max_logit = logits_cpu[token_idx];
        max_token_idx = token_idx;
      }
    }

    return max_token_idx;
}