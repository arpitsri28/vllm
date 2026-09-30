#pragma once

#include <cublas_v2.h>
#include <vector>
#include "weights.cuh"

std::vector<int> generate(
    cublasHandle_t cublas_handle,
    const ModelWeights& weights,
    const std::vector<int>& prompt_tokens,
    int max_new_tokens,
    int max_seq_len
);
