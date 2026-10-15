#pragma once

#include "engine.cuh"

std::vector<int> generatePaged(cublasHandle_t cublas_handle, const ModelWeights& weights,
                               const std::vector<int>& prompt_tokens, int max_new_tokens,
                               int max_seq_len);
