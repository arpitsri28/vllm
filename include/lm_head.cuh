#pragma once

#include <cuda_bf16.h>
#include <cublas_v2.h>

void lmHeadProjection(
    cublasHandle_t cublas_handle,
    const __nv_bfloat16* final_hidden,
    const __nv_bfloat16* embedding_weights,
    float* logits
);