#pragma once

#include <cuda_bf16.h>
#include <cublas_v2.h>

void gateProjection(cublasHandle_t cublas_handle, const __nv_bfloat16* input, const __nv_bfloat16* weights, __nv_bfloat16* output, int num_tokens);
void upProjection(cublasHandle_t cublas_handle, const __nv_bfloat16* input, const __nv_bfloat16* weights, __nv_bfloat16* output, int num_tokens);
void downProjection(cublasHandle_t cublas_handle, const __nv_bfloat16* input, const __nv_bfloat16* weights, __nv_bfloat16* output, int num_tokens);
