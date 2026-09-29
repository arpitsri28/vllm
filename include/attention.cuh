#pragma once

#include <cuda_bf16.h>
#include <cublas_v2.h>

void computeAttentionScores(cublasHandle_t cublas_handle, __nv_bfloat16* q_proj, __nv_bfloat16* k_proj_layer, __nv_bfloat16* attn_scores, int num_tokens);
void attentionScoresTimesV(cublasHandle_t cublas_handle, __nv_bfloat16* attn_scores, __nv_bfloat16* v_proj_layer, __nv_bfloat16* attn_scores_v, int num_tokens);
void outputProjection(cublasHandle_t cublas_handle, __nv_bfloat16* attention_output, __nv_bfloat16* w_o, __nv_bfloat16* output, int num_tokens);
void qProjection(
    cublasHandle_t cublas_handle,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weights,
    __nv_bfloat16* output,
    int num_tokens
);

void kProjection(
    cublasHandle_t cublas_handle,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weights,
    __nv_bfloat16* output,
    int num_tokens
);

void vProjection(
    cublasHandle_t cublas_handle,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weights,
    __nv_bfloat16* output,
    int num_tokens
);

void computeDecodeAttentionScores(
    cublasHandle_t cublas_handle,
    const __nv_bfloat16* q_proj,
    const __nv_bfloat16* k_cache_layer,
    __nv_bfloat16* attn_scores,
    int total_seq_len
);

void decodeAttentionScoresTimesV(
    cublasHandle_t cublas_handle,
    const __nv_bfloat16* attn_scores,
    const __nv_bfloat16* v_cache_layer,
    __nv_bfloat16* attention_output,
    int total_seq_len
);