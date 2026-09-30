#pragma once

#include <cuda_bf16.h>
#include <cublas_v2.h>
#include "weights.cuh"
#include "kvcache.cuh"

struct TransformerBuffers
{
    __nv_bfloat16* rms_output;

    __nv_bfloat16* q;
    __nv_bfloat16* k;
    __nv_bfloat16* v;

    __nv_bfloat16* attn_scores;
    __nv_bfloat16* attn_output;
    __nv_bfloat16* o_output;

    __nv_bfloat16* gate;
    __nv_bfloat16* up;
    __nv_bfloat16* down;
    __nv_bfloat16* residual;
};

void allocateTransformerBuffers(
    TransformerBuffers& buffers,
    int num_tokens,
    int max_seq_len
);

void freeTransformerBuffers(
    TransformerBuffers& buffers
);

int transformerForward(
    cublasHandle_t cublas_handle,
    __nv_bfloat16* hidden_states,
    const ModelWeights& weights,
    KVCache& cache,
    TransformerBuffers& buffers,
    float* logits,
    int num_tokens
);

int decodeForward(
    cublasHandle_t cublas_handle,
    __nv_bfloat16* hidden_states,
    const ModelWeights& weights,
    KVCache& cache,
    TransformerBuffers& buffers,
    float* logits
);
