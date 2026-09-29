#include "../../include/transformer.cuh"
#include "weights.cuh"
#include "rmsnorm.cuh"
#include "attention.cuh"
#include "rope.cuh"
#include "residual.cuh"
#include "mlp.cuh"
#include "silu.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include "lm_head.cuh"
#include "argmax.cuh"
#include "kv_cache.cuh"
constexpr int NUM_LAYERS = 16;


void allocateTransformerBuffers(TransformerBuffers& buffers, int num_tokens) {
  cudaMalloc((void**)&buffers.rms_output,
            num_tokens * 2048 * sizeof(__nv_bfloat16));

  cudaMalloc((void**)&buffers.q,
            num_tokens * 2048 * sizeof(__nv_bfloat16));

  cudaMalloc((void**)&buffers.k,
            num_tokens * 512 * sizeof(__nv_bfloat16));

  cudaMalloc((void**)&buffers.v,
            num_tokens * 512 * sizeof(__nv_bfloat16));

  cudaMalloc((void**)&buffers.attn_scores, NUM_Q_HEADS * max_seq_len * sizeof(__nv_bfloat16));

  cudaMalloc((void**)&buffers.attn_output,
            num_tokens * 2048 * sizeof(__nv_bfloat16));

  cudaMalloc((void**)&buffers.o_output,
            num_tokens * 2048 * sizeof(__nv_bfloat16));

  cudaMalloc((void**)&buffers.gate,
            num_tokens * 8192 * sizeof(__nv_bfloat16));

  cudaMalloc((void**)&buffers.up,
            num_tokens * 8192 * sizeof(__nv_bfloat16));

  cudaMalloc((void**)&buffers.down,
            num_tokens * 2048 * sizeof(__nv_bfloat16));

  cudaMalloc((void**)&buffers.residual,
            num_tokens * 2048 * sizeof(__nv_bfloat16));
}

void freeTransformerBuffers(TransformerBuffers& buffers) {
    cudaFree(buffers.rms_output);
    cudaFree(buffers.q);
    cudaFree(buffers.k);
    cudaFree(buffers.v);
    cudaFree(buffers.attn_scores);
    cudaFree(buffers.attn_output);
    cudaFree(buffers.o_output);
    cudaFree(buffers.gate);
    cudaFree(buffers.up);
    cudaFree(buffers.down);
    cudaFree(buffers.residual);

    buffers.rms_output = nullptr;
    buffers.q = nullptr;
    buffers.k = nullptr;
    buffers.v = nullptr;
    buffers.attn_scores = nullptr;
    buffers.attn_output = nullptr;
    buffers.o_output = nullptr;
    buffers.gate = nullptr;
    buffers.up = nullptr;
    buffers.down = nullptr;
    buffers.residual = nullptr;
}

int transformerForward(cublasHandle_t cublas_handle, __nv_bfloat16* hidden_states, const ModelWeights& weights, KVCache& cache, TransformerBuffers& buffers, float* logits, int num_tokens) {
  for (int layer = 0; layer < NUM_LAYERS; ++layer) {
    rmsNormKernel<<<num_tokens, 1024>>>(hidden_states, buffers.rms_output, weights.layers[layer].input_norm);
    qProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_q, buffers.q, num_tokens);
    kProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_k, buffers.k, num_tokens);
    vProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_v, buffers.v, num_tokens);
    rope(buffers.q, num_tokens, 2048, 0);
    rope(buffers.k, num_tokens, 512, 0);
    storeKV(cache, layer, buffers.k, buffers.v, num_tokens);
    computeAttentionScores(cublas_handle, buffers.q, buffers.k, buffers.attn_scores, num_tokens);
    causalMask(buffers.attn_scores, num_tokens);
    softmax(buffers.attn_scores, num_tokens);
    attentionScoresTimesV(cublas_handle, buffers.attn_scores, buffers.v, buffers.attn_output, num_tokens);
    outputProjection(cublas_handle, buffers.attn_output, weights.layers[layer].w_o, buffers.o_output, num_tokens);
    residualAdd(hidden_states, buffers.o_output, num_tokens);
    rmsNormKernel<<<num_tokens, 1024>>>(hidden_states, buffers.rms_output, weights.layers[layer].post_attention_norm);
    gateProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_gate, buffers.gate, num_tokens);
    upProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_up, buffers.up, num_tokens);
    silu(buffers.gate, buffers.up, num_tokens);
    downProjection(cublas_handle, buffers.gate, weights.layers[layer].w_down, buffers.down, num_tokens);
    residualAdd(hidden_states, buffers.down, num_tokens);
  }
  cache.current_seq_len += num_tokens;
  rmsNormKernel<<<num_tokens, 1024>>>(hidden_states, buffers.rms_output, weights.final_norm);
  __nv_bfloat16* last_hidden = buffers.rms_output + (num_tokens - 1) * EMBEDDING_LENGTH;
  lmHeadProjection(cublas_handle, last_hidden, weights.embedding, logits);
  int next_token = argmaxLogits(logits);
  return next_token;
}

int decodeForward(cublasHandle_t cublas_handle, __nv_bfloat16* hidden_states, const ModelWeights& weights, KVCache& cache, TransformerBuffers& buffers, float* logits) {
    int position = cache.current_seq_len;
    int total_seq_len = cache.current_seq_len + 1;

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        rmsNormKernel<<<1, 1024>>>(hidden_states, buffers.rms_output, weights.layers[layer].input_norm);
        qProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_q, buffers.q, 1);
        kProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_k, buffers.k, 1);
        vProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_v, buffers.v, 1);
        rope(buffers.q, 1, 2048, position );
        rope(buffers.k, 1, 512, position);
        storeKV(cache, layer, buffers.k, buffers.v, 1);
        const __nv_bfloat16* k_cache_layer = cache.k_cache + static_cast<size_t>(layer) * cache.max_seq_len * KV_DIM;
        const __nv_bfloat16* v_cache_layer = cache.v_cache + static_cast<size_t>(layer) * cache.max_seq_len * KV_DIM;
        computeDecodeAttentionScores(cublas_handle, buffers.q, k_cache_layer, buffers.attn_scores, total_seq_len);
        softmaxRows(buffers.attn_scores, NUM_Q_HEADS, total_seq_len);
        decodeAttentionScoresTimesV(cublas_handle, buffers.attn_scores, v_cache_layer, buffers.attn_output, total_seq_len);
        outputProjection(cublas_handle, buffers.attn_output, weights.layers[layer].w_o, buffers.o_output, 1);
        residualAdd(hidden_states, buffers.o_output, 1);
        rmsNormKernel<<<1, 1024>>>(hidden_states, buffers.rms_output, weights.layers[layer].post_attention_norm);
        gateProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_gate, buffers.gate, 1);
        upProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_up, buffers.up, 1);
        silu(buffers.gate, buffers.up, 1);
        downProjection(cublas_handle, buffers.gate, weights.layers[layer].w_down, buffers.down, 1);
        residualAdd(hidden_states, buffers.down, 1);
    }
    cache.current_seq_len += 1;
    rmsNormKernel<<<1, 1024>>>(hidden_states, buffers.rms_output, weights.final_norm);
    lmHeadProjection(cublas_handle, buffers.rms_output, weights.embedding, logits);
    return argmaxLogits(logits);
}