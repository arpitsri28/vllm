#include <cstddef>
#include "operation_profile.cuh"
#include <cuda_bf16.h>
#include "numerical_trace.cuh"
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
#include "kvcache.cuh"
#include "causalMask.cuh"
#include "softmax.cuh"
#include <algorithm>
#include <stdexcept>
constexpr int NUM_LAYERS = 16;
constexpr int NUM_Q_HEADS = 32;
constexpr int KV_DIM = 512;
constexpr int EMBEDDING_LENGTH = 2048;


void allocateTransformerBuffers(TransformerBuffers& buffers, int num_tokens, int max_seq_len) {
    OP_SCOPE("workspace_allocate",false);
  auto allocate = [](void** pointer, size_t bytes) {
    operationAllocation();
    cudaError_t status = cudaMalloc(pointer, bytes);
    if (status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(status));
    }
  };
  allocate((void**)&buffers.rms_output,
            num_tokens * 2048 * sizeof(__nv_bfloat16));

  allocate((void**)&buffers.q,
            num_tokens * 2048 * sizeof(__nv_bfloat16));

  allocate((void**)&buffers.k,
            num_tokens * 512 * sizeof(__nv_bfloat16));

  allocate((void**)&buffers.v,
            num_tokens * 512 * sizeof(__nv_bfloat16));

  allocate((void**)&buffers.attn_scores, NUM_Q_HEADS * std::max(static_cast<size_t>(num_tokens) * num_tokens, static_cast<size_t>(max_seq_len)) * sizeof(__nv_bfloat16));

  allocate((void**)&buffers.attn_output,
            num_tokens * 2048 * sizeof(__nv_bfloat16));

  allocate((void**)&buffers.o_output,
            num_tokens * 2048 * sizeof(__nv_bfloat16));

  allocate((void**)&buffers.gate,
            num_tokens * 8192 * sizeof(__nv_bfloat16));

  allocate((void**)&buffers.up,
            num_tokens * 8192 * sizeof(__nv_bfloat16));

  allocate((void**)&buffers.down,
            num_tokens * 2048 * sizeof(__nv_bfloat16));

  allocate((void**)&buffers.residual,
            num_tokens * 2048 * sizeof(__nv_bfloat16));
}

void freeTransformerBuffers(TransformerBuffers& buffers) {
    OP_SCOPE("workspace_free",false);
    operationFree(11);
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
    OP_KERNEL("rmsnorm", rmsNormKernel<<<num_tokens, 1024>>>(hidden_states, buffers.rms_output, weights.layers[layer].input_norm));
    qProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_q, buffers.q, num_tokens);
    kProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_k, buffers.k, num_tokens);
    vProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_v, buffers.v, num_tokens);
    rope(buffers.q, num_tokens, 2048, 0);
    rope(buffers.k, num_tokens, 512, 0);
    storeKV(cache, layer, buffers.k, buffers.v, num_tokens);
    computeAttentionScores(cublas_handle, buffers.q, buffers.k, buffers.attn_scores, num_tokens);
    causalMask(buffers.attn_scores, num_tokens);
    softmaxRows(buffers.attn_scores, NUM_Q_HEADS * num_tokens, num_tokens);
    attentionScoresTimesV(cublas_handle, buffers.attn_scores, buffers.v, buffers.attn_output, num_tokens);
    outputProjection(cublas_handle, buffers.attn_output, weights.layers[layer].w_o, buffers.o_output, num_tokens);
    residualAdd(hidden_states, buffers.o_output, num_tokens);
    OP_KERNEL("rmsnorm", rmsNormKernel<<<num_tokens, 1024>>>(hidden_states, buffers.rms_output, weights.layers[layer].post_attention_norm));
    gateProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_gate, buffers.gate, num_tokens);
    upProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_up, buffers.up, num_tokens);
    silu(buffers.gate, buffers.up, num_tokens);
    downProjection(cublas_handle, buffers.gate, weights.layers[layer].w_down, buffers.down, num_tokens);
    residualAdd(hidden_states, buffers.down, num_tokens);
  }
  cache.current_seq_len += num_tokens;
  OP_KERNEL("rmsnorm", rmsNormKernel<<<num_tokens, 1024>>>(hidden_states, buffers.rms_output, weights.final_norm));
  __nv_bfloat16* last_hidden = buffers.rms_output + (num_tokens - 1) * EMBEDDING_LENGTH;
  lmHeadProjection(cublas_handle, last_hidden, weights.embedding, logits);
  int next_token = argmaxLogits(logits);
  return next_token;
}

int decodeForward(cublasHandle_t cublas_handle, __nv_bfloat16* hidden_states, const ModelWeights& weights, KVCache& cache, TransformerBuffers& buffers, float* logits) {
    int position = cache.current_seq_len;
    int total_seq_len = cache.current_seq_len + 1;

    for (int layer = 0; layer < NUM_LAYERS; ++layer) {
        OP_KERNEL("rmsnorm", rmsNormKernel<<<1, 1024>>>(hidden_states, buffers.rms_output, weights.layers[layer].input_norm));
        traceBF16(layer, "input_norm", buffers.rms_output, 2048);
        qProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_q, buffers.q, 1);
        traceBF16(layer, "q", buffers.q, 2048);
        kProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_k, buffers.k, 1);
        traceBF16(layer, "k", buffers.k, 512);
        vProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_v, buffers.v, 1);
        traceBF16(layer, "v", buffers.v, 512);
        rope(buffers.q, 1, 2048, position );
        traceBF16(layer, "rope_q", buffers.q, 2048);
        rope(buffers.k, 1, 512, position);
        traceBF16(layer, "rope_k", buffers.k, 512);
        storeKV(cache, layer, buffers.k, buffers.v, 1);
        const __nv_bfloat16* k_cache_layer = cache.k_cache + static_cast<size_t>(layer) * cache.max_seq_len * KV_DIM;
        const __nv_bfloat16* v_cache_layer = cache.v_cache + static_cast<size_t>(layer) * cache.max_seq_len * KV_DIM;
        computeDecodeAttentionScores(cublas_handle, buffers.q, k_cache_layer, buffers.attn_scores, total_seq_len);
        traceBF16(layer, "scores", buffers.attn_scores, total_seq_len, 32);
        softmaxRows(buffers.attn_scores, NUM_Q_HEADS, total_seq_len);
        traceBF16(layer, "probabilities", buffers.attn_scores, total_seq_len, 32);
        decodeAttentionScoresTimesV(cublas_handle, buffers.attn_scores, v_cache_layer, buffers.attn_output, total_seq_len);
        traceBF16(layer, "weighted_v", buffers.attn_output, 2048);
        outputProjection(cublas_handle, buffers.attn_output, weights.layers[layer].w_o, buffers.o_output, 1);
        traceBF16(layer, "output_projection", buffers.o_output, 2048);
        residualAdd(hidden_states, buffers.o_output, 1);
        traceBF16(layer, "attention_residual", hidden_states, 2048);
        OP_KERNEL("rmsnorm", rmsNormKernel<<<1, 1024>>>(hidden_states, buffers.rms_output, weights.layers[layer].post_attention_norm));
        traceBF16(layer, "post_attention_norm", buffers.rms_output, 2048);
        gateProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_gate, buffers.gate, 1);
        traceBF16(layer, "gate", buffers.gate, 8192);
        upProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_up, buffers.up, 1);
        traceBF16(layer, "up", buffers.up, 8192);
        silu(buffers.gate, buffers.up, 1);
        traceBF16(layer, "silu", buffers.gate, 8192);
        downProjection(cublas_handle, buffers.gate, weights.layers[layer].w_down, buffers.down, 1);
        traceBF16(layer, "down", buffers.down, 2048);
        residualAdd(hidden_states, buffers.down, 1);
        traceBF16(layer, "mlp_residual", hidden_states, 2048);
    }
    cache.current_seq_len += 1;
    OP_KERNEL("rmsnorm", rmsNormKernel<<<1, 1024>>>(hidden_states, buffers.rms_output, weights.final_norm));
        traceBF16(16, "final_norm", buffers.rms_output, 2048);
    lmHeadProjection(cublas_handle, buffers.rms_output, weights.embedding, logits);
    traceFloat(16, "logits", logits, VOCAB_SIZE);
    return argmaxLogits(logits);
}
