#include <cstddef>
#include "operation_profile.cuh"
#include <cuda_bf16.h>
#include "numerical_trace.cuh"
#include "paged_transformer.cuh"
#include "paged_attention.cuh"
#include "rmsnorm.cuh"
#include "attention.cuh"
#include "rope.cuh"
#include "residual.cuh"
#include "mlp.cuh"
#include "silu.cuh"
#include "lm_head.cuh"
#include "argmax.cuh"
#include "causalMask.cuh"
#include "softmax.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

constexpr int NUM_LAYERS = 16;
constexpr int NUM_Q_HEADS = 32;
constexpr int EMBEDDING_LENGTH = 2048;

int transformerForwardPaged(cublasHandle_t cublas_handle, __nv_bfloat16* hidden_states, const ModelWeights& weights, PagedKVCache& cache, SequenceKVState& sequence, TransformerBuffers& buffers, float* logits, int num_tokens) {
  if (sequence.current_seq_len != 0 || num_tokens <= 0 || num_tokens > 1024) {
    throw std::invalid_argument("Paged prefill requires an empty sequence and 1..1024 tokens");
  }
  reserveSequenceBlocks(cache, sequence, num_tokens);
  for (int layer = 0; layer < NUM_LAYERS; ++layer) {
    OP_KERNEL("rmsnorm", rmsNormKernel<<<num_tokens, 1024>>>(hidden_states, buffers.rms_output, weights.layers[layer].input_norm));
    qProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_q, buffers.q, num_tokens);
    kProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_k, buffers.k, num_tokens);
    vProjection(cublas_handle, buffers.rms_output, weights.layers[layer].w_v, buffers.v, num_tokens);
    rope(buffers.q, num_tokens, 2048, 0);
    rope(buffers.k, num_tokens, 512, 0);
    storePagedKV(cache, sequence, layer, buffers.k, buffers.v, num_tokens);
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
  sequence.current_seq_len += num_tokens;
  OP_KERNEL("rmsnorm", rmsNormKernel<<<num_tokens, 1024>>>(hidden_states, buffers.rms_output, weights.final_norm));
  __nv_bfloat16* last_hidden = buffers.rms_output + (num_tokens - 1) * EMBEDDING_LENGTH;
  lmHeadProjection(cublas_handle, last_hidden, weights.embedding, logits);
  int next_token = argmaxLogits(logits);
  return next_token;
}


std::vector<int> decodeBatchPaged(cublasHandle_t cublas_handle, __nv_bfloat16* hidden_states,
                                 const ModelWeights& weights, PagedKVCache& cache,
                                 const std::vector<SequenceKVState*>& sequences,
                                 TransformerBuffers& buffers, float* logits) {
    PagedBatchMetadata metadata{};
    try {
        preparePagedBatch(cache, sequences, metadata);
        int batch_size = metadata.batch_size;
        for (int layer = 0; layer < NUM_LAYERS; ++layer) {
            OP_KERNEL("rmsnorm", rmsNormKernel<<<batch_size, 1024>>>(hidden_states, buffers.rms_output, weights.layers[layer].input_norm));
            traceBF16(layer, "input_norm", buffers.rms_output + numericalTraceRow * 2048, 2048);
            // Row-stable cuBLAS shape preserves the validated BF16 rounding contract.
            for (int row = 0; row < batch_size; ++row)
                qProjection(cublas_handle, buffers.rms_output + row * 2048, weights.layers[layer].w_q, buffers.q + row * 2048, 1);
            traceBF16(layer, "q", buffers.q + numericalTraceRow * 2048, 2048);
            for (int row = 0; row < batch_size; ++row)
                kProjection(cublas_handle, buffers.rms_output + row * 2048, weights.layers[layer].w_k, buffers.k + row * 512, 1);
            traceBF16(layer, "k", buffers.k + numericalTraceRow * 512, 512);
            for (int row = 0; row < batch_size; ++row)
                vProjection(cublas_handle, buffers.rms_output + row * 2048, weights.layers[layer].w_v, buffers.v + row * 512, 1);
            traceBF16(layer, "v", buffers.v + numericalTraceRow * 512, 512);
            ropeBatch(buffers.q, batch_size, 2048, metadata.positions);
            traceBF16(layer, "rope_q", buffers.q + numericalTraceRow * 2048, 2048);
            ropeBatch(buffers.k, batch_size, 512, metadata.positions);
            traceBF16(layer, "rope_k", buffers.k + numericalTraceRow * 512, 512);
            storePagedKVBatch(cache, metadata, layer, buffers.k, buffers.v);
            computePagedAttentionScores(cublas_handle, buffers.q, cache, metadata, layer, buffers.attn_scores);
                traceBF16(layer, "scores", buffers.attn_scores + numericalTraceRow * 32 * metadata.score_stride, sequences[numericalTraceRow]->current_seq_len + 1, 32, metadata.score_stride);
            for (int row = 0; row < batch_size; ++row) {
                int length = sequences[row]->current_seq_len + 1;
                for (int head = 0; head < NUM_Q_HEADS; ++head) {
                    softmaxRows(buffers.attn_scores + (row * NUM_Q_HEADS + head) * metadata.score_stride, 1, length);
                }
            }
                traceBF16(layer, "probabilities", buffers.attn_scores + numericalTraceRow * 32 * metadata.score_stride, sequences[numericalTraceRow]->current_seq_len + 1, 32, metadata.score_stride);
            pagedAttentionScoresTimesV(cublas_handle, buffers.attn_scores, cache, metadata, layer, buffers.attn_output);
            traceBF16(layer, "weighted_v", buffers.attn_output + numericalTraceRow * 2048, 2048);
            for (int row = 0; row < batch_size; ++row)
                outputProjection(cublas_handle, buffers.attn_output + row * 2048, weights.layers[layer].w_o, buffers.o_output + row * 2048, 1);
            traceBF16(layer, "output_projection", buffers.o_output + numericalTraceRow * 2048, 2048);
            residualAdd(hidden_states, buffers.o_output, batch_size);
            traceBF16(layer, "attention_residual", hidden_states + numericalTraceRow * 2048, 2048);
            OP_KERNEL("rmsnorm", rmsNormKernel<<<batch_size, 1024>>>(hidden_states, buffers.rms_output, weights.layers[layer].post_attention_norm));
            traceBF16(layer, "post_attention_norm", buffers.rms_output + numericalTraceRow * 2048, 2048);
            for (int row = 0; row < batch_size; ++row)
                gateProjection(cublas_handle, buffers.rms_output + row * 2048, weights.layers[layer].w_gate, buffers.gate + row * 8192, 1);
            traceBF16(layer, "gate", buffers.gate + numericalTraceRow * 8192, 8192);
            for (int row = 0; row < batch_size; ++row)
                upProjection(cublas_handle, buffers.rms_output + row * 2048, weights.layers[layer].w_up, buffers.up + row * 8192, 1);
            traceBF16(layer, "up", buffers.up + numericalTraceRow * 8192, 8192);
            silu(buffers.gate, buffers.up, batch_size);
            traceBF16(layer, "silu", buffers.gate + numericalTraceRow * 8192, 8192);
            for (int row = 0; row < batch_size; ++row)
                downProjection(cublas_handle, buffers.gate + row * 8192, weights.layers[layer].w_down, buffers.down + row * 2048, 1);
            traceBF16(layer, "down", buffers.down + numericalTraceRow * 2048, 2048);
            residualAdd(hidden_states, buffers.down, batch_size);
            traceBF16(layer, "mlp_residual", hidden_states + numericalTraceRow * 2048, 2048);
        }
        OP_KERNEL("rmsnorm", rmsNormKernel<<<batch_size, 1024>>>(hidden_states, buffers.rms_output, weights.final_norm));
        traceBF16(16, "final_norm", buffers.rms_output + numericalTraceRow * 2048, 2048);
        for (int row = 0; row < batch_size; ++row)
            lmHeadProjection(cublas_handle, buffers.rms_output + row * EMBEDDING_LENGTH,
                weights.embedding, logits + static_cast<size_t>(row) * VOCAB_SIZE);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(error));
        }
        traceFloat(16, "logits", logits + numericalTraceRow * VOCAB_SIZE, VOCAB_SIZE);
        std::vector<int> tokens;
        tokens.reserve(batch_size);
        for (int row = 0; row < batch_size; ++row) {
            tokens.push_back(argmaxLogits(logits + static_cast<size_t>(row) * VOCAB_SIZE));
        }
        for (auto* sequence : sequences) {
            sequence->current_seq_len += 1;
        }
        freePagedBatch(metadata);
        return tokens;
    } catch (...) {
        freePagedBatch(metadata);
        throw;
    }
}

int decodeForwardPaged(cublasHandle_t cublas_handle, __nv_bfloat16* hidden_states,
                       const ModelWeights& weights, PagedKVCache& cache,
                       SequenceKVState& sequence, TransformerBuffers& buffers, float* logits) {
    return decodeBatchPaged(cublas_handle, hidden_states, weights, cache, {&sequence}, buffers, logits)[0];
}
