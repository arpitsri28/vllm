#pragma once

#include "transformer.cuh"
#include "paged_kvcache.cuh"

int transformerForwardPaged(cublasHandle_t cublas_handle, __nv_bfloat16* hidden_states,
                            const ModelWeights& weights, PagedKVCache& cache,
                            SequenceKVState& sequence, TransformerBuffers& buffers,
                            float* logits, int num_tokens);

int decodeForwardPaged(cublasHandle_t cublas_handle, __nv_bfloat16* hidden_states,
                       const ModelWeights& weights, PagedKVCache& cache,
                       SequenceKVState& sequence, TransformerBuffers& buffers, float* logits);
std::vector<int> decodeBatchPaged(cublasHandle_t cublas_handle, __nv_bfloat16* hidden_states,
                                 const ModelWeights& weights, PagedKVCache& cache,
                                 const std::vector<SequenceKVState*>& sequences,
                                 TransformerBuffers& buffers, float* logits);
