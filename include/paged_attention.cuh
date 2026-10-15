#pragma once

#include "paged_kvcache.cuh"
#include <cublas_v2.h>

struct PagedBatchMetadata {
    int* block_tables = nullptr;
    int* positions = nullptr;
    int batch_size = 0;
    int score_stride = 0;
    // Exact BF16 logical cache views for the canonical cuBLAS decode arithmetic.
    __nv_bfloat16* linear_k = nullptr;
    __nv_bfloat16* linear_v = nullptr;
    std::vector<int> lengths;
};

void preparePagedBatch(PagedKVCache& cache, const std::vector<SequenceKVState*>& sequences,
                       PagedBatchMetadata& metadata);
void freePagedBatch(PagedBatchMetadata& metadata);
void storePagedKVBatch(PagedKVCache& cache, const PagedBatchMetadata& metadata,
                       int layer, const __nv_bfloat16* k, const __nv_bfloat16* v);
void computePagedAttentionScores(cublasHandle_t handle, const __nv_bfloat16* q, const PagedKVCache& cache,
                                 const PagedBatchMetadata& metadata, int layer,
                                 __nv_bfloat16* scores);
void pagedAttentionScoresTimesV(cublasHandle_t handle, const __nv_bfloat16* scores, const PagedKVCache& cache,
                               const PagedBatchMetadata& metadata, int layer,
                               __nv_bfloat16* attention_output);
