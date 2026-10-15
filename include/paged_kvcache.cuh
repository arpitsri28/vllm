#pragma once

#include <cuda_bf16.h>
#include <vector>

constexpr int BLOCK_SIZE = 16;

struct PagedKVCache {
    __nv_bfloat16* k_blocks = nullptr;
    __nv_bfloat16* v_blocks = nullptr;
    int num_blocks = 0;
    int block_size = BLOCK_SIZE;
    std::vector<int> free_blocks;
};

struct SequenceKVState {
    std::vector<int> block_table;
    int current_seq_len = 0;
};

void allocatePagedKVCache(PagedKVCache& cache, int num_blocks);
void freePagedKVCache(PagedKVCache& cache);
int allocatePhysicalBlock(PagedKVCache& cache);
void freePhysicalBlock(PagedKVCache& cache, int physical_block);
void freeSequenceBlocks(PagedKVCache& cache, SequenceKVState& sequence);
void reserveSequenceBlocks(PagedKVCache& cache, SequenceKVState& sequence, int total_seq_len);
void storePagedKV(PagedKVCache& cache, SequenceKVState& sequence, int layer,
                  const __nv_bfloat16* k, const __nv_bfloat16* v, int num_tokens);
