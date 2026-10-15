#include <cstddef>
#include "operation_profile.cuh"
#include "paged_kvcache.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

constexpr int NUM_LAYERS = 16;
constexpr int KV_DIM = 512;

static void checkPagedCUDA(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

static void validatePagedCache(const PagedKVCache& cache) {
    if (!cache.k_blocks || !cache.v_blocks || cache.num_blocks <= 0 || cache.block_size != BLOCK_SIZE) {
        throw std::invalid_argument("Paged cache is not allocated");
    }
}

static void validateSequenceBlocks(const PagedKVCache& cache, const SequenceKVState& sequence) {
    validatePagedCache(cache);
    for (size_t i = 0; i < sequence.block_table.size(); ++i) {
        int block = sequence.block_table[i];
        if (block < 0 || block >= cache.num_blocks ||
            std::find(cache.free_blocks.begin(), cache.free_blocks.end(), block) != cache.free_blocks.end() ||
            std::find(sequence.block_table.begin(), sequence.block_table.begin() + i, block) != sequence.block_table.begin() + i) {
            throw std::invalid_argument("Invalid or duplicate sequence block");
        }
    }
}

void allocatePagedKVCache(PagedKVCache& cache, int num_blocks) {
    OP_SCOPE("kv_allocate",false);
    operationAllocation(2);
    if (cache.k_blocks || cache.v_blocks || cache.num_blocks != 0 || !cache.free_blocks.empty()) {
        throw std::invalid_argument("Paged cache already owns storage");
    }
    constexpr size_t bytes_per_block = static_cast<size_t>(NUM_LAYERS) * BLOCK_SIZE * KV_DIM * sizeof(__nv_bfloat16);
    if (num_blocks <= 0 || static_cast<size_t>(num_blocks) > std::numeric_limits<size_t>::max() / bytes_per_block) {
        throw std::invalid_argument("Invalid paged cache capacity");
    }
    PagedKVCache pending{};
    try {
        pending.free_blocks.reserve(num_blocks);
        for (int block = num_blocks - 1; block >= 0; --block) {
            pending.free_blocks.push_back(block);
        }
        size_t bytes = static_cast<size_t>(num_blocks) * bytes_per_block;
        checkPagedCUDA(cudaMalloc(reinterpret_cast<void**>(&pending.k_blocks), bytes));
        checkPagedCUDA(cudaMalloc(reinterpret_cast<void**>(&pending.v_blocks), bytes));
        pending.num_blocks = num_blocks;
    } catch (...) {
        freePagedKVCache(pending);
        throw;
    }
    cache = std::move(pending);
}

void freePagedKVCache(PagedKVCache& cache) {
    OP_SCOPE("kv_free",false);
    operationFree(2);
    cudaFree(cache.k_blocks);
    cudaFree(cache.v_blocks);
    cache = PagedKVCache{};
}

int allocatePhysicalBlock(PagedKVCache& cache) {
    validatePagedCache(cache);
    if (cache.free_blocks.empty()) {
        throw std::runtime_error("Paged KV cache has no free blocks");
    }
    int block = cache.free_blocks.back();
    cache.free_blocks.pop_back();
    return block;
}

void freePhysicalBlock(PagedKVCache& cache, int physical_block) {
    validatePagedCache(cache);
    if (physical_block < 0 || physical_block >= cache.num_blocks ||
        std::find(cache.free_blocks.begin(), cache.free_blocks.end(), physical_block) != cache.free_blocks.end()) {
        throw std::invalid_argument("Invalid block or double free");
    }
    cache.free_blocks.push_back(physical_block);
}

void freeSequenceBlocks(PagedKVCache& cache, SequenceKVState& sequence) {
    validateSequenceBlocks(cache, sequence);
    for (int block : sequence.block_table) {
        freePhysicalBlock(cache, block);
    }
    sequence.block_table.clear();
    sequence.current_seq_len = 0;
}

void reserveSequenceBlocks(PagedKVCache& cache, SequenceKVState& sequence, int total_seq_len) {
    validateSequenceBlocks(cache, sequence);
    if (sequence.current_seq_len < 0 || total_seq_len < sequence.current_seq_len || total_seq_len > 1024 ||
        static_cast<size_t>((sequence.current_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE) > sequence.block_table.size()) {
        throw std::invalid_argument("Invalid sequence length or incomplete block table");
    }
    size_t needed = (total_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (needed <= sequence.block_table.size()) {
        return;
    }
    if (needed - sequence.block_table.size() > cache.free_blocks.size()) {
        throw std::runtime_error("Insufficient free paged KV blocks");
    }
    sequence.block_table.reserve(needed);
    while (sequence.block_table.size() < needed) {
        sequence.block_table.push_back(allocatePhysicalBlock(cache));
    }
}

void storePagedKV(PagedKVCache& cache, SequenceKVState& sequence, int layer,
                  const __nv_bfloat16* k, const __nv_bfloat16* v, int num_tokens) {
    OP_SCOPE("kv_store",true);
    if (layer < 0 || layer >= NUM_LAYERS || num_tokens <= 0 || !k || !v ||
        sequence.current_seq_len < 0 || sequence.current_seq_len > 1024 || num_tokens > 1024 - sequence.current_seq_len) {
        throw std::invalid_argument("Invalid paged KV store");
    }
    reserveSequenceBlocks(cache, sequence, sequence.current_seq_len + num_tokens);
    for (int token = 0; token < num_tokens;) {
        int position = sequence.current_seq_len + token;
        int slot = position % BLOCK_SIZE;
        int physical_block = sequence.block_table[position / BLOCK_SIZE];
        int count = std::min(num_tokens - token, BLOCK_SIZE - slot);
        size_t offset = ((static_cast<size_t>(layer) * cache.num_blocks + physical_block) * BLOCK_SIZE + slot) * KV_DIM;
        size_t source = static_cast<size_t>(token) * KV_DIM;
        size_t bytes = static_cast<size_t>(count) * KV_DIM * sizeof(__nv_bfloat16);
        checkPagedCUDA(cudaMemcpy(cache.k_blocks + offset, k + source, bytes, cudaMemcpyDeviceToDevice));
        checkPagedCUDA(cudaMemcpy(cache.v_blocks + offset, v + source, bytes, cudaMemcpyDeviceToDevice));
        token += count;
    }
}
