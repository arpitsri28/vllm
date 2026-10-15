#include <cstddef>
#include "operation_profile.cuh"
#include "benchmark_trace.cuh"
#include "paged_attention.cuh"
#include "attention.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

constexpr int KV_DIM = 512;
constexpr int NUM_Q_HEADS = 32;
constexpr int MAX_BLOCKS_PER_SEQUENCE = 1024 / BLOCK_SIZE;

static void checkPagedAttentionCUDA(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

void freePagedBatch(PagedBatchMetadata& metadata) {
    OP_SCOPE("metadata_free",false);
    operationFree(4);
    cudaFree(metadata.block_tables);
    cudaFree(metadata.positions);
    cudaFree(metadata.linear_k);
    cudaFree(metadata.linear_v);
    metadata = PagedBatchMetadata{};
}

void preparePagedBatch(PagedKVCache& cache, const std::vector<SequenceKVState*>& sequences,
                       PagedBatchMetadata& metadata) {
    OP_SCOPE("metadata_allocate_upload",false);
    operationAllocation(4);
    if (!cache.k_blocks || !cache.v_blocks || cache.num_blocks <= 0 || cache.block_size != BLOCK_SIZE ||
        sequences.empty() || sequences.size() > 4 || metadata.block_tables || metadata.positions || metadata.linear_k || metadata.linear_v) {
        throw std::invalid_argument("Invalid paged decode batch");
    }
    metadata = PagedBatchMetadata{};
    size_t needed = 0;
    std::vector<int> owners(cache.num_blocks, -1);
    for (size_t row = 0; row < sequences.size(); ++row) {
        const auto* sequence = sequences[row];
        if (!sequence || sequence->current_seq_len < 1 || sequence->current_seq_len >= 1024) {
            throw std::invalid_argument("Paged decode requires 1..1023 cached tokens");
        }
        for (int block : sequence->block_table) {
            if (block < 0 || block >= cache.num_blocks || owners[block] != -1) {
                throw std::invalid_argument("Paged decode sequences must own distinct blocks");
            }
            owners[block] = static_cast<int>(row);
        }
        size_t required = (sequence->current_seq_len + BLOCK_SIZE) / BLOCK_SIZE;
        if (required > sequence->block_table.size()) {
            needed += required - sequence->block_table.size();
        }
    }
    if (needed > cache.free_blocks.size()) {
        throw std::runtime_error("Insufficient blocks for decode batch");
    }
    std::vector<int> tables(sequences.size() * MAX_BLOCKS_PER_SEQUENCE, -1);
    std::vector<int> positions;
    for (size_t row = 0; row < sequences.size(); ++row) {
        auto& sequence = *sequences[row];
        reserveSequenceBlocks(cache, sequence, sequence.current_seq_len + 1);
        if (sequence.block_table.size() > MAX_BLOCKS_PER_SEQUENCE) {
            throw std::invalid_argument("Paged block table exceeds context limit");
        }
        std::copy(sequence.block_table.begin(), sequence.block_table.end(), tables.begin() + row * MAX_BLOCKS_PER_SEQUENCE);
        positions.push_back(sequence.current_seq_len);
        metadata.score_stride = std::max(metadata.score_stride, sequence.current_seq_len + 1);
    }
    try {
        checkPagedAttentionCUDA(cudaMalloc(reinterpret_cast<void**>(&metadata.block_tables), tables.size() * sizeof(int)));
        checkPagedAttentionCUDA(cudaMalloc(reinterpret_cast<void**>(&metadata.positions), positions.size() * sizeof(int)));
        checkPagedAttentionCUDA(cudaMemcpy(metadata.block_tables, tables.data(), tables.size() * sizeof(int), cudaMemcpyHostToDevice));
        checkPagedAttentionCUDA(cudaMemcpy(metadata.positions, positions.data(), positions.size() * sizeof(int), cudaMemcpyHostToDevice));
        metadata.batch_size = static_cast<int>(sequences.size());
        for (int position : positions) metadata.lengths.push_back(position + 1);
        size_t linear_bytes = sequences.size() * metadata.score_stride * KV_DIM * sizeof(__nv_bfloat16);
        checkPagedAttentionCUDA(cudaMalloc(reinterpret_cast<void**>(&metadata.linear_k), linear_bytes));
        checkPagedAttentionCUDA(cudaMalloc(reinterpret_cast<void**>(&metadata.linear_v), linear_bytes));
        benchMemory();
    } catch (...) {
        freePagedBatch(metadata);
        throw;
    }
}

static __global__ void storePagedKVBatchKernel(__nv_bfloat16* k_blocks, __nv_bfloat16* v_blocks,
                                               int num_blocks, int layer, const int* tables,
                                               const int* positions, const __nv_bfloat16* k,
                                               const __nv_bfloat16* v) {
    int row = blockIdx.x;
    int position = positions[row];
    int physical = tables[row * MAX_BLOCKS_PER_SEQUENCE + position / BLOCK_SIZE];
    size_t offset = ((static_cast<size_t>(layer) * num_blocks + physical) * BLOCK_SIZE + position % BLOCK_SIZE) * KV_DIM;
    k_blocks[offset + threadIdx.x] = k[row * KV_DIM + threadIdx.x];
    v_blocks[offset + threadIdx.x] = v[row * KV_DIM + threadIdx.x];
}

// Gather changes layout only: BF16 payload bits are copied without arithmetic.
static __global__ void gatherPagedKernel(const __nv_bfloat16* blocks, int num_blocks,
                                         int layer, const int* tables, const int* positions,
                                         int stride, __nv_bfloat16* linear) {
    int row = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int token = index / KV_DIM;
    if (token > positions[row]) return;
    int physical = tables[row * MAX_BLOCKS_PER_SEQUENCE + token / BLOCK_SIZE];
    size_t source = ((static_cast<size_t>(layer) * num_blocks + physical) * BLOCK_SIZE + token % BLOCK_SIZE) * KV_DIM;
    linear[static_cast<size_t>(row) * stride * KV_DIM + index] = blocks[source + index % KV_DIM];
}

static void validatePagedAttention(const PagedKVCache& cache, const PagedBatchMetadata& metadata, int layer) {
    if (!cache.k_blocks || !cache.v_blocks || !metadata.block_tables || !metadata.positions || !metadata.linear_k || !metadata.linear_v ||
        metadata.lengths.size() != static_cast<size_t>(metadata.batch_size) ||
        metadata.batch_size < 1 || metadata.batch_size > 4 || metadata.score_stride < 1 ||
        metadata.score_stride > 1024 || layer < 0 || layer >= 16) {
        throw std::invalid_argument("Invalid paged attention arguments");
    }
}

void storePagedKVBatch(PagedKVCache& cache, const PagedBatchMetadata& metadata,
                       int layer, const __nv_bfloat16* k, const __nv_bfloat16* v) {
    OP_SCOPE("kv_store",true);
    validatePagedAttention(cache, metadata, layer);
    operationKernel();
    storePagedKVBatchKernel<<<metadata.batch_size, KV_DIM>>>(cache.k_blocks, cache.v_blocks, cache.num_blocks,
        layer, metadata.block_tables, metadata.positions, k, v);
    checkPagedAttentionCUDA(cudaGetLastError());
}

void computePagedAttentionScores(cublasHandle_t handle, const __nv_bfloat16* q, const PagedKVCache& cache,
                                 const PagedBatchMetadata& metadata, int layer, __nv_bfloat16* scores) {
    validatePagedAttention(cache, metadata, layer);
    dim3 grid((metadata.score_stride * KV_DIM + 255) / 256, metadata.batch_size);
    { OP_SCOPE("kv_gather",true); operationKernel();
    gatherPagedKernel<<<grid, 256>>>(cache.k_blocks, cache.num_blocks, layer, metadata.block_tables,
        metadata.positions, metadata.score_stride, metadata.linear_k); }
    checkPagedAttentionCUDA(cudaGetLastError());
    for (int row = 0; row < metadata.batch_size; ++row) {
        computeDecodeAttentionScores(handle, q + row * 2048,
            metadata.linear_k + static_cast<size_t>(row) * metadata.score_stride * KV_DIM,
            scores + row * NUM_Q_HEADS * metadata.score_stride, metadata.lengths[row], metadata.score_stride);
    }
}

void pagedAttentionScoresTimesV(cublasHandle_t handle, const __nv_bfloat16* scores, const PagedKVCache& cache,
                               const PagedBatchMetadata& metadata, int layer, __nv_bfloat16* attention_output) {
    validatePagedAttention(cache, metadata, layer);
    dim3 grid((metadata.score_stride * KV_DIM + 255) / 256, metadata.batch_size);
    { OP_SCOPE("kv_gather",true); operationKernel();
    gatherPagedKernel<<<grid, 256>>>(cache.v_blocks, cache.num_blocks, layer, metadata.block_tables,
        metadata.positions, metadata.score_stride, metadata.linear_v); }
    checkPagedAttentionCUDA(cudaGetLastError());
    for (int row = 0; row < metadata.batch_size; ++row) {
        decodeAttentionScoresTimesV(handle, scores + row * NUM_Q_HEADS * metadata.score_stride,
            metadata.linear_v + static_cast<size_t>(row) * metadata.score_stride * KV_DIM,
            attention_output + row * 2048, metadata.lengths[row], metadata.score_stride);
    }
}
