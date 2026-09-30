#include "kvcache.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

constexpr int NUM_LAYERS = 16;
constexpr int KV_DIM = 512;

void allocateKVCache(KVCache& cache, int max_seq_len) {
  cache.max_seq_len = max_seq_len;
  cache.current_seq_len = 0;

  size_t cache_elements = NUM_LAYERS * max_seq_len * KV_DIM;

  cudaError_t k_status = cudaMalloc(reinterpret_cast<void**>(&cache.k_cache), cache_elements * sizeof(__nv_bfloat16));

  cudaError_t v_status = cudaMalloc(reinterpret_cast<void**>(&cache.v_cache), cache_elements * sizeof(__nv_bfloat16));

  if (k_status != cudaSuccess || v_status != cudaSuccess) {
      throw std::runtime_error(cudaGetErrorString(k_status != cudaSuccess ? k_status : v_status));
  }
}

void freeKVCache(KVCache& cache) {
  cudaFree(cache.k_cache);
  cudaFree(cache.v_cache);

  cache.k_cache = nullptr;
  cache.v_cache = nullptr;

  cache.max_seq_len = 0;
  cache.current_seq_len = 0;
}

void storeKV(KVCache& cache, int layer, const __nv_bfloat16* k, const __nv_bfloat16* v, int num_tokens) {
  size_t offset = static_cast<size_t>(layer) * cache.max_seq_len * KV_DIM + static_cast<size_t>(cache.current_seq_len) * KV_DIM;
  size_t copy_size = static_cast<size_t>(num_tokens) * KV_DIM * sizeof(__nv_bfloat16);
  cudaError_t status = cudaMemcpy(cache.k_cache + offset, k, copy_size, cudaMemcpyDeviceToDevice);
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
  status = cudaMemcpy(cache.v_cache + offset, v, copy_size, cudaMemcpyDeviceToDevice);
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}