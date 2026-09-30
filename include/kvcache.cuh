#pragma once

#include <cuda_bf16.h>

struct KVCache {
    __nv_bfloat16* k_cache;
    __nv_bfloat16* v_cache;

    int max_seq_len;
    int current_seq_len;
};

void allocateKVCache(
    KVCache& cache,
    int max_seq_len
);

void freeKVCache(
    KVCache& cache
);

void storeKV(
    KVCache& cache,
    int layer,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    int num_tokens
);
