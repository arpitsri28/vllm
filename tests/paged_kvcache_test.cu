#include "paged_kvcache.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

static void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

static __nv_bfloat16 value(int layer, int position, int d, bool v) {
    int number = v ? (layer * 11 + position * 7 + d % 23) % 103 - 51
                   : (layer * 13 + position * 3 + d % 17) % 101 - 50;
    return __float2bfloat16_rn(number / 8.0f);
}

static void runBoundary(int prompt_length) {
    PagedKVCache cache{};
    SequenceKVState sequence{};
    __nv_bfloat16* k = nullptr;
    __nv_bfloat16* v = nullptr;
    auto cleanup = [&]() {
        cudaFree(k);
        cudaFree(v);
        freePagedKVCache(cache);
    };
    try {
        allocatePagedKVCache(cache, 9);
        for (int block = 0; block < 9; ++block) {
            require(allocatePhysicalBlock(cache) == block, "Unexpected initial allocation order");
        }
        for (int block : {8, 2, 5}) {
            freePhysicalBlock(cache, block);
        }
        check(cudaMalloc(reinterpret_cast<void**>(&k), prompt_length * 512 * sizeof(__nv_bfloat16)));
        check(cudaMalloc(reinterpret_cast<void**>(&v), prompt_length * 512 * sizeof(__nv_bfloat16)));
        for (int pass = 0; pass < 3; ++pass) {
            int count = pass == 0 ? prompt_length : 1;
            int start = sequence.current_seq_len;
            auto previous_table = sequence.block_table;
            for (int layer = 0; layer < 16; ++layer) {
                std::vector<__nv_bfloat16> host_k(count * 512), host_v(count * 512);
                for (int token = 0; token < count; ++token) {
                    for (int d = 0; d < 512; ++d) {
                        host_k[token * 512 + d] = value(layer, start + token, d, false);
                        host_v[token * 512 + d] = value(layer, start + token, d, true);
                    }
                }
                check(cudaMemcpy(k, host_k.data(), host_k.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
                check(cudaMemcpy(v, host_v.data(), host_v.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
                storePagedKV(cache, sequence, layer, k, v, count);
                require(sequence.current_seq_len == start, "Store advanced sequence length per layer");
                require(sequence.block_table.size() == static_cast<size_t>((start + count + 15) / 16), "Incorrect lazy page allocation");
            }
            require(std::equal(previous_table.begin(), previous_table.end(), sequence.block_table.begin()), "Existing page mappings changed");
            sequence.current_seq_len += count;
            std::vector<int> expected = {5, 2, 8};
            require(std::equal(sequence.block_table.begin(), sequence.block_table.end(), expected.begin()), "Fragmented block translation failed");
            require(cache.free_blocks.size() == 3 - sequence.block_table.size(), "Unexpected free block count");
            for (int layer = 0; layer < 16; ++layer) {
                for (int token = 0; token < sequence.current_seq_len; ++token) {
                    int physical = sequence.block_table[token / 16];
                    size_t offset = ((static_cast<size_t>(layer) * cache.num_blocks + physical) * 16 + token % 16) * 512;
                    std::vector<__nv_bfloat16> actual_k(512), actual_v(512);
                    check(cudaMemcpy(actual_k.data(), cache.k_blocks + offset, 512 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
                    check(cudaMemcpy(actual_v.data(), cache.v_blocks + offset, 512 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
                    for (int d = 0; d < 512; ++d) {
                        require(__bfloat162float(actual_k[d]) == __bfloat162float(value(layer, token, d, false)), "K page data mismatch");
                        require(__bfloat162float(actual_v[d]) == __bfloat162float(value(layer, token, d, true)), "V page data mismatch");
                    }
                }
            }
            std::cout << "prompt=" << prompt_length << " stored_length=" << sequence.current_seq_len
                      << " pages=" << sequence.block_table.size() << '\n';
        }
        auto released = sequence.block_table;
        freeSequenceBlocks(cache, sequence);
        require(sequence.current_seq_len == 0 && sequence.block_table.empty(), "Sequence release failed");
        require(cache.free_blocks.size() == 3, "Blocks were not returned");
        for (auto it = released.rbegin(); it != released.rend(); ++it) {
            require(allocatePhysicalBlock(cache) == *it, "Released page was not reused");
        }
        check(cudaDeviceSynchronize());
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
}

static void testExhaustion() {
    PagedKVCache cache{};
    SequenceKVState sequence{};
    try {
        allocatePagedKVCache(cache, 2);
        bool rejected = false;
        try {
            reserveSequenceBlocks(cache, sequence, 33);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected && sequence.block_table.empty() && cache.free_blocks.size() == 2, "Exhaustion partially mutated the pool");
        int first = allocatePhysicalBlock(cache);
        allocatePhysicalBlock(cache);
        rejected = false;
        try {
            allocatePhysicalBlock(cache);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        require(rejected, "Exhausted pool did not fail");
        freePhysicalBlock(cache, first);
        rejected = false;
        try {
            freePhysicalBlock(cache, first);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected && cache.free_blocks.size() == 1, "Double free was accepted");
    } catch (...) {
        freePagedKVCache(cache);
        throw;
    }
    freePagedKVCache(cache);
}

int main() {
    try {
        for (int length : {1, 15, 16, 17, 31, 32, 33}) {
            runBoundary(length);
        }
        testExhaustion();
        std::cout << "Paged cache boundary, fragmentation, reuse, and exhaustion checks passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
