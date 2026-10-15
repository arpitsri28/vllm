#include "paged_attention.cuh"
#include "attention.cuh"
#include "softmax.cuh"
#include "rope.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

static void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    PagedKVCache cache{};
    PagedBatchMetadata metadata{};
    std::vector<SequenceKVState> states(3);
    std::vector<int> lengths = {16, 17, 33};
    cublasHandle_t handle = nullptr;
    __nv_bfloat16 *q = nullptr, *reference_q = nullptr, *k = nullptr, *v = nullptr;
    __nv_bfloat16 *scores = nullptr, *reference_scores = nullptr, *output = nullptr, *reference_output = nullptr;
    __nv_bfloat16 *last_k = nullptr, *last_v = nullptr;
    auto cleanup = [&]() {
        freePagedBatch(metadata);
        freePagedKVCache(cache);
        for (auto* pointer : {q, reference_q, k, v, scores, reference_scores, output, reference_output, last_k, last_v}) {
            cudaFree(pointer);
        }
        if (handle) {
            cublasDestroy(handle);
        }
    };
    try {
        require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cuBLAS initialization failed");
        allocatePagedKVCache(cache, 8);
        for (auto& state : states) {
            reserveSequenceBlocks(cache, state, 1);
        }
        auto allocate = [&](auto** pointer, size_t count) {
            check(cudaMalloc(reinterpret_cast<void**>(pointer), count * sizeof(__nv_bfloat16)));
        };
        allocate(&q, 3 * 2048);
        allocate(&reference_q, 3 * 2048);
        allocate(&k, 3 * 33 * 512);
        allocate(&v, 3 * 33 * 512);
        allocate(&scores, 3 * 32 * 33);
        allocate(&reference_scores, 32 * 33);
        allocate(&output, 3 * 2048);
        allocate(&reference_output, 2048);
        allocate(&last_k, 3 * 512);
        allocate(&last_v, 3 * 512);
        std::vector<__nv_bfloat16> host_q(3 * 2048), host_k(3 * 33 * 512), host_v(3 * 33 * 512);
        std::vector<__nv_bfloat16> host_last_k(3 * 512), host_last_v(3 * 512);
        for (int row = 0; row < 3; ++row) {
            for (int d = 0; d < 2048; ++d) {
                host_q[row * 2048 + d] = __float2bfloat16_rn(((d * 3 + row) % 11 - 5) / 16.0f);
            }
            for (int token = 0; token < 33; ++token) {
                for (int d = 0; d < 512; ++d) {
                    size_t index = (row * 33 + token) * 512 + d;
                    host_k[index] = __float2bfloat16_rn(((token * 3 + d + row * 7) % 19 - 9) / 16.0f);
                    host_v[index] = __float2bfloat16_rn(((token + d * 5 + row * 11) % 23 - 11) / 16.0f);
                    if (token == lengths[row] - 1) {
                        host_last_k[row * 512 + d] = host_k[index];
                        host_last_v[row * 512 + d] = host_v[index];
                    }
                }
            }
        }
        check(cudaMemcpy(q, host_q.data(), host_q.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        check(cudaMemcpy(reference_q, q, host_q.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
        check(cudaMemcpy(k, host_k.data(), host_k.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        check(cudaMemcpy(v, host_v.data(), host_v.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        check(cudaMemcpy(last_k, host_last_k.data(), host_last_k.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        check(cudaMemcpy(last_v, host_last_v.data(), host_last_v.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        std::vector<SequenceKVState*> sequences;
        for (int row = 0; row < 3; ++row) {
            storePagedKV(cache, states[row], 0, k + row * 33 * 512, v + row * 33 * 512, lengths[row] - 1);
            states[row].current_seq_len = lengths[row] - 1;
            sequences.push_back(&states[row]);
        }
        preparePagedBatch(cache, sequences, metadata);
        storePagedKVBatch(cache, metadata, 0, last_k, last_v);
        ropeBatch(q, 3, 2048, metadata.positions);
        for (int row = 0; row < 3; ++row) {
            rope(reference_q + row * 2048, 1, 2048, lengths[row] - 1);
        }
        std::vector<__nv_bfloat16> a(3 * 2048), b(3 * 2048);
        check(cudaMemcpy(a.data(), q, a.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(b.data(), reference_q, b.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        require(std::memcmp(a.data(), b.data(), a.size() * sizeof(__nv_bfloat16)) == 0, "Batched RoPE changed row positions or math");
        check(cudaMemset(scores, 0x7f, 3 * 32 * 33 * sizeof(__nv_bfloat16)));
        computePagedAttentionScores(handle, q, cache, metadata, 0, scores);
        std::vector<__nv_bfloat16> actual_scores(3 * 32 * 33);
        check(cudaMemcpy(actual_scores.data(), scores, actual_scores.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        float max_score_error = 0.0f;
        for (int row = 0; row < 3; ++row) {
            computeDecodeAttentionScores(handle, q + row * 2048, k + row * 33 * 512, reference_scores, lengths[row]);
            std::vector<__nv_bfloat16> expected(32 * lengths[row]);
            check(cudaMemcpy(expected.data(), reference_scores, expected.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
            for (int head = 0; head < 32; ++head) {
                for (int token = 0; token < lengths[row]; ++token) {
                    float actual = __bfloat162float(actual_scores[(row * 32 + head) * 33 + token]);
                    float reference = __bfloat162float(expected[head * lengths[row] + token]);
                    require(std::isfinite(actual), "Non-finite paged attention score");
                    max_score_error = std::max(max_score_error, std::abs(actual - reference));
                }
                softmaxRows(scores + (row * 32 + head) * 33, 1, lengths[row]);
            }
        }
        pagedAttentionScoresTimesV(handle, scores, cache, metadata, 0, output);
        float max_value_error = 0.0f;
        for (int row = 0; row < 3; ++row) {
            computeDecodeAttentionScores(handle, q + row * 2048, k + row * 33 * 512, reference_scores, lengths[row]);
            softmaxRows(reference_scores, 32, lengths[row]);
            decodeAttentionScoresTimesV(handle, reference_scores, v + row * 33 * 512, reference_output, lengths[row]);
            check(cudaMemcpy(a.data(), output + row * 2048, 2048 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(b.data(), reference_output, 2048 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
            for (int d = 0; d < 2048; ++d) {
                float actual = __bfloat162float(a[d]);
                require(std::isfinite(actual), "Padding or invalid pages entered attention");
                max_value_error = std::max(max_value_error, std::abs(actual - __bfloat162float(b[d])));
            }
            require(states[row].current_seq_len == lengths[row] - 1, "Attention or store advanced sequence length");
        }
        check(cudaDeviceSynchronize());
        std::cout << "Ragged GQA score max error=" << max_score_error << ", attention output max error=" << max_value_error << '\n';
        require(max_score_error <= 0.001f && max_value_error <= 0.005f, "Paged attention exceeds synthetic-test tolerance");
    } catch (const std::exception& error) {
        cleanup();
        std::cerr << error.what() << '\n';
        return 1;
    }
    cleanup();
    return 0;
}
