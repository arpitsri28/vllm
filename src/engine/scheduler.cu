#include <cstddef>
#include "operation_profile.cuh"
#include "benchmark_trace.cuh"
#include "scheduler.cuh"
#include "paged_transformer.cuh"
#include "embedding.cuh"
#include "argmax.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

static bool requestFinished(const Request& request) {
    return request.generated_tokens.size() >= static_cast<size_t>(request.max_new_tokens) ||
           request.last_token == 128001 || request.last_token == 128008 || request.last_token == 128009;
}

void generateContinuous(cublasHandle_t cublas_handle, const ModelWeights& weights,
                        std::vector<Request>& requests, int max_batch_size, int max_seq_len,
                        std::vector<SchedulerEvent>* events) {
    if (max_batch_size < 1 || max_batch_size > MAX_BATCH_SIZE || max_seq_len < 1 || max_seq_len > 1024) {
        throw std::invalid_argument("Invalid scheduler capacity");
    }
    Scheduler scheduler;
    int max_prompt_len = 1;
    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& request = requests[i];
        if (request.prompt_tokens.empty() || request.prompt_tokens.size() > static_cast<size_t>(max_seq_len) ||
            request.max_new_tokens < 0 || request.max_new_tokens > max_seq_len - static_cast<int>(request.prompt_tokens.size()) ||
            !request.kv_state.block_table.empty() || request.kv_state.current_seq_len != 0 ||
            request.prefilled || request.finished || !request.generated_tokens.empty()) {
            throw std::invalid_argument("Requests must be fresh and fit the context limit");
        }
        for (size_t previous = 0; previous < i; ++previous) {
            if (requests[previous].request_id == request.request_id) {
                throw std::invalid_argument("Duplicate request ID");
            }
        }
        for (int token : request.prompt_tokens) {
            if (token < 0 || token >= VOCAB_SIZE) {
                throw std::invalid_argument("Prompt token is outside the vocabulary");
            }
        }
        max_prompt_len = std::max(max_prompt_len, static_cast<int>(request.prompt_tokens.size()));
        scheduler.waiting_queue.push(static_cast<int>(i));
    }
    if (events) {
        events->clear();
    }
    if (requests.empty()) {
        return;
    }
    scheduler.active_slots.assign(max_batch_size, -1);
    for (int slot = max_batch_size - 1; slot >= 0; --slot) {
        scheduler.free_slots.push_back(slot);
    }
    PagedKVCache cache{};
    TransformerBuffers prefill_buffers{};
    TransformerBuffers decode_buffers{};
    __nv_bfloat16* hidden = nullptr;
    int* gpu_tokens = nullptr;
    float* logits = nullptr;
    auto check = [](cudaError_t status) {
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }
    };
    auto cleanup = [&]() {
        if (cache.k_blocks && cache.v_blocks) {
            for (auto& request : requests) {
                freeSequenceBlocks(cache, request.kv_state);
            }
        }
        freePagedKVCache(cache);
        freeTransformerBuffers(prefill_buffers);
        freeTransformerBuffers(decode_buffers);
        OP_FREE(cudaFree(hidden));
        OP_FREE(cudaFree(gpu_tokens));
        OP_FREE(cudaFree(logits));
    };
    try {
        allocatePagedKVCache(cache, max_batch_size * ((max_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE));
        allocateTransformerBuffers(prefill_buffers, max_prompt_len, max_seq_len);
        allocateTransformerBuffers(decode_buffers, max_batch_size, max_batch_size * max_seq_len);
        int token_capacity = std::max(max_prompt_len, max_batch_size);
        OP_ALLOCATE(check(cudaMalloc(reinterpret_cast<void**>(&hidden), static_cast<size_t>(token_capacity) * 2048 * sizeof(__nv_bfloat16))));
        OP_ALLOCATE(check(cudaMalloc(reinterpret_cast<void**>(&gpu_tokens), token_capacity * sizeof(int))));
        OP_ALLOCATE(check(cudaMalloc(reinterpret_cast<void**>(&logits), static_cast<size_t>(max_batch_size) * VOCAB_SIZE * sizeof(float))));
        int iteration = 0;
        auto release = [&](int slot) {
            auto& request = requests[scheduler.active_slots[slot]];
            std::vector<int> blocks = request.kv_state.block_table;
            freeSequenceBlocks(cache, request.kv_state);
            scheduler.active_slots[slot] = -1;
            scheduler.free_slots.push_back(slot);
            if (events) {
                events->push_back({iteration, slot, request.request_id, false, blocks, static_cast<int>(cache.free_blocks.size())});
            }
        };
        while (!scheduler.waiting_queue.empty() || scheduler.free_slots.size() != static_cast<size_t>(max_batch_size)) {
            for (int slot = 0; slot < max_batch_size; ++slot) {
                int index = scheduler.active_slots[slot];
                if (index >= 0 && requests[index].finished) {
                    release(slot);
                }
            }
            while (!scheduler.waiting_queue.empty() && !scheduler.free_slots.empty()) {
                int slot = scheduler.free_slots.back();
                scheduler.free_slots.pop_back();
                int index = scheduler.waiting_queue.front();
                scheduler.waiting_queue.pop();
                scheduler.active_slots[slot] = index;
                auto& request = requests[index];
                if (request.max_new_tokens > 0) {
                    int length = static_cast<int>(request.prompt_tokens.size());
                    benchPhaseBegin("prefill");
                    check(cudaMemcpy(gpu_tokens, request.prompt_tokens.data(), length * sizeof(int), cudaMemcpyHostToDevice));
                    OP_KERNEL("embedding", embeddingGatherKernel<<<length, 1024>>>(gpu_tokens, hidden, weights.embedding));
                    check(cudaGetLastError());
                    request.last_token = transformerForwardPaged(cublas_handle, hidden, weights, cache,
                        request.kv_state, prefill_buffers, logits, length);
                    request.generated_tokens.push_back(request.last_token);
                    benchToken(request.request_id);
                    request.prefilled = true;
                }
                request.finished = requestFinished(request);
                if (events) {
                    events->push_back({iteration, slot, request.request_id, true, request.kv_state.block_table,
                                       static_cast<int>(cache.free_blocks.size())});
                }
                if (request.finished) {
                    release(slot);
                }
            }
            std::vector<int> active;
            std::vector<int> tokens;
            std::vector<SequenceKVState*> sequences;
            for (int index : scheduler.active_slots) {
                if (index >= 0) {
                    active.push_back(index);
                    tokens.push_back(requests[index].last_token);
                    sequences.push_back(&requests[index].kv_state);
                }
            }
            if (!active.empty()) {
                benchPhaseBegin("decode");
                check(cudaMemcpy(gpu_tokens, tokens.data(), tokens.size() * sizeof(int), cudaMemcpyHostToDevice));
                OP_KERNEL("embedding", embeddingGatherKernel<<<active.size(), 1024>>>(gpu_tokens, hidden, weights.embedding));
                check(cudaGetLastError());
                auto generated = decodeBatchPaged(cublas_handle, hidden, weights, cache, sequences, decode_buffers, logits);
                for (size_t row = 0; row < active.size(); ++row) {
                    auto& request = requests[active[row]];
                    request.last_token = generated[row];
                    request.generated_tokens.push_back(generated[row]);
                    benchToken(request.request_id);
                    request.finished = requestFinished(request);
                }
            }
            ++iteration;
        }
        check(cudaDeviceSynchronize());
        if (cache.free_blocks.size() != static_cast<size_t>(cache.num_blocks)) {
            throw std::runtime_error("Scheduler leaked physical blocks");
        }
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
}
