#include "engine.cuh"
#include "embedding.cuh"
#include "transformer.cuh"
#include "argmax.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>

constexpr int EMBEDDING_LENGTH = 2048;

std::vector<int> generate(cublasHandle_t cublas_handle, const ModelWeights& weights, const std::vector<int>& prompt_tokens, int max_new_tokens, int max_seq_len) {
    if (prompt_tokens.empty()) {
        throw std::invalid_argument("Prompt must not be empty");
    }
    if (max_new_tokens < 0 || max_seq_len <= 0 || max_seq_len > 1024) {
        throw std::invalid_argument("Require max_new_tokens >= 0 and 0 < max_seq_len <= 1024");
    }
    if (prompt_tokens.size() > static_cast<size_t>(max_seq_len) ||
        max_new_tokens > max_seq_len - static_cast<int>(prompt_tokens.size())) {
        throw std::invalid_argument("Prompt plus generated tokens exceeds max_seq_len");
    }
    for (int token : prompt_tokens) {
        if (token < 0 || token >= VOCAB_SIZE) {
            throw std::invalid_argument("Prompt token is outside the vocabulary");
        }
    }
    std::vector<int> generated_tokens;
    if (max_new_tokens == 0) {
        return generated_tokens;
    }
    generated_tokens.reserve(max_new_tokens);
    int prompt_len = static_cast<int>(prompt_tokens.size());
    KVCache cache{};
    TransformerBuffers buffers{};
    float* logits = nullptr;
    int* gpu_prompt_tokens = nullptr;
    int* gpu_decode_token = nullptr;
    __nv_bfloat16* prompt_hidden = nullptr;
    __nv_bfloat16* decode_hidden = nullptr;

    auto check = [](cudaError_t status) {
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }
    };
    auto cleanup = [&]() {
        cudaFree(gpu_prompt_tokens);
        cudaFree(prompt_hidden);
        cudaFree(gpu_decode_token);
        cudaFree(decode_hidden);
        cudaFree(logits);
        freeTransformerBuffers(buffers);
        freeKVCache(cache);
    };

    try {
        std::ofstream trace;
        const char* trace_path = std::getenv("TINY_VLLM_LOGITS_TRACE");
        if (trace_path != nullptr && *trace_path != '\0') {
            trace.open(trace_path, std::ios::binary);
            if (!trace) {
                throw std::runtime_error("Failed to open logits trace");
            }
        }
        auto trace_logits = [&]() {
            if (!trace.is_open()) {
                return;
            }
            std::vector<float> host_logits(VOCAB_SIZE);
            check(cudaMemcpy(host_logits.data(), logits, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            trace.write(reinterpret_cast<const char*>(host_logits.data()), VOCAB_SIZE * sizeof(float));
            if (!trace) {
                throw std::runtime_error("Failed to write logits trace");
            }
        };
        allocateKVCache(cache, max_seq_len);
        check(cudaMalloc(reinterpret_cast<void**>(&logits), VOCAB_SIZE * sizeof(float)));
        allocateTransformerBuffers(buffers, prompt_len, max_seq_len);
        check(cudaMalloc(reinterpret_cast<void**>(&prompt_hidden), prompt_len * EMBEDDING_LENGTH * sizeof(__nv_bfloat16)));
        check(cudaMalloc(reinterpret_cast<void**>(&gpu_prompt_tokens), prompt_len * sizeof(int)));
        check(cudaMemcpy(gpu_prompt_tokens, prompt_tokens.data(), prompt_len * sizeof(int), cudaMemcpyHostToDevice));
        check(cudaMalloc(reinterpret_cast<void**>(&gpu_decode_token), sizeof(int)));
        check(cudaMalloc(reinterpret_cast<void**>(&decode_hidden), EMBEDDING_LENGTH * sizeof(__nv_bfloat16)));

        embeddingGatherKernel<<<prompt_len, 1024>>>(gpu_prompt_tokens, prompt_hidden, weights.embedding);
        check(cudaGetLastError());
        int next_token = transformerForward(cublas_handle, prompt_hidden, weights, cache, buffers, logits, prompt_len);
        check(cudaGetLastError());
        generated_tokens.push_back(next_token);
        trace_logits();

        while (generated_tokens.size() < static_cast<size_t>(max_new_tokens) &&
               next_token != 128001 && next_token != 128008 && next_token != 128009) {
            check(cudaMemcpy(gpu_decode_token, &next_token, sizeof(int), cudaMemcpyHostToDevice));
            embeddingGatherKernel<<<1, 1024>>>(gpu_decode_token, decode_hidden, weights.embedding);
            check(cudaGetLastError());
            next_token = decodeForward(cublas_handle, decode_hidden, weights, cache, buffers, logits);
            check(cudaGetLastError());
            generated_tokens.push_back(next_token);
            trace_logits();
        }
        if (trace.is_open()) {
            trace.close();
            if (!trace) {
                throw std::runtime_error("Failed to close logits trace");
            }
        }
        const char* dump_path = std::getenv("TINY_VLLM_LOGITS_DUMP");
        if (dump_path != nullptr && *dump_path != '\0') {
            std::vector<float> host_logits(VOCAB_SIZE);
            check(cudaMemcpy(host_logits.data(), logits, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            std::vector<int> prefix = prompt_tokens;
            prefix.insert(prefix.end(), generated_tokens.begin(), generated_tokens.end() - 1);
            nlohmann::json dump = {
                {"prefix", prefix},
                {"generated_tokens", generated_tokens},
                {"logits", host_logits}
            };
            std::ofstream file(dump_path);
            file << dump << '\n';
            file.close();
            if (!file) {
                throw std::runtime_error("Failed to write logits dump");
            }
        }
    } catch (...) {
        cleanup();
        throw;
    }

    cleanup();
    return generated_tokens;
}
