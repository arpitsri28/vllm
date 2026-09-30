#include "argmax.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    float* device = nullptr;
    auto check = [](cudaError_t status) {
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }
    };
    try {
        check(cudaMalloc(reinterpret_cast<void**>(&device), VOCAB_SIZE * sizeof(float)));
        std::vector<float> scores(VOCAB_SIZE, -100.0f);
        auto expect = [&](int token) {
            check(cudaMemcpy(device, scores.data(), scores.size() * sizeof(float), cudaMemcpyHostToDevice));
            if (argmaxLogits(device) != token) {
                throw std::runtime_error("Unexpected argmax token");
            }
        };
        expect(0);
        scores[469] = 16.497030f;
        scores[9928] = 16.497549f;
        expect(469);
        scores[9928] = 16.6f;
        expect(9928);
        std::fill(scores.begin(), scores.end(), -100.0f);
        scores[469] = 16.5f;
        scores[9928] = 16.5625f;
        expect(469);
        scores[9928] = 16.5626f;
        expect(9928);
        cudaFree(device);
        std::cout << "Five BF16 argmax checks passed\n";
    } catch (const std::exception& error) {
        cudaFree(device);
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
