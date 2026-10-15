#include "argmax.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cuda_bf16.h>
#include <limits>
#include <random>
#include <cstring>

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
        std::cout << "Five BF16 argmax checks passed\n";
        auto reference = [&]() {
            float maximum=__bfloat162float(__float2bfloat16_rn(scores[0]));int selected=0;
            for(int i=1;i<VOCAB_SIZE;++i){float value=__bfloat162float(__float2bfloat16_rn(scores[i]));if(value>maximum){maximum=value;selected=i;}}
            return selected;
        };
        int extended_checks=0;
        auto validate = [&]() { expect(reference()); ++extended_checks; };
        const float nan=std::numeric_limits<float>::quiet_NaN();
        const float inf=std::numeric_limits<float>::infinity();
        std::fill(scores.begin(),scores.end(),-inf);validate();
        scores[255]=inf;scores[256]=inf;validate();
        scores[0]=nan;validate(); // A first-token NaN must dominate later +Inf.
        std::fill(scores.begin(),scores.end(),nan);validate();
        scores[0]=-inf;validate();
        scores[700]=inf;scores[900]=inf;validate();
        scores[0]=std::numeric_limits<float>::signaling_NaN();validate();
        std::fill(scores.begin(),scores.end(),-inf);scores[0]=-0.0f;scores[1000]=0.0f;validate();
        std::fill(scores.begin(),scores.end(),0.0f);scores[127000]=std::numeric_limits<float>::max();scores[128000]=inf;validate();
        uint32_t tiny_bits=0x00010000u;float tiny;std::memcpy(&tiny,&tiny_bits,sizeof(tiny));
        std::fill(scores.begin(),scores.end(),-1.0f);scores[0]=0.0f;scores[42]=tiny;validate();
        std::fill(scores.begin(),scores.end(),-1.0f);scores[0]=-tiny;scores[42]=-std::numeric_limits<float>::denorm_min();validate();
        std::mt19937 rng(20261015);
        for(int trial=0;trial<100;++trial){
            for(float& value:scores){uint32_t bits=rng();std::memcpy(&value,&bits,sizeof(value));}
            validate();
        }
        cudaFree(device);
        std::cout << extended_checks << " extended NaN/Inf/tie/random-bit-pattern argmax checks passed\n";
    } catch (const std::exception& error) {
        cudaFree(device);
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
