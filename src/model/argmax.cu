#include <cstddef>
#include "operation_profile.cuh"
#include "benchmark_trace.cuh"
#include "argmax.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cuda_bf16.h>
#include <map>
#include <stdexcept>

namespace {
constexpr int ARGMAX_THREADS = 256;
constexpr int ARGMAX_BLOCKS = 128;
struct Candidate { float value; int index; };

__device__ Candidate better(Candidate a, Candidate b) {
    return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
}

__global__ void argmaxPartials(const float* logits, Candidate* partials) {
    Candidate best{-CUDART_INF_F, VOCAB_SIZE};
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < VOCAB_SIZE;
         index += blockDim.x * gridDim.x) {
        float value = __bfloat162float(__float2bfloat16_rn(logits[index]));
        // The original CPU scan ignores NaNs except when token zero is NaN.
        if (!isnan(value)) best = better(best, {value, index});
    }
    __shared__ Candidate values[ARGMAX_THREADS];
    values[threadIdx.x] = best;
    __syncthreads();
    for (int stride = ARGMAX_THREADS / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) values[threadIdx.x] = better(values[threadIdx.x], values[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = values[0];
}

__global__ void argmaxFinish(const float* logits, const Candidate* partials, int* selected) {
    __shared__ Candidate values[ARGMAX_THREADS];
    values[threadIdx.x] = threadIdx.x < ARGMAX_BLOCKS ? partials[threadIdx.x] : Candidate{-CUDART_INF_F, VOCAB_SIZE};
    __syncthreads();
    for (int stride = ARGMAX_THREADS / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) values[threadIdx.x] = better(values[threadIdx.x], values[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // CPU initializes its running maximum from token zero. If that is NaN,
        // every subsequent strict greater-than comparison is false: return zero.
        selected[0] = isnan(__bfloat162float(__float2bfloat16_rn(logits[0]))) ? 0 : values[0].index;
    }
}

struct Workspace {
    Candidate* partials = nullptr;
    int device = -1;
    ~Workspace() {
        if (!partials) return;
        int current = device;
        cudaGetDevice(&current);
        if (current != device) cudaSetDevice(device);
        cudaFree(partials);
        if (current != device) cudaSetDevice(current);
    }
};

Workspace& workspace() {
    // One allocation per host thread/device, reused across every token and request.
    static thread_local std::map<int, Workspace> workspaces;
    int device;
    benchmarkCheck(cudaGetDevice(&device));
    auto& result = workspaces[device];
    if (!result.partials) {
        OP_SCOPE("argmax_workspace_allocate", false);
        operationAllocation();
        benchmarkCheck(cudaMalloc(reinterpret_cast<void**>(&result.partials),
            ARGMAX_BLOCKS * sizeof(Candidate) + sizeof(int)));
        result.device = device;
    }
    return result;
}
} // namespace

int argmaxLogits(const float* logits) {
    benchGpuEnd();
    operationFlush();
    auto& scratch = workspace();
    int* selected = reinterpret_cast<int*>(scratch.partials + ARGMAX_BLOCKS);
    {
        OP_SCOPE("gpu_greedy_reduction", true);
        operationKernel();
        argmaxPartials<<<ARGMAX_BLOCKS, ARGMAX_THREADS>>>(logits, scratch.partials);
        benchmarkCheck(cudaGetLastError());
        operationKernel();
        argmaxFinish<<<1, ARGMAX_THREADS>>>(logits, scratch.partials, selected);
        benchmarkCheck(cudaGetLastError());
    }
    operationFlush();
    int token;
    {
        OP_SCOPE("token_d2h", true);
        operationCopy(sizeof(int));
        benchmarkCheck(cudaMemcpy(&token, selected, sizeof(int), cudaMemcpyDeviceToHost));
    }
    return token;
}
