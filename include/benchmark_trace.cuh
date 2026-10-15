#pragma once

// Opt-in observations only. Production calls leave activeBenchmarkTrace null.
#include <cuda_runtime.h>
#include "operation_profile.cuh"
#include <chrono>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>

inline void benchmarkCheck(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
using BenchmarkClock = std::chrono::steady_clock;
inline double benchmarkMs(BenchmarkClock::time_point start) {
    return std::chrono::duration<double, std::milli>(BenchmarkClock::now() - start).count();
}
struct BenchmarkTrace {
    cudaEvent_t begin{}, end{};
    BenchmarkClock::time_point start;
    std::string phase;
    bool open = false;
    double prefill_gpu_stream_ms = 0, decode_gpu_stream_ms = 0;
    double prefill_wall_ms = 0;
    BenchmarkClock::time_point phase_start;
    size_t prefill_used = 0, decode_used = 0;
    std::map<int, std::vector<double>> token_ms;
    BenchmarkTrace() {
        benchmarkCheck(cudaEventCreate(&begin));
        benchmarkCheck(cudaEventCreate(&end));
    }
    ~BenchmarkTrace() { cudaEventDestroy(begin); cudaEventDestroy(end); }
    BenchmarkTrace(const BenchmarkTrace&) = delete;
    BenchmarkTrace& operator=(const BenchmarkTrace&) = delete;
};
inline thread_local BenchmarkTrace* activeBenchmarkTrace = nullptr;
inline void benchMemory() {
    auto* t = activeBenchmarkTrace;
    if (!t) return;
    size_t free, total;
    benchmarkCheck(cudaMemGetInfo(&free, &total));
    auto& peak = t->phase == "prefill" ? t->prefill_used : t->decode_used;
    peak = std::max(peak, total - free);
}
inline void benchPhaseBegin(const char* phase) {
    operationPhase(phase);
    auto* t = activeBenchmarkTrace;
    if (!t) return;
    t->phase = phase;
    t->phase_start = BenchmarkClock::now();
    benchMemory();
    benchmarkCheck(cudaEventRecord(t->begin));
    t->open = true;
}
// Called immediately before argmax copies logits to the host. Batched rows
// share one interval, stopped at the first row's argmax.
inline void benchGpuEnd() {
    auto* t = activeBenchmarkTrace;
    if (!t || !t->open) return;
    benchmarkCheck(cudaEventRecord(t->end));
    benchmarkCheck(cudaEventSynchronize(t->end));
    float elapsed;
    benchmarkCheck(cudaEventElapsedTime(&elapsed, t->begin, t->end));
    (t->phase == "prefill" ? t->prefill_gpu_stream_ms : t->decode_gpu_stream_ms) += elapsed;
    t->open = false;
    benchMemory();
}
inline void benchToken(int request_id) {
    auto* t = activeBenchmarkTrace;
    if (!t) return;
    t->token_ms[request_id].push_back(benchmarkMs(t->start));
    if (t->phase == "prefill") t->prefill_wall_ms += benchmarkMs(t->phase_start);
}
