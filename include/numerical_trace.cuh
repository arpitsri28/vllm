#pragma once
// Diagnostic-only snapshots. No code or synchronization in normal builds.
#ifdef TINY_VLLM_NUMERICAL_TRACE
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <functional>
#include <vector>
#include <string>
#include <stdexcept>
inline thread_local int numericalTraceRow = 0;
inline thread_local std::function<void(int,const std::string&,const std::vector<float>&)> numericalTraceSink;
inline void traceBF16(int layer, const char* name, const __nv_bfloat16* pointer,
                      int width, int rows=1, int stride=0) {
    if (!numericalTraceSink) return;
    if (!stride) stride=width;
    std::vector<__nv_bfloat16> host(static_cast<size_t>(rows)*width);
    auto error=cudaMemcpy2D(host.data(),width*sizeof(__nv_bfloat16),pointer,stride*sizeof(__nv_bfloat16),
                            width*sizeof(__nv_bfloat16),rows,cudaMemcpyDeviceToHost);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    std::vector<float> values(host.size());
    for(size_t i=0;i<host.size();++i) values[i]=__bfloat162float(host[i]);
    numericalTraceSink(layer,name,values);
}
inline void traceFloat(int layer,const char* name,const float* pointer,int count) {
    if (!numericalTraceSink) return;
    std::vector<float> values(count);
    auto error=cudaMemcpy(values.data(),pointer,count*sizeof(float),cudaMemcpyDeviceToHost);
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    numericalTraceSink(layer,name,values);
}
#else
inline constexpr int numericalTraceRow=0;
inline void traceBF16(int,const char*,const __nv_bfloat16*,int,int=1,int=0) {}
inline void traceFloat(int,const char*,const float*,int) {}
#endif
