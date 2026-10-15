#pragma once
// Only the dedicated operation_profiler target enables these observations.
#ifdef TINY_VLLM_OPERATION_PROFILE
#include <cuda_runtime.h>
#include <chrono>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
using OpClock=std::chrono::steady_clock;
inline void opCheck(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
struct OpStats {double host_wall_ms=0,gpu_stream_ms=0;size_t calls=0,explicit_kernels=0,cublas_calls=0,allocations=0,frees=0,copy_bytes=0;};
struct OpEvent {cudaEvent_t start{},stop{};OpStats* stats=nullptr;};
struct OperationProfile {
    std::string phase="setup";std::map<std::string,OpStats> stats;std::vector<OpEvent> events;size_t used=0;
    OpStats* current=nullptr;
    ~OperationProfile(){for(auto& e:events){cudaEventDestroy(e.start);cudaEventDestroy(e.stop);}}
    void flush(){
        if(!used)return;
        auto begin=OpClock::now();opCheck(cudaEventSynchronize(events[used-1].stop));
        auto& wait=stats[phase+".profiler_readiness_wait"];++wait.calls;
        wait.host_wall_ms+=std::chrono::duration<double,std::milli>(OpClock::now()-begin).count();
        for(size_t i=0;i<used;++i){float ms;opCheck(cudaEventElapsedTime(&ms,events[i].start,events[i].stop));events[i].stats->gpu_stream_ms+=ms;}
        used=0;
    }
};
inline thread_local OperationProfile* activeOperationProfile=nullptr;
struct OperationScope {
    OperationProfile* p;OpStats* stats=nullptr;OpStats* previous=nullptr;size_t slot=0;bool gpu;OpClock::time_point begin;
    OperationScope(const char* name,bool use_gpu):p(activeOperationProfile),gpu(use_gpu){
        if(!p)return;
        begin=OpClock::now();stats=&p->stats[p->phase+"."+name];++stats->calls;previous=p->current;p->current=stats;
        if(gpu){
            slot=p->used++;
            if(slot==p->events.size()){OpEvent e;opCheck(cudaEventCreate(&e.start));opCheck(cudaEventCreate(&e.stop));p->events.push_back(e);}
            p->events[slot].stats=stats;opCheck(cudaEventRecord(p->events[slot].start));
        }
    }
    ~OperationScope(){if(!p)return;if(gpu)cudaEventRecord(p->events[slot].stop);stats->host_wall_ms+=std::chrono::duration<double,std::milli>(OpClock::now()-begin).count();p->current=previous;}
};
inline void operationPhase(const char* phase){if(activeOperationProfile){activeOperationProfile->flush();activeOperationProfile->phase=phase;}}
inline void operationFlush(){if(activeOperationProfile)activeOperationProfile->flush();}
inline void operationKernel(){if(activeOperationProfile&&activeOperationProfile->current)++activeOperationProfile->current->explicit_kernels;}
inline void operationBlas(){if(activeOperationProfile&&activeOperationProfile->current)++activeOperationProfile->current->cublas_calls;}
inline void operationAllocation(size_t count=1){if(activeOperationProfile&&activeOperationProfile->current)activeOperationProfile->current->allocations+=count;}
inline void operationFree(size_t count=1){if(activeOperationProfile&&activeOperationProfile->current)activeOperationProfile->current->frees+=count;}
inline void operationCopy(size_t bytes){if(activeOperationProfile&&activeOperationProfile->current)activeOperationProfile->current->copy_bytes+=bytes;}
#define OP_SCOPE(name,gpu) OperationScope op_scope(name,gpu)
#define OP_ALLOCATE(...) do {OperationScope allocation_scope("buffer_allocate",false);operationAllocation();__VA_ARGS__;} while(0)
#define OP_FREE(...) do {OperationScope free_scope("buffer_free",false);operationFree();__VA_ARGS__;} while(0)
#define OP_KERNEL(name,...) do {OperationScope kernel_scope(name,true);operationKernel();__VA_ARGS__;} while(0)
#else
inline void operationPhase(const char*){}
inline void operationFlush(){}
inline void operationKernel(){}
inline void operationBlas(){}
inline void operationAllocation(size_t=1){}
inline void operationFree(size_t=1){}
inline void operationCopy(size_t){}
#define OP_SCOPE(name,gpu)
#define OP_ALLOCATE(...) __VA_ARGS__
#define OP_FREE(...) __VA_ARGS__
#define OP_KERNEL(name,...) __VA_ARGS__
#endif
