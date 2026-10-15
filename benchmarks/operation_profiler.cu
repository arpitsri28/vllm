#define main tinyVllmCliForProfile
#include "../src/main.cu"
#undef main
#include "operation_profile.cuh"
#include "benchmark_trace.cuh"
#include <sstream>
int main(int argc,char** argv){
    char* gpu=nullptr;cublasHandle_t handle=nullptr;
    try{
        if(argc!=2)throw std::invalid_argument("Usage: operation_profiler OUTPUT.json");
        std::ostringstream quiet;auto* old=std::cout.rdbuf(quiet.rdbuf());
        auto loaded=loadSafeTensors("models/llama-3.2-1b-instruct/model.safetensors");std::cout.rdbuf(old);
        gpu=copyModelToGPU(loaded);auto weights=mapWeights(loaded,gpu);
        if(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS initialization failed");
        std::ifstream file("benchmarks/prompts.json");json prompts;file>>prompts;json report=json::array();
        for(auto c:std::vector<json>{{{"mode","contiguous"},{"prompt",256},{"batch",1}},{{"mode","batch"},{"prompt",128},{"batch",1}},{{"mode","batch"},{"prompt",128},{"batch",4}}}){
            auto prompt=prompts.at(std::to_string(c.at("prompt").get<int>())).get<std::vector<int>>();
            auto run=[&](){
                if(c.at("mode")=="contiguous")return std::vector<std::vector<int>>{generate(handle,weights,prompt,128,1024)};
                std::vector<Request> requests(c.at("batch").get<int>());
                for(size_t i=0;i<requests.size();++i){requests[i].request_id=i;requests[i].prompt_tokens=prompt;requests[i].max_new_tokens=128;}
                generateContinuous(handle,weights,requests,requests.size(),1024);
                std::vector<std::vector<int>> tokens;for(auto& request:requests)tokens.push_back(request.generated_tokens);return tokens;
            };
            auto expected=run();run();
            OperationProfile profile;
            // Prepare event storage outside the measured interval; reuse it each token.
            for(int i=0;i<4096;++i){OpEvent event;opCheck(cudaEventCreate(&event.start));opCheck(cudaEventCreate(&event.stop));profile.events.push_back(event);}
            benchmarkCheck(cudaDeviceSynchronize());activeOperationProfile=&profile;
            auto start=OpClock::now();auto actual=run();profile.flush();benchmarkCheck(cudaDeviceSynchronize());
            double wall=std::chrono::duration<double,std::milli>(OpClock::now()-start).count();activeOperationProfile=nullptr;
            if(actual!=expected)throw std::runtime_error("Profiling changed generated tokens");
            c["wall_ms_with_operation_probes"]=wall;c["operations"]=json::object();
            for(auto& item:profile.stats){const auto& x=item.second;c["operations"][item.first]={{"host_api_wall_ms",x.host_wall_ms},{"gpu_stream_elapsed_ms",x.gpu_stream_ms},{"calls",x.calls},{"explicit_kernel_launches",x.explicit_kernels},{"cublas_calls",x.cublas_calls},{"observed_allocation_calls",x.allocations},{"observed_free_calls",x.frees},{"tracked_copy_bytes",x.copy_bytes}};}
            c["tokens_unchanged"]=true;c["actual_output_tokens"]=0;for(auto& t:actual)c["actual_output_tokens"]=c["actual_output_tokens"].get<int>()+t.size();
            c["method"]="Full 128-output-token call after two warmups. CUDA events surround operations; host timers cover API submission/CPU work. Pending GPU intervals are completed at existing argmax dependency boundaries using an explicit profiler readiness wait. Intervals include queue idle/submission gaps and are not kernel-active time. Host wall and GPU intervals overlap and must not be summed. Profiling overhead makes total wall unsuitable for performance claims. Counts cover explicit source launches and cuBLAS API calls; physical cuBLAS kernel counts require external profiling.";
            report.push_back(c);std::cout<<c.at("mode")<<" batch="<<c.at("batch")<<" done\n";
        }
        std::ofstream out(argv[1]);out<<report.dump(2)<<'\n';cublasDestroy(handle);cudaFree(gpu);return 0;
    }catch(const std::exception& e){activeOperationProfile=nullptr;std::cerr<<e.what()<<'\n';if(handle)cublasDestroy(handle);cudaFree(gpu);return 1;}
}
