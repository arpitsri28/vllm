#define main tinyVllmCliForDiagnostic
#include "../src/main.cu"
#undef main
#include "numerical_trace.cuh"
#include "paged_transformer.cuh"
#include "embedding.cuh"
#include "argmax.cuh"
#include "benchmark_trace.cuh"
#include <sstream>
#include <map>
#include <cmath>
#include <cstring>

using Snapshots=std::map<std::string,std::vector<float>>;
static Snapshots reference, observed;
static json differences=json::array();
static std::string comparison;
static int current_step;
static bool recording;
static void observe(int layer,const std::string& name,const std::vector<float>& values) {
    std::string key=std::to_string(layer)+":"+name;
    if(recording) {reference[key]=values;return;}
    observed[key]=values;
    const auto& expected=reference.at(key);
    if(expected.size()!=values.size()) throw std::runtime_error("Trace tensor shape mismatch");
    double mae=0,max_error=0; size_t changed=0; json examples=json::array();
    for(size_t i=0;i<values.size();++i) {
        if(!std::isfinite(values[i])||!std::isfinite(expected[i])) throw std::runtime_error("Nonfinite diagnostic tensor");
        double error=std::abs(double(values[i])-expected[i]);mae+=error;max_error=std::max(error,max_error);
        if(error) {++changed;if(examples.size()<3) examples.push_back({{"index",i},{"reference",expected[i]},{"actual",values[i]}});}
    }
    differences.push_back({{"comparison",comparison},{"decode_step",current_step},{"layer",layer},{"tensor",name},
        {"elements",values.size()},{"changed",changed},{"mae",mae/values.size()},{"max_error",max_error},{"examples",examples}});
}
struct State {
    KVCache flat{}; PagedKVCache pages{}; std::vector<SequenceKVState> sequences;
    TransformerBuffers prefill{},decode{}; int* ids=nullptr; __nv_bfloat16* hidden=nullptr;float* logits=nullptr;
    bool paged;
    State(int batch,int prompt,bool use_pages):sequences(batch),paged(use_pages) {
        if(paged) allocatePagedKVCache(pages,batch*8);else allocateKVCache(flat,128);
        allocateTransformerBuffers(prefill,prompt,128);allocateTransformerBuffers(decode,batch,batch*128);
        benchmarkCheck(cudaMalloc((void**)&ids,std::max(batch,prompt)*sizeof(int)));
        benchmarkCheck(cudaMalloc((void**)&hidden,std::max(batch,prompt)*2048*sizeof(__nv_bfloat16)));
        benchmarkCheck(cudaMalloc((void**)&logits,batch*VOCAB_SIZE*sizeof(float)));
    }
    ~State(){freeKVCache(flat);freePagedKVCache(pages);freeTransformerBuffers(prefill);freeTransformerBuffers(decode);cudaFree(ids);cudaFree(hidden);cudaFree(logits);}
    int prefillOne(cublasHandle_t handle,const ModelWeights& weights,const std::vector<int>& prompt,int row=0){
        benchmarkCheck(cudaMemcpy(ids,prompt.data(),prompt.size()*sizeof(int),cudaMemcpyHostToDevice));
        embeddingGatherKernel<<<prompt.size(),1024>>>(ids,hidden,weights.embedding);
        return paged?transformerForwardPaged(handle,hidden,weights,pages,sequences[row],prefill,logits,prompt.size()):
            transformerForward(handle,hidden,weights,flat,prefill,logits,prompt.size());
    }
    std::vector<int> step(cublasHandle_t handle,const ModelWeights& weights,const std::vector<int>& tokens){
        benchmarkCheck(cudaMemcpy(ids,tokens.data(),tokens.size()*sizeof(int),cudaMemcpyHostToDevice));
        embeddingGatherKernel<<<tokens.size(),1024>>>(ids,hidden,weights.embedding);
        if(!paged) return {decodeForward(handle,hidden,weights,flat,decode,logits)};
        std::vector<SequenceKVState*> seq;for(auto& s:sequences)seq.push_back(&s);
        return decodeBatchPaged(handle,hidden,weights,pages,seq,decode,logits);
    }
};
static std::vector<__nv_bfloat16> cacheValues(const State& state,int row,int layer,bool v){
    int length=state.paged?state.sequences[row].current_seq_len:state.flat.current_seq_len;
    std::vector<__nv_bfloat16> values(length*512);
    for(int pos=0;pos<length;++pos){
        const __nv_bfloat16* pointer;
        if(state.paged){int physical=state.sequences[row].block_table[pos/16];
            pointer=(v?state.pages.v_blocks:state.pages.k_blocks)+((size_t(layer)*state.pages.num_blocks+physical)*16+pos%16)*512;
        }else pointer=(v?state.flat.v_cache:state.flat.k_cache)+(size_t(layer)*128+pos)*512;
        benchmarkCheck(cudaMemcpy(values.data()+pos*512,pointer,512*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
    }return values;
}
static void compareCaches(const State& a,int rowA,const State& b,int rowB,json& reports){
    for(int layer=0;layer<16;++layer)for(bool v:{false,true}){
        auto x=cacheValues(a,rowA,layer,v),y=cacheValues(b,rowB,layer,v);
        if(x.size()!=y.size())throw std::runtime_error("Cache length mismatch");
        size_t changed=0;double maximum=0;
        for(size_t i=0;i<x.size();++i){double error=std::abs(double(__bfloat162float(x[i]))-__bfloat162float(y[i]));changed+=error!=0;maximum=std::max(maximum,error);}
        reports.push_back({{"comparison",comparison},{"decode_step",current_step},{"layer",layer},{"cache",v?"V":"K"},{"changed",changed},{"max_error",maximum}});
    }
}
static std::vector<float> floats(const std::vector<__nv_bfloat16>& x){std::vector<float> y;for(auto v:x)y.push_back(__bfloat162float(v));return y;}
static json scoreOracle(const State& flat){
    int layer=2,length=flat.flat.current_seq_len;auto k=floats(cacheValues(flat,0,layer,false));
    const auto& q=reference.at("2:rope_q");const auto& scores=reference.at("2:scores");const auto& actual=observed.at("2:scores");
    json cases=json::array();
    for(size_t i=0;i<scores.size();++i)if(scores[i]!=actual[i]){
        int head=i/length,token=i%length;double dot=0;
        for(int d=0;d<64;++d)dot+=double(q[head*64+d])*k[token*512+(head/4)*64+d];
        dot*=0.125;float rounded=__bfloat162float(__float2bfloat16_rn(float(dot)));
        cases.push_back({{"index",i},{"fp64_scaled_dot",dot},{"nearest_bf16",rounded},{"contiguous",scores[i]},{"paged",actual[i]}});
    }return cases;
}
static json projectionOracle(const ModelWeights& weights){
    const auto& input=reference.at("0:input_norm");const auto& q=reference.at("0:q");const auto& actual=observed.at("0:q");json cases=json::array();
    for(size_t i=0;i<q.size();++i)if(q[i]!=actual[i]){
        std::vector<__nv_bfloat16> host(2048);benchmarkCheck(cudaMemcpy(host.data(),weights.layers[0].w_q+i*2048,2048*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
        double dot=0;for(int d=0;d<2048;++d)dot+=double(input[d])*__bfloat162float(host[d]);
        cases.push_back({{"index",i},{"fp64_dot",dot},{"nearest_bf16",__bfloat162float(__float2bfloat16_rn(float(dot)))},{"single",q[i]},{"batch",actual[i]}});
    }return cases;
}
int main(int argc,char** argv){
    char* gpu=nullptr;cublasHandle_t handle=nullptr;json report;
    try{
        std::ostringstream quiet;auto* old=std::cout.rdbuf(quiet.rdbuf());
        auto model=loadSafeTensors("models/llama-3.2-1b-instruct/model.safetensors");std::cout.rdbuf(old);
        gpu=copyModelToGPU(model);auto weights=mapWeights(model,gpu);
        if(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS init");
        if(argc>2 && std::string(argv[2])=="strict-reduction"){
            if(cublasSetMathMode(handle,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Math mode rejected");
        }
        if(argc>2 && std::string(argv[2])=="generation-check") {
            std::ifstream prompt_file("benchmarks/prompts.json");json prompts;prompt_file>>prompts;
            auto prompt=prompts.at("128").get<std::vector<int>>();
            auto contiguous=generate(handle,weights,prompt,128,1024);
            auto paged=generatePaged(handle,weights,prompt,128,1024);
            if(contiguous.size()!=128 || paged!=contiguous)throw std::runtime_error("128-token single-mode agreement failed");
            report["contiguous"]=contiguous;report["paged"]=paged;report["batches"]=json::array();
            for(int size:{1,2,4}) {
                std::vector<Request> requests(size);
                for(int row=0;row<size;++row){requests[row].request_id=row;requests[row].prompt_tokens=prompt;requests[row].max_new_tokens=128;}
                generateContinuous(handle,weights,requests,size,1024);
                for(const auto& request:requests) {
                    if(!request.finished || request.generated_tokens!=contiguous)throw std::runtime_error("128-token batch agreement failed");
                    report["batches"].push_back({{"batch_size",size},{"request_id",request.request_id},{"generated_tokens",request.generated_tokens},{"matches_contiguous",true}});
                }
            }
            benchmarkCheck(cudaDeviceSynchronize());report["passed"]=true;
            std::ofstream out(argv[1]);out<<report.dump(2)<<'\n';cublasDestroy(handle);cudaFree(gpu);return 0;
        }
        report["math_mode"]=argc>2?argv[2]:"default";
        report["cache_comparisons"]=json::array();report["tokens"]=json::array();
        comparison="contiguous_vs_paged";
        {
            std::vector<int> prompt={128000};const std::vector<int> seed={791,6864,315,9822,374};
            while(prompt.size()<16)prompt.push_back(seed[(prompt.size()-1)%seed.size()]);
            State flat(1,16,false),paged(1,16,true);
            int teacher=flat.prefillOne(handle,weights,prompt);int pt=paged.prefillOne(handle,weights,prompt);
            current_step=0;compareCaches(flat,0,paged,0,report["cache_comparisons"]);
            report["prefill_tokens_equal"]=teacher==pt;
            numericalTraceSink=observe;
            for(current_step=1;current_step<=8;++current_step){
                reference.clear();recording=true;numericalTraceRow=0;int next=flat.step(handle,weights,{teacher})[0];
                recording=false;int actual=paged.step(handle,weights,{teacher})[0];
                report["tokens"].push_back({{"comparison",comparison},{"decode_step",current_step},{"teacher_input",teacher},{"reference",next},{"actual",actual}});
                numericalTraceSink=nullptr;compareCaches(flat,0,paged,0,report["cache_comparisons"]);numericalTraceSink=observe;
                if(current_step==1)report["score_oracle"]=scoreOracle(flat);
                teacher=next;
            }numericalTraceSink=nullptr;
        }
        comparison="single_paged_vs_batch";
        {
            std::ifstream f("tests/paged_requests.json");json inputs;f>>inputs;
            std::vector<std::vector<int>> prompts;int maxprompt=1;
            for(int i=0;i<3;++i){prompts.push_back(inputs[i]["prompt_tokens"].get<std::vector<int>>());maxprompt=std::max(maxprompt,int(prompts.back().size()));}
            State one0(1,maxprompt,true),one1(1,maxprompt,true),one2(1,maxprompt,true),batch(3,maxprompt,true);
            std::vector<State*> singles={&one0,&one1,&one2};std::vector<int> teacher;
            for(int i=0;i<3;++i){teacher.push_back(singles[i]->prefillOne(handle,weights,prompts[i]));batch.prefillOne(handle,weights,prompts[i],i);}
            current_step=0;compareCaches(one2,0,batch,2,report["cache_comparisons"]);
            for(current_step=1;current_step<=12;++current_step){
                numericalTraceSink=nullptr;std::vector<int> next={one0.step(handle,weights,{teacher[0]})[0],one1.step(handle,weights,{teacher[1]})[0]};
                reference.clear();numericalTraceRow=0;recording=true;numericalTraceSink=observe;next.push_back(one2.step(handle,weights,{teacher[2]})[0]);
                numericalTraceRow=2;recording=false;auto actual=batch.step(handle,weights,teacher);
                report["tokens"].push_back({{"comparison",comparison},{"decode_step",current_step},{"teacher_input",teacher[2]},{"reference",next[2]},{"actual",actual[2]}});
                numericalTraceSink=nullptr;compareCaches(one2,0,batch,2,report["cache_comparisons"]);
                if(current_step==1)report["projection_oracle"]=projectionOracle(weights);
                teacher=next;
            }
        }
        benchmarkCheck(cudaDeviceSynchronize());report["cuda_error"]=nullptr;
        report["tensor_comparisons"]=differences;
        report["method"]="Decode teacher forcing: each path consumes identical reference tokens; selected batch row 2. All logits and layer tensors compared; cache data gathered by logical position independently.";
        std::ofstream out(argc>1?argv[1]:"numerical_diagnostic.json");out<<report.dump(2)<<'\n';
        cublasDestroy(handle);cudaFree(gpu);return 0;
    }catch(const std::exception& e){numericalTraceSink=nullptr;std::cerr<<e.what()<<'\n';if(handle)cublasDestroy(handle);cudaFree(gpu);return 1;}
}
