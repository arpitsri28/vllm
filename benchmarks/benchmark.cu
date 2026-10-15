// Reuse the existing loader exactly as the repository's inference tests do.
#define main tinyVllmCliForBenchmark
#include "../src/main.cu"
#undef main
#include "benchmark_trace.cuh"
#include <sstream>
#include <iomanip>

static json execute(cublasHandle_t handle, const ModelWeights& weights,
                    const std::vector<int>& prompt, int output, int batch,
                    const std::string& mode, bool instrument, size_t model_used,
                    int max_seq_len) {
    BenchmarkTrace trace;
    benchmarkCheck(cudaDeviceSynchronize());
    trace.start = BenchmarkClock::now();
    activeBenchmarkTrace = instrument ? &trace : nullptr;
    std::vector<std::vector<int>> generated;
    try {
        if (mode == "batch") {
            std::vector<Request> requests(batch);
            for (int i = 0; i < batch; ++i) {
                requests[i].request_id = i;
                requests[i].prompt_tokens = prompt;
                requests[i].max_new_tokens = output;
            }
            generateContinuous(handle, weights, requests, batch, max_seq_len);
            for (const auto& r : requests) generated.push_back(r.generated_tokens);
        } else {
            generated.push_back(mode == "paged"
                ? generatePaged(handle, weights, prompt, output, max_seq_len)
                : generate(handle, weights, prompt, output, max_seq_len));
        }
        benchmarkCheck(cudaGetLastError());
        benchmarkCheck(cudaDeviceSynchronize());
    } catch (...) {
        activeBenchmarkTrace = nullptr;
        throw;
    }
    double e2e = benchmarkMs(trace.start);
    activeBenchmarkTrace = nullptr;
    int actual = 0, completed = 0;
    for (const auto& tokens : generated) {
        actual += static_cast<int>(tokens.size());
        completed += tokens.size() == static_cast<size_t>(output);
    }
    json r = {{"status", completed == batch ? "ok" : "early_termination"},
        {"actual_output_tokens", actual}, {"completed_requests", completed},
        {"failed_or_incomplete_requests", batch - completed},
        {"generated_tokens", generated}, {"e2e_wall_ms", e2e},
        {"e2e_output_tokens_per_s", actual * 1000.0 / e2e},
        {"model_loaded_device_used_mib", model_used / 1048576.0}};
    if (!instrument) return r;
    r["prefill_gpu_stream_ms"] = trace.prefill_gpu_stream_ms;
    r["decode_gpu_stream_ms"] = trace.decode_gpu_stream_ms;
    r["prefill_wall_ms"] = trace.prefill_wall_ms;
    r["prefill_observed_peak_device_used_mib"] = trace.prefill_used / 1048576.0;
    r["decode_observed_peak_device_used_mib"] = trace.decode_used / 1048576.0;
    r["observed_peak_device_used_mib"] = std::max({model_used, trace.prefill_used, trace.decode_used}) / 1048576.0;
    r["token_timestamps_wall_ms"] = json::object();
    std::vector<double> ttfts, itls;
    for (const auto& [id, times] : trace.token_ms) {
        r["token_timestamps_wall_ms"][std::to_string(id)] = times;
        if (!times.empty()) ttfts.push_back(times.front());
        for (size_t i = 1; i < times.size(); ++i) itls.push_back(times[i] - times[i-1]);
    }
    auto mean = [](const std::vector<double>& v) {
        double s = 0; for (double x : v) s += x; return v.empty() ? 0.0 : s / v.size();
    };
    auto p95 = [](std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[static_cast<size_t>(std::ceil(0.95 * v.size())) - 1];
    };
    r["mean_ttft_wall_ms"] = mean(ttfts);
    r["p95_ttft_wall_ms"] = p95(ttfts);
    r["mean_itl_wall_ms"] = mean(itls);
    r["p95_itl_wall_ms"] = p95(itls);
    if (mode != "batch" && !trace.token_ms[0].empty()) {
        const auto& t = trace.token_ms[0];
        double decode = t.back() - t.front();
        r["ttft_wall_ms"] = t.front();
        r["decode_wall_ms"] = decode;
        r["decode_output_tokens"] = static_cast<int>(t.size()) - 1;
        r["decode_wall_tokens_per_s"] = decode > 0 ? (t.size()-1) * 1000.0 / decode : 0.0;
        r["decode_gpu_stream_tokens_per_s"] = trace.decode_gpu_stream_ms > 0 ? (t.size()-1) * 1000.0 / trace.decode_gpu_stream_ms : 0.0;
    }
    return r;
}

int main(int argc, char** argv) {
    char* gpu = nullptr;
    cublasHandle_t handle = nullptr;
    try {
        if (argc != 4) throw std::invalid_argument("Usage: inference_benchmark MODEL CONFIG.json OUTPUT.jsonl");
        std::ifstream config_file(argv[2]); json config; config_file >> config;
        std::ifstream prompts_file(config.at("prompts_file").get<std::string>()); json prompts; prompts_file >> prompts;
        std::ofstream results(argv[3]);
        if (!results) throw std::runtime_error("Cannot open results file");
        unsetenv("TINY_VLLM_LOGITS_TRACE"); unsetenv("TINY_VLLM_LOGITS_DUMP");
        cudaDeviceProp prop{}; benchmarkCheck(cudaGetDeviceProperties(&prop, 0));
        int driver, runtime; benchmarkCheck(cudaDriverGetVersion(&driver)); benchmarkCheck(cudaRuntimeGetVersion(&runtime));
        // Avoid loader chatter without changing the production loader.
        std::ostringstream quiet; auto* original = std::cout.rdbuf(quiet.rdbuf());
        SafeTensorModel loaded;
        try { loaded = loadSafeTensors(argv[1]); } catch (...) { std::cout.rdbuf(original); throw; }
        std::cout.rdbuf(original);
        if (loaded.tensor_data.empty()) throw std::runtime_error("Empty model");
        // Verify BF16 and the hard-coded model shape before interpreting tensor bytes.
        for (const auto& item : loaded.header.items()) {
            if (item.key() != "__metadata__" && item.value().at("dtype") != "BF16")
                throw std::runtime_error("Expected BF16 checkpoint");
        }
        if (loaded.header.at("model.embed_tokens.weight").at("shape") != json::array({128256,2048}))
            throw std::runtime_error("Unsupported embedding shape");
        gpu = copyModelToGPU(loaded);
        auto weights = mapWeights(loaded, gpu);
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS initialization failed");
        benchmarkCheck(cudaDeviceSynchronize());
        size_t free, total; benchmarkCheck(cudaMemGetInfo(&free, &total));
        size_t model_used = total-free;
        json metadata = {{"type", "metadata"}, {"gpu", prop.name}, {"vram_mib", prop.totalGlobalMem / 1048576.0},
            {"cuda_runtime", runtime}, {"cuda_driver", driver}, {"precision", "BF16"},
            {"model_tensor_bytes", loaded.tensor_data.size()}, {"config", config},
            {"build", "Release; CUDA architecture 86; no fast-math"},
            {"model_loaded_device_used_mib", model_used / 1048576.0}};
        results << metadata.dump() << '\n'; results.flush();
        loaded.tensor_data.clear(); loaded.tensor_data.shrink_to_fit();
        int errors = 0;
        for (const auto& c : config.at("cases")) {
            int length = c.at("prompt_tokens").get<int>(), output = c.at("output_tokens").get<int>(), batch = c.value("batch_size", 1);
            std::string mode = c.at("mode").get<std::string>();
            json base = c;
            if ((mode == "batch" && batch > MAX_BATCH_SIZE) || c.contains("skip_reason")) {
                base["type"] = "unsupported"; base["status"] = "unsupported";
                base["reason"] = c.value("skip_reason", "Scheduler and paged kernels support at most 4 active requests");
                results << base.dump() << '\n'; results.flush(); continue;
            }
            if (mode != "batch" && mode != "paged" && mode != "contiguous") throw std::invalid_argument("Unknown mode");
            auto prompt = prompts.at(std::to_string(length)).get<std::vector<int>>();
            if (prompt.size() != static_cast<size_t>(length)) throw std::runtime_error("Wrong prompt length");
            std::cout << "Running " << mode << " P=" << length << " O=" << output << " B=" << batch << std::endl;
            bool healthy = true;
            try {
                auto plain = execute(handle, weights, prompt, output, batch, mode, false, model_used, config.at("max_seq_len").get<int>());
                auto observed = execute(handle, weights, prompt, output, batch, mode, true, model_used, config.at("max_seq_len").get<int>());
                if (plain.at("generated_tokens") != observed.at("generated_tokens")) throw std::runtime_error("Instrumentation changes generated tokens");
                base["type"] = "instrumentation_check";
                base["plain_e2e_wall_ms"] = plain.at("e2e_wall_ms");
                base["observed_e2e_wall_ms"] = observed.at("e2e_wall_ms");
                base["tokens_equal"] = true;
                results << base.dump() << '\n'; results.flush();
                for (int i = 0; i < config.at("warmups").get<int>(); ++i)
                    execute(handle, weights, prompt, output, batch, mode, true, model_used, config.at("max_seq_len").get<int>());
            } catch (const std::exception& e) {
                base["type"] = "error"; base["status"] = "error"; base["error"] = e.what();
                results << base.dump() << '\n'; results.flush(); ++errors; healthy = false;
            }
            if (!healthy) continue;
            for (int rep = 0; rep < config.at("repetitions").get<int>(); ++rep) {
                json row = c; row["type"] = "run"; row["repetition"] = rep;
                try {
                    auto observed = execute(handle, weights, prompt, output, batch, mode, true, model_used, config.at("max_seq_len").get<int>());
                    auto plain = execute(handle, weights, prompt, output, batch, mode, false, model_used, config.at("max_seq_len").get<int>());
                    if (plain.at("generated_tokens") != observed.at("generated_tokens")) throw std::runtime_error("Measured instrumentation changes tokens");
                    row.update(observed);
                    row["uninstrumented_e2e_wall_ms"] = plain.at("e2e_wall_ms");
                    row["uninstrumented_output_tokens_per_s"] = plain.at("e2e_output_tokens_per_s");
                }
                catch (const std::exception& e) { row["status"] = "error"; row["error"] = e.what(); ++errors; }
                results << row.dump() << '\n'; results.flush();
                std::cout << "  repetition " << rep << ": " << row.at("status") << std::endl;
                if (row.at("status") == "error") break;
            }
        }
        cublasDestroy(handle); handle = nullptr; benchmarkCheck(cudaFree(gpu)); gpu = nullptr;
        return errors ? 2 : 0;
    } catch (const std::exception& e) {
        activeBenchmarkTrace = nullptr;
        if (handle) cublasDestroy(handle);
        cudaFree(gpu); std::cerr << e.what() << '\n'; return 1;
    }
}
