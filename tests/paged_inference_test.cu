#define main tinyVllmMainForTest
#include "../src/main.cu"
#undef main

#include "paged_transformer.cuh"
#include "embedding.cuh"
#include "argmax.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <set>

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

static void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static std::vector<int> boundaryPrompt(int length) {
    const std::vector<int> seed = {791, 6864, 315, 9822, 374};
    std::vector<int> result = {128000};
    while (result.size() < static_cast<size_t>(length)) {
        result.push_back(seed[(result.size() - 1) % seed.size()]);
    }
    return result;
}

static int firstMismatch(const std::vector<int>& a, const std::vector<int>& b) {
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        if (a[i] != b[i]) {
            return static_cast<int>(i + 1);
        }
    }
    return a.size() == b.size() ? -1 : static_cast<int>(std::min(a.size(), b.size()) + 1);
}

static void testPrefill(cublasHandle_t handle, const ModelWeights& weights, int length, json& report) {
    KVCache contiguous{};
    PagedKVCache paged{};
    SequenceKVState sequence{};
    TransformerBuffers buffers{};
    int* ids = nullptr;
    __nv_bfloat16* hidden = nullptr;
    float* logits = nullptr;
    auto cleanup = [&]() {
        freeKVCache(contiguous);
        freePagedKVCache(paged);
        freeTransformerBuffers(buffers);
        cudaFree(ids);
        cudaFree(hidden);
        cudaFree(logits);
    };
    try {
        allocateKVCache(contiguous, 64);
        allocatePagedKVCache(paged, 8);
        int held = allocatePhysicalBlock(paged);
        allocatePhysicalBlock(paged);
        freePhysicalBlock(paged, held);
        allocateTransformerBuffers(buffers, length, 64);
        check(cudaMalloc(reinterpret_cast<void**>(&ids), length * sizeof(int)));
        check(cudaMalloc(reinterpret_cast<void**>(&hidden), static_cast<size_t>(length) * 2048 * sizeof(__nv_bfloat16)));
        check(cudaMalloc(reinterpret_cast<void**>(&logits), VOCAB_SIZE * sizeof(float)));
        auto prompt = boundaryPrompt(length);
        check(cudaMemcpy(ids, prompt.data(), length * sizeof(int), cudaMemcpyHostToDevice));
        embeddingGatherKernel<<<length, 1024>>>(ids, hidden, weights.embedding);
        int token = transformerForward(handle, hidden, weights, contiguous, buffers, logits, length);
        std::vector<float> expected(VOCAB_SIZE), actual(VOCAB_SIZE);
        check(cudaMemcpy(expected.data(), logits, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        embeddingGatherKernel<<<length, 1024>>>(ids, hidden, weights.embedding);
        int paged_token = transformerForwardPaged(handle, hidden, weights, paged, sequence, buffers, logits, length);
        check(cudaMemcpy(actual.data(), logits, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < VOCAB_SIZE; ++i) {
            require(std::isfinite(expected[i]) && std::isfinite(actual[i]), "Non-finite prefill logits");
        }
        require(token == paged_token, "Paged prefill token differs");
        require(std::memcmp(expected.data(), actual.data(), VOCAB_SIZE * sizeof(float)) == 0, "Paged prefill logits differ");
        require(sequence.current_seq_len == length, "Prefill sequence length advanced incorrectly");
        for (int layer = 0; layer < 16; ++layer) {
            for (int position = 0; position < length; ++position) {
                size_t flat = (static_cast<size_t>(layer) * 64 + position) * 512;
                size_t physical = ((static_cast<size_t>(layer) * paged.num_blocks + sequence.block_table[position / 16]) * 16 + position % 16) * 512;
                std::vector<__nv_bfloat16> a(512), b(512);
                check(cudaMemcpy(a.data(), contiguous.k_cache + flat, 512 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
                check(cudaMemcpy(b.data(), paged.k_blocks + physical, 512 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
                require(std::memcmp(a.data(), b.data(), 512 * sizeof(__nv_bfloat16)) == 0, "Paged prefill K differs");
                check(cudaMemcpy(a.data(), contiguous.v_cache + flat, 512 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
                check(cudaMemcpy(b.data(), paged.v_blocks + physical, 512 * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
                require(std::memcmp(a.data(), b.data(), 512 * sizeof(__nv_bfloat16)) == 0, "Paged prefill V differs");
            }
        }
        check(cudaDeviceSynchronize());
        report.push_back({{"prompt_length", length}, {"token", token}, {"paged_token", paged_token},
                          {"logits_bitwise_equal", true}, {"kv_bitwise_equal", true}, {"block_table", sequence.block_table}});
        std::cout << "Prefill length " << length << ": identical logits and KV\n";
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
}

static bool testGeneration(cublasHandle_t handle, const ModelWeights& weights, const std::vector<int>& prompt,
                           const std::filesystem::path& directory, json& report) {
    auto flat_path = directory / "contiguous.f32";
    auto paged_path = directory / "paged.f32";
    setenv("TINY_VLLM_LOGITS_TRACE", flat_path.c_str(), 1);
    auto flat = generate(handle, weights, prompt, 40, 128);
    setenv("TINY_VLLM_LOGITS_TRACE", paged_path.c_str(), 1);
    auto paged = generatePaged(handle, weights, prompt, 40, 128);
    unsetenv("TINY_VLLM_LOGITS_TRACE");
    check(cudaDeviceSynchronize());
    int mismatch = firstMismatch(flat, paged);
    size_t common_steps = std::min(flat.size(), paged.size());
    if (mismatch > 0) {
        common_steps = std::min(common_steps, static_cast<size_t>(mismatch));
    }
    require(std::filesystem::file_size(flat_path) == flat.size() * VOCAB_SIZE * sizeof(float), "Incomplete contiguous trace");
    require(std::filesystem::file_size(paged_path) == paged.size() * VOCAB_SIZE * sizeof(float), "Incomplete paged trace");
    std::ifstream flat_file(flat_path, std::ios::binary), paged_file(paged_path, std::ios::binary);
    double error = 0;
    double max_error = 0;
    for (size_t step = 0; step < common_steps; ++step) {
        std::vector<float> a(VOCAB_SIZE), b(VOCAB_SIZE);
        flat_file.read(reinterpret_cast<char*>(a.data()), VOCAB_SIZE * sizeof(float));
        paged_file.read(reinterpret_cast<char*>(b.data()), VOCAB_SIZE * sizeof(float));
        require(static_cast<bool>(flat_file) && static_cast<bool>(paged_file), "Logit trace read failed");
        for (int i = 0; i < VOCAB_SIZE; ++i) {
            require(std::isfinite(a[i]) && std::isfinite(b[i]), "Non-finite inference logits");
            double difference = std::abs(static_cast<double>(a[i]) - b[i]);
            error += difference;
            max_error = std::max(max_error, difference);
        }
    }
    double mae = common_steps ? error / (common_steps * VOCAB_SIZE) : 0;
    report.push_back({{"prompt_tokens", prompt}, {"contiguous", flat}, {"paged", paged},
                      {"first_mismatch", mismatch}, {"shared_prefix_steps", common_steps},
                      {"logit_mae", mae}, {"max_logit_error", max_error}});
    std::cout << "Prompt length " << prompt.size() << ": first mismatch " << mismatch << ", MAE " << mae << '\n';
    return mismatch == -1 && mae <= 0.01;
}

static bool testBatch(cublasHandle_t handle, const ModelWeights& weights, const json& inputs, json& report) {
    std::vector<Request> requests;
    for (const auto& item : inputs) {
        Request request;
        request.request_id = item.at("request_id").get<int>();
        request.prompt_tokens = item.at("prompt_tokens").get<std::vector<int>>();
        request.max_new_tokens = item.at("max_new_tokens").get<int>();
        requests.push_back(request);
    }
    std::vector<SchedulerEvent> events;
    generateContinuous(handle, weights, requests, 3, 128, &events);
    bool passed = true;
    std::set<size_t> output_lengths;
    report["requests"] = json::array();
    for (size_t i = 0; i < requests.size(); ++i) {
        auto& request = requests[i];
        output_lengths.insert(request.generated_tokens.size());
        auto single = generatePaged(handle, weights, request.prompt_tokens, request.max_new_tokens, 128);
        int mismatch = firstMismatch(single, request.generated_tokens);
        passed = passed && mismatch == -1;
        require(request.finished && request.kv_state.block_table.empty() && request.kv_state.current_seq_len == 0, "Finished request retained pages");
        report["requests"].push_back({{"request_id", request.request_id}, {"text", inputs[i].at("text")},
                                      {"single_paged", single}, {"continuous", request.generated_tokens}, {"first_mismatch", mismatch}});
    }
    require(output_lengths.size() >= 4, "Batch test did not exercise four different output lengths");
    std::vector<int> slots(3, -1);
    std::vector<int> released_at(3, -1);
    std::set<int> released_pages;
    std::set<int> admitted;
    bool reused = false;
    int admissions = 0;
    report["events"] = json::array();
    for (const auto& event : events) {
        if (event.admitted) {
            require(slots[event.slot] == -1 && admitted.insert(event.request_id).second, "Slot or request admitted twice");
            if (admissions < 3) {
                require(event.iteration == 0 && event.request_id == admissions, "Initial requests were not admitted together");
            } else {
                require(released_at[event.slot] == event.iteration, "Waiting request did not immediately reuse a freed slot");
            }
            slots[event.slot] = event.request_id;
            for (int block : event.blocks) {
                if (released_pages.count(block)) {
                    reused = true;
                }
            }
            ++admissions;
        } else {
            require(slots[event.slot] == event.request_id, "Released the wrong request");
            slots[event.slot] = -1;
            released_at[event.slot] = event.iteration;
            released_pages.insert(event.blocks.begin(), event.blocks.end());
        }
        report["events"].push_back({{"iteration", event.iteration}, {"slot", event.slot},
                                    {"request_id", event.request_id}, {"admitted", event.admitted},
                                    {"blocks", event.blocks}, {"free_blocks", event.free_blocks}});
    }
    require(admissions == static_cast<int>(requests.size()) && reused, "Waiting queue or physical block reuse was not exercised");
    require(std::all_of(slots.begin(), slots.end(), [](int slot) { return slot == -1; }), "Active slots remain occupied");
    require(!events.empty() && events.back().free_blocks == 24, "Final block pool was not fully restored");
    report["block_reuse_observed"] = reused;
    std::vector<Request> edges(2);
    edges[0].request_id = 100;
    edges[0].prompt_tokens = {128000};
    edges[0].max_new_tokens = 0;
    edges[1].request_id = 101;
    edges[1].prompt_tokens = inputs[0].at("prompt_tokens").get<std::vector<int>>();
    edges[1].max_new_tokens = 1;
    generateContinuous(handle, weights, edges, 1, 128);
    require(edges[0].finished && !edges[0].prefilled && edges[0].generated_tokens.empty(), "Zero-token request performed inference");
    auto one_token = generatePaged(handle, weights, edges[1].prompt_tokens, 1, 128);
    require(edges[1].finished && edges[1].generated_tokens == one_token, "One-token request decoded beyond prefill");
    report["zero_and_one_token_requests_passed"] = true;
    check(cudaDeviceSynchronize());
    return passed;
}

int main(int argc, char** argv) {
    std::string stage = argc > 1 ? argv[1] : "prefill";
    std::string model = argc > 2 ? argv[2] : "models/llama-3.2-1b-instruct/model.safetensors";
    std::string prompts = argc > 3 ? argv[3] : "tests/paged_requests.json";
    json report = {{"stage", stage}, {"passed", false}};
    char* gpu_weights = nullptr;
    cublasHandle_t handle = nullptr;
    auto directory = std::filesystem::temp_directory_path() /
        ("tiny_vllm_paged_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    bool passed = false;
    try {
        unsetenv("TINY_VLLM_LOGITS_DUMP");
        unsetenv("TINY_VLLM_LOGITS_TRACE");
        std::filesystem::create_directories(directory);
        SafeTensorModel loaded = loadSafeTensors(model);
        require(!loaded.tensor_data.empty(), "Model loading failed");
        gpu_weights = copyModelToGPU(loaded);
        ModelWeights weights = mapWeights(loaded, gpu_weights);
        require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cuBLAS initialization failed");
        std::ifstream input(prompts);
        json inputs;
        input >> inputs;
        passed = true;
        if (stage == "prefill") {
            report["cases"] = json::array();
            for (int length : {1, 15, 16, 17, 31, 32, 33}) {
                testPrefill(handle, weights, length, report["cases"]);
            }
        } else if (stage == "paged") {
            report["cases"] = json::array();
            for (int length : {1, 15, 16, 17, 31, 32, 33}) {
                passed = testGeneration(handle, weights, boundaryPrompt(length), directory, report["cases"]) && passed;
            }
            for (const auto& item : inputs) {
                passed = testGeneration(handle, weights, item.at("prompt_tokens").get<std::vector<int>>(), directory, report["cases"]) && passed;
            }
        } else if (stage == "batch") {
            passed = testBatch(handle, weights, inputs, report);
        } else {
            throw std::invalid_argument("Stage must be prefill, paged, or batch");
        }
        check(cudaDeviceSynchronize());
        report["cuda_runtime_error"] = nullptr;
    } catch (const std::exception& error) {
        report["error"] = error.what();
        std::cerr << error.what() << '\n';
        passed = false;
    }
    unsetenv("TINY_VLLM_LOGITS_TRACE");
    if (handle) {
        cublasDestroy(handle);
    }
    cudaFree(gpu_weights);
    std::filesystem::remove_all(directory);
    report["passed"] = passed;
    std::ofstream output("paged_" + stage + "_report.json");
    output << report.dump(2) << '\n';
    return passed ? 0 : 1;
}
