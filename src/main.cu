#include "../include/weights.cuh"
#include "engine.cuh"
#include "paged_engine.cuh"
#include "scheduler.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdlib>
#include <filesystem>
#include <cuda_bf16.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>

#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct SafeTensorModel {
    std::vector<char> tensor_data;
    json header;
};

SafeTensorModel loadSafeTensors(const std::string& model_path) {
  // 1. Open SafeTensors file in BINARY mode
    std::ifstream file(model_path, std::ios::binary);

    if (!file.is_open())
    {
        std::cerr << "Could not open model file\n";
        return {};
    }

    // --------------------------------------------------
    // 2. Read first 8 bytes
    // --------------------------------------------------
    // SafeTensors stores the JSON header size
    // as a 64-bit unsigned integer.
    uint64_t header_size = 0;

    file.read(
        reinterpret_cast<char*>(&header_size),
        sizeof(uint64_t)
    );

    std::cout << "Header size: "
              << header_size
              << " bytes\n";

    // --------------------------------------------------
    // 3. Read JSON header
    // --------------------------------------------------
    std::string header_string(header_size, '\0');

    file.read(
        header_string.data(),
        header_size
    );

    // Convert JSON text into a C++ JSON object
    json header = json::parse(header_string);

    std::cout << "Successfully parsed JSON header\n";

    // --------------------------------------------------
    // 4. Print some tensor information
    // --------------------------------------------------
    for (const auto& [name, info] : header.items())
    {
        if (name == "__metadata__")
            continue;

        std::cout << "\nTensor: " << name << "\n";

        std::cout << "dtype: "
                  << info.at("dtype").get<std::string>()
                  << "\n";

        std::cout << "shape: ";

        for (const auto& dim : info.at("shape"))
        {
            std::cout << dim.get<uint64_t>() << " ";
        }

        std::cout << "\n";

        auto offsets =
            info.at("data_offsets")
                .get<std::vector<uint64_t>>();

        std::cout << "offset: "
                  << offsets[0]
                  << " -> "
                  << offsets[1]
                  << "\n";
    }

    // --------------------------------------------------
    // 5. We are NOW positioned at beginning of tensor data
    // --------------------------------------------------

    // Find how many bytes remain in the file.
    std::streampos tensor_data_start = file.tellg();

    file.seekg(0, std::ios::end);

    std::streampos file_end = file.tellg();

    uint64_t tensor_data_size =
        static_cast<uint64_t>(
            file_end - tensor_data_start
        );

    std::cout << "\nTensor data size: "
              << tensor_data_size
              << " bytes\n";

    // --------------------------------------------------
    // 6. Allocate CPU memory for raw tensor bytes
    // --------------------------------------------------
    std::vector<char> tensor_data(tensor_data_size);

    // Return to beginning of tensor data
    file.seekg(tensor_data_start);

    // Read all model weights into CPU RAM
    file.read(
        tensor_data.data(),
        tensor_data_size
    );

    if (!file)
    {
        std::cerr << "Failed while reading tensor data\n";
        return {};
    }

    std::cout << "Successfully loaded model weights into CPU memory\n";

    file.close();

    return {std::move(tensor_data), std::move(header)};
}

char* copyModelToGPU(const SafeTensorModel& Model)
{
    size_t size_of_tensors = Model.tensor_data.size();
    char* gpu_tensor_data = nullptr;
    cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&gpu_tensor_data),
        size_of_tensors
    );
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
    status = cudaMemcpy(
        gpu_tensor_data,
        Model.tensor_data.data(),
        size_of_tensors,
        cudaMemcpyHostToDevice
    );
    if (status != cudaSuccess) {
        cudaFree(gpu_tensor_data);
        throw std::runtime_error(cudaGetErrorString(status));
    }
    return gpu_tensor_data;
}

ModelWeights mapWeights(const SafeTensorModel &model, char *gpu_model_weights) {
  size_t embed_offset = model.header["model.embed_tokens.weight"]["data_offsets"][0].get<size_t>();
  ModelWeights mw;
  mw.embedding =
    reinterpret_cast<__nv_bfloat16*>(
        gpu_model_weights + embed_offset
    );
  
  for (int N=0; N < 16; ++N) {
    LayerWeights layer;
    size_t input_norm_offset = model.header["model.layers." + std::to_string(N) + ".input_layernorm.weight"]["data_offsets"][0].get<size_t>();
    layer.input_norm = reinterpret_cast<__nv_bfloat16*>(gpu_model_weights + input_norm_offset);
    size_t q_proj_offset = model.header["model.layers." + std::to_string(N) + ".self_attn.q_proj.weight"]["data_offsets"][0].get<size_t>();
    layer.w_q = reinterpret_cast<__nv_bfloat16*>(gpu_model_weights + q_proj_offset);
    size_t k_proj_offset = model.header["model.layers." + std::to_string(N) + ".self_attn.k_proj.weight"]["data_offsets"][0].get<size_t>();
    layer.w_k = reinterpret_cast<__nv_bfloat16*>(gpu_model_weights + k_proj_offset);
    size_t v_proj_offset = model.header["model.layers." + std::to_string(N) + ".self_attn.v_proj.weight"]["data_offsets"][0].get<size_t>();
    layer.w_v = reinterpret_cast<__nv_bfloat16*>(gpu_model_weights + v_proj_offset);
    size_t output_proj_offset = model.header["model.layers." + std::to_string(N) + ".self_attn.o_proj.weight"]["data_offsets"][0].get<size_t>();
    layer.w_o = reinterpret_cast<__nv_bfloat16*>(gpu_model_weights + output_proj_offset);
    size_t gate_proj_offset = model.header["model.layers." + std::to_string(N) + ".mlp.gate_proj.weight"]["data_offsets"][0].get<size_t>();
    layer.w_gate = reinterpret_cast<__nv_bfloat16*>(gpu_model_weights + gate_proj_offset);
    size_t up_proj_offset = model.header["model.layers." + std::to_string(N) + ".mlp.up_proj.weight"]["data_offsets"][0].get<size_t>();
    layer.w_up = reinterpret_cast<__nv_bfloat16*>(gpu_model_weights + up_proj_offset);
    size_t down_proj_offset = model.header["model.layers." + std::to_string(N) + ".mlp.down_proj.weight"]["data_offsets"][0].get<size_t>();
    layer.w_down = reinterpret_cast<__nv_bfloat16*>(gpu_model_weights + down_proj_offset);
    size_t post_att_offset = model.header["model.layers." + std::to_string(N) + ".post_attention_layernorm.weight"]["data_offsets"][0].get<size_t>();
    layer.post_attention_norm = reinterpret_cast<__nv_bfloat16*>(gpu_model_weights + post_att_offset);
    mw.layers.push_back(layer);
  }
  size_t final_norm_offset = model.header["model.norm.weight"]["data_offsets"][0].get<size_t>();
  mw.final_norm = reinterpret_cast<__nv_bfloat16*>(
        gpu_model_weights + final_norm_offset
    );
  
  return mw;
}

std::vector<int> readTokens(const std::string& tokens_path) {
  std::ifstream file(tokens_path);
  json data;
  file >> data;
  return data.get<std::vector<int>>();
}

int main(int argc, char** argv)
{
    const std::string model_path = argc > 1 ? argv[1] : "models/llama-3.2-1b-instruct/model.safetensors";
    const std::string tokens_path = argc > 2 ? argv[2] : "src/tokens.json";
    char* gpu_tensor_data = nullptr;
    cublasHandle_t cublas_handle = nullptr;

    try {
        int max_new_tokens = argc > 3 ? std::stoi(argv[3]) : 32;
        int max_seq_len = argc > 4 ? std::stoi(argv[4]) : 1024;
        std::string mode = argc > 5 ? argv[5] : "contiguous";
        if (mode != "contiguous" && mode != "paged" && mode != "batch") {
            throw std::invalid_argument("Mode must be contiguous, paged, or batch");
        }
        SafeTensorModel Model = loadSafeTensors(model_path);
        if (Model.tensor_data.empty()) {
            throw std::runtime_error("Model contains no tensor data");
        }
        gpu_tensor_data = copyModelToGPU(Model);
        ModelWeights w = mapWeights(Model, gpu_tensor_data);
        if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS handle");
        }
        std::vector<int> generated_tokens;
        if (mode == "batch") {
            std::ifstream file(tokens_path);
            json data;
            file >> data;
            std::vector<Request> requests;
            for (const auto& item : data) {
                Request request;
                request.request_id = item.at("request_id").get<int>();
                request.prompt_tokens = item.at("prompt_tokens").get<std::vector<int>>();
                request.max_new_tokens = item.value("max_new_tokens", max_new_tokens);
                requests.push_back(std::move(request));
            }
            int batch_size = argc > 6 ? std::stoi(argv[6]) : MAX_BATCH_SIZE;
            generateContinuous(cublas_handle, w, requests, batch_size, max_seq_len);
            for (const auto& request : requests) {
                std::cout << json({{"request_id", request.request_id}, {"generated_tokens", request.generated_tokens}}) << '\n';
            }
        } else {
            std::vector<int> input_tokens = readTokens(tokens_path);
            generated_tokens = mode == "paged"
                ? generatePaged(cublas_handle, w, input_tokens, max_new_tokens, max_seq_len)
                : generate(cublas_handle, w, input_tokens, max_new_tokens, max_seq_len);
        }
        for (int token : generated_tokens) {
            std::cout << token << " ";
        }
        std::cout << std::endl;

        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
        cudaFree(gpu_tensor_data);
        gpu_tensor_data = nullptr;

        if (!generated_tokens.empty()) {
            auto quote = [](const std::string& value) {
                std::string result = "'";
                for (char c : value) {
                    result += c == '\'' ? "'\\''" : std::string(1, c);
                }
                return result + "'";
            };
            std::string command = "python3 src/python/tokenizer.py --decode --model " +
                quote(std::filesystem::absolute(model_path).parent_path().string()) + " --ids";
            for (int token : generated_tokens) {
                command += " " + std::to_string(token);
            }
            if (std::system(command.c_str()) != 0) {
                std::cerr << "Detokenization failed; generated token IDs are printed above" << std::endl;
            }
        }
    } catch (const std::exception& error) {
        if (cublas_handle != nullptr) {
            cublasDestroy(cublas_handle);
        }
        cudaFree(gpu_tensor_data);
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}
