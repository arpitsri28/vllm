#include "../include/weights.cuh"
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
    cudaMalloc(
        reinterpret_cast<void**>(&gpu_tensor_data),
        size_of_tensors
    );
    cudaMemcpy(
        gpu_tensor_data,
        Model.tensor_data.data(),
        size_of_tensors,
        cudaMemcpyHostToDevice
    );
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
    mw.layers.push_back(layer)
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

int main()
{
    const std::string model_path =
        "/Users/arpitsrivastav/Desktop/tiny_vLLM/models/"
        "llama-3.2-1b-instruct/model.safetensors";

    SafeTensorModel Model = loadSafeTensors(model_path);
    char *gpu_tensor_data = copyModelToGPU(Model);
    ModelWeights w = mapWeights(Model, gpu_tensor_data);
    std::vector<int> input_tokens = readTokens("/Users/arpitsrivastav/Desktop/tiny_vLLM/src/tokens.json");
    constexpr int MAX_PROMPT_LEN = 512;
    int *gpu_input_tokens = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&gpu_input_tokens), MAX_PROMPT_LEN * sizeof(int));
    cudaMemcpy(gpu_input_tokens, input_tokens.data(), input_tokens.size() * sizeof(int), cudaMemcpyHostToDevice);
}
