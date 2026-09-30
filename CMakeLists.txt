cmake_minimum_required(VERSION 3.18)
project(tiny_vLLM LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp REQUIRED)

add_executable(tiny_vLLM
    src/main.cu
    src/engine/engine.cu
    src/model/attention.cu
    src/model/causalMask.cu
    src/model/embedding.cu
    src/model/resConnection.cu
    src/model/rmsNorm.cu
    src/model/RoPe.cu
    src/model/SiLU.cu
    src/model/softmax.cu
    src/model/transformer.cu
    src/model/mlp.cu
    src/model/kvcache.cu
    src/model/lm_head.cu
    src/model/argmax.cu
)

target_include_directories(tiny_vLLM PRIVATE include ${NLOHMANN_JSON_INCLUDE_DIR})
target_link_libraries(tiny_vLLM PRIVATE CUDA::cudart CUDA::cublas)
set_target_properties(tiny_vLLM PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
