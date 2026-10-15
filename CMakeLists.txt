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
    src/model/paged_kvcache.cu
    src/model/paged_attention.cu
    src/model/paged_transformer.cu
    src/engine/paged_engine.cu
    src/engine/scheduler.cu
)

option(BUILD_PAGED_TESTS "Build staged paged inference validation" OFF)
if(BUILD_PAGED_TESTS)
    enable_testing()
    get_target_property(PAGED_TEST_SOURCES tiny_vLLM SOURCES)
    list(REMOVE_ITEM PAGED_TEST_SOURCES src/main.cu)
    add_executable(paged_kvcache_test tests/paged_kvcache_test.cu src/model/paged_kvcache.cu)
    add_executable(paged_attention_test tests/paged_attention_test.cu src/model/paged_kvcache.cu
        src/model/paged_attention.cu src/model/attention.cu src/model/softmax.cu src/model/RoPe.cu)
    add_executable(paged_inference_test tests/paged_inference_test.cu ${PAGED_TEST_SOURCES})
    foreach(test_target paged_kvcache_test paged_attention_test paged_inference_test)
        target_include_directories(${test_target} PRIVATE include ${NLOHMANN_JSON_INCLUDE_DIR})
        target_link_libraries(${test_target} PRIVATE CUDA::cudart CUDA::cublas)
        set_target_properties(${test_target} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    endforeach()
    add_test(NAME paged_kvcache COMMAND paged_kvcache_test)
    add_test(NAME paged_attention COMMAND paged_attention_test)
    foreach(stage prefill paged batch)
        add_test(NAME paged_${stage} COMMAND paged_inference_test ${stage}
            ${CMAKE_CURRENT_SOURCE_DIR}/models/llama-3.2-1b-instruct/model.safetensors
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/paged_requests.json)
        set_tests_properties(paged_${stage} PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    endforeach()
    set_tests_properties(paged_kvcache PROPERTIES FIXTURES_SETUP paged_cache_validated)
    set_tests_properties(paged_prefill PROPERTIES FIXTURES_REQUIRED paged_cache_validated FIXTURES_SETUP paged_prefill_validated)
    set_tests_properties(paged_attention PROPERTIES FIXTURES_REQUIRED paged_prefill_validated FIXTURES_SETUP paged_attention_validated)
    set_tests_properties(paged_paged PROPERTIES FIXTURES_REQUIRED paged_attention_validated FIXTURES_SETUP paged_decode_validated)
    set_tests_properties(paged_batch PROPERTIES FIXTURES_REQUIRED paged_decode_validated)
    set_tests_properties(paged_kvcache paged_prefill paged_attention paged_paged paged_batch
        PROPERTIES FAIL_REGULAR_EXPRESSION "projection failed|Projection failed|GEMM failed|CUDA last error")
endif()

target_include_directories(tiny_vLLM PRIVATE include ${NLOHMANN_JSON_INCLUDE_DIR})
target_link_libraries(tiny_vLLM PRIVATE CUDA::cudart CUDA::cublas)
set_target_properties(tiny_vLLM PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

option(BUILD_BENCHMARKS "Build opt-in inference benchmarks" OFF)
if(BUILD_BENCHMARKS)
    get_target_property(BENCHMARK_SOURCES tiny_vLLM SOURCES)
    list(REMOVE_ITEM BENCHMARK_SOURCES src/main.cu)
    add_executable(inference_benchmark benchmarks/benchmark.cu ${BENCHMARK_SOURCES})
    target_include_directories(inference_benchmark PRIVATE include ${NLOHMANN_JSON_INCLUDE_DIR})
    target_link_libraries(inference_benchmark PRIVATE CUDA::cudart CUDA::cublas)
    set_target_properties(inference_benchmark PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()

if(BUILD_PAGED_TESTS)
    add_executable(numerical_diagnostic tests/numerical_diagnostic.cu ${PAGED_TEST_SOURCES})
    target_compile_definitions(numerical_diagnostic PRIVATE TINY_VLLM_NUMERICAL_TRACE=1)
    target_include_directories(numerical_diagnostic PRIVATE include ${NLOHMANN_JSON_INCLUDE_DIR})
    target_link_libraries(numerical_diagnostic PRIVATE CUDA::cudart CUDA::cublas)
    set_target_properties(numerical_diagnostic PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()

if(BUILD_PAGED_TESTS)
    add_executable(bf16_argmax_test tests/argmax_test.cu src/model/argmax.cu)
    target_include_directories(bf16_argmax_test PRIVATE include)
    target_link_libraries(bf16_argmax_test PRIVATE CUDA::cudart)
    set_target_properties(bf16_argmax_test PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    add_test(NAME bf16_argmax COMMAND bf16_argmax_test)
endif()

option(BUILD_OPERATION_PROFILER "Build diagnostic operation-level timing executable" OFF)
if(BUILD_OPERATION_PROFILER)
    get_target_property(OPERATION_SOURCES tiny_vLLM SOURCES)
    list(REMOVE_ITEM OPERATION_SOURCES src/main.cu)
    add_executable(operation_profiler benchmarks/operation_profiler.cu ${OPERATION_SOURCES})
    target_compile_definitions(operation_profiler PRIVATE TINY_VLLM_OPERATION_PROFILE=1)
    target_include_directories(operation_profiler PRIVATE include ${NLOHMANN_JSON_INCLUDE_DIR})
    target_link_libraries(operation_profiler PRIVATE CUDA::cudart CUDA::cublas)
    set_target_properties(operation_profiler PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endif()
