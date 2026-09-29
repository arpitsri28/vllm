#include "lm_head.cuh"
constexpr int VOCAB_SIZE = 128256;
constexpr int EMBEDDING_LENGTH = 2048;

void lmHeadProjection(cublasHandle_t cublas_handle, const __nv_bfloat16* final_hidden, const __nv_bfloat16* embedding_weights, float* logits) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        VOCAB_SIZE,          
        1,                   
        EMBEDDING_LENGTH,    
        &alpha,
        embedding_weights,
        CUDA_R_16BF,
        EMBEDDING_LENGTH,
        final_hidden,
        CUDA_R_16BF,
        EMBEDDING_LENGTH,
        &beta,
        logits,
        CUDA_R_32F,
        VOCAB_SIZE,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "LM head projection failed" << std::endl;
    }
}