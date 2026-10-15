#include <cstddef>
#include "operation_profile.cuh"
#include "../../include/mlp.cuh"
#include <iostream>

constexpr int EMBEDDING_LENGTH = 2048;
constexpr int MLP_DIM = 8192;
void gateProjection(cublasHandle_t cublas_handle, const __nv_bfloat16* input, const __nv_bfloat16* weights, __nv_bfloat16* output, int num_tokens) {
    OP_SCOPE("gate_projection",true);
  const float alpha = 1.0f;
  const float beta = 0.0f;
    operationBlas();
  cublasStatus_t status = cublasGemmEx(
      cublas_handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      MLP_DIM,   
      num_tokens,
      EMBEDDING_LENGTH,   
      &alpha,
      weights,
      CUDA_R_16BF,
      EMBEDDING_LENGTH,
      input,
      CUDA_R_16BF,
      EMBEDDING_LENGTH,
      &beta,
      output,
      CUDA_R_16BF,
      MLP_DIM,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT
  );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Gate Projection failed" << std::endl;
    }
}

void upProjection(cublasHandle_t cublas_handle, const __nv_bfloat16* input, const __nv_bfloat16* weights, __nv_bfloat16* output, int num_tokens) {
    OP_SCOPE("up_projection",true);
  const float alpha = 1.0f;
  const float beta = 0.0f;
    operationBlas();
  cublasStatus_t status = cublasGemmEx(
      cublas_handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      MLP_DIM,   
      num_tokens,
      EMBEDDING_LENGTH,   
      &alpha,
      weights,
      CUDA_R_16BF,
      EMBEDDING_LENGTH,
      input,
      CUDA_R_16BF,
      EMBEDDING_LENGTH,
      &beta,
      output,
      CUDA_R_16BF,
      MLP_DIM,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT
  );

  if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "Up Projection failed" << std::endl;
  }
}

void downProjection(cublasHandle_t cublas_handle, const __nv_bfloat16* input, const __nv_bfloat16* weights, __nv_bfloat16* output, int num_tokens) {
    OP_SCOPE("down_projection",true);
  const float alpha = 1.0f;
  const float beta = 0.0f;
    operationBlas();
  cublasStatus_t status = cublasGemmEx(
      cublas_handle,
      CUBLAS_OP_T,
      CUBLAS_OP_N,
      EMBEDDING_LENGTH,  
      num_tokens,
      MLP_DIM, 
      &alpha,
      weights,
      CUDA_R_16BF,
      MLP_DIM,
      input,
      CUDA_R_16BF,
      MLP_DIM,
      &beta,
      output,
      CUDA_R_16BF,
      EMBEDDING_LENGTH,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT
  );

  if (status != CUBLAS_STATUS_SUCCESS) {
      std::cerr << "Up Projection failed" << std::endl;
  }
}
