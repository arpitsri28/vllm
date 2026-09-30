#include "../../include/attention.cuh"
#include <iostream>

constexpr int NUM_Q_HEADS = 32;
constexpr int KV_DIM = 512;
constexpr int EMBEDDING_LENGTH = 2048;

void computeAttentionScores(cublasHandle_t cublas_handle, __nv_bfloat16* q_proj, __nv_bfloat16* k_proj_layer, __nv_bfloat16* attn_scores, int num_tokens) {
    constexpr int GQA_Q_TO_K_RATIO = 4;
    constexpr int HEAD_DIM = 64;

    const float attn_alpha = 1.0f / 8.0f; 
    const float attn_beta = 0.0f;

    for (int head = 0; head < NUM_Q_HEADS; ++head) {
        int k_head_idx = head / GQA_Q_TO_K_RATIO;
        __nv_bfloat16* q_head = q_proj + head * HEAD_DIM;
        __nv_bfloat16* k_head = k_proj_layer + k_head_idx * HEAD_DIM;
        __nv_bfloat16* attn_score_head = attn_scores + num_tokens * num_tokens * head;
        cublasStatus_t status = cublasGemmEx(
            cublas_handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            num_tokens,
            num_tokens,
            HEAD_DIM,
            &attn_alpha,
            k_head,
            CUDA_R_16BF,
            KV_DIM,
            q_head,
            CUDA_R_16BF,
            EMBEDDING_LENGTH,
            &attn_beta,
            attn_score_head,
            CUDA_R_16BF,
            num_tokens,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );

        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cuBLAS attention score GEMM failed for head "
                      << head << std::endl;
        }
    }
}

void computeDecodeAttentionScores(cublasHandle_t cublas_handle, const __nv_bfloat16* q_proj, const __nv_bfloat16* k_cache_layer, __nv_bfloat16* attn_scores, int total_seq_len) {
    constexpr int GQA_Q_TO_K_RATIO = 4;
    constexpr int HEAD_DIM = 64;

    const float attn_alpha = 1.0f / 8.0f; 
    const float attn_beta = 0.0f;
    for (int head = 0; head < NUM_Q_HEADS; ++head) {
        int k_head_idx = head / GQA_Q_TO_K_RATIO;
        const __nv_bfloat16* q_head = q_proj + head * HEAD_DIM;
        const __nv_bfloat16* k_head = k_cache_layer + k_head_idx * HEAD_DIM;
        __nv_bfloat16* attn_score_head = attn_scores + head * total_seq_len;
        cublasStatus_t status = cublasGemmEx(
            cublas_handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            total_seq_len,
            1,
            HEAD_DIM,
            &attn_alpha,
            k_head,
            CUDA_R_16BF,
            KV_DIM,
            q_head,
            CUDA_R_16BF,
            HEAD_DIM,    
            &attn_beta,
            attn_score_head,
            CUDA_R_16BF,
            total_seq_len,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr
                << "Decode attention score GEMM failed for head "
                << head << std::endl;
        }
    }
}

void decodeAttentionScoresTimesV(cublasHandle_t cublas_handle, const __nv_bfloat16* attn_scores, const __nv_bfloat16* v_cache_layer, __nv_bfloat16* attention_output, int total_seq_len) {
    constexpr int GQA_Q_TO_V_RATIO = 4;
    constexpr int HEAD_DIM = 64;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int head = 0; head < NUM_Q_HEADS; ++head) {
        int v_head_idx = head / GQA_Q_TO_V_RATIO;
        const __nv_bfloat16* attn_scores_head = attn_scores + head * total_seq_len;


        const __nv_bfloat16* v_head = v_cache_layer + v_head_idx * HEAD_DIM;

        __nv_bfloat16* output_head = attention_output + head * HEAD_DIM;

        cublasStatus_t status = cublasGemmEx(
            cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            HEAD_DIM,        
            1,               
            total_seq_len,   
            &alpha,
            v_head,
            CUDA_R_16BF,
            KV_DIM,          
            attn_scores_head,
            CUDA_R_16BF,
            total_seq_len,
            &beta,
            output_head,
            CUDA_R_16BF,
            EMBEDDING_LENGTH,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );

        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr
                << "Decode attention x V GEMM failed for head "
                << head << std::endl;
        }
    }
}

void attentionScoresTimesV(cublasHandle_t cublas_handle, __nv_bfloat16* attn_scores, __nv_bfloat16* v_proj_layer, __nv_bfloat16* attn_scores_v, int num_tokens) {
    constexpr int GQA_ATTN_SCORES_TO_V_RATIO = 4;
    constexpr int HEAD_DIM = 64;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int head = 0; head < NUM_Q_HEADS; ++head) {
        int v_head_idx =
            head / GQA_ATTN_SCORES_TO_V_RATIO;

        __nv_bfloat16* attn_scores_head = attn_scores + head * num_tokens * num_tokens;

        __nv_bfloat16* v_head = v_proj_layer + v_head_idx * HEAD_DIM;

        __nv_bfloat16* output_attn_head = attn_scores_v + head * HEAD_DIM;

        cublasStatus_t status = cublasGemmEx(
            cublas_handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            HEAD_DIM,
            num_tokens,
            num_tokens,
            &alpha,
            v_head,
            CUDA_R_16BF,
            KV_DIM,
            attn_scores_head,
            CUDA_R_16BF,
            num_tokens,
            &beta,
            output_attn_head,
            CUDA_R_16BF,
            EMBEDDING_LENGTH,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        );

        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr
                << "attention × V GEMM failed for head "
                << head
                << std::endl;
        }
    }
}

void outputProjection(cublasHandle_t cublas_handle, __nv_bfloat16* attention_output, __nv_bfloat16* w_o, __nv_bfloat16* output, int num_tokens) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        EMBEDDING_LENGTH,   
        num_tokens,
        EMBEDDING_LENGTH,   
        &alpha,
        w_o,
        CUDA_R_16BF,
        EMBEDDING_LENGTH,
        attention_output,
        CUDA_R_16BF,
        EMBEDDING_LENGTH,
        &beta,
        output,
        CUDA_R_16BF,
        EMBEDDING_LENGTH,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "O projection GEMM failed" << std::endl;
    }
}

void qProjection(cublasHandle_t cublas_handle, const __nv_bfloat16* input, const __nv_bfloat16* weights, __nv_bfloat16* output, int num_tokens) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        EMBEDDING_LENGTH,   
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
        EMBEDDING_LENGTH,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "Q projection failed" << std::endl;
    }
}

void kProjection(cublasHandle_t cublas_handle, const __nv_bfloat16* input, const __nv_bfloat16* weights, __nv_bfloat16* output, int num_tokens) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        KV_DIM,             
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
        KV_DIM,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "K projection failed" << std::endl;
    }
}

void vProjection(cublasHandle_t cublas_handle, const __nv_bfloat16* input, const __nv_bfloat16* weights, __nv_bfloat16* output, int num_tokens) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        KV_DIM,             
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
        KV_DIM,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "V projection failed" << std::endl;
    }
}
