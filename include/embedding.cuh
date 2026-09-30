#pragma once

#include <cuda_bf16.h>

__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens);
