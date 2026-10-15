#pragma once

#include <cuda_bf16.h>

void rope(__nv_bfloat16 *input, int num_tokens, int proj_dim, int position_offset);
void ropeBatch(__nv_bfloat16* input, int batch_size, int proj_dim, const int* positions);
