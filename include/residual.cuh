#pragma once

#include <cuda_bf16.h>

void residualAdd(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds, int num_tokens);
