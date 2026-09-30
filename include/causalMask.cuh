#pragma once

#include <cuda_bf16.h>

void causalMask(__nv_bfloat16 *input, int num_tokens);
