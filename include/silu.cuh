#pragma once

#include <cuda_bf16.h>

void silu(__nv_bfloat16 *a, __nv_bfloat16 *b, int num_tokens);
