#pragma once

#include <cuda_bf16.h>

void softmaxRows(__nv_bfloat16* input, int num_rows, int row_length);
