#pragma once

#include <cuda_bf16.h>
#include <vector>

struct LayerWeights {
    __nv_bfloat16* w_q;
    __nv_bfloat16* w_k;
    __nv_bfloat16* w_v;
    __nv_bfloat16* w_o;

    __nv_bfloat16* w_gate;
    __nv_bfloat16* w_up;
    __nv_bfloat16* w_down;

    __nv_bfloat16* input_norm;
    __nv_bfloat16* post_attention_norm;
};

struct ModelWeights {
    __nv_bfloat16* embedding;
    __nv_bfloat16* final_norm;

    std::vector<LayerWeights> layers;
};
