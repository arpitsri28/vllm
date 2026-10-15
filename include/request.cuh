#pragma once

#include "paged_kvcache.cuh"

struct Request {
    int request_id = 0;
    std::vector<int> prompt_tokens;
    std::vector<int> generated_tokens;
    SequenceKVState kv_state;
    int max_new_tokens = 0;
    int last_token = -1;
    bool prefilled = false;
    bool finished = false;
};
