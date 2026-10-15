#pragma once

#include "request.cuh"
#include "weights.cuh"
#include <cublas_v2.h>
#include <queue>

constexpr int MAX_BATCH_SIZE = 4;

struct SchedulerEvent {
    int iteration;
    int slot;
    int request_id;
    bool admitted;
    std::vector<int> blocks;
    int free_blocks;
};

struct Scheduler {
    std::queue<int> waiting_queue;
    std::vector<int> active_slots;
    std::vector<int> free_slots;
};

void generateContinuous(cublasHandle_t cublas_handle, const ModelWeights& weights,
                        std::vector<Request>& requests, int max_batch_size, int max_seq_len,
                        std::vector<SchedulerEvent>* events = nullptr);
