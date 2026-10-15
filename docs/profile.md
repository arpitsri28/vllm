# Operation profiling evidence

Times are measured over one full generation call after two warmups. Host API wall and GPU stream elapsed overlap; do not add them. Event intervals include submission gaps and are not summed kernel-active execution. Profiler readiness waits move the existing token-copy dependency to an explicit observation point. Operation profiling perturbs performance; no throughput claim uses its total wall time.

Nsight Systems was absent; Nsight Compute 2025.1.1 returned ERR_NVGPUCTRPERM. Physical cuBLAS kernel counts, occupancy and hardware bandwidth were unavailable. Explicit launch counts and cuBLAS API calls below are measured at source call sites.

## before: contiguous, batch 1

Actual output tokens: 128; profiled full-call wall: 1504.207 ms (perturbed). Tokens match warmup. All phases remain in raw JSON; the table focuses on decode.

| Operation | Host API/CPU wall ms | GPU stream elapsed ms | Calls | Explicit kernels | cuBLAS calls | Alloc/free calls | Tracked selection-copy bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| attention_scores | 335.769 | 184.250 | 2032 | 0 | 65024 | 0/0 | 0 |
| buffer_free | 2.304 | 0.000 | 5 | 0 | 0 | 0/5 | 0 |
| cpu_greedy_scan | 107.172 | 0.000 | 127 | 0 | 0 | 0/0 | 0 |
| down_projection | 22.685 | 140.057 | 2032 | 0 | 2032 | 0/0 | 0 |
| embedding | 0.963 | 0.921 | 127 | 127 | 0 | 0/0 | 0 |
| gate_projection | 19.611 | 129.994 | 2032 | 0 | 2032 | 0/0 | 0 |
| k_projection | 17.290 | 17.752 | 2032 | 0 | 2032 | 0/0 | 0 |
| kv_free | 0.583 | 0.000 | 1 | 0 | 0 | 0/2 | 0 |
| kv_store | 20.492 | 11.352 | 2032 | 0 | 0 | 0/0 | 0 |
| lm_head | 1.563 | 118.619 | 127 | 0 | 127 | 0/0 | 0 |
| logits_d2h | 16.730 | 16.314 | 127 | 0 | 0 | 0/0 | 65154048 |
| logits_host_allocation | 4.130 | 0.000 | 127 | 0 | 0 | 0/0 | 0 |
| o_projection | 17.395 | 42.863 | 2032 | 0 | 2032 | 0/0 | 0 |
| profiler_readiness_wait | 277.726 | 0.000 | 254 | 0 | 0 | 0/0 | 0 |
| q_projection | 18.048 | 43.042 | 2032 | 0 | 2032 | 0/0 | 0 |
| residual | 21.543 | 18.444 | 4064 | 4064 | 0 | 0/0 | 0 |
| rmsnorm | 21.805 | 28.553 | 4191 | 4191 | 0 | 0/0 | 0 |
| rope | 22.767 | 19.263 | 4064 | 4064 | 0 | 0/0 | 0 |
| softmax | 11.005 | 14.151 | 2032 | 2032 | 0 | 0/0 | 0 |
| swiglu | 10.655 | 15.342 | 2032 | 2032 | 0 | 0/0 | 0 |
| up_projection | 18.568 | 129.820 | 2032 | 0 | 2032 | 0/0 | 0 |
| v_projection | 17.080 | 17.815 | 2032 | 0 | 2032 | 0/0 | 0 |
| weighted_v | 478.999 | 326.059 | 2032 | 0 | 65024 | 0/0 | 0 |
| workspace_free | 0.906 | 0.000 | 1 | 0 | 0 | 0/11 | 0 |
## before: batch, batch 1

Actual output tokens: 128; profiled full-call wall: 1966.954 ms (perturbed). Tokens match warmup. All phases remain in raw JSON; the table focuses on decode.

| Operation | Host API/CPU wall ms | GPU stream elapsed ms | Calls | Explicit kernels | cuBLAS calls | Alloc/free calls | Tracked selection-copy bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| attention_scores | 293.347 | 173.558 | 2032 | 0 | 65024 | 0/0 | 0 |
| buffer_free | 0.091 | 0.000 | 3 | 0 | 0 | 0/3 | 0 |
| cpu_greedy_scan | 88.434 | 0.000 | 127 | 0 | 0 | 0/0 | 0 |
| down_projection | 19.610 | 140.204 | 2032 | 0 | 2032 | 0/0 | 0 |
| embedding | 1.022 | 0.974 | 127 | 127 | 0 | 0/0 | 0 |
| gate_projection | 17.109 | 131.677 | 2032 | 0 | 2032 | 0/0 | 0 |
| k_projection | 15.127 | 17.803 | 2032 | 0 | 2032 | 0/0 | 0 |
| kv_free | 0.707 | 0.000 | 1 | 0 | 0 | 0/2 | 0 |
| kv_gather | 20.133 | 19.465 | 4064 | 4064 | 0 | 0/0 | 0 |
| kv_store | 12.992 | 9.052 | 2032 | 2032 | 0 | 0/0 | 0 |
| lm_head | 1.473 | 118.641 | 127 | 0 | 127 | 0/0 | 0 |
| logits_d2h | 16.188 | 15.758 | 127 | 0 | 0 | 0/0 | 65154048 |
| logits_host_allocation | 4.372 | 0.000 | 127 | 0 | 0 | 0/0 | 0 |
| metadata_allocate_upload | 2.657 | 0.000 | 127 | 0 | 0 | 508/0 | 0 |
| metadata_free | 1.881 | 0.000 | 127 | 0 | 0 | 0/508 | 0 |
| o_projection | 15.154 | 42.696 | 2032 | 0 | 2032 | 0/0 | 0 |
| profiler_readiness_wait | 592.820 | 0.000 | 254 | 0 | 0 | 0/0 | 0 |
| q_projection | 15.992 | 43.270 | 2032 | 0 | 2032 | 0/0 | 0 |
| residual | 19.245 | 18.382 | 4064 | 4064 | 0 | 0/0 | 0 |
| rmsnorm | 19.637 | 28.936 | 4191 | 4191 | 0 | 0/0 | 0 |
| rope | 21.084 | 19.415 | 4064 | 4064 | 0 | 0/0 | 0 |
| softmax | 334.558 | 404.036 | 65024 | 65024 | 0 | 0/0 | 0 |
| swiglu | 9.522 | 15.446 | 2032 | 2032 | 0 | 0/0 | 0 |
| up_projection | 16.145 | 129.889 | 2032 | 0 | 2032 | 0/0 | 0 |
| v_projection | 15.008 | 17.849 | 2032 | 0 | 2032 | 0/0 | 0 |
| weighted_v | 353.448 | 249.546 | 2032 | 0 | 65024 | 0/0 | 0 |
| workspace_free | 0.686 | 0.000 | 2 | 0 | 0 | 0/22 | 0 |
## before: batch, batch 4

Actual output tokens: 512; profiled full-call wall: 7642.467 ms (perturbed). Tokens match warmup. All phases remain in raw JSON; the table focuses on decode.

| Operation | Host API/CPU wall ms | GPU stream elapsed ms | Calls | Explicit kernels | cuBLAS calls | Alloc/free calls | Tracked selection-copy bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| attention_scores | 1442.408 | 696.349 | 8128 | 0 | 260096 | 0/0 | 0 |
| buffer_free | 0.203 | 0.000 | 3 | 0 | 0 | 0/3 | 0 |
| cpu_greedy_scan | 362.530 | 0.000 | 508 | 0 | 0 | 0/0 | 0 |
| down_projection | 78.730 | 559.207 | 8128 | 0 | 8128 | 0/0 | 0 |
| embedding | 1.638 | 1.436 | 127 | 127 | 0 | 0/0 | 0 |
| gate_projection | 68.556 | 525.054 | 8128 | 0 | 8128 | 0/0 | 0 |
| k_projection | 61.832 | 60.348 | 8128 | 0 | 8128 | 0/0 | 0 |
| kv_free | 1.898 | 0.000 | 1 | 0 | 0 | 0/2 | 0 |
| kv_gather | 25.018 | 27.461 | 4064 | 4064 | 0 | 0/0 | 0 |
| kv_store | 10.087 | 8.936 | 2032 | 2032 | 0 | 0/0 | 0 |
| lm_head | 5.345 | 474.555 | 508 | 0 | 508 | 0/0 | 0 |
| logits_d2h | 89.697 | 88.450 | 508 | 0 | 0 | 0/0 | 260616192 |
| logits_host_allocation | 9.172 | 0.000 | 508 | 0 | 0 | 0/0 | 0 |
| metadata_allocate_upload | 92.908 | 0.000 | 127 | 0 | 0 | 508/0 | 0 |
| metadata_free | 162.318 | 0.000 | 127 | 0 | 0 | 0/508 | 0 |
| o_projection | 61.775 | 171.538 | 8128 | 0 | 8128 | 0/0 | 0 |
| profiler_readiness_wait | 998.914 | 0.000 | 635 | 0 | 0 | 0/0 | 0 |
| q_projection | 63.951 | 172.615 | 8128 | 0 | 8128 | 0/0 | 0 |
| residual | 20.217 | 18.614 | 4064 | 4064 | 0 | 0/0 | 0 |
| rmsnorm | 21.291 | 30.141 | 4191 | 4191 | 0 | 0/0 | 0 |
| rope | 20.098 | 19.002 | 4064 | 4064 | 0 | 0/0 | 0 |
| softmax | 2240.383 | 1616.222 | 260096 | 260096 | 0 | 0/0 | 0 |
| swiglu | 9.992 | 16.149 | 2032 | 2032 | 0 | 0/0 | 0 |
| up_projection | 65.964 | 517.956 | 8128 | 0 | 8128 | 0/0 | 0 |
| v_projection | 60.881 | 60.353 | 8128 | 0 | 8128 | 0/0 | 0 |
| weighted_v | 1438.470 | 999.311 | 8128 | 0 | 260096 | 0/0 | 0 |
| workspace_free | 1.884 | 0.000 | 2 | 0 | 0 | 0/22 | 0 |
## after: contiguous, batch 1

Actual output tokens: 128; profiled full-call wall: 1430.773 ms (perturbed). Tokens match warmup. All phases remain in raw JSON; the table focuses on decode.

| Operation | Host API/CPU wall ms | GPU stream elapsed ms | Calls | Explicit kernels | cuBLAS calls | Alloc/free calls | Tracked selection-copy bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| attention_scores | 340.988 | 190.358 | 2032 | 0 | 65024 | 0/0 | 0 |
| buffer_free | 1.486 | 0.000 | 5 | 0 | 0 | 0/5 | 0 |
| down_projection | 23.194 | 140.235 | 2032 | 0 | 2032 | 0/0 | 0 |
| embedding | 0.824 | 0.736 | 127 | 127 | 0 | 0/0 | 0 |
| gate_projection | 20.072 | 130.174 | 2032 | 0 | 2032 | 0/0 | 0 |
| gpu_greedy_reduction | 1.401 | 1.390 | 127 | 254 | 0 | 0/0 | 0 |
| k_projection | 17.739 | 17.903 | 2032 | 0 | 2032 | 0/0 | 0 |
| kv_free | 4.077 | 0.000 | 1 | 0 | 0 | 0/2 | 0 |
| kv_store | 21.727 | 11.379 | 2032 | 0 | 0 | 0/0 | 0 |
| lm_head | 1.598 | 118.618 | 127 | 0 | 127 | 0/0 | 0 |
| o_projection | 17.939 | 44.515 | 2032 | 0 | 2032 | 0/0 | 0 |
| profiler_readiness_wait | 293.352 | 0.000 | 381 | 0 | 0 | 0/0 | 0 |
| q_projection | 18.257 | 43.103 | 2032 | 0 | 2032 | 0/0 | 0 |
| residual | 23.953 | 18.632 | 4064 | 4064 | 0 | 0/0 | 0 |
| rmsnorm | 24.183 | 28.787 | 4191 | 4191 | 0 | 0/0 | 0 |
| rope | 23.920 | 19.350 | 4064 | 4064 | 0 | 0/0 | 0 |
| softmax | 12.114 | 14.337 | 2032 | 2032 | 0 | 0/0 | 0 |
| swiglu | 11.813 | 15.397 | 2032 | 2032 | 0 | 0/0 | 0 |
| token_d2h | 2.138 | 1.803 | 127 | 0 | 0 | 0/0 | 508 |
| up_projection | 18.925 | 129.972 | 2032 | 0 | 2032 | 0/0 | 0 |
| v_projection | 17.626 | 17.969 | 2032 | 0 | 2032 | 0/0 | 0 |
| weighted_v | 483.634 | 353.178 | 2032 | 0 | 65024 | 0/0 | 0 |
| workspace_free | 12.630 | 0.000 | 1 | 0 | 0 | 0/11 | 0 |
## after: batch, batch 1

Actual output tokens: 128; profiled full-call wall: 1865.038 ms (perturbed). Tokens match warmup. All phases remain in raw JSON; the table focuses on decode.

| Operation | Host API/CPU wall ms | GPU stream elapsed ms | Calls | Explicit kernels | cuBLAS calls | Alloc/free calls | Tracked selection-copy bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| attention_scores | 296.634 | 173.624 | 2032 | 0 | 65024 | 0/0 | 0 |
| buffer_free | 0.092 | 0.000 | 3 | 0 | 0 | 0/3 | 0 |
| down_projection | 20.699 | 140.311 | 2032 | 0 | 2032 | 0/0 | 0 |
| embedding | 0.788 | 0.732 | 127 | 127 | 0 | 0/0 | 0 |
| gate_projection | 17.658 | 131.783 | 2032 | 0 | 2032 | 0/0 | 0 |
| gpu_greedy_reduction | 1.400 | 1.407 | 127 | 254 | 0 | 0/0 | 0 |
| k_projection | 15.783 | 17.882 | 2032 | 0 | 2032 | 0/0 | 0 |
| kv_free | 0.724 | 0.000 | 1 | 0 | 0 | 0/2 | 0 |
| kv_gather | 21.372 | 19.697 | 4064 | 4064 | 0 | 0/0 | 0 |
| kv_store | 10.728 | 9.158 | 2032 | 2032 | 0 | 0/0 | 0 |
| lm_head | 1.809 | 118.603 | 127 | 0 | 127 | 0/0 | 0 |
| metadata_allocate_upload | 2.496 | 0.000 | 127 | 0 | 0 | 508/0 | 0 |
| metadata_free | 2.064 | 0.000 | 127 | 0 | 0 | 0/508 | 0 |
| o_projection | 15.915 | 42.809 | 2032 | 0 | 2032 | 0/0 | 0 |
| profiler_readiness_wait | 572.970 | 0.000 | 381 | 0 | 0 | 0/0 | 0 |
| q_projection | 16.510 | 43.222 | 2032 | 0 | 2032 | 0/0 | 0 |
| residual | 21.215 | 18.575 | 4064 | 4064 | 0 | 0/0 | 0 |
| rmsnorm | 21.673 | 28.750 | 4191 | 4191 | 0 | 0/0 | 0 |
| rope | 21.282 | 19.683 | 4064 | 4064 | 0 | 0/0 | 0 |
| softmax | 334.016 | 407.189 | 65024 | 65024 | 0 | 0/0 | 0 |
| swiglu | 10.607 | 15.406 | 2032 | 2032 | 0 | 0/0 | 0 |
| token_d2h | 2.331 | 2.027 | 127 | 0 | 0 | 0/0 | 508 |
| up_projection | 16.839 | 129.995 | 2032 | 0 | 2032 | 0/0 | 0 |
| v_projection | 15.653 | 17.989 | 2032 | 0 | 2032 | 0/0 | 0 |
| weighted_v | 365.123 | 249.567 | 2032 | 0 | 65024 | 0/0 | 0 |
| workspace_free | 0.671 | 0.000 | 2 | 0 | 0 | 0/22 | 0 |
## after: batch, batch 4

Actual output tokens: 512; profiled full-call wall: 7085.117 ms (perturbed). Tokens match warmup. All phases remain in raw JSON; the table focuses on decode.

| Operation | Host API/CPU wall ms | GPU stream elapsed ms | Calls | Explicit kernels | cuBLAS calls | Alloc/free calls | Tracked selection-copy bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| attention_scores | 1431.957 | 698.727 | 8128 | 0 | 260096 | 0/0 | 0 |
| buffer_free | 0.182 | 0.000 | 3 | 0 | 0 | 0/3 | 0 |
| down_projection | 84.225 | 559.745 | 8128 | 0 | 8128 | 0/0 | 0 |
| embedding | 1.265 | 1.093 | 127 | 127 | 0 | 0/0 | 0 |
| gate_projection | 72.212 | 525.539 | 8128 | 0 | 8128 | 0/0 | 0 |
| gpu_greedy_reduction | 5.238 | 5.925 | 508 | 1016 | 0 | 0/0 | 0 |
| k_projection | 64.756 | 59.550 | 8128 | 0 | 8128 | 0/0 | 0 |
| kv_free | 1.256 | 0.000 | 1 | 0 | 0 | 0/2 | 0 |
| kv_gather | 26.154 | 27.840 | 4064 | 4064 | 0 | 0/0 | 0 |
| kv_store | 11.215 | 9.105 | 2032 | 2032 | 0 | 0/0 | 0 |
| lm_head | 5.786 | 474.543 | 508 | 0 | 508 | 0/0 | 0 |
| metadata_allocate_upload | 68.922 | 0.000 | 127 | 0 | 0 | 508/0 | 0 |
| metadata_free | 64.375 | 0.000 | 127 | 0 | 0 | 0/508 | 0 |
| o_projection | 65.597 | 172.154 | 8128 | 0 | 8128 | 0/0 | 0 |
| profiler_readiness_wait | 1003.127 | 0.000 | 1143 | 0 | 0 | 0/0 | 0 |
| q_projection | 67.084 | 173.223 | 8128 | 0 | 8128 | 0/0 | 0 |
| residual | 22.125 | 18.801 | 4064 | 4064 | 0 | 0/0 | 0 |
| rmsnorm | 23.164 | 29.478 | 4191 | 4191 | 0 | 0/0 | 0 |
| rope | 22.011 | 19.276 | 4064 | 4064 | 0 | 0/0 | 0 |
| softmax | 2157.256 | 1632.182 | 260096 | 260096 | 0 | 0/0 | 0 |
| swiglu | 11.111 | 16.208 | 2032 | 2032 | 0 | 0/0 | 0 |
| token_d2h | 7.768 | 6.514 | 508 | 0 | 0 | 0/0 | 2032 |
| up_projection | 69.364 | 518.430 | 8128 | 0 | 8128 | 0/0 | 0 |
| v_projection | 64.083 | 59.369 | 8128 | 0 | 8128 | 0/0 | 0 |
| weighted_v | 1512.702 | 997.595 | 8128 | 0 | 260096 | 0/0 | 0 |
| workspace_free | 2.321 | 0.000 | 2 | 0 | 0 | 0/22 | 0 |
