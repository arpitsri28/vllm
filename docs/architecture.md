# Architecture

The implementation specializes to Llama 3.2 1B Instruct: 16 transformer layers, hidden size 2,048, intermediate size 8,192, 32 Q heads, eight KV heads, head dimension 64, and vocabulary 128,256. Model configuration is not a general runtime model loader. The CLI accepts a context limit up to 1,024.

## Loading and tokenization

[src/main.cu](../src/main.cu) reads a SafeTensors JSON header and tensor payload, allocates device weight storage, and maps named tensors into `ModelWeights`. The embedding matrix is also the LM-head matrix. Tokenization and CLI detokenization use [src/python/tokenizer.py](../src/python/tokenizer.py) and Transformers. Run the CLI from the repository root with an explicit JSON token array; the historical `src/tokens.json` is an empty placeholder.

## Transformer execution

[transformer.cu](../src/model/transformer.cu) implements contiguous prefill and decode. Each layer computes input RMSNorm, Q/K/V projections, RoPE on Q/K, cache storage, scaled grouped-query attention, output projection, an attention residual, post-attention RMSNorm, gated SwiGLU MLP and its residual. Final RMSNorm and a tied LM head produce FP32 logits. BF16 storage and rounding occur between operations; FP32 cuBLAS accumulation does not eliminate sensitivity to reduction order.

Prefill processes the prompt and selects the first generated token. Decode consumes the previous token, projects only new Q/K/V, reads prior cache entries and selects the next token. EOS stopping remains enabled. [engine.cu](../src/engine/engine.cu) owns generation workspaces and cleanup. The GPU selector preserves the original CPU scan's BF16 selection rule, including its token-zero NaN behavior.

## Cache layouts

Contiguous K and V cache pools have logical layout `[layer, position, kv_dimension]`, where `kv_dimension = 8 × 64 = 512`. Paged pools use `[layer, physical_block, token_in_block, kv_dimension]`, with 16 tokens per block. K and V are separate BF16 allocations. A request's host block table maps `logical_position / 16` to a physical block; the offset within that block is `logical_position % 16`.

[The allocator](../src/model/paged_kvcache.cu) tracks free physical blocks and releases a request's blocks on retirement. Tests cover boundaries, fragmentation, exhaustion and reuse. There is no prefix sharing, copy-on-write or cache offload.

## Paged compatibility path

During decode, [paged_attention.cu](../src/model/paged_attention.cu) uploads positions and block tables and gathers BF16 cache entries into contiguous temporary K/V views. It calls the same cuBLAS score and weighted-V helpers as contiguous attention, with explicit per-head score strides for ragged rows. This preserves the tested numerical trajectory at a gathering/allocation cost; it is not fused direct page-aware attention.

## Static scheduler

[scheduler.cu](../src/engine/scheduler.cu) receives a fixed vector of requests, admits up to four active slots, performs serial prefills, advances active requests through a shared decode-layer loop and reuses retired slots. Elementwise operations and cache writes can operate on multiple rows. Decode projections and LM-head evaluation use one-row GEMMs to reproduce single-request BF16 results.

This establishes queue retirement/reuse and static concurrency. There is no network ingress, external arrival stream, independently verified dynamic serving loop, multi-GPU execution or production isolation guarantee.

## Instrumentation

[benchmark_trace.cuh](../include/benchmark_trace.cuh) supports opt-in wall timestamps, CUDA events and sampled device usage. [numerical_trace.cuh](../include/numerical_trace.cuh) and [operation_profile.cuh](../include/operation_profile.cuh) enable dedicated diagnostic targets. Operation profiling compiles away in ordinary targets. Instrumented and uninstrumented calls are measured separately; see [methodology](benchmarks.md).
