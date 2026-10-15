# Correctness and numerical compatibility

## Gates and scope

| CTest gate | Main coverage |
|---|---|
| `paged_kvcache` | Page boundaries, stores, logical mapping, fragmentation, exhaustion, free/reuse |
| `paged_prefill` | Contiguous/paged prefill logits and bitwise KV agreement |
| `paged_attention` | Ragged lengths 16/17/33; scores and attention outputs versus reference |
| `paged_paged` | 13 prompts; exact generated tokens and logit MAE ≤ 0.01 |
| `paged_batch` | Single-paged versus multi-request output, mixed budgets, slot and block reuse |
| `bf16_argmax` | Five original and 111 additional edge/random-bit-pattern cases |

Saved [optimized-stage gate output](../benchmarks/results/correctness/optimized/ctest.txt) records all six passing. [Paged regression evidence](../benchmarks/results/correctness/optimized/paged_paged_report.json) records 13/13 exact trajectories with zero observed logit error. Thresholds and expected outputs were not weakened.

[Intermediate comparison summaries](../benchmarks/results/correctness/optimized/intermediate_summary.json) record 5,480 matching tensors and 704 matching cache comparisons. Teacher forcing supplies identical tokens to each path, comparing normalized inputs, Q/K/V, RoPE, scores, softmax, weighted V, output projection, residuals, MLP, final norm and logits. Cache comparisons independently gather logical positions. Large original per-tensor records are locally archived, with the original trace hash retained.

[Generation agreement](../benchmarks/results/correctness/optimized/generation-agreement.json) records exact 128-token agreement for contiguous, single paged and all seven request trajectories across batches 1/2/4. [Argmax output](../benchmarks/results/correctness/optimized/argmax.txt) records 116 cases: BF16 rounding/ties, NaN/Inf, signed zeros, subnormals, overflow and seeded random float-bit patterns. Original raw benchmark pairs additionally check instrumented/uninstrumented output equality and CPU/GPU trajectory equality.

## Historical failures and evidence

Initially, three of 13 paged prompts diverged in tokens, nine exceeded the 0.01 logit-MAE threshold, and batch request 2 diverged at generated token 8. [Before-fix reports](../benchmarks/results/correctness/numerical_before/paged_paged_report.json) preserve the failures.

The earliest contiguous/paged difference was decode step 1, layer 2, attention-score index 169:

| Value | Numerical result |
|---|---:|
| Contiguous cuBLAS | 0.00396728515625 |
| Original paged kernel | 0.003936767578125 |
| FP64 scaled dot-product oracle | 0.0039517879486083984 |

The oracle rounds to the paged value; prior Q/K/V, RoPE and cache inputs matched. This established a reduction/backend rounding difference rather than a cache mapping error. Errors propagated into later layer caches.

The first single-paged/batch difference occurred at decode step 1, layer 0, Q projection: three of 2,048 entries differed, maximum error 0.001953125, despite identical normalized inputs. At index 767, the FP64 oracle was 0.21923926201498034, rounding to the single-row value 0.2197265625; the batch-shaped GEMM produced 0.21875. The changed GEMM shape selected a numerically different computation. Disallowing reduced-precision reductions alone did not resolve both paths.

The compatibility fix gathered paged BF16 K/V and reused reference attention; batched projections use the same one-row GEMM shapes as the single-request path. The result is exact tested agreement, at a throughput cost. This is an empirical backend/reduction diagnosis supported by intermediate comparisons and FP64 oracles, not an assumption that every mismatch is harmless floating-point noise.

## Selection semantics and external reference

GPU selection rounds FP32 logits to BF16 with round-to-nearest-even before comparing; ties choose the lowest token ID. NaNs other than token zero are ignored, while token-zero NaN returns zero to match the historical CPU initialization rule. Infinity behavior is preserved.

Historical validation versus Hugging Face recorded 376/379 selections matching (99.2%) **on shared engine prefixes**. It does not establish free-running generation equivalence. The original report is retained in the local archive; [compact reference evidence](../benchmarks/results/correctness/hf_shared_prefix.json) retains the count and original report hash. No universal numerical equivalence across prompts, libraries or GPU architectures is claimed.

## Reproduction

Configure with `-DBUILD_PAGED_TESTS=ON`, build, and run `ctest --test-dir build --output-on-failure --stop-on-failure -j1` from the repository root. For layer and cross-mode diagnostics:

```bash
mkdir -p benchmarks/local
./build/numerical_diagnostic benchmarks/local/numerical.json
./build/numerical_diagnostic benchmarks/local/generation.json generation-check
./build/bf16_argmax_test
```

Checkpoint location is fixed in CTest and numerical diagnostics to `models/llama-3.2-1b-instruct/model.safetensors`. Test results generated by a new run are distinct from preserved historical evidence. See [release validation](release-validation.md) for checks performed after repository cleanup.
