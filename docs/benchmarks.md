# Benchmark methodology and evidence

## Experimental stages

| Stage | Saved workload and repetitions | Purpose |
|---|---|---|
| Initial contiguous | P128/256/512 × O64/128/256, five repetitions per case; five additional P256/O128 smoke repetitions | Original contiguous baseline; paged/batch failures explicitly excluded |
| Numerical compatibility fixes | P256/O128 contiguous regression, five repetitions; P128/O128 static batches 1/2/4, five each | Verify correctness before batching measurements |
| GPU greedy selection | P256/O128 contiguous, ten CPU and ten GPU repetitions; P128/O128 batches 1/2/4, five CPU and five GPU each; late CPU control, five | Isolate the measured CPU-to-GPU selection improvement |

Every performance configuration uses two warmups, the same BF16 checkpoint and checked-in prompt IDs. No batch-eight performance is claimed. The counter-permission probe is preserved but excluded from measurements. Exploratory Hugging Face records are retained separately; they do not support a cross-framework speedup claim.

Original JSONL records are copied byte-for-byte under [measurements](../benchmarks/results/measurements/). [Their manifest](../benchmarks/results/provenance/measurements.json) records original paths and SHA-256 hashes. All 11 files total 557,769 bytes. Summaries retain medians, means, sample standard deviations and counts; source/metadata paths inside preserved statistics describe the historical experiment layout. Current navigation and commands use the paths in this document.

The initial P256/O128 pooled ten-run baseline achieved approximately 96.905 decode tok/s, 16.555 ms TTFT and 98.395 uninstrumented E2E tok/s. The later matched CPU-versus-GPU experiment is the headline comparison; it must not be mixed with that earlier baseline. The post-correctness-fix contiguous regression measured approximately 95.947 decode tok/s and 97.483 uninstrumented E2E tok/s, about 0.9% below the initial pooled E2E baseline.

## Hardware, checkpoint and build

- Single NVIDIA A40, nominal 48 GB; recorded `nvidia-smi` total 46,068 MiB. CUDA-reported usable total in run metadata can differ from the management tool total.
- Driver 580.178.04; CUDA toolkit 12.8.93; GNU C++ 13.3.0; Release build, CUDA architecture 86, C++17/CUDA17. No fast-math flag.
- BF16 Llama 3.2 1B Instruct, tied embedding/LM head; checkpoint SHA-256 `1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f`.
- Prompt IDs in [prompts.json](../benchmarks/prompts.json) are authoritative. P denotes prompt tokens per request; O denotes requested output tokens per request. Sampling is disabled; seed metadata does not imply a randomized sampling workload.
- Parent Git commit for measured dirty working trees: `1586d95eaa76354d1dcd0d000d50f368b0a60d3b`. This commit alone does **not** identify either measured implementation. [Revision manifest](../benchmarks/results/provenance/revisions.json) supplies exact hashed build-input trees and reconstruction patches. [Reconstruction instructions](../benchmarks/README.md#historical-source-reconstruction) identify the CPU/GPU comparison sources.

Historical CPU, GPU, CPU operation-profile, and post-fix build inputs are reconstructable. The earliest experiment's exact `CMakeLists.txt` and `src/model/paged_transformer.cu` were not recovered with their recorded hashes; its complete exact-source reconstruction remains unverified. Its raw measurements, configs and recorded source hashes remain available. No replacement content is presented as an exact historical file.

## Harness and warmups

[inference_benchmark](../benchmarks/benchmark.cu) loads the checkpoint once outside timed calls. For each configuration it checks instrumented/uninstrumented generated-token agreement, performs two complete warmup calls, then executes the configured repetitions. Each measured repetition includes an observed call and a paired call without timing/memory observers, with token equality required. Thus five measured repetitions involve more than five generation calls; paired calls are not independent additional repetitions.

Recorded metadata captures GPU, build, precision, configuration, model allocation and runtime versions. EOS stopping is preserved. Complete fixed-length statistics exclude failures and early termination, while raw records retain statuses and actual outputs. Count actual generated tokens, never prompt tokens or requested tokens that were not generated.

## Timing and synchronization

| Metric | Definition |
|---|---|
| TTFT wall | Generation entry to first selected token, including allocation, prompt upload, prefill and selection |
| Decode wall throughput | `(actual generated tokens − 1) / (last token timestamp − first token timestamp)` for a single request |
| E2E output throughput | Actual generated tokens across requests divided by full generation-call wall time |
| E2E wall | Generation entry through cleanup; includes prefill, scheduling, selection and decode |
| Request ITL | Difference between consecutive instrumented token timestamps for that request |
| CUDA-event interval | Elapsed GPU-stream interval around the observed phase; not summed active-kernel duration |

Model loading, compilation, tokenization and text rendering are outside benchmark generation timing. First-token selection belongs to prefill; O128 has 127 decode selections. Synchronous token-ID copies and phase/observer synchronization establish completion dependencies. CUDA events close phase work before selection; stream intervals can include copies, host submission gaps and metadata work. Event-based rates must not be interpreted as full request throughput.

Headline decode and TTFT values use **instrumented wall timing**. Headline E2E throughput/latency use **paired uninstrumented wall timing**. There is no probe-free per-token latency measurement; per-request TTFT and ITL below are instrumented and can be perturbed. Operation-profile timing is a separate diagnostic run, never the source of headline speedups.

## Verified CPU-to-GPU comparison

Medians recomputed from ten runs per stage:

| Metric | CPU selector | GPU selector | Change |
|---|---:|---:|---:|
| Instrumented decode tok/s | 95.238 | 105.420 | +10.69% |
| Instrumented TTFT ms | 16.992 | 15.739 | −7.38% |
| Uninstrumented E2E tok/s | 97.394 | 107.600 | +10.48% |
| Uninstrumented E2E ms | 1314.251 | 1189.587 | −9.49% |

The throughput ratio is the ratio of medians, not the median of paired ratios. Saved statistics use 10,000 independent bootstrap resamples with seed 20261015. The E2E ratio's 95% interval is approximately `[1.0878, 1.1539]`. This describes within-measurement-block variation, not all clock, temperature, ordering or system drift. A later five-run CPU control measured 97.700 E2E tok/s, versus 107.600 optimized (ratio 1.1013). Before/after collection was not randomized or interleaved; no causal certainty beyond this controlled workload is implied.

The [full summary](../benchmarks/results/optimization_summary.json) preserves dispersion and outliers, rather than selecting only fast runs. The [verification script](../benchmarks/scripts/release_checks.py) recomputes every retained configuration's reported statistics and checks raw before/after/control token equality.

## Static batching latency

Fixed simultaneous submissions; P128/O128 per request, B1/2/4, two warmups and five repetitions each. Prefill is serial, then active requests advance together. Aggregate throughput counts B × 128 actual tokens and divides by the full uninstrumented call. No dynamic arrivals or network server are involved.

| B | E2E aggregate tok/s | E2E completion ms | Scaling vs B1 | Mean/p95 TTFT ms | Mean/p95 ITL ms | Observed peak MiB | Successful/failed requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 92.600 | 1382.288 | 1.000× | 27.450 / 82.541 | 11.059 / 12.699 | 2686.25 | 5 / 0 |
| 2 | 95.418 | 2682.920 | 1.030× | 18.294 / 27.954 | 21.519 / 23.488 | 2720.25 | 10 / 0 |
| 4 | 94.316 | 5428.552 | 1.019× | 30.021 / 45.224 | 42.910 / 49.469 | 2786.25 | 20 / 0 |

Throughput/completion values are medians of uninstrumented calls. Latency means and p95 pool instrumented observations over all requests/repetitions; p95 uses nearest rank. B1 TTFT includes a slow observed prefill and is not evidence that B2 generally lowers TTFT. ITL rises approximately 1.95× at B2 and 3.88× at B4; full-call completion grows approximately 1.94× and 3.93×. Larger batches yield modest aggregate throughput gains at a per-request latency cost, not linear scaling.

Device usage comes from `cudaMemGetInfo` at observed phase and live-workspace boundaries. It is device-wide usage and an **observed lower bound on peak**, not a continuous high-water mark or weight-only allocation. No allocator equivalence with another framework is assumed.

## Profiling scope and limitations

Lightweight source instrumentation measured host CPU/API wall time, CUDA-stream elapsed intervals, explicit kernel-launch counts, cuBLAS API calls, allocation/free counts and tracked selection-copy bytes. Host and GPU intervals overlap; do not add them. cuBLAS calls do not reveal physical kernel counts. CUDA event pools and explicit readiness waits perturb execution; profiler wall totals are diagnostic only.

The pre-optimization contiguous decode profile recorded 107.172 ms of CPU greedy scanning across 127 selections. Selection transfers totaled 65,154,048 bytes (127 × 513,024). Optimized decode returns 508 bytes (127 × 4), with two reduction launches per selection. The 128,256× payload ratio is not a speedup claim.

Nsight Systems was absent. Nsight Compute 2025.1.1 was installed, but the attempted hardware-counter capture failed with `ERR_NVGPUCTRPERM`. Occupancy, active-kernel time, hardware bandwidth and counter-derived bottlenecks were unavailable. [Operation tables](profile.md) and [raw operation records](../benchmarks/results/profiling/) retain the available evidence.

## Reproduction and preservation

Follow [benchmarks/README.md](../benchmarks/README.md) for current, historical and short smoke commands. Run correctness gates before performance. The standalone diagnostic compares intermediate tensors and generated trajectories. The benchmark verifier checks counters/timestamps/formulas; release checks compare saved tables and hashes.

Original artifacts were copied into a local archive outside the repository; every copy was SHA-256 verified before removal from the working tree. The external archive includes full tensor diagnostics, binaries, source tarballs, superseded reports and logs, with original-path mapping. Checkpoints were left intact and excluded from the archive/public file set.
