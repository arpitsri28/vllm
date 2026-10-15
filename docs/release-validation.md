# Release candidate validation

Validated on 2026-10-15 using a new isolated directory containing the proposed public source files, a fresh CMake build directory, an NVIDIA A40 and a local checkpoint symlink. CUDA 12.8.93, GNU C++ 13.3.0, Release SM86, Python 3.12 and Transformers 5.19.0 were available in the existing environment.

This validates the isolated source package and documented paths with installed prerequisites. It does **not** establish fresh operating-system provisioning, a newly installed Python environment, or a Git clone of a published release commit. No release commit exists yet.

## Executed checks

| Check | Result |
|---|---|
| Clean CMake configure/build | Passed; inference CLI, all test/diagnostic targets, benchmark and operation profiler built |
| `paged_kvcache`, `paged_prefill`, `paged_attention`, `paged_paged`, `paged_batch`, `bf16_argmax` | 6/6 passed; 21.21 seconds total |
| Tokenizer and CLI detokenization | Passed with local checkpoint tokenizer |
| Contiguous/paged CLI smoke | Exact matching 32-token outputs |
| Four-slot CLI with six queued requests | All six completed at their respective output budgets |
| Intermediate diagnostics | 5,480 tensors and 704 cache comparisons matched |
| P128/O128 cross-mode generation diagnostic | Contiguous, paged and all requests in batches 1/2/4 matched |
| GPU quick benchmark | Complete actual 128-token run; counters/timestamps/formulas verified; summarizer ran |
| Historical CPU baseline | Reconstructed, source hashes verified, benchmark target compiled, quick benchmark completed |
| Reconstructed CPU versus GPU quick benchmark | Exact generated-token agreement |
| Fixture regeneration | Byte-identical checked-in prompt fixtures |
| Historical reconstruction patches | CPU, GPU, CPU-profile and post-fix inputs matched retained hashes |
| Preserved evidence | 11 JSONL files, 557,769 bytes; hashes verified; 125 performance runs valid |
| Published statistics | All three stages' retained statistics and optimization bootstrap intervals recomputed |
| Documentation | Relative links and headline numbers checked; Mermaid flowchart parsed with Mermaid 11.12.0; SVG verified against saved data |
| Git hygiene | `git diff --check` passed; proposed public files checked for credentials and excluded artifacts |

[Machine-readable validation](../benchmarks/results/correctness/release/validation.json) records build-input hashes. [CTest output](../benchmarks/results/correctness/release/ctest.txt) retains the six gate results. Quick smoke measurements are local validation outputs, not new performance claims. The operation profiler was compiled; its historical operation measurements were preserved, rather than spending GPU time repeating the full profile.

## Preservation and security

The original local archive contains 185 copied files totaling 16,316,328 bytes, plus a working-tree patch and a manifest with original relative paths and SHA-256 hashes. Every repository copy removed during cleanup matched its verified archive copy. Model checkpoints were not moved or deleted. The local checkpoint hash matches the recorded experiment hash.

The public candidate retains source, tests, benchmark/profiling executables' source, original configurations, prompt IDs, selected raw measurements, original summary statistics, compact correctness evidence, sanitized environment/source provenance, and historical reconstruction patches. Bulky numerical dumps, executable binaries, source tarballs, redundant reports and temporary logs remain in the external archive.

Credential-pattern checks of current public candidates and reachable Git history found no matches for inspected private-key and API-token formats. The branch history contains no model checkpoint or file above 1 MB. No history was rewritten. These checks are not a guarantee that every possible credential format was detected. The public file list contains no binary, model, environment file or file above 1 MB. Ignored files are excluded from the proposed staging list.

## Validation limitations

Licensing remains unresolved; no `LICENSE` was created. The earliest experiment's exact source reconstruction is incomplete for two recorded inputs, as documented in [methodology](benchmarks.md). The matched headline CPU/GPU comparison has complete hashed build-input reconstruction, and the CPU baseline was rebuilt and run during validation.

Nsight hardware-counter profiling remains unavailable due to `ERR_NVGPUCTRPERM`. Static batches do not establish dynamic serving. Model-specific, context, concurrency and hardware validation limits are unchanged. The README retains 10.69% decode and 10.48% E2E improvements from the original measured comparison; no smoke run replaces those statistics.
