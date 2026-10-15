# Benchmark reproduction

Run from the repository root. Build commands and authorized checkpoint placement are in the [main README](../README.md#build-and-run). The harness takes exactly three arguments: checkpoint, configuration JSON, output JSONL. Preserved measurements live under `results/measurements/`; write new output to ignored `benchmarks/local/`.

## Current implementation

```bash
ctest --test-dir build --output-on-failure --stop-on-failure -j1
mkdir -p benchmarks/local
./build/inference_benchmark models/llama-3.2-1b-instruct/model.safetensors \
  benchmarks/configs/optimization_primary.json benchmarks/local/primary.jsonl
./build/inference_benchmark models/llama-3.2-1b-instruct/model.safetensors \
  benchmarks/configs/batching.json benchmarks/local/batching.jsonl
python3 benchmarks/scripts/verify_results.py \
  benchmarks/local/primary.jsonl benchmarks/local/batching.jsonl
python3 benchmarks/scripts/summarize.py benchmarks/local/primary.jsonl \
  benchmarks/local/batching.jsonl --output benchmarks/local/summary
```

The primary configuration uses two warmups and ten repetitions; batching uses two warmups and five repetitions at each of B1/2/4, P128/O128. For a short workflow check use `configs/quick.json` (two warmups, one measured repetition). A smoke run does not reproduce a ten-run performance claim.

`configs/full.json` and `configs/smoke.json` preserve the **initial** experiment's gate exclusions; they intentionally retain historical skip reasons. `configs/contiguous_regression.json`, `optimization_control.json` and `optimization_profile.json` preserve later experiment settings. The profile configuration is a one-call counter probe, not a performance sample.

## Verify saved evidence

```bash
python3 benchmarks/scripts/release_checks.py
python3 benchmarks/scripts/summarize.py \
  benchmarks/results/measurements/optimization/after/primary.jsonl \
  benchmarks/results/measurements/optimization/after/batching.jsonl \
  --output benchmarks/local/saved-summary
```

`release_checks.py` verifies original JSONL hashes, statistics for all three stages, CPU/GPU/control tokens, profiling counts, local Markdown links, documentation path hygiene and the data-derived SVG. It does not rerun GPU correctness tests. See [methodology](../docs/benchmarks.md) and [correctness](../docs/correctness.md).

## Historical source reconstruction

Measured source trees were dirty working trees based on parent commit `1586d95eaa76354d1dcd0d000d50f368b0a60d3b`; neither result should be attributed to that commit alone. [revisions.json](results/provenance/revisions.json) identifies exact build-input SHA-256 hashes and a content-derived tree identifier per stage:

- `cpu`: source snapshot used by the saved CPU-argmax performance executable; headline **before** and late CPU control.
- `gpu`: recorded optimized source hashes; headline **after** and optimized static batches.
- `cpu-profile`: CPU implementation plus operation timing hooks, used for profiling only.
- `post-fix`: recorded build inputs after the paged/batch compatibility fix, before GPU selection optimization.

Exact comparison build-input identifiers (SHA-256 over the canonical path/hash map):

| Measured implementation | Build-input tree SHA-256 |
|---|---|
| cpu | `c22a0ed26f82227d97e6d8cb17917d247ad526628defba11e6138fc38c6b5a45` |
| gpu | `17bba865669889d6a0c90008d9ad6c9497d925f721ec5ce1f971cc61d7ae545c` |

Patches transform this release's source inputs into those historical trees, including test/CMake versions; no compiled executable is distributed. Install `patch` if needed. Reconstruction refuses an existing destination and verifies every historical input hash. These commands build the baseline in a separate directory without altering the current checkout:

```bash
mkdir -p benchmarks/local
python3 benchmarks/scripts/reconstruct.py --stage cpu --output benchmarks/local/cpu
# Run these next commands from the reconstructed root.
cd benchmarks/local/cpu
mkdir -p models
ln -s ../../../../models/llama-3.2-1b-instruct models/llama-3.2-1b-instruct
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DBUILD_PAGED_TESTS=ON -DBUILD_BENCHMARKS=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure --stop-on-failure -j1
./build/inference_benchmark models/llama-3.2-1b-instruct/model.safetensors \
  benchmarks/configs/optimization_primary.json before.jsonl
```

To reconstruct the optimized historical tree, use `--stage gpu --output benchmarks/local/gpu` from the release root and repeat the same build/run steps there. Prompt fixtures/configurations are copied from the retained release inputs. They match the recorded comparison settings. The checkpoint symlink is local convenience; model files are never copied into a public source set.

The initial experiment's source manifest is preserved, but two exact hashed inputs (`CMakeLists.txt` and `src/model/paged_transformer.cu`) were not recovered. That stage's complete exact-source rebuild remains unverified; its results/configurations must not be mislabeled as independently reconstructed. [Detailed provenance](../docs/benchmarks.md#hardware-checkpoint-and-build).

## Optional profiling and fixture regeneration

```bash
mkdir -p benchmarks/local
./build/operation_profiler benchmarks/local/operations.json
python3 benchmarks/scripts/make_config.py --output benchmarks/local/regenerated
```

Operation profiling uses two warmups and one observed generation for each diagnostic workload; its perturbed timings do not support headline throughput. Fixture regeneration writes a new directory and must be compared with checked-in `prompts.json` before reuse. Optional `scripts/hf_baseline.py` additionally needs PyTorch and NumPy; it is an exploratory eager reference, not an official framework benchmark or a speedup claim.
