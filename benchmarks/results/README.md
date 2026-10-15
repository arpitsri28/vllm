# Preserved evidence

- `contiguous_summary.json`: original nine-configuration contiguous statistics, including pooled P256/O128 measurements.
- `batching_summary.json`: post-numerical-fix regression/batching statistics, prior artifact integrity findings and per-request latency.
- `optimization_summary.json`: matched CPU/GPU comparison, late CPU control, bootstrap intervals, batching scalability and pooled latency.
- `measurements/contiguous/`: original smoke/full measurements and separate exploratory Hugging Face baseline.
- `measurements/numerical/`: contiguous regression and static batching after correctness fixes.
- `measurements/optimization/{before,after}/`: CPU/GPU comparison measurements. `before/ncu-run.jsonl` is an excluded counter-permission probe, not a valid performance run. `control_primary.jsonl` is the late CPU control.
- `correctness/`: original small reports, compact intermediate-comparison summaries, token agreement, gate output and shared-prefix reference counts.
- `profiling/`: original CPU/GPU source-instrumented operation measurements; these are perturbed diagnostic runs.
- `provenance/`: source, checkpoint, environment and original artifact hashes; reconstruction identifiers and raw measurement integrity manifest.

Selected original JSONL files and the three summary files are preserved byte-for-byte. References inside historical JSON metadata use the old experiment's logical paths; current commands and evidence navigation are documented in [benchmark reproduction](../README.md). Large tensor records, compiled binaries, snapshots and redundant logs are kept in the verified external local archive and are not distributed here.

The recorded dirty working trees share a parent commit; use the source hashes and reconstruction patches to distinguish the actual measured CPU/GPU implementations. No fabricated experiment commit IDs are assigned.
