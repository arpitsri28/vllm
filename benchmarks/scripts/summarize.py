"""Standard-library-only summaries. Never includes incomplete runs in fixed-length statistics."""
import argparse
import collections
import json
import statistics
from pathlib import Path

METRICS = ['prefill_wall_ms','prefill_gpu_stream_ms','ttft_wall_ms','decode_wall_ms',
           'decode_gpu_stream_ms','decode_wall_tokens_per_s','decode_gpu_stream_tokens_per_s',
           'e2e_wall_ms','e2e_output_tokens_per_s','mean_ttft_wall_ms','p95_ttft_wall_ms',
           'mean_itl_wall_ms','p95_itl_wall_ms','model_loaded_device_used_mib',
           'prefill_observed_peak_device_used_mib','decode_observed_peak_device_used_mib',
           'observed_peak_device_used_mib','uninstrumented_e2e_wall_ms','uninstrumented_output_tokens_per_s']

def summarize(paths):
    groups = collections.defaultdict(list)
    metadata, exclusions = [], []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            if r['type'] == 'metadata': metadata.append(r)
            elif r['type'] == 'run':
                key = (r['mode'], r['prompt_tokens'], r['output_tokens'], r['batch_size'])
                groups[key].append(r)
            elif r['type'] in ('unsupported','error'): exclusions.append(r)
    summary = []
    for key, rows in sorted(groups.items()):
        valid = [r for r in rows if r['status'] == 'ok']
        stats = {}
        for field in METRICS:
            values = [r[field] for r in valid if field in r]
            if values:
                stats[field] = dict(median=statistics.median(values), mean=statistics.mean(values),
                                    stdev=statistics.stdev(values) if len(values)>1 else None, n=len(values))
        summary.append(dict(mode=key[0], prompt_tokens=key[1], output_tokens=key[2], batch_size=key[3],
                            successful_repetitions=len(valid), attempted_repetitions=len(rows),
                            incomplete_repetitions=sum(r['status']=='early_termination' for r in rows),
                            errors=[r.get('error') for r in rows if r['status']=='error'], statistics=stats))
    return dict(metadata=metadata, configurations=summary, exclusions=exclusions)

def main():
    p = argparse.ArgumentParser(); p.add_argument('inputs', nargs='+'); p.add_argument('--output', default='benchmarks/local/summary')
    a=p.parse_args(); result=summarize(a.inputs)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Measured benchmark results','',
           'All table statistics are medians of complete fixed-output runs. Full mean, sample standard deviation and counts are in `summary.json`. Latencies include instrumentation unless explicitly labeled uninstrumented.','',
           '| Mode | Prompt | Output/request | Batch | Successful/attempted | Prefill wall ms | TTFT wall ms | Decode wall tok/s | E2E wall ms | Output tok/s | Observed device peak MiB |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in result['configurations']:
        s=c['statistics']
        def val(f): return f"{s[f]['median']:.3f}" if f in s else '—'
        lines.append(f"| {c['mode']} | {c['prompt_tokens']} | {c['output_tokens']} | {c['batch_size']} | {c['successful_repetitions']}/{c['attempted_repetitions']} | {val('prefill_wall_ms')} | {val('ttft_wall_ms' if c['mode']!='batch' else 'mean_ttft_wall_ms')} | {val('decode_wall_tokens_per_s')} | {val('e2e_wall_ms')} | {val('e2e_output_tokens_per_s')} | {val('observed_peak_device_used_mib')} |")
    lines+=['','Batch TTFT in this table is the per-run mean across requests. Per-run p95 TTFT and pooled per-request token-gap p95 ITL are preserved in raw data and summarized in JSON.','',
            'CUDA event values are GPU stream elapsed intervals, not summed kernel-active execution time. They exclude host argmax and detokenization but include submission gaps, copies and metadata management inside each interval.','',
            'Unsupported cases and errors:','']
    for r in result['exclusions']: lines.append('- '+json.dumps(r))
    (out/'TABLE.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
