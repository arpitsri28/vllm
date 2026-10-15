"""Verify published evidence and relative documentation links without modifying inputs."""
import argparse
import hashlib
import json
import math
import re
import random
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT/'benchmarks/results'
NONPORTABLE_TEXT = re.compile(r'/(?:Users|workspace|home)/|(?:root|ubuntu)@|(?:\d{1,3}\.){3}\d{1,3}', re.I)


def read(path):
    return json.loads(path.read_text())


def runs(path):
    return [r for line in path.read_text().splitlines()
            if (r := json.loads(line)).get('type') == 'run']


def key(row):
    return tuple(row[k] for k in ('mode','prompt_tokens','output_tokens','batch_size'))


def verify_statistics(saved, rows):
    for field, expected in saved.items():
        values = [r[field] for r in rows]
        actual = dict(n=len(values),median=statistics.median(values),mean=statistics.mean(values),
                      stdev=statistics.stdev(values) if len(values)>1 else None)
        for metric, value in expected.items():
            if value is None:
                assert actual[metric] is None
            else:
                assert math.isclose(value,actual[metric],rel_tol=1e-12,abs_tol=1e-12), (field,metric)


def ratio_ci(before, after):
    rng=random.Random(20261015)
    values=[]
    for _ in range(10000):
        b=statistics.median(rng.choices(before,k=len(before)))
        a=statistics.median(rng.choices(after,k=len(after)))
        values.append(a/b)
    values.sort()
    return [values[249],values[9749]]


def chart_text(primary):
    rows=[('Decode (instrumented)', 'decode_wall_tokens_per_s'),
          ('End-to-end (uninstrumented)', 'uninstrumented_output_tokens_per_s')]
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="700" height="240" viewBox="0 0 700 240" role="img" aria-labelledby="title desc">',
           '<title id="title">CPU and GPU argmax throughput on A40</title>',
           '<desc id="desc">Median tokens per second; 256 prompt and 128 output tokens, ten repetitions per implementation.</desc>',
           '<rect width="700" height="240" fill="#ffffff"/>',
           '<g font-family="sans-serif" font-size="14" fill="#172033">',
           '<text x="20" y="25">Median generated tokens/s · A40 · P256/O128</text>',
           '<rect x="20" y="44" width="14" height="14" fill="#64748b"/><text x="42" y="56">CPU argmax</text>',
           '<rect x="170" y="44" width="14" height="14" fill="#2563eb"/><text x="192" y="56">GPU argmax</text>']
    for i,(label,metric) in enumerate(rows):
        y=88+i*76
        parts.append(f'<text x="20" y="{y}">{label}</text>')
        for j,(stage,color) in enumerate([('before','#64748b'),('after','#2563eb')]):
            value=primary['statistics'][stage][metric]['median']; width=value*3.2
            parts.extend([f'<rect x="250" y="{y-15+j*25}" width="{width:.3f}" height="19" fill="{color}"/>',
                          f'<text x="{260+width:.3f}" y="{y+j*25}">{value:.3f}</text>'])
    parts.extend(['</g>','</svg>'])
    return '\n'.join(parts)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-chart',action='store_true',help='Regenerate only the saved-data SVG')
    args=parser.parse_args()
    manifest=read(RESULTS/'provenance/measurements.json')
    checked=[]
    for entry in manifest['files']:
        p=ROOT/entry['path']
        assert p.stat().st_size == entry['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == entry['sha256'],p
        if entry['role']=='measured benchmark':checked.append(p)
    subprocess.run([sys.executable,str(ROOT/'benchmarks/scripts/verify_results.py'),*map(str,checked)],check=True)
    for name,paths in [('contiguous',['contiguous/smoke.jsonl','contiguous/full.jsonl']),
                       ('batching',['numerical/batching.jsonl','numerical/contiguous_regression.jsonl'])]:
        rows=[r for p in paths for r in runs(RESULTS/'measurements'/p) if r['status']=='ok']
        for c in read(RESULTS/(name+'_summary.json'))['configurations']:
            group=[r for r in rows if key(r)==key(c)]
            verify_statistics(c['statistics'],group)
    summary=read(RESULTS/'optimization_summary.json')
    groups={stage:runs(RESULTS/'measurements/optimization'/stage/'primary.jsonl')+
                  runs(RESULTS/'measurements/optimization'/stage/'batching.jsonl') for stage in ['before','after']}
    for config in summary['configurations']:
        token_reference=None
        for stage in ['before','after']:
            rows=[r for r in groups[stage] if key(r)==key(config)]
            assert all(r['status']=='ok' for r in rows)
            verify_statistics(config['statistics'][stage],rows)
            for row in rows:
                if token_reference is None:token_reference=row['generated_tokens']
                assert row['generated_tokens']==token_reference
            stamps=[t for row in rows for t in row['token_timestamps_wall_ms'].values()]
            values={'ttft':[t[0] for t in stamps],'itl':[b-a for t in stamps for a,b in zip(t,t[1:])]}
            for metric,observations in values.items():
                expected=config['latency'][stage][metric]
                assert math.isclose(statistics.mean(observations),expected['mean'],rel_tol=1e-12)
                assert math.isclose(sorted(observations)[math.ceil(.95*len(observations))-1],expected['p95'],rel_tol=1e-12)
        before=config['statistics']['before']['uninstrumented_output_tokens_per_s']['median']
        after=config['statistics']['after']['uninstrumented_output_tokens_per_s']['median']
        assert math.isclose(after/before,config['throughput_ratio'],rel_tol=1e-12)
        ci=ratio_ci(*[[r['uninstrumented_output_tokens_per_s'] for r in groups[stage] if key(r)==key(config)] for stage in ['before','after']])
        assert ci==config['bootstrap_95pct_ratio_interval']
    # Confirm the selector work/payload reported in the case study.
    profiles={stage:read(RESULTS/'profiling'/(stage+'.json')) for stage in ['before','after']}
    for stage,workloads in profiles.items():
        for workload in workloads:
            count=workload['actual_output_tokens']-workload['batch']
            ops=workload['operations']
            transfer=ops['decode.logits_d2h' if stage=='before' else 'decode.token_d2h']
            assert transfer['calls']==count
            assert transfer['tracked_copy_bytes']==count*(128256*4 if stage=='before' else 4)
            selector=ops['decode.cpu_greedy_scan' if stage=='before' else 'decode.gpu_greedy_reduction']
            assert selector['calls']==count
            assert selector['explicit_kernel_launches']==(0 if stage=='before' else count*2)
    cpu_profile=next(w for w in profiles['before'] if w['mode']=='contiguous')
    assert f"{cpu_profile['operations']['decode.cpu_greedy_scan']['host_api_wall_ms']:.1f}"=='107.2'
    primary=next(c for c in summary['configurations'] if c['mode']=='contiguous')
    control=runs(RESULTS/'measurements/optimization/control_primary.jsonl')
    verify_statistics(summary['control_statistics'],control)
    ref=groups['after'][0]['generated_tokens']
    assert all(r['generated_tokens']==ref for r in control)
    chart=ROOT/'docs/assets/argmax-throughput.svg'
    if args.write_chart:
        chart.parent.mkdir(parents=True,exist_ok=True);chart.write_text(chart_text(primary))
    assert chart.read_text()==chart_text(primary)
    for stage in ['numerical_after','optimized']:
        c=read(RESULTS/'correctness'/stage/'intermediate_summary.json')
        assert (c['tensor_comparisons'],c['cache_comparisons'],c['tensor_mismatches'],c['cache_mismatches'])==(5480,704,0,0)
    # Check actual numbers displayed in the headline and batching tables.
    readme=(ROOT/'README.md').read_text()
    for c in summary['configurations']:
        if c['mode']=='contiguous':
            for field in ['decode_wall_tokens_per_s','ttft_wall_ms','uninstrumented_output_tokens_per_s','uninstrumented_e2e_wall_ms']:
                for stage in ['before','after']:
                    assert f"{c['statistics'][stage][field]['median']:.3f}" in readme
        else:
            for field in ['uninstrumented_output_tokens_per_s','uninstrumented_e2e_wall_ms']:
                assert f"{c['statistics']['after'][field]['median']:.3f}" in readme
    # Check rounded changes and concurrency scaling, not just the median values.
    for field in ['decode_wall_tokens_per_s','uninstrumented_output_tokens_per_s',
                  'ttft_wall_ms','uninstrumented_e2e_wall_ms']:
        old=primary['statistics']['before'][field]['median']
        new=primary['statistics']['after'][field]['median']
        change=f'{(new/old-1)*100:+.2f}%'.replace('-', '−')
        assert change in readme,(field,change)
    batches=[c for c in summary['configurations'] if c['mode']=='batch']
    base=next(c for c in batches if c['batch_size']==1)['statistics']['after']['uninstrumented_output_tokens_per_s']['median']
    for batch in batches:
        value=batch['statistics']['after']['uninstrumented_output_tokens_per_s']['median']
        assert f'{value/base:.3f}×' in readme
    checked_links=0
    markdown_paths=[ROOT/'README.md',*sorted((ROOT/'docs').rglob('*.md')),
                    *sorted(p for p in (ROOT/'benchmarks').rglob('*.md')
                            if 'local' not in p.relative_to(ROOT).parts)]
    for p in markdown_paths:
        text=p.read_text()
        assert not NONPORTABLE_TEXT.search(text),p
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            if target.startswith(('https://','http://','mailto:')):continue
            filename,_,anchor=target.partition('#');dest=p.parent/filename if filename else p
            assert dest.exists(),(p,target)
            if anchor and dest.is_file() and dest.suffix=='.md':
                headings=re.findall(r'^#+\s+(.+)$',dest.read_text(),flags=re.M)
                slugs=[re.sub(r'[^\w\- ]','',h.lower()).replace(' ','-') for h in headings]
                assert anchor in slugs,(p,target)
            checked_links+=1
    for p in (ROOT/'benchmarks/scripts').glob('*.py'):
        if p.name!=Path(__file__).name:
            assert not NONPORTABLE_TEXT.search(p.read_text()),p
    print(json.dumps({'original_jsonl_files_verified':len(manifest['files']),
                      'original_jsonl_bytes':sum(x['bytes'] for x in manifest['files']),
                      'all_saved_statistics_verified':True,'cpu_gpu_control_tokens_equal':True,
                      'relative_links_verified':checked_links,'chart_matches_saved_data':True,'profiling_counts_and_payloads_verified':True}))


if __name__=='__main__':main()
