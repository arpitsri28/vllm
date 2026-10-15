"""Check real benchmark records for counter, timestamp and throughput consistency."""
import json
import math
import sys
from pathlib import Path

checked = incomplete = 0
for path in sys.argv[1:]:
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if r['type'] != 'run' or r['status'] == 'error': continue
        tokens = r['generated_tokens']
        assert r['actual_output_tokens'] == sum(map(len,tokens))
        assert r['completed_requests'] == sum(len(t)==r['output_tokens'] for t in tokens)
        assert (r['status']=='ok') == (r['completed_requests']==r['batch_size'])
        assert r['e2e_wall_ms']>0
        assert math.isclose(r['e2e_output_tokens_per_s'],r['actual_output_tokens']*1000/r['e2e_wall_ms'])
        for key, stamps in r['token_timestamps_wall_ms'].items():
            assert len(stamps)==len(tokens[int(key)])
            assert all(0<t<=r['e2e_wall_ms'] for t in stamps)
            assert all(a<b for a,b in zip(stamps,stamps[1:]))
        if r['mode']!='batch':
            assert r['decode_output_tokens']==r['actual_output_tokens']-1
            stamps=r['token_timestamps_wall_ms']['0']
            assert math.isclose(r['decode_wall_ms'],stamps[-1]-stamps[0])
            if r['decode_wall_ms']:
                assert math.isclose(r['decode_wall_tokens_per_s'],r['decode_output_tokens']*1000/r['decode_wall_ms'])
        for metric, value in r.items():
            if isinstance(value,(int,float)): assert math.isfinite(value), metric
        incomplete += r['status']!='ok'
        checked += 1
assert checked, 'No measured runs found'
print(json.dumps(dict(checked_runs=checked,incomplete_runs=incomplete,counters_and_timing_consistent=True),indent=2))
