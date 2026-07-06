#!/usr/bin/env python3
# Parse M* benchmark results.json -> ttft/itl (ms), tok/s, req/s. Values in JSON
# are SECONDS, nested under [metric]['text'].
import json, sys, glob, os, statistics

def stat(d, metric, key):
    v = ((d.get(metric) or {}).get('text') or {}).get(key)
    return v

def row(path):
    d = json.load(open(path))
    b = d.get('max_concurrency') or d.get('batch_size')
    ttft = stat(d, 'ttft', 'mean'); ttft_p50 = stat(d, 'ttft', 'p50')
    itl = stat(d, 'itl', 'mean'); itl_p50 = stat(d, 'itl', 'p50')
    return {
        'B': b,
        'reqps': d.get('request_throughput') or 0.0,
        'tokps': d.get('text_token_throughput') or 0.0,
        'ttft_ms': (ttft*1000) if ttft is not None else float('nan'),
        'ttft_p50_ms': (ttft_p50*1000) if ttft_p50 is not None else float('nan'),
        'itl_ms': (itl*1000) if itl is not None else float('nan'),
        'itl_p50_ms': (itl_p50*1000) if itl_p50 is not None else float('nan'),
        'jct_ms': d.get('jct_mean_ms') or 0.0,
        'n': d.get('completed') or d.get('num_requests'),
    }

def fmt(r):
    return ("i2t B%-2s | req/s %6.3f | tok/s %7.1f | ttft_ms mean %7.1f p50 %7.1f | "
            "itl_ms mean %6.2f p50 %6.2f | jct_ms %5.0f | n=%s" % (
        r['B'], r['reqps'], r['tokps'], r['ttft_ms'], r['ttft_p50_ms'],
        r['itl_ms'], r['itl_p50_ms'], r['jct_ms'], r['n']))

if __name__ == '__main__':
    files = []
    for a in sys.argv[1:]:
        files += sorted(glob.glob(a)) if ('*' in a) else [a]
    rows = []
    for f in files:
        if os.path.exists(f):
            try: rows.append((f, row(f)))
            except Exception as e: print("ERR", f, e)
    rows.sort(key=lambda x: int(x[1]['B']))
    for f, r in rows:
        print(fmt(r))
