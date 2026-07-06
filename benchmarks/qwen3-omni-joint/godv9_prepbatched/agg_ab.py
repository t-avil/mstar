#!/usr/bin/env python3
# Aggregate an interleaved lab_ab A/B: median over rounds, A vs B, per i2t cell.
# A = baseline (flag off), B = fix (flag on). Metrics from results.json.
import json, glob, os, statistics, re, sys

ABDIRS = sys.argv[1:] if len(sys.argv) > 1 else ["/m-coriander/coriander/tim/lab_godv9/ab_prepbatched"]

def sec(d, m, k):
    v = ((d.get(m) or {}).get('text') or {}).get(k)
    return v

def load(path):
    d = json.load(open(path))
    return {
        'reqps': d.get('request_throughput') or 0.0,
        'tokps': d.get('text_token_throughput') or 0.0,
        'ttft_ms': (sec(d,'ttft','mean') or float('nan'))*1000,
        'itl_ms': (sec(d,'itl','mean') or float('nan'))*1000,
        'jct_ms': d.get('jct_mean_ms') or 0.0,
        'n': d.get('completed'),
    }

# group[(side,B)] = list of metric dicts
groups = {}
for ABDIR in ABDIRS:
    for f in glob.glob(os.path.join(ABDIR, "*_i2t_B*_r*/results.json")):
        m = re.search(r'/([AB])_i2t_B(\d+)_r(\d+)/results.json$', f)
        if not m: continue
        side, b, rnd = m.group(1), int(m.group(2)), int(m.group(3))
        groups.setdefault((side, b), []).append(load(f))

def med(rows, key):
    vals = [r[key] for r in rows if r[key] == r[key]]  # drop nan
    return statistics.median(vals) if vals else float('nan')

bs = sorted({b for (_, b) in groups})
print(f"{'cell':>7} | {'metric':>8} | {'A(off)':>9} | {'B(on)':>9} | {'delta':>8} | rounds")
print("-"*66)
for b in bs:
    A = groups.get(('A', b), []); B = groups.get(('B', b), [])
    for key, label, better_lower in [('reqps','req/s',False),('tokps','tok/s',False),
                                      ('ttft_ms','ttft_ms',True),('itl_ms','itl_ms',True),
                                      ('jct_ms','jct_ms',True)]:
        a = med(A, key); bb = med(B, key)
        if a==a and bb==bb and a:
            d = (bb-a)/a*100
            arrow = ""
            if abs(d) >= 1:
                good = (d<0) if better_lower else (d>0)
                arrow = " WIN" if good else " (worse)"
            print(f"i2t B{b:<4} | {label:>8} | {a:9.3f} | {bb:9.3f} | {d:+7.1f}%{arrow}  ({len(A)}v{len(B)})")
    print("-"*66)
