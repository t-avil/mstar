#!/usr/bin/env python3
"""Verify M*-new throughput is not worse than M*-old at every batch size.

Reads aggregates from raw_<path>.json. For text paths uses text_token_throughput,
for speech paths uses audio_throughput.  Threshold: new/old >= 0.97.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_all, PATHS, BATCHES

THRESHOLD = 0.97
issues = []

for path in PATHS:
    data = load_all()[path]
    agg = data["aggregates"]
    modality = "speech" if "speech" in path else "text"
    metric = "audio_throughput" if modality == "speech" else "text_token_throughput"

    for b in BATCHES:
        bkey = f"B{b}"
        cell_new = agg.get(bkey, {}).get("mstar_new")
        cell_old = agg.get(bkey, {}).get("mstar_old")
        if not cell_new or not cell_old:
            continue
        v_new = cell_new["recomputed"].get(metric)
        v_old = cell_old["recomputed"].get(metric)
        if v_new is None or v_old is None or v_old == 0:
            continue
        ratio = v_new / v_old
        status = "PASS" if ratio >= THRESHOLD else "FAIL"
        if status == "FAIL":
            issues.append(f"{path} B={b}: {ratio:.3f}x")
        print(f"  {path:20s} B={b:<3d}  {metric:25s}  new={v_new:10.2f}  old={v_old:10.2f}  ratio={ratio:.3f}x  {status}")

print()
if issues:
    print(f"VERDICT: FAIL — {len(issues)} point(s) below {THRESHOLD}x threshold")
    for i in issues:
        print(f"  {i}")
    sys.exit(1)
else:
    print(f"VERDICT: PASS — all points >= {THRESHOLD}x")
