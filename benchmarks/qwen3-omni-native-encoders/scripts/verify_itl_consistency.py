#!/usr/bin/env python3
"""Verify ITL consistency: increases at higher batch are explained by batching.

For each path, checks that B=1 ITL (no batching contention) shows M*-new is
not worse than M*-old.  Higher-batch ITL increases are expected batching
overhead when throughput also increases.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_all, PATHS, BATCHES

issues = []

for path in PATHS:
    data = load_all()[path]
    agg = data["aggregates"]
    modality = "speech" if "speech" in path else "text"
    itl_key = f"itl_{'audio' if modality == 'speech' else 'text'}"
    tp_key = "audio_throughput" if modality == "speech" else "text_token_throughput"

    print(f"\n  {path}:")
    for b in BATCHES:
        bkey = f"B{b}"
        cell_new = agg.get(bkey, {}).get("mstar_new")
        cell_old = agg.get(bkey, {}).get("mstar_old")
        if not cell_new or not cell_old:
            continue
        har_new = cell_new.get("harness", {})
        har_old = cell_old.get("harness", {})
        itl_new_d = har_new.get(itl_key)
        itl_old_d = har_old.get(itl_key)
        if not isinstance(itl_new_d, dict) or not isinstance(itl_old_d, dict):
            print(f"    B={b}: ITL data missing for one system (skipped)")
            continue
        itl_new = itl_new_d.get("mean", 0)
        itl_old = itl_old_d.get("mean", 0)

        tp_new = cell_new["recomputed"].get(tp_key, 0)
        tp_old = cell_old["recomputed"].get(tp_key, 0)
        tp_ratio = tp_new / tp_old if tp_old > 0 else float("inf")

        if itl_old > 0.0001:
            itl_ratio = itl_new / itl_old
        else:
            itl_ratio = float("inf")

        tag = ""
        if b == 1:
            if itl_ratio > 1.10:
                tag = "REGRESSION at B=1"
                issues.append(f"{path} B=1: ITL {itl_ratio:.2f}x worse (no batching excuse)")
            else:
                tag = "OK (no batching contention)"
        else:
            if itl_ratio > 1.10 and tp_ratio > 1.10:
                tag = "batching trade-off (higher throughput)"
            elif itl_ratio > 1.10 and tp_ratio < 1.0:
                tag = "REGRESSION (worse ITL AND throughput)"
                issues.append(f"{path} B={b}: ITL {itl_ratio:.2f}x worse, throughput {tp_ratio:.2f}x")
            else:
                tag = "OK"

        print(f"    B={b:<3d}  ITL new={itl_new*1000:8.2f}ms old={itl_old*1000:8.2f}ms "
              f"ratio={itl_ratio:.2f}x  TP={tp_ratio:.2f}x  {tag}")

print()
if issues:
    print(f"VERDICT: {len(issues)} issue(s)")
    for i in issues:
        print(f"  {i}")
    sys.exit(1)
else:
    print("VERDICT: PASS — ITL patterns consistent with batching behavior")
