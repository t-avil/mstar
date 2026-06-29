#!/usr/bin/env python3
"""
Deep ITL analysis: confirm ITL increases are explained by batching,
not by per-request degradation.

Key insight: if M*-old processes requests sequentially, then at batch=N:
  - M*-old wall_time ≈ N * single_request_time
  - M*-old ITL ≈ single_request_ITL (low, because no contention)
  - M*-old throughput ≈ single_request_throughput (doesn't scale)

If M*-new batches requests:
  - M*-new wall_time << N * single_request_time
  - M*-new ITL > single_request_ITL (contention from batching)
  - M*-new throughput >> single_request_throughput (scales with batch)

Diagnostic: plot ITL * throughput_ratio to see if the "ITL cost" is
proportional to the throughput gain. If so, it's pure batching overhead.
"""
import json
import os
import sys
from collections import defaultdict
import statistics

sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_raw

PATHS = [
    ("audio_to_text",   "S2T", False),
    ("audio_to_speech",  "S2S", True),
    ("image_to_text",    "I2T", False),
    ("image_to_speech",  "I2S", True),
]
BATCH_SIZES = [1, 2, 4, 8, 16, 32]

print("=" * 100)
print("DEEP ITL ANALYSIS: Is the ITL increase explained by batching?")
print("=" * 100)

for path_name, label, is_speech in PATHS:
    data = load_raw(path_name)
    agg = data["aggregates"]

    print(f"\n{'='*100}")
    print(f"  {label} ({path_name})")
    print(f"{'='*100}")

    itl_key = "itl_audio" if is_speech else "itl_text"
    tp_key = "audio_throughput" if is_speech else "text_token_throughput"
    ttft_key = "ttft_audio" if is_speech else "ttft_text"

    rows = []
    for bs in BATCH_SIZES:
        bkey = f"B{bs}"
        if bkey not in agg:
            continue
        new = agg[bkey].get("mstar_new", {})
        old = agg[bkey].get("mstar_old", {})
        if not new or not old:
            continue

        new_r, old_r = new["recomputed"], old["recomputed"]
        new_h, old_h = new.get("harness", {}), old.get("harness", {})

        new_tp = new_r.get(tp_key)
        old_tp = old_r.get(tp_key)
        new_itl = new_h.get(itl_key, {}).get("mean")
        old_itl = old_h.get(itl_key, {}).get("mean")
        new_ttft = new_h.get(ttft_key, {}).get("p50")
        old_ttft = old_h.get(ttft_key, {}).get("p50")
        new_req = new_r.get("request_throughput")
        old_req = old_r.get("request_throughput")
        new_wall = new_r.get("wall_time_s")
        old_wall = old_r.get("wall_time_s")

        if None in (new_tp, old_tp, new_itl, old_itl):
            continue

        rows.append({
            "bs": bs,
            "new_tp": new_tp, "old_tp": old_tp,
            "new_itl": new_itl, "old_itl": old_itl,
            "new_ttft": new_ttft, "old_ttft": old_ttft,
            "new_req": new_req, "old_req": old_req,
            "new_wall": new_wall, "old_wall": old_wall,
        })

    if not rows:
        continue

    # Get B=1 as baseline
    b1 = rows[0] if rows[0]["bs"] == 1 else None

    print(f"\n  {'B':>3s} | {'TP ratio':>9s} | {'ITL new':>9s} | {'ITL old':>9s} | {'ITL ratio':>9s} | {'Req/s ratio':>10s} | {'TTFT ratio':>10s} | {'Wall ratio':>10s} | {'Verdict':>10s}")
    print(f"  {'-'*3}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for r in rows:
        tp_ratio = r["new_tp"] / r["old_tp"] if r["old_tp"] > 0 else 0
        itl_ratio = r["new_itl"] / r["old_itl"] if r["old_itl"] > 0.0001 else float('inf')
        req_ratio = r["new_req"] / r["old_req"] if r["old_req"] and r["old_req"] > 0 else 0
        ttft_ratio = r["old_ttft"] / r["new_ttft"] if r["new_ttft"] and r["new_ttft"] > 0 else 0
        wall_ratio = r["old_wall"] / r["new_wall"] if r["new_wall"] and r["new_wall"] > 0 else 0

        # Verdict logic:
        # If throughput is better AND (ITL * req_ratio) is reasonable, it's batching
        # "Reasonable" = the per-request ITL contribution doesn't exceed the throughput gain
        if tp_ratio >= 0.97:
            if r["old_itl"] < 0.001:
                verdict = "SEQ->BATCH"
            elif itl_ratio < 2.0:
                verdict = "OK"
            elif tp_ratio > itl_ratio:
                verdict = "BATCH-OK"
            else:
                verdict = "CHECK"
        else:
            verdict = "REGRESSION"

        itl_old_str = f"{r['old_itl']*1000:.2f}ms" if r['old_itl'] > 0.001 else f"{r['old_itl']*1000:.3f}ms"
        print(f"  {r['bs']:3d} | {tp_ratio:9.3f}x | {r['new_itl']*1000:7.2f}ms | {itl_old_str:>9s} | {itl_ratio:9.2f}x | {req_ratio:10.3f}x | {ttft_ratio:10.2f}x | {wall_ratio:10.2f}x | {verdict:>10s}")

    # Analysis of batching efficiency
    print(f"\n  Batching efficiency analysis:")
    if b1:
        for r in rows:
            if r["bs"] == 1:
                continue
            # Ideal linear scaling: throughput = B * single_throughput
            ideal_tp = r["bs"] * b1["new_tp"]
            actual_tp = r["new_tp"]
            efficiency = actual_tp / ideal_tp * 100
            old_efficiency = (r["old_tp"] / (r["bs"] * b1["old_tp"])) * 100 if b1["old_tp"] > 0 else 0
            print(f"    B={r['bs']:2d}: new efficiency={efficiency:.1f}% old efficiency={old_efficiency:.1f}%")

    # Cross-metric consistency: does the throughput ratio track with wall-time ratio?
    print(f"\n  Cross-metric consistency (throughput ratio should ≈ wall-time ratio * N/N):")
    for r in rows:
        tp_ratio = r["new_tp"] / r["old_tp"] if r["old_tp"] > 0 else 0
        wall_ratio = r["old_wall"] / r["new_wall"] if r["new_wall"] and r["new_wall"] > 0 else 0
        deviation = abs(tp_ratio - wall_ratio) / max(tp_ratio, wall_ratio) * 100 if max(tp_ratio, wall_ratio) > 0 else 0
        status = "OK" if deviation < 20 else "CHECK"
        print(f"    B={r['bs']:2d}: tp_ratio={tp_ratio:.3f} wall_ratio={wall_ratio:.3f} deviation={deviation:.1f}% [{status}]")

# Now check: for the audio paths, is ITL at B=1 comparable between old and new?
# B=1 is the cleanest comparison (no batching effects)
print(f"\n{'='*100}")
print("  B=1 BASELINE COMPARISON (no batching effects)")
print(f"{'='*100}")
print("  At B=1, ITL should be similar between old and new (no batching contention)")
print()

for path_name, label, is_speech in PATHS:
    data = load_raw(path_name)
    agg = data["aggregates"]
    itl_key = "itl_audio" if is_speech else "itl_text"
    ttft_key = "ttft_audio" if is_speech else "ttft_text"

    b1 = agg.get("B1", {})
    new_h = b1.get("mstar_new", {}).get("harness", {})
    old_h = b1.get("mstar_old", {}).get("harness", {})

    new_itl = new_h.get(itl_key, {})
    old_itl = old_h.get(itl_key, {})
    new_ttft = new_h.get(ttft_key, {})
    old_ttft = old_h.get(ttft_key, {})

    print(f"  {label} B=1:")
    for metric_name, new_m, old_m in [("ITL", new_itl, old_itl), ("TTFT", new_ttft, old_ttft)]:
        for stat in ["mean", "p50", "p95", "p99"]:
            nv = new_m.get(stat)
            ov = old_m.get(stat)
            if nv is not None and ov is not None:
                ratio = nv / ov if ov > 0 else float('inf')
                status = "OK" if ratio < 1.15 else ("BETTER" if ratio < 1.0 else "WORSE")
                print(f"    {metric_name} {stat:4s}: new={nv*1000:8.2f}ms  old={ov*1000:8.2f}ms  ratio={ratio:.3f}x [{status}]")
    print()
