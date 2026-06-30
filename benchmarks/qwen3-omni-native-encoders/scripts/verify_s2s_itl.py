#!/usr/bin/env python3
"""S2S ITL deep dive: verify B=1 proves no per-token regression."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_raw, BATCHES

data = load_raw("audio_to_speech")
agg = data["aggregates"]
issues = []

print("=" * 80)
print("S2S ITL DEEP DIVE")
print("=" * 80)

for b in BATCHES:
    bkey = f"B{b}"
    cell_new = agg.get(bkey, {}).get("mstar_new")
    cell_old = agg.get(bkey, {}).get("mstar_old")
    if not cell_new or not cell_old:
        continue
    har_new = cell_new.get("harness", {})
    har_old = cell_old.get("harness", {})
    itl_new_d = har_new.get("itl_audio")
    itl_old_d = har_old.get("itl_audio")
    ttft_new_d = har_new.get("ttft_audio")
    ttft_old_d = har_old.get("ttft_audio")

    tp_new = cell_new["recomputed"]["audio_throughput"]
    tp_old = cell_old["recomputed"]["audio_throughput"]
    tp_ratio = tp_new / tp_old

    req_new = cell_new["recomputed"]["request_throughput"]
    req_old = cell_old["recomputed"]["request_throughput"]
    req_ratio = req_new / req_old

    print(f"\n  B={b}:")
    print(f"    Throughput: {tp_ratio:.3f}x  (new={tp_new:.2f} old={tp_old:.2f} aud-sec/s)")
    print(f"    Req/s:      {req_ratio:.3f}x  (new={req_new:.2f} old={req_old:.2f})")

    if isinstance(ttft_new_d, dict) and isinstance(ttft_old_d, dict):
        ttft_r = ttft_old_d["p50"] / ttft_new_d["p50"] if ttft_new_d["p50"] > 0 else float("inf")
        print(f"    TTFT p50:   old/new={ttft_r:.2f}x  (new={ttft_new_d['p50']*1000:.0f}ms old={ttft_old_d['p50']*1000:.0f}ms)")

    if isinstance(itl_new_d, dict) and isinstance(itl_old_d, dict):
        itl_new = itl_new_d["mean"]
        itl_old = itl_old_d["mean"]
        itl_ratio = itl_new / itl_old if itl_old > 0.001 else float("inf")
        print(f"    ITL mean:   new/old={itl_ratio:.3f}x  (new={itl_new*1000:.1f}ms old={itl_old*1000:.1f}ms)")

        if b == 1:
            # B=1 has no batching contention, so a worse ITL here is a genuine
            # per-token regression with no excuse. This is the claim the script
            # exists to substantiate, so it must be able to FAIL.
            if itl_ratio > 1.10:
                issues.append(f"S2S B=1: ITL {itl_ratio:.2f}x worse (no batching excuse)")
                print(f"    >>> B=1 ITL {itl_ratio:.2f}x worse — REGRESSION (no batching contention)")
            else:
                print(f"    >>> B=1 ITL {itl_ratio:.2f}x — OK (no batching contention)")
        elif itl_ratio > 1.1 and tp_ratio > 1.1:
            print(f"    >>> ITL {itl_ratio:.1f}x worse BUT throughput {tp_ratio:.1f}x better — BATCHING TRADE-OFF")
        elif itl_ratio > 1.1 and tp_ratio < 1.0:
            issues.append(f"S2S B={b}: ITL {itl_ratio:.2f}x worse AND throughput {tp_ratio:.2f}x worse")
            print(f"    >>> ITL {itl_ratio:.1f}x worse AND throughput {tp_ratio:.1f}x worse — REAL REGRESSION")
        else:
            print(f"    >>> ITL within acceptable range relative to throughput gain")

print(f"\n{'='*80}")
if issues:
    print(f"VERDICT: FAIL — {len(issues)} S2S ITL issue(s):")
    for i in issues:
        print(f"  {i}")
    print(f"{'='*80}")
    sys.exit(1)
print("VERDICT: PASS — B=1 (no batching) shows no per-token regression; "
      "higher-B ITL increases are covered by the throughput trade-off.")
print(f"{'='*80}")
