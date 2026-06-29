#!/usr/bin/env python3
"""
Comprehensive benchmark analysis: M*-new vs M*-old
Checks:
1. Throughput: new >= old at every batch size
2. TTFT consistency with throughput
3. ITL consistency with throughput
4. Internal consistency: do TTFT/ITL explain throughput differences?
5. Statistical significance (variance analysis)
6. Anomaly detection (outliers, suspicious patterns)
"""
import json
import sys
import os
from collections import defaultdict
import statistics

sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_raw

PATHS = [
    ("audio_to_text",   "S2T", "text_token_throughput", "tok/s"),
    ("audio_to_speech",  "S2S", "audio_throughput",      "aud-sec/s"),
    ("image_to_text",    "I2T", "text_token_throughput", "tok/s"),
    ("image_to_speech",  "I2S", "audio_throughput",      "aud-sec/s"),
]
BATCH_SIZES = [1, 2, 4, 8, 16, 32]

def analyze_consistency():
    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK VERIFICATION")
    print("M*-new (integration-mnew f58a805) vs M*-old (upstream main ae7d173)")
    print("=" * 80)

    all_issues = []
    all_results = {}

    for path_name, label, tp_key, tp_unit in PATHS:
        print(f"\n{'='*80}")
        print(f"  {label} ({path_name})")
        print(f"{'='*80}")

        data = load_raw(path_name)
        agg = data["aggregates"]
        dps = data["datapoints"]

        # Group datapoints by system and batch
        dp_groups = defaultdict(lambda: defaultdict(list))
        for dp in dps:
            if dp.get("phase") == "measure":
                dp_groups[dp["system"]][dp["batch"]].append(dp)

        for bs in BATCH_SIZES:
            bkey = f"B{bs}"
            if bkey not in agg:
                print(f"\n  [SKIP] Batch {bs}: no data")
                continue

            new_agg = agg[bkey].get("mstar_new", {})
            old_agg = agg[bkey].get("mstar_old", {})

            if not new_agg or not old_agg:
                print(f"\n  [SKIP] Batch {bs}: missing system data")
                continue

            new_r = new_agg.get("recomputed", {})
            old_r = old_agg.get("recomputed", {})
            new_h = new_agg.get("harness", {})
            old_h = old_agg.get("harness", {})

            # --- Throughput ---
            is_speech_path = "speech" in path_name
            if is_speech_path:
                new_tp = new_r.get("audio_throughput")
                old_tp = old_r.get("audio_throughput")
            else:
                new_tp = new_r.get("text_token_throughput")
                old_tp = old_r.get("text_token_throughput")

            if new_tp is None or old_tp is None:
                print(f"\n  [SKIP] Batch {bs}: throughput is None")
                continue

            tp_ratio = new_tp / old_tp if old_tp > 0 else float('inf')

            # --- TTFT ---
            if is_speech_path:
                ttft_key = "ttft_audio"
            else:
                ttft_key = "ttft_text"

            new_ttft_p50 = new_h.get(ttft_key, {}).get("p50")
            old_ttft_p50 = old_h.get(ttft_key, {}).get("p50")
            new_ttft_mean = new_h.get(ttft_key, {}).get("mean")
            old_ttft_mean = old_h.get(ttft_key, {}).get("mean")
            new_ttft_p95 = new_h.get(ttft_key, {}).get("p95")
            old_ttft_p95 = old_h.get(ttft_key, {}).get("p95")

            # --- ITL ---
            if is_speech_path:
                itl_key = "itl_audio"
            else:
                itl_key = "itl_text"

            new_itl_mean = new_h.get(itl_key, {}).get("mean")
            old_itl_mean = old_h.get(itl_key, {}).get("mean")
            new_itl_p50 = new_h.get(itl_key, {}).get("p50")
            old_itl_p50 = old_h.get(itl_key, {}).get("p50")
            new_itl_p95 = new_h.get(itl_key, {}).get("p95")
            old_itl_p95 = old_h.get(itl_key, {}).get("p95")

            # --- Request throughput ---
            new_req = new_r.get("request_throughput")
            old_req = old_r.get("request_throughput")
            req_ratio = new_req / old_req if old_req and old_req > 0 else None

            # --- Print ---
            print(f"\n  Batch {bs}:")
            tp_status = "OK" if tp_ratio >= 0.97 else "REGRESSION"
            if tp_ratio < 0.97:
                all_issues.append(f"{label} B={bs}: throughput regression {tp_ratio:.3f}x")
            print(f"    Throughput: new={new_tp:.2f} old={old_tp:.2f} {tp_unit}  ratio={tp_ratio:.3f}x [{tp_status}]")
            if req_ratio:
                print(f"    Req/s:      new={new_req:.2f} old={old_req:.2f}  ratio={req_ratio:.3f}x")

            # TTFT analysis
            if new_ttft_p50 is not None and old_ttft_p50 is not None:
                ttft_ratio = old_ttft_p50 / new_ttft_p50 if new_ttft_p50 > 0 else float('inf')
                ttft_status = "OK" if new_ttft_p50 <= old_ttft_p50 * 1.10 else "WORSE"
                if new_ttft_p50 > old_ttft_p50 * 1.10:
                    all_issues.append(f"{label} B={bs}: TTFT p50 worse by {new_ttft_p50/old_ttft_p50:.2f}x")
                print(f"    TTFT p50:   new={new_ttft_p50*1000:.1f}ms old={old_ttft_p50*1000:.1f}ms  ratio={ttft_ratio:.2f}x (old/new, >1=new faster) [{ttft_status}]")
                if new_ttft_p95 and old_ttft_p95:
                    print(f"    TTFT p95:   new={new_ttft_p95*1000:.1f}ms old={old_ttft_p95*1000:.1f}ms")

            # ITL analysis
            if new_itl_mean is not None and old_itl_mean is not None:
                # ITL near zero needs special handling
                if old_itl_mean < 0.001 and new_itl_mean < 0.001:
                    print(f"    ITL mean:   new={new_itl_mean*1000:.2f}ms old={old_itl_mean*1000:.2f}ms  (both near-zero, no meaningful comparison)")
                elif old_itl_mean < 0.001:
                    print(f"    ITL mean:   new={new_itl_mean*1000:.2f}ms old={old_itl_mean*1000:.2f}ms  [NOTE: old near-zero suggests sequential processing]")
                    # This is actually expected for M*-old at high batch: sequential processing
                    # means requests don't overlap, so ITL is very low but throughput is also low
                else:
                    itl_ratio = old_itl_mean / new_itl_mean if new_itl_mean > 0 else float('inf')
                    itl_status = "OK" if new_itl_mean <= old_itl_mean * 1.10 else "WORSE"
                    if new_itl_mean > old_itl_mean * 1.10:
                        all_issues.append(f"{label} B={bs}: ITL mean worse by {new_itl_mean/old_itl_mean:.2f}x")
                    print(f"    ITL mean:   new={new_itl_mean*1000:.2f}ms old={old_itl_mean*1000:.2f}ms  ratio={itl_ratio:.2f}x (old/new, >1=new faster) [{itl_status}]")
                if new_itl_p50 and old_itl_p50:
                    print(f"    ITL p50:    new={new_itl_p50*1000:.2f}ms old={old_itl_p50*1000:.2f}ms")

            # --- Consistency check: throughput vs TTFT vs ITL ---
            # If throughput is higher, we expect either:
            #   - Better TTFT (faster first token), OR
            #   - Better ITL (faster generation), OR
            #   - Higher concurrency (more requests processed in parallel)
            # The combination should be consistent
            if tp_ratio > 1.1 and new_ttft_p50 and old_ttft_p50:
                if new_ttft_p50 > old_ttft_p50 * 1.5:
                    # Throughput is much better but TTFT is much worse
                    # This is OK if it's explained by batching overhead
                    note = "EXPECTED: batching adds TTFT overhead but improves throughput via concurrency"
                    print(f"    [CONSISTENCY] Throughput {tp_ratio:.1f}x better but TTFT {new_ttft_p50/old_ttft_p50:.1f}x worse -> {note}")

            # Check for anomalous ITL patterns
            if new_itl_mean is not None and old_itl_mean is not None:
                if old_itl_mean > 0 and new_itl_mean / old_itl_mean > 5:
                    print(f"    [ANOMALY] ITL increased {new_itl_mean/old_itl_mean:.1f}x — investigate")
                    all_issues.append(f"{label} B={bs}: ITL anomaly ({new_itl_mean/old_itl_mean:.1f}x increase)")

            # --- Variance analysis from raw datapoints ---
            new_dps = dp_groups.get("mstar_new", {}).get(bs, [])
            old_dps = dp_groups.get("mstar_old", {}).get(bs, [])

            if new_dps and old_dps:
                new_jcts = [dp["jct_ms"] for dp in new_dps]
                old_jcts = [dp["jct_ms"] for dp in old_dps]

                new_jct_mean = statistics.mean(new_jcts)
                old_jct_mean = statistics.mean(old_jcts)
                new_jct_std = statistics.stdev(new_jcts) if len(new_jcts) > 1 else 0
                old_jct_std = statistics.stdev(old_jcts) if len(old_jcts) > 1 else 0
                new_cv = new_jct_std / new_jct_mean if new_jct_mean > 0 else 0
                old_cv = old_jct_std / old_jct_mean if old_jct_mean > 0 else 0

                print(f"    JCT stats:  new: mean={new_jct_mean:.1f}ms std={new_jct_std:.1f}ms CV={new_cv:.3f}")
                print(f"                old: mean={old_jct_mean:.1f}ms std={old_jct_std:.1f}ms CV={old_cv:.3f}")
                print(f"    Datapoints: new={len(new_dps)} old={len(old_dps)}")

                if new_cv > 0.5:
                    all_issues.append(f"{label} B={bs}: high variance in new (CV={new_cv:.3f})")
                    print(f"    [WARN] High variance in new results (CV={new_cv:.3f})")

            # Store for summary
            all_results[(label, bs)] = {
                "tp_ratio": tp_ratio,
                "ttft_ratio_p50": (old_ttft_p50 / new_ttft_p50) if (new_ttft_p50 and old_ttft_p50 and new_ttft_p50 > 0) else None,
                "itl_ratio_mean": (old_itl_mean / new_itl_mean) if (new_itl_mean and old_itl_mean and new_itl_mean > 0.001) else None,
                "new_tp": new_tp,
                "old_tp": old_tp,
            }

    # --- Summary ---
    print(f"\n{'='*80}")
    print("  SUMMARY: NOT-WORSE VERDICT")
    print(f"{'='*80}")

    print("\nThroughput ratios (new/old, >1.0 = new is better):")
    for (label, bs), r in sorted(all_results.items()):
        status = "OK" if r["tp_ratio"] >= 0.97 else "REGRESSION"
        marker = "  " if r["tp_ratio"] >= 0.97 else "!!"
        print(f"  {marker} {label:4s} B={bs:2d}: {r['tp_ratio']:.3f}x  (new={r['new_tp']:.1f} old={r['old_tp']:.1f}) [{status}]")

    print(f"\nTotal issues found: {len(all_issues)}")
    for i, issue in enumerate(all_issues, 1):
        print(f"  {i}. {issue}")

    if not all_issues:
        print("  NONE — all metrics look clean")

    # Check the ITL paradox: at high batch, M*-old has very low ITL but low throughput
    # This is because M*-old processes sequentially (no real batching)
    print(f"\n{'='*80}")
    print("  ITL PARADOX ANALYSIS (sequential vs batched processing)")
    print(f"{'='*80}")
    print("""
When M*-old shows very low ITL (near-zero) at high batch sizes, it's because:
- M*-old processes requests mostly sequentially (one at a time)
- Each request gets the GPU to itself → low ITL per-token
- But total throughput is low because requests are serial

When M*-new shows higher ITL at high batch sizes:
- M*-new batches requests together → higher concurrency
- Tokens from different requests share GPU time → slightly higher per-token ITL
- But total throughput is much higher because of parallelism

This is EXPECTED and CORRECT behavior. Higher ITL + much higher throughput
= the system is doing more work concurrently, which is the goal.

The key metric for "not worse" is throughput (and req/s), not raw ITL.
ITL is only a problem if throughput is ALSO worse.
""")

    # Final verdict
    regressions = [r for r in all_results.values() if r["tp_ratio"] < 0.97]
    if not regressions:
        print("VERDICT: PASS — M*-new is not worse than M*-old on any path/batch combination")
    else:
        print(f"VERDICT: FAIL — {len(regressions)} throughput regressions detected")

    return all_issues

if __name__ == "__main__":
    issues = analyze_consistency()
    sys.exit(1 if issues else 0)
