#!/usr/bin/env python3
"""
Verify raw datapoints match the aggregates in the JSON files.
Also check: no duplicate datapoints, warmup properly excluded,
counts match expected (batch_size * reps), etc.
"""
import json
import os
import sys
import statistics
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_raw

PATHS = [
    ("audio_to_text",   "S2T", False),
    ("audio_to_speech",  "S2S", True),
    ("image_to_text",    "I2T", False),
    ("image_to_speech",  "I2S", True),
]
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
issues = []

for path_name, label, is_speech in PATHS:
    print(f"\n{'='*70}")
    print(f"  {label} ({path_name}) — DATA INTEGRITY")
    print(f"{'='*70}")

    data = load_raw(path_name)
    dps = data["datapoints"]
    agg = data["aggregates"]

    # 1. Check for duplicates
    seen = set()
    dupes = 0
    for dp in dps:
        key = (dp["system"], dp["batch"], dp["phase"], dp["request_id"], dp["jct_ms"])
        if key in seen:
            dupes += 1
        seen.add(key)
    if dupes:
        issues.append(f"{label}: {dupes} duplicate datapoints")
        print(f"  [WARN] {dupes} duplicate datapoints found")
    else:
        print(f"  [OK] No duplicate datapoints")

    # 2. Check warmup vs measure split
    warmup_count = sum(1 for dp in dps if dp.get("phase") == "warmup")
    measure_count = sum(1 for dp in dps if dp.get("phase") == "measure")
    print(f"  Datapoints: {len(dps)} total, {warmup_count} warmup, {measure_count} measure")
    if warmup_count + measure_count != len(dps):
        other = len(dps) - warmup_count - measure_count
        print(f"  [WARN] {other} datapoints with unknown phase")
        issues.append(f"{label}: {other} datapoints with unknown phase")

    # 3. Verify aggregates match measured datapoints
    for bs in BATCH_SIZES:
        bkey = f"B{bs}"
        if bkey not in agg:
            continue

        for system in ["mstar_new", "mstar_old", "vllm"]:
            if system not in agg[bkey]:
                continue

            sys_agg = agg[bkey][system]
            recomp = sys_agg.get("recomputed", {})

            # Get measured datapoints for this system+batch
            sys_dps = [dp for dp in dps
                       if dp["system"] == system
                       and dp["batch"] == bs
                       and dp.get("phase") == "measure"]

            reported_n = recomp.get("n")
            actual_n = len(sys_dps)

            if reported_n != actual_n:
                msg = f"{label} {system} B={bs}: n mismatch: reported={reported_n} actual={actual_n}"
                print(f"  [WARN] {msg}")
                issues.append(msg)

            if not sys_dps:
                continue

            # Verify JCT mean
            jcts = [dp["jct_ms"] / 1000.0 for dp in sys_dps]  # convert to seconds
            computed_mean = statistics.mean(jcts)
            reported_mean = recomp.get("jct_mean_s")

            if reported_mean is not None:
                diff = abs(computed_mean - reported_mean) / max(abs(reported_mean), 1e-9)
                if diff > 0.01:  # 1% tolerance
                    msg = f"{label} {system} B={bs}: JCT mean mismatch {diff:.4f} (computed={computed_mean:.4f} reported={reported_mean:.4f})"
                    print(f"  [WARN] {msg}")
                    issues.append(msg)

            # Verify throughput is consistent with wall_time and data volume
            wall_time = recomp.get("wall_time_s")
            if is_speech:
                reported_tp = recomp.get("audio_throughput")
                # For speech: audio_throughput = sum(audio_seconds) / wall_time
                audio_secs = [dp.get("audio_seconds", 0) for dp in sys_dps]
                total_audio = sum(audio_secs)
                if wall_time and wall_time > 0:
                    computed_tp = total_audio / wall_time
                    if reported_tp and reported_tp > 0:
                        tp_diff = abs(computed_tp - reported_tp) / reported_tp
                        if tp_diff > 0.05:
                            msg = f"{label} {system} B={bs}: audio_throughput mismatch {tp_diff:.3f} (computed={computed_tp:.2f} reported={reported_tp:.2f})"
                            print(f"  [WARN] {msg}")
                            issues.append(msg)
            else:
                reported_tp = recomp.get("text_token_throughput")
                text_bytes = [dp.get("text_bytes", 0) for dp in sys_dps]
                # For text: throughput ≈ total_tokens / wall_time
                # We don't know exact token count from bytes, but can check request_throughput
                req_tp = recomp.get("request_throughput")
                if wall_time and wall_time > 0 and actual_n > 0:
                    computed_req_tp = actual_n / wall_time
                    if req_tp and req_tp > 0:
                        req_diff = abs(computed_req_tp - req_tp) / req_tp
                        if req_diff > 0.05:
                            msg = f"{label} {system} B={bs}: req_throughput mismatch {req_diff:.3f} (computed={computed_req_tp:.2f} reported={req_tp:.2f})"
                            print(f"  [WARN] {msg}")
                            issues.append(msg)

    # 4. Check per-system datapoint counts are balanced
    print(f"\n  Per-system counts:")
    for system in ["mstar_new", "mstar_old", "vllm"]:
        for bs in BATCH_SIZES:
            sys_dps = [dp for dp in dps
                       if dp["system"] == system
                       and dp["batch"] == bs
                       and dp.get("phase") == "measure"]
            expected = bs * 10  # typical: 10 reps per request in batch
            if sys_dps:
                print(f"    {system:12s} B={bs:2d}: {len(sys_dps):4d} measured", end="")
                if len(sys_dps) != expected:
                    print(f"  (expected ~{expected})")
                else:
                    print()

    # 5. Check for negative JCTs or suspiciously fast/slow values
    for dp in dps:
        if dp.get("phase") != "measure":
            continue
        jct = dp["jct_ms"]
        if jct < 0:
            issues.append(f"{label} {dp['system']} B={dp['batch']}: negative JCT {jct}")
        if jct < 1:
            issues.append(f"{label} {dp['system']} B={dp['batch']}: suspiciously fast JCT {jct:.4f}ms")

print(f"\n{'='*70}")
print(f"DATA INTEGRITY SUMMARY")
print(f"{'='*70}")
print(f"Issues found: {len(issues)}")
for i, issue in enumerate(issues, 1):
    print(f"  {i}. {issue}")
if not issues:
    print("  ALL CLEAN — aggregates match raw data, no duplicates, no anomalies")
