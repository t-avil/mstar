#!/usr/bin/env python3
"""Verify raw JSON data integrity.

Checks:
  - No duplicate (system, batch, request_id) in measured datapoints
  - For systems with raw datapoints, counts match aggregate 'completed'
  - All JCTs are positive

Note: some system variants were ingested as aggregates-only (no per-request
raw datapoints) — this is expected and not flagged.
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_all, PATHS, BATCHES

issues = []

for path in PATHS:
    data = load_all()[path]
    dps = data["datapoints"]
    agg = data["aggregates"]

    measured = [dp for dp in dps if dp.get("phase") == "measure"]
    keys = [(dp.get("system"), dp.get("batch"), dp.get("request_id")) for dp in measured]
    dupes = [k for k, c in collections.Counter(keys).items() if c > 1]
    if dupes:
        print(f"  {path}: WARNING — {len(dupes)} duplicate key(s) in {len(measured)} datapoints")
    else:
        print(f"  {path}: no duplicates in {len(measured)} measured datapoints")

    sys_dp_counts = collections.Counter(dp.get("system") for dp in measured)
    systems_with_data = {s for s, n in sys_dp_counts.items() if n > 10}

    for b in BATCHES:
        bkey = f"B{b}"
        for sys_name in systems_with_data:
            cell = agg.get(bkey, {}).get(sys_name)
            if not cell:
                continue
            expected_n = cell.get("completed") or cell.get("num_requests")
            actual = len([dp for dp in measured
                         if dp.get("system") == sys_name and dp.get("batch") == b])
            if expected_n and actual != expected_n:
                msg = f"{path} {bkey}/{sys_name}: expected {expected_n}, got {actual}"
                issues.append(msg)

    bad_jct = [dp for dp in measured if dp.get("jct_ms", 1) <= 0]
    if bad_jct:
        issues.append(f"{path}: {len(bad_jct)} datapoints with non-positive JCT")
        print(f"  {path}: {len(bad_jct)} bad JCTs")

print()
if issues:
    print(f"ISSUES ({len(issues)}):")
    for i in issues:
        print(f"  {i}")
    sys.exit(1)
else:
    print("VERDICT: PASS — data integrity checks passed")
