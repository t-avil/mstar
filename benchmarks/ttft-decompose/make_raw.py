#!/usr/bin/env python3
"""Build raw.json (per-request, per-stage TTFT datapoints) from the M* server log
TTFT_TRACE + node-timing lines. CLAUDE.md raw schema: every datapoint, no
aggregates baked in (charts recompute median/p99 at plot time).

Usage: make_raw.py <server.log> <out_raw.json> [warmup_drop=5]
"""
import re, sys, json
from collections import defaultdict
from datetime import datetime, timezone

LOG, OUT = sys.argv[1], sys.argv[2]
WARMUP = int(sys.argv[3]) if len(sys.argv) > 3 else 5

trace_re = re.compile(r"TTFT_TRACE rid=(\S+) ev=(\S+) t_ns=(\d+)(.*)$")
node_re = re.compile(r"node-timing: (.+)$")

ev = defaultdict(dict)
rid_walk = {}
node_best_total, node_best_map = -1, {}

with open(LOG, errors="replace") as f:
    for line in f:
        m = trace_re.search(line)
        if m:
            rid, event, t_ns, extra = m.group(1), m.group(2), int(m.group(3)), m.group(4)
            if event not in ev[rid]:
                ev[rid][event] = t_ns
            for kv in extra.strip().split():
                if kv.startswith("walk="):
                    rid_walk[rid] = kv.split("=", 1)[1]
            continue
        m = node_re.search(line)
        if m:
            entries, total = {}, 0
            for part in m.group(1).split("|"):
                mm = re.match(r"(\S+)/(\S+): p50=([\d.]+)ms mean=([\d.]+)ms n=(\d+)", part.strip())
                if mm:
                    entries[(mm.group(1), mm.group(2))] = (int(mm.group(5)), float(mm.group(3)), float(mm.group(4)))
                    total += int(mm.group(5))
            if total > node_best_total:
                node_best_total, node_best_map = total, entries

# Stage definitions: label -> (ev_a, ev_b). Sub-stages prefixed "  ".
STAGES = [
    ("api_to_preproc",        "api_recv",      "preproc_start"),
    ("preprocess_cpu",        "preproc_start", "preproc_end"),
    ("admission_total",       "preproc_end",   "worker_submit"),
    ("  preproc_to_conductor","preproc_end",   "cond_ingest"),
    ("  conductor_dispatch",  "cond_ingest",   "cond_dispatch"),
    ("  conductor_to_submit", "cond_dispatch", "worker_submit"),
    ("encoder_plus_prefill_wall", "worker_submit", "worker_emit"),
    ("emit_to_client_total",  "worker_emit",   "api_first_chunk"),
    ("  worker_emit_to_chunk","worker_emit",   "chunk_ready"),
    ("  chunk_to_api",        "chunk_ready",   "api_first_chunk"),
    ("total_server_ttft",     "api_recv",      "api_first_chunk"),
]

def dms(rid, a, b):
    if a in ev[rid] and b in ev[rid]:
        return (ev[rid][b] - ev[rid][a]) / 1e6
    return None

datapoints = []
for walk, path in (("prefill_audio", "S2T"), ("prefill_vision", "I2T")):
    rids = [r for r in ev if rid_walk.get(r) == walk
            and "api_recv" in ev[r] and "api_first_chunk" in ev[r]]
    rids.sort(key=lambda r: ev[r]["api_recv"])
    for i, r in enumerate(rids):
        phase = "warmup" if i < WARMUP else "measure"
        stages = {label: dms(r, a, b) for label, a, b in STAGES}
        datapoints.append({"path": path, "walk": walk, "iter": i,
                           "phase": phase, "rid": r, "stages_ms": stages})

raw = {
    "benchmark": "ttft-decompose",
    "description": "B=1 per-stage TTFT decomposition for Qwen3-Omni S2T/I2T on M* (8xH200, GPUs 0,1, isolated single server). Cross-process monotonic-clock boundary traces + per-node GPU CUDA-event timing.",
    "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    "device": {"cuda_visible_devices": "0,1", "gpu_name": "NVIDIA H200", "n_gpu": 2},
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "units": "ms",
    "warmup_iters": WARMUP,
    "stage_order": [s[0] for s in STAGES],
    "node_timing_gpu_ms": {f"{k[0]}/{k[1]}": {"n": v[0], "p50_ms": v[1], "mean_ms": v[2]}
                            for k, v in sorted(node_best_map.items())},
    "datapoints": datapoints,
    "status": "complete",
}
with open(OUT, "w") as f:
    json.dump(raw, f, indent=2)
print(f"wrote {OUT}: {len(datapoints)} datapoints, node_timing n={node_best_total}")
