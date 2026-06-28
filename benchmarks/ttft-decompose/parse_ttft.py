#!/usr/bin/env python3
"""Parse TTFT_TRACE + node-timing lines from the M* server log into a per-stage
B=1 TTFT breakdown (median + p99) for S2T (prefill_audio) and I2T (prefill_vision).

Usage: parse_ttft.py <server.log> [warmup_drop]
"""
import re
import sys
import json
from collections import defaultdict
from statistics import median

LOG = sys.argv[1]
WARMUP = int(sys.argv[2]) if len(sys.argv) > 2 else 5

trace_re = re.compile(r"TTFT_TRACE rid=(\S+) ev=(\S+) t_ns=(\d+)(.*)$")
node_re = re.compile(r"node-timing: (.+)$")

# rid -> {event: t_ns}, rid -> walk
ev = defaultdict(dict)
rid_walk = {}
rid_extra = defaultdict(dict)

# node-timing: keep the aggregate line with the largest total n (most cumulative)
node_best = {}  # (node,walk) -> (n, p50, mean)
node_best_total = -1
node_best_map = {}

with open(LOG, errors="replace") as f:
    for line in f:
        m = trace_re.search(line)
        if m:
            rid, event, t_ns, extra = m.group(1), m.group(2), int(m.group(3)), m.group(4)
            # first occurrence wins for events that could repeat
            if event not in ev[rid]:
                ev[rid][event] = t_ns
            for kv in extra.strip().split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    rid_extra[rid][k] = v
                    if k == "walk":
                        rid_walk[rid] = v
            continue
        m = node_re.search(line)
        if m:
            body = m.group(1)
            entries = {}
            total = 0
            for part in body.split("|"):
                part = part.strip()
                mm = re.match(r"(\S+)/(\S+): p50=([\d.]+)ms mean=([\d.]+)ms n=(\d+)", part)
                if mm:
                    node, walk, p50, mean, n = mm.group(1), mm.group(2), float(mm.group(3)), float(mm.group(4)), int(mm.group(5))
                    entries[(node, walk)] = (n, p50, mean)
                    total += n
            if total > node_best_total:
                node_best_total = total
                node_best_map = entries

def ms(rid, a, b):
    if a in ev[rid] and b in ev[rid]:
        return (ev[rid][b] - ev[rid][a]) / 1e6
    return None

def pct(vals, p):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * p
    lo = int(k); hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)

# stages as (label, ev_a, ev_b)
STAGES = [
    ("api->preproc enqueue",   "api_recv",     "preproc_start"),
    ("preprocess (CPU)",       "preproc_start", "preproc_end"),
    ("  preproc->conductor",   "preproc_end",  "cond_ingest"),
    ("  conductor dispatch",   "cond_ingest",  "cond_dispatch"),
    ("  conductor->worker submit", "cond_dispatch", "worker_submit"),
    ("ADMISSION total (preproc_end->submit)", "preproc_end", "worker_submit"),
    ("encoder+prefill wall (submit->emit)",   "worker_submit", "worker_emit"),
    ("  worker_emit->chunk_ready",  "worker_emit", "chunk_ready"),
    ("  chunk_ready->api_first_chunk", "chunk_ready", "api_first_chunk"),
    ("emit->client total (emit->api_chunk)",  "worker_emit", "api_first_chunk"),
    ("TOTAL server TTFT (api_recv->api_chunk)", "api_recv", "api_first_chunk"),
]

def report(path_name, walk_filter):
    rids = [r for r in ev if rid_walk.get(r) == walk_filter and "api_recv" in ev[r] and "api_first_chunk" in ev[r]]
    rids.sort(key=lambda r: ev[r]["api_recv"])
    measured = rids[WARMUP:]
    print(f"\n===== {path_name}  (walk={walk_filter})  total_traced={len(rids)} measured(after {WARMUP} warmup)={len(measured)} =====")
    print(f"{'stage':<42} {'median_ms':>10} {'p99_ms':>10} {'n':>4}")
    out = {}
    for label, a, b in STAGES:
        vals = [ms(r, a, b) for r in measured]
        vals = [v for v in vals if v is not None]
        med = pct(vals, 0.5); p99 = pct(vals, 0.99)
        n = len(vals)
        ms_s = f"{med:10.2f}" if med is not None else f"{'--':>10}"
        p99_s = f"{p99:10.2f}" if p99 is not None else f"{'--':>10}"
        print(f"{label:<42} {ms_s} {p99_s} {n:>4}")
        out[label] = {"median_ms": med, "p99_ms": p99, "n": n}
    return out, len(measured)

print(f"# parsed {len(ev)} traced rids; node-timing best total n={node_best_total}")
print("\n# Node-timing GPU-kernel split (p50/mean ms per node/walk, cumulative):")
for (node, walk), (n, p50, mean) in sorted(node_best_map.items()):
    print(f"  {node:<18}/{walk:<16} p50={p50:7.2f}ms mean={mean:7.2f}ms n={n}")

s2t, _ = report("S2T (audio->text)", "prefill_audio")
i2t, _ = report("I2T (image->text)", "prefill_vision")

with open(sys.argv[1] + ".breakdown.json", "w") as f:
    json.dump({"s2t": s2t, "i2t": i2t,
               "node_timing": {f"{k[0]}/{k[1]}": {"n": v[0], "p50_ms": v[1], "mean_ms": v[2]}
                               for k, v in node_best_map.items()}}, f, indent=2)
