#!/usr/bin/env python3
"""V10 head-to-head i2t chart: three series, established 2x2 convention.

Panels: tok/s | req/s | TTFT(text) p50 [ms] | ITL(text) mean [ms].

Series:
  M* current best (BLUE, solid, circle)  = rm_out/final i2t cells (this run's
      best config: PD topology + ordered emit + batched-vision grids + all
      session fixes; medians of repeats).
  M* encoders baseline (BLUE, DOTTED, triangle) = sweep_mstar_new_v2{,b,c}
      (encoders-implemeneted @4c33b33), medians across the three sweeps.
  vLLM-Omni 0.22 (GREEN, solid, square) = committed healthy h2h_* aggregates.

Colors/style: shared chartstyle.mplstyle; BLUE #1f77b4 = M*, GREEN #2ca02c =
vLLM-Omni; marker shape + line style are redundant CVD-safe encodings.
Reads committed/warmed raw only; nothing re-benchmarked here.
"""
import glob
import json
import os
import statistics
import re
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
plt.style.use("../chartstyle.mplstyle")
BLUE, GREEN = "#1f77b4", "#2ca02c"
BATCHES = [1, 2, 4, 8, 16, 32]
BEST_DIR = os.environ.get(
    "V10_BEST_DIR", "/m-coriander/coriander/tim/rm_out/final"
)
ENC_GLOBS = [
    "/m-coriander/coriander/tim/sweep_mstar_new_v2/i2t/B*/results.json",
    "/m-coriander/coriander/tim/sweep_mstar_new_v2b/i2t/B*/results.json",
    "/m-coriander/coriander/tim/sweep_mstar_new_v2c/i2t/B*/results.json",
]


def _sec(d, m, k):
    return ((d.get(m) or {}).get("text") or {}).get(k)


def collect(files, batch_of):
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if (d.get("completed") or 0) <= 0 or (d.get("request_throughput") or 0) <= 0:
            continue
        B = batch_of(f, d)
        if B is None:
            continue
        g[B]["req_s"].append(d["request_throughput"])
        if d.get("text_token_throughput"):
            g[B]["tok"].append(d["text_token_throughput"])
        tt = _sec(d, "ttft", "p50")
        it = _sec(d, "itl", "mean")
        if tt is not None:
            g[B]["ttft"].append(tt * 1000)
        if it is not None:
            g[B]["itl"].append(it * 1000)
    out = {}
    for B, m in g.items():
        med = lambda k: statistics.median(m[k]) if m[k] else None
        out[B] = {"req_s": med("req_s"), "tok": med("tok"),
                  "ttft": med("ttft"), "itl": med("itl")}
    return out


def dirbatch(f, d):
    m = re.search(r"i2t_B(\d+)(?:_r\d+)?/", f)
    return int(m.group(1)) if m else None


def sweepbatch(f, d):
    m = re.search(r"/B(\d+)/results.json$", f)
    return int(m.group(1)) if m else None


best = collect(glob.glob(os.path.join(BEST_DIR, "i2t_B*/results.json")) +
               glob.glob(os.path.join(BEST_DIR, "i2t_B*_r*/results.json")), dirbatch)
enc = collect(sum((glob.glob(g) for g in ENC_GLOBS), []), sweepbatch)


def vllm_batch(f, d):
    if d.get("inference_system") != "vllm_omni":
        return None
    if d.get("request_type") != "image_to_text":
        return None
    return d.get("max_concurrency") or d.get("batch_size")

vfiles = glob.glob("h2h_*/**/results.json", recursive=True)
vllm = collect(vfiles, vllm_batch)

data = {"note": "V10 h2h i2t; best=rm_out/final (PD+fixes), enc=sweep_mstar_new_v2{,b,c} "
                "(encoders 4c33b33), vLLM=committed 0.22 healthy; medians",
        "batches": BATCHES,
        "mstar_best": {str(b): best.get(b) for b in BATCHES},
        "mstar_encoders": {str(b): enc.get(b) for b in BATCHES},
        "vllm022": {str(b): vllm.get(b) for b in BATCHES}}
json.dump(data, open("chart_v10_h2h_data.json", "w"), indent=2)

PANELS = [
    ("tok",   "Text throughput (tok/s)",    "tokens / s",   "%.0f", False),
    ("req_s", "Request throughput (req/s)", "requests / s", "%.2f", False),
    ("ttft",  "TTFT p50 (ms)  — lower is better", "ms",     "%.0f", True),
    ("itl",   "ITL mean (ms)  — lower is better", "ms",     "%.1f", True),
]
SERIES = [
    (best, BLUE,  "-",  "o", "M* current best"),
    (enc,  BLUE,  ":",  "^", "M* encoders baseline"),
    (vllm, GREEN, "-",  "s", "vLLM-Omni 0.22"),
]
xs = list(range(len(BATCHES)))
fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
for ax, (key, title, ylab, fmt, lower_better) in zip(axes.flat, PANELS):
    for series, color, ls, marker, label in SERIES:
        ys = [series.get(b, {}).get(key) for b in BATCHES]
        ax.plot([x for x, y in zip(xs, ys) if y is not None],
                [y for y in ys if y is not None], color=color, linestyle=ls,
                marker=marker, markersize=7, markeredgecolor="white",
                markeredgewidth=0.8, label=label, zorder=3,
                alpha=0.75 if ls == ":" else 1.0)
    ax.set_title(title)
    ax.set_ylabel(ylab)
    ax.set_xlabel("batch size (concurrency)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in BATCHES])
    ax.margins(y=0.22)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=8)
    # direct labels: best above in blue, vLLM below in green; encoders
    # unlabeled (dotted context line) to keep the panel readable.
    bv = [best.get(b, {}).get(key) for b in BATCHES]
    vv = [vllm.get(b, {}).get(key) for b in BATCHES]
    for x, (a, b) in enumerate(zip(bv, vv)):
        close = (a is not None and b is not None
                 and abs(a - b) < 0.10 * max(a, b, 1e-9))
        for y, color, is_m in ((a, BLUE, True), (b, GREEN, False)):
            if y is None:
                continue
            above = is_m
            dy = 9 if above else -15
            dx = (-0.10 if is_m else 0.10) if close else 0.0
            ax.annotate(fmt % y, (x, y), textcoords="offset points",
                        xytext=(dx * 30, dy), ha="center",
                        va="bottom" if dy > 0 else "top",
                        fontsize=7.5, color=color, zorder=4)

fig.suptitle(
    "Qwen3-Omni  I2T (image → text):  M* best vs encoders baseline vs "
    "vLLM-Omni 0.22 — warmed, all batches", fontsize=12.5, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
os.makedirs("charts", exist_ok=True)
out = "charts/v10_h2h_i2t_2x2.png"
fig.savefig(out)
print("wrote", out)
print(json.dumps(data["mstar_best"], indent=0))
