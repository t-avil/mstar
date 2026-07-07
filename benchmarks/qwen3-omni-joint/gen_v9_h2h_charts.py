#!/usr/bin/env python3
"""V9 head-to-head i2t chart: best TRUTHFUL M* (blue) vs vLLM-Omni 0.22 (green),
all batches, 2x2 grid.  Panels (established i2t convention):
    tok/s | req/s | TTFT(text) p50 [ms] | ITL(text) mean [ms]

Data (both warmed, same benchmark.runner, medians of repeats):
  M* (blue, circle) = verify_warm_e1  (E1 build opt/prep-pos-batched-v9 @915ab8f;
                      MSTAR_PREP_DEVICE_POS_BATCHED on; 5 measured rounds/cell,
                      strict warm-in, isolated on GPUs 6,7).
  vLLM 0.22 (green, square) = committed healthy h2h_* aggregates (thr>0, completed>0).

Colors + style from the shared workspace convention (chartstyle.mplstyle):
  BLUE #1f77b4 = M*, GREEN #2ca02c = vLLM-Omni. Marker shape is a redundant
  (CVD-safe) encoding on top of hue; every point is direct-labeled.
Nothing re-benchmarked here; reads committed/warmed raw only.
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
MSTAR_DIR = "/m-coriander/coriander/tim/verify_warm_e1"


def _sec(d, m, k):
    return ((d.get(m) or {}).get("text") or {}).get(k)


def collect(files, batch_of):
    """files -> {B: {req_s, tok, ttft_p50_ms, itl_ms}} using medians."""
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


# M*: verify_warm_e1/i2t_B<b>_r<r>
def mstar_batch(f, d):
    m = re.search(r"i2t_B(\d+)_r\d+", f)
    return int(m.group(1)) if m else None

mstar = collect(glob.glob(os.path.join(MSTAR_DIR, "i2t_B*_r*/results.json")), mstar_batch)

# vLLM 0.22: committed healthy h2h_* image_to_text
def vllm_batch(f, d):
    if d.get("inference_system") != "vllm_omni":
        return None
    if d.get("request_type") != "image_to_text":
        return None
    return d.get("max_concurrency") or d.get("batch_size")

vfiles = glob.glob("h2h_*/**/results.json", recursive=True) + glob.glob("h2h_*/*/results.json")
vllm = collect(vfiles, vllm_batch)

# persist the exact plotted datapoints (regenerable + auditable)
data = {"note": "V9 h2h i2t; M*=verify_warm_e1 (E1), vLLM=committed 0.22 healthy; medians",
        "batches": BATCHES,
        "mstar": {str(b): mstar.get(b) for b in BATCHES},
        "vllm022": {str(b): vllm.get(b) for b in BATCHES}}
json.dump(data, open("chart_v9_h2h_data.json", "w"), indent=2)

# ---- plot 2x2 ----
PANELS = [
    ("tok",   "Text throughput (tok/s)",   "tokens / s",       "%.0f", False),
    ("req_s", "Request throughput (req/s)", "requests / s",     "%.2f", False),
    ("ttft",  "TTFT p50 (ms)  — lower is better", "ms",    "%.0f", True),
    ("itl",   "ITL mean (ms)  — lower is better", "ms",    "%.1f", True),
]
xs = list(range(len(BATCHES)))  # even spacing; batch labels on ticks
fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
for ax, (key, title, ylab, fmt, lower_better) in zip(axes.flat, PANELS):
    mv = [mstar.get(b, {}).get(key) for b in BATCHES]
    vv = [vllm.get(b, {}).get(key) for b in BATCHES]
    ax.plot([x for x, y in zip(xs, mv) if y is not None],
            [y for y in mv if y is not None], color=BLUE, marker="o", markersize=7,
            markeredgecolor="white", markeredgewidth=0.8, label="M* (E1)", zorder=3)
    ax.plot([x for x, y in zip(xs, vv) if y is not None],
            [y for y in vv if y is not None], color=GREEN, marker="s", markersize=7,
            markeredgecolor="white", markeredgewidth=0.8, label="vLLM-Omni 0.22", zorder=3)
    ax.set_title(title)
    ax.set_ylabel(ylab)
    ax.set_xlabel("batch size (concurrency)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in BATCHES])
    ax.margins(y=0.20)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    ymax = ax.get_ylim()[1]
    # collision-aware direct labels: near-floor labels float above; close pairs
    # (within 10%) nudge apart horizontally (M* left, vLLM right).
    for x, (a, b) in enumerate(zip(mv, vv)):
        close = (a is not None and b is not None
                 and abs(a - b) < 0.10 * max(a, b, 1e-9))
        for y, color, is_m in ((a, BLUE, True), (b, GREEN, False)):
            if y is None:
                continue
            above = is_m or (y < 0.14 * ymax) or (not is_m and b is not None and a is not None and b >= a)
            dy = 8 if above else -14
            dx = 0.0
            if close:
                dx = -0.10 if is_m else 0.10
                above = is_m  # M* above, vLLM below to separate
                dy = 9 if above else -15
            ax.annotate(fmt % y, (x, y), textcoords="offset points",
                        xytext=(dx * 30, dy), ha="center",
                        va="bottom" if dy > 0 else "top",
                        fontsize=7.5, color=color, zorder=4)

fig.suptitle("Qwen3-Omni  I2T (image → text):  M* (E1) vs vLLM-Omni 0.22  — warmed, all batches",
             fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = "charts/v9_h2h_i2t_2x2.png"
os.makedirs("charts", exist_ok=True)
fig.savefig(out)
print("wrote", out)
print(json.dumps(data["mstar"], indent=0))
print(json.dumps(data["vllm022"], indent=0))
