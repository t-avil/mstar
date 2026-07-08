#!/usr/bin/env python3
"""V9 head-to-head S2T chart — identical style to gen_v9_h2h_charts.py (i2t):
M* (blue circle) vs vLLM-Omni 0.22 (green square), 2x2 grid, direct-labeled.
Panels: tok/s | req/s | TTFT p50 [ms] | ITL mean [ms].
Reads chart_v9_h2h_s2t_data.json (M* = current build, this session; vLLM = committed
raw_audio_to_text.json aggregates). Same plotting code as the i2t generator so the two
figures are visually consistent.
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
plt.style.use("../chartstyle.mplstyle")
BLUE, GREEN = "#1f77b4", "#2ca02c"
BATCHES = [1, 2, 4, 8, 16, 32]
D = json.load(open("chart_v9_h2h_s2t_data.json"))
mstar = {int(k): v for k, v in D["mstar"].items()}
vllm = {int(k): v for k, v in D["vllm022"].items()}

PANELS = [
    ("tok",   "Text throughput (tok/s)",   "tokens / s",   "%.0f", False),
    ("req_s", "Request throughput (req/s)", "requests / s", "%.2f", False),
    ("ttft",  "TTFT p50 (ms)  — lower is better", "ms",     "%.0f", True),
    ("itl",   "ITL mean (ms)  — lower is better", "ms",     "%.1f", True),
]
xs = list(range(len(BATCHES)))
fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
for ax, (key, title, ylab, fmt, lower_better) in zip(axes.flat, PANELS):
    mv = [mstar.get(b, {}).get(key) for b in BATCHES]
    vv = [vllm.get(b, {}).get(key) for b in BATCHES]
    ax.plot([x for x, y in zip(xs, mv) if y is not None], [y for y in mv if y is not None],
            color=BLUE, marker="o", markersize=7, markeredgecolor="white", markeredgewidth=0.8,
            label="M* (current)", zorder=3)
    ax.plot([x for x, y in zip(xs, vv) if y is not None], [y for y in vv if y is not None],
            color=GREEN, marker="s", markersize=7, markeredgecolor="white", markeredgewidth=0.8,
            label="vLLM-Omni 0.22", zorder=3)
    ax.set_title(title); ax.set_ylabel(ylab); ax.set_xlabel("batch size (concurrency)")
    ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BATCHES])
    ax.margins(y=0.20); ax.set_ylim(bottom=0); ax.legend(loc="upper left")
    ymax = ax.get_ylim()[1]
    for x, (a, b) in enumerate(zip(mv, vv)):
        close = (a is not None and b is not None and abs(a - b) < 0.10 * max(a, b, 1e-9))
        for y, color, is_m in ((a, BLUE, True), (b, GREEN, False)):
            if y is None: continue
            above = is_m or (y < 0.14 * ymax) or (not is_m and b is not None and a is not None and b >= a)
            dy = 8 if above else -14; dx = 0.0
            if close:
                dx = -0.10 if is_m else 0.10; above = is_m; dy = 9 if above else -15
            ax.annotate(fmt % y, (x, y), textcoords="offset points", xytext=(dx * 30, dy),
                        ha="center", va="bottom" if dy > 0 else "top", fontsize=7.5, color=color, zorder=4)
fig.suptitle("Qwen3-Omni  S2T (speech → text):  M* (current) vs vLLM-Omni 0.22  — warmed, all batches",
             fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
os.makedirs("charts", exist_ok=True); fig.savefig("charts/v9_h2h_s2t_2x2.png")
print("wrote charts/v9_h2h_s2t_2x2.png")
