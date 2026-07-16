#!/usr/bin/env python3
"""vLLM-Omni 0.22 vs 0.24 — same style as gen_v9_h2h_charts.py.
Two green lines per panel: 0.22 = SOLID green, 0.24 = DASHED green.
2x2 grid per path (i2t, s2t): tok/s | req/s | TTFT p50 [ms] | ITL mean [ms].
0.22 = committed chart data; 0.24 = freshly measured (chart_vllm_024_data.json)."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
plt.style.use("../chartstyle.mplstyle")
GREEN = "#2ca02c"
BATCHES = [1, 2, 4, 8, 16, 32]
xs = list(range(len(BATCHES)))

V024 = json.load(open("chart_vllm_024_data.json"))
SRC022 = {
    "i2t": json.load(open("chart_v9_h2h_data.json"))["vllm022"],
    "s2t": json.load(open("chart_v9_h2h_s2t_data.json"))["vllm022"],
}
PANELS = [
    ("tok",   "Text throughput (tok/s)",   "tokens / s",   "%.0f", False),
    ("req_s", "Request throughput (req/s)", "requests / s", "%.2f", False),
    ("ttft",  "TTFT p50 (ms)  — lower is better", "ms",     "%.0f", True),
    ("itl",   "ITL mean (ms)  — lower is better", "ms",     "%.1f", True),
]

def series(d, key):
    return [d.get(str(b), {}).get(key) for b in BATCHES]

for tag, title in (("i2t", "I2T (image → text)"), ("s2t", "S2T (speech → text)")):
    v22, v24 = SRC022[tag], V024.get(tag, {})
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    for ax, (key, ptitle, ylab, fmt, lower_better) in zip(axes.flat, PANELS):
        a = series(v22, key); b = series(v24, key)
        ax.plot([x for x, y in zip(xs, a) if y is not None], [y for y in a if y is not None],
                color=GREEN, linestyle="-", marker="s", markersize=7, linewidth=2,
                markeredgecolor="white", markeredgewidth=0.8, label="vLLM-Omni 0.22", zorder=3)
        ax.plot([x for x, y in zip(xs, b) if y is not None], [y for y in b if y is not None],
                color=GREEN, linestyle="--", marker="^", markersize=7, linewidth=2,
                markeredgecolor="white", markeredgewidth=0.8, label="vLLM-Omni 0.24", zorder=3)
        ax.set_title(ptitle); ax.set_ylabel(ylab); ax.set_xlabel("batch size (concurrency)")
        ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in BATCHES])
        ax.margins(y=0.20); ax.set_ylim(bottom=0); ax.legend(loc="upper left")
        for x, (ya, yb) in enumerate(zip(a, b)):
            for y, dy in ((ya, 8), (yb, -14)):
                if y is None: continue
                ax.annotate(fmt % y, (x, y), textcoords="offset points", xytext=(0, dy),
                            ha="center", va="bottom" if dy > 0 else "top", fontsize=7.5,
                            color=GREEN, zorder=4)
    fig.suptitle(f"Qwen3-Omni  {title}:  vLLM-Omni 0.22 (solid) vs 0.24 (dashed)  — warmed",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs("charts", exist_ok=True)
    out = f"charts/vllm_022_vs_024_{tag}_2x2.png"
    fig.savefig(out); print("wrote", out)
