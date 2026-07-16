#!/usr/bin/env python3
"""All-stacks 2x2 per path (i2t, s2t) — same style as gen_v9_h2h_charts.py.
Four lines:
  vLLM-Omni 0.22   solid green   (committed)
  vLLM-Omni 0.24   dashed green  (freshly measured, chart_vllm_024_data.json)
  M* encoders-impl dashed blue   (committed baseline = mstar_old raw aggregates)
  M* newest (best) solid blue    (committed E1 = chart_v9_h2h[_s2t]_data.json)
Panels: tok/s | req/s | TTFT p50 [ms] | ITL mean [ms]."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
plt.style.use("../chartstyle.mplstyle")
GREEN, BLUE = "#2ca02c", "#1f77b4"
BATCHES = [1, 2, 4, 8, 16, 32]
xs = list(range(len(BATCHES)))

V024 = json.load(open("chart_vllm_024_data.json"))
V022 = {"i2t": json.load(open("chart_v9_h2h_data.json"))["vllm022"],
        "s2t": json.load(open("chart_v9_h2h_s2t_data.json"))["vllm022"]}
NEW  = {"i2t": json.load(open("chart_v9_h2h_data.json"))["mstar"],
        "s2t": json.load(open("chart_v9_h2h_s2t_data.json"))["mstar"]}

# encoders-implemented = AUTHORITATIVE M*-new (1f66ce6, native encoders) parsed from the
# encoders-implemeneted-benchmarked branch NUMBERS.md (parse_encoders_numbers.py).
_ENC = json.load(open("chart_encoders_impl_data.json"))
BASE = {"i2t": _ENC["i2t"], "s2t": _ENC["s2t"]}

PANELS = [
    ("tok",   "Text throughput (tok/s)",   "tokens / s",   "%.0f"),
    ("req_s", "Request throughput (req/s)", "requests / s", "%.2f"),
    ("ttft",  "TTFT p50 (ms)  — lower is better", "ms",     "%.0f"),
    ("itl",   "ITL mean (ms)  — lower is better", "ms",     "%.1f"),
]
# (data, color, linestyle, marker, label)
def lines(tag):
    return [
        (NEW[tag],  BLUE,  "-",  "o", "M* newest (best)"),
        (BASE[tag], BLUE,  "--", "v", "M* encoders-implemented"),
        (V022[tag], GREEN, "-",  "s", "vLLM-Omni 0.22"),
        (V024.get(tag, {}), GREEN, "--", "^", "vLLM-Omni 0.24"),
    ]

def series(d, key):
    return [d.get(str(b), {}).get(key) for b in BATCHES]

for tag, title in (("i2t", "I2T (image → text)"), ("s2t", "S2T (speech → text)")):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    for ax, (key, ptitle, ylab, fmt) in zip(axes.flat, PANELS):
        for d, color, ls, mk, lbl in lines(tag):
            ys = series(d, key)
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not pts: continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linestyle=ls,
                    marker=mk, markersize=6.5, linewidth=2, markeredgecolor="white",
                    markeredgewidth=0.7, label=lbl, zorder=3)
        ax.set_title(ptitle); ax.set_ylabel(ylab); ax.set_xlabel("batch size (concurrency)")
        ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BATCHES])
        ax.margins(y=0.15); ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle(f"Qwen3-Omni  {title}:  M* (blue) vs vLLM-Omni (green) — newest/0.22 solid, "
                 f"encoders/0.24 dashed", fontsize=12.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs("charts", exist_ok=True)
    out = f"charts/allstacks_{tag}_2x2.png"
    fig.savefig(out); print("wrote", out)
