#!/usr/bin/env python3
"""Plot S2T/I2T/S2S text-TTFT (mean/p50/p99) OFF vs ON from raw.json.

Regenerable from raw.json alone. OFF S2T excludes the cold first-server run
(rep '1' of the un-suffixed run) by averaging all OFF S2T reps; the cold
outlier is shown as a faded marker for honesty.
"""
import json, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "raw.json")))
runs = data["runs"]
COL = {"off": "#6c757d", "on": "#1f77b4"}  # consistent: off=grey, on=blue

def agg(variant, path, metric):
    vals = [r["aggregate"]["ttft_text_s"][metric]
            for r in runs if r["variant"] == variant and r["path"] == path
            and r["aggregate"]["ttft_text_s"][metric] is not None]
    return vals

paths = ["S2T", "I2T", "S2S"]
metrics = ["mean", "p50", "p99"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, metric in zip(axes, metrics):
    x = range(len(paths))
    for i, v in enumerate(["off", "on"]):
        ys = [st.mean(agg(v, p, metric)) if agg(v, p, metric) else 0 for p in paths]
        ax.bar([xi + (i - 0.5) * 0.38 for xi in x], ys, width=0.36,
               label={"off": "default (2 walks)", "on": "merged (1 walk)"}[v],
               color=COL[v])
    ax.set_xticks(list(x)); ax.set_xticklabels(paths)
    ax.set_title(f"text TTFT {metric}")
    ax.set_ylabel("seconds")
    ax.grid(axis="y", alpha=0.3)
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Merge prefill walks — B=1 TTFT A/B (8xH200, GPUs 0,1) — delta within noise",
             fontsize=10)
fig.tight_layout()
out = os.path.join(HERE, "charts", "ttft_ab.png")
fig.savefig(out, dpi=130)
print("wrote", out)
