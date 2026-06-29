#!/usr/bin/env python3
"""Generate TTFT A/B charts from raw.json for the vision CUDA-graph alignment bench.

Reads raw.json (one entry per (variant, batch_size, metric)) and writes charts/.
All styling comes from ../../chartstyle.mplstyle (shared, off main).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = os.path.join(HERE, "..", "chartstyle.mplstyle")
plt.style.use(STYLE)

# Fixed per-variant color + label mapping, reused across every chart.
VARIANT_COLOR = {"baseline": "#4C72B0", "aligned": "#55A868"}
VARIANT_LABEL = {"baseline": "Baseline (pow2 buckets)",
                 "aligned": "Aligned (MSTAR_VISION_GRAPH_ALIGN=1)"}


def load():
    with open(os.path.join(HERE, "raw.json")) as f:
        return json.load(f)


def ttft_grouped_bar(data):
    dps = data["datapoints"]
    batch_sizes = sorted({d["batch_size"] for d in dps})
    variants = ["baseline", "aligned"]
    fig, ax = plt.subplots()
    width = 0.38
    for vi, v in enumerate(variants):
        xs, ys = [], []
        for bi, bs in enumerate(batch_sizes):
            m = [d for d in dps if d["variant"] == v and d["batch_size"] == bs
                 and d["metric"] == "ttft" and d["stat"] == "mean"]
            if not m:
                continue
            xs.append(bi + (vi - 0.5) * width)
            ys.append(m[0]["value"] * 1000.0)
        ax.bar(xs, ys, width, label=VARIANT_LABEL[v], color=VARIANT_COLOR[v])
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels([f"B={b}" for b in batch_sizes])
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("I2T TTFT — vision prefill CUDA-graph bucket alignment")
    ax.legend()
    fig.savefig(os.path.join(HERE, "charts", "ttft_ab.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "charts"), exist_ok=True)
    data = load()
    ttft_grouped_bar(data)
    print("charts written to", os.path.join(HERE, "charts"))
