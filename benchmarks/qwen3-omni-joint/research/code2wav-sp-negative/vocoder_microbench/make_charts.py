"""Charts for the Code2Wav vocoder microbenchmark (reads raw.json)."""
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = "/home/tim/exp_3way/chartstyle.mplstyle"
if os.path.exists(STYLE):
    plt.style.use(STYLE)

# Fixed per-variant color/label mapping (consistent across all figures).
VARIANTS = {
    "sp_off":    ("SP off (single GPU, compiled)", "#1f77b4"),
    "sp_single": ("SP on, 2 shards, 1 GPU",        "#ff7f0e"),
    "sp_xdev":   ("SP on, 2 shards, 2 GPUs",       "#2ca02c"),
}


def load():
    with open(os.path.join(HERE, "raw.json")) as f:
        return json.load(f)


def medians(dp):
    """(variant, chunk) -> median measure-phase ms."""
    acc = {}
    for d in dp:
        if d["phase"] != "measure":
            continue
        acc.setdefault((d["variant"], d["chunk_frames"]), []).append(d["value"])
    return {k: statistics.median(v) for k, v in acc.items()}


def main():
    data = load()
    dp = data["datapoints"]
    med = medians(dp)
    chunks = data["chunk_frames"]
    present = [v for v in VARIANTS if any(k[0] == v for k in med)]

    # Fig 1: median latency vs chunk size.
    fig, ax = plt.subplots()
    for v in present:
        ys = [med[(v, c)] for c in chunks]
        ax.plot(chunks, ys, marker="o", label=VARIANTS[v][0], color=VARIANTS[v][1])
    ax.axvline(50, ls="--", color="grey", lw=1)
    ax.text(52, ax.get_ylim()[1] * 0.95, "serving chunk\n(50 frames)",
            fontsize=8, va="top", color="grey")
    ax.set_xlabel("codec chunk size (frames)")
    ax.set_ylabel("vocoder forward latency (ms, median)")
    ax.set_title("Code2Wav vocoder: SP is slower at every chunk size")
    ax.legend()
    fig.savefig(os.path.join(HERE, "charts", "latency_vs_chunk.png"))
    plt.close(fig)

    # Fig 2: speedup (sp_off / variant); <1 means slower than baseline.
    fig, ax = plt.subplots()
    for v in present:
        if v == "sp_off":
            continue
        ys = [med[("sp_off", c)] / med[(v, c)] for c in chunks]
        ax.plot(chunks, ys, marker="s", label=VARIANTS[v][0], color=VARIANTS[v][1])
    ax.axhline(1.0, ls="--", color="black", lw=1, label="SP-off baseline (1.0)")
    ax.set_xlabel("codec chunk size (frames)")
    ax.set_ylabel("speedup vs SP-off  (>1 = faster)")
    ax.set_title("Code2Wav SP speedup vs single-GPU baseline (<1 = regression)")
    ax.set_ylim(0, 1.2)
    ax.legend()
    fig.savefig(os.path.join(HERE, "charts", "speedup_vs_chunk.png"))
    plt.close(fig)
    print("wrote charts/latency_vs_chunk.png charts/speedup_vs_chunk.png")


if __name__ == "__main__":
    main()
