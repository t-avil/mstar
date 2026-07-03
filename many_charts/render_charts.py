#!/usr/bin/env python3
"""Self-contained chart renderer for the many-charts branch.

Reads ONLY ./data/*.json (all committed on this branch). Per path, one
figure with every measured metric: req/s, TTFT p50, ITL mean, RTF p50
(speech only). ALL benchmarked batches drawn for every series.

Series/style (as instructed):
  M* new            solid  BLUE   = best current: mstar_v3 final-stack cells
                                    where measured, else mstar_v2 (canonical).
                                    Latency uses canonical v2 sweeps (v3
                                    latency at B32 is a closed-loop preview
                                    artifact; provenance in data/*.json).
  M* new (previous) dotted BLUE   = mstar_new (encoders-implemeneted 4c33b33)
  M* old            dashdot light BLUE = mstar_old
  vLLM-Omni         solid  GREEN  = vllm 0.22 rebenchmark
  vLLM-Omni (prev)  dotted GREEN  = 0.21-era, extracted from git history
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("../benchmarks/chartstyle.mplstyle")

BLUE, LBLUE, GREEN = "#1f77b4", "#7fb3d8", "#2ca02c"
PATHS = ["audio_to_text", "image_to_text", "audio_to_speech", "image_to_speech"]
SHORT = {"audio_to_text": "s2t", "image_to_text": "i2t",
         "audio_to_speech": "s2s", "image_to_speech": "i2s"}
BATCHES = [1, 2, 4, 8, 16, 32]

D = {name: json.load(open(f"data/{name}.json"))
     for name in ["mstar_v3", "mstar_v2", "mstar_new", "mstar_old", "vllm_022", "vllm_021"]}


def merged_current(path):
    """M* new (current): v3 req/s where measured, else v2; latency from v2."""
    out = {}
    v2, v3 = D["mstar_v2"].get(path, {}), D["mstar_v3"].get(path, {})
    for b in BATCHES:
        k = f"B{b}"
        c2, c3 = v2.get(k, {}), v3.get(k, {})
        out[b] = {
            "req_s": c3.get("req_s") or c2.get("req_s"),
            "ttft_p50": c2.get("ttft_p50") or c3.get("ttft_p50"),
            "itl_mean": c2.get("itl_mean") or c3.get("itl_mean"),
            "rtf_p50": c2.get("rtf_p50") or c3.get("rtf_p50"),
        }
    return out


def series(path):
    def plain(name):
        return {b: D[name].get(path, {}).get(f"B{b}", {}) for b in BATCHES}
    return [
        ("M* new", merged_current(path), BLUE, "-", 2.6),
        ("M* new (previous)", plain("mstar_new"), BLUE, ":", 1.6),
        ("M* old", plain("mstar_old"), LBLUE, "-.", 1.4),
        ("vLLM-Omni", plain("vllm_022"), GREEN, "-", 2.6),
        ("vLLM-Omni (previous)", plain("vllm_021"), GREEN, ":", 1.6),
    ]


METRICS = [("req_s", "requests / s", "throughput"),
           ("ttft_p50", "TTFT p50 (s)", "TTFT p50 (lower is better)"),
           ("itl_mean", "ITL mean (s)", "ITL mean (lower is better)"),
           ("rtf_p50", "RTF p50", "RTF p50 (lower = faster than realtime)")]


def main():
    for path in PATHS:
        speech = path.endswith("speech")
        metrics = METRICS if speech else METRICS[:3]
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.8))
        ss = series(path)
        for ax, (mk, ylabel, title) in zip(axes, metrics):
            for label, cells, color, ls, lw in ss:
                xs = [b for b in BATCHES if cells.get(b, {}).get(mk) is not None]
                ys = [cells[b][mk] for b in xs]
                if xs:
                    ax.plot(xs, ys, ls, color=color, linewidth=lw,
                            marker="o", markersize=4.5, label=label)
            ax.set_xscale("log", base=2)
            ax.set_xticks(BATCHES)
            ax.set_xticklabels([str(b) for b in BATCHES])
            ax.set_xlabel("concurrency (closed loop)")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=10)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{SHORT[path]} ({path})", y=1.03, fontsize=13)
        fig.tight_layout()
        out = f"charts/{path}.png"
        fig.savefig(out)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
