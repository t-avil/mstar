#!/usr/bin/env python3
"""Render comparison charts from data/*.json (see extract_data.py). 2x2 grids.

 text  (s2t,i2t): req/s | tok/s | TTFT(text) p50 | ITL(text) mean
 speech(s2s,i2s): req/s | RTF p50 | TTFT(audio) p50 | ITL(audio) mean

Series/style (fixed by instruction — do not change without one):
  M* new              solid BLUE   = mstar_v3 cells where measured else
                                     mstar_v2 (canonical); latency from v2.
  M* new (previous)   dotted BLUE  = mstar_new
  M* old              dash-dot GREY
  vLLM-Omni (v0.22)   solid GREEN
  vLLM-Omni (v0.21)   dotted GREEN
Missing/anomalous points are silently omitted — no markers, no footnotes.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
plt.style.use("../benchmarks/chartstyle.mplstyle")

BLUE, GREY, GREEN = "#1f77b4", "#8a8f94", "#2ca02c"
PATHS = ["audio_to_text", "image_to_text", "audio_to_speech", "image_to_speech"]
SHORT = {"audio_to_text": "s2t", "image_to_text": "i2t",
         "audio_to_speech": "s2s", "image_to_speech": "i2s"}
BATCHES = [1, 2, 4, 8, 16, 32]

D = {n: json.load(open(f"data/{n}.json"))
     for n in ["mstar_v3", "mstar_v2", "mstar_new", "mstar_old", "vllm_022", "vllm_021"]}


def merged_current(path):
    out = {}
    v2, v3 = D["mstar_v2"].get(path, {}), D["mstar_v3"].get(path, {})
    for b in BATCHES:
        c2, c3 = v2.get(f"B{b}", {}), v3.get(f"B{b}", {})
        out[b] = {
            "req_s": c3.get("req_s") or c2.get("req_s"),
            "tok_s": c3.get("tok_s") or c2.get("tok_s"),
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
        ("M* old", plain("mstar_old"), GREY, "-.", 1.4),
        ("vLLM-Omni (v0.22)", plain("vllm_022"), GREEN, "-", 2.6),
        ("vLLM-Omni (v0.21)", plain("vllm_021"), GREEN, ":", 1.6),
    ]


def panel(ax, ss, mk, ylabel, title):
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
    ax.set_title(title, fontsize=11)


def main():
    os.makedirs("charts", exist_ok=True)
    for path in PATHS:
        speech = path.endswith("speech")
        ss = series(path)
        fig, ax = plt.subplots(2, 2, figsize=(9.6, 7.6))
        panel(ax[0, 0], ss, "req_s", "requests / s", "throughput (req/s)")
        if speech:
            panel(ax[0, 1], ss, "rtf_p50", "RTF p50",
                  "RTF p50 (lower = faster than realtime)")
            panel(ax[1, 0], ss, "ttft_p50", "s", "TTFT p50, audio stream (lower better)")
            panel(ax[1, 1], ss, "itl_mean", "s", "ITL mean, audio stream (lower better)")
        else:
            panel(ax[0, 1], ss, "tok_s", "tokens / s", "throughput (tok/s)")
            panel(ax[1, 0], ss, "ttft_p50", "s", "TTFT p50 (lower better)")
            panel(ax[1, 1], ss, "itl_mean", "s", "ITL mean (lower better)")
        ax[0, 0].legend(fontsize=8)
        fig.suptitle(f"{SHORT[path]} ({path})", fontsize=14)
        fig.tight_layout()
        out = f"charts/{path}.png"
        fig.savefig(out)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
