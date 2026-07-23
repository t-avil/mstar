#!/usr/bin/env python3
"""Regenerate the 5 Qwen3-Omni comparison charts from raw/.

Three series:
  m-star main (upstream)  -> grey, dotted
  this work (#131)        -> blue, solid
  vLLM-Omni 0.24          -> green, solid

Text panels: tok/s | req/s | TTFT p50 | ITL mean.
Speech panels: audio-s/s | req/s | RTF p50 | (this work / vLLM-0.24) req/s ratio.
Reads raw/<label>/<rt>_b<B>/results.json ; writes charts/final_*.png .
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot import)

HERE = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(HERE, "..", "..", "chartstyle.mplstyle"))

BLUE, GREEN, GREY = "#1f77b4", "#2ca02c", "#7f7f7f"
BATCHES = [1, 2, 4, 8, 16, 32]
xs = list(range(len(BATCHES)))
E = os.path.join(HERE, "..", "raw")

# (label, colour, linestyle, marker, legend) — draw order: main, this-work, vLLM
SERIES = [
    ("upmain", GREY, ":", "^", "m-star main (upstream)"),
    ("this-work", BLUE, "-", "o", "this work (#131)"),
    ("vllm024", GREEN, "-", "s", "vLLM-Omni 0.24"),
]


def val(label, rt, b, *path):
    f = f"{E}/{label}/{rt}_b{b}/results.json"
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        d = json.load(fh)
    for k in path:
        if not isinstance(d, dict) or d.get(k) is None:
            return None
        d = d[k]
    return d


def scaled_ttft(label, rt, b):
    v = val(label, rt, b, "ttft", "text", "p50")
    return v * 1000 if (v is not None and v < 5) else v


def scaled_itl(label, rt, b):
    v = val(label, rt, b, "itl", "text", "mean")
    return v * 1000 if (v is not None and v < 2) else v


def line(ax, ys, c, ls, mk, lbl):
    pts = [(x, y) for x, y in zip(xs, ys, strict=True) if y is not None]
    if not pts:
        return
    ax.plot(
        [p[0] for p in pts], [p[1] for p in pts], color=c, linestyle=ls,
        marker=mk, markersize=6.5, linewidth=2, markeredgecolor="white",
        markeredgewidth=0.7, label=lbl, zorder=3,
    )


def finish(ax, title, ylab):
    ax.set_title(title)
    ax.set_ylabel(ylab)
    ax.set_xlabel("batch size (concurrency)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in BATCHES])
    ax.margins(y=0.15)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=7.5)


def suptitle(fig, title):
    fig.suptitle(
        f"Qwen3-Omni  {title}  —  this work (blue) vs m-star main (grey) "
        "vs vLLM-Omni 0.24 (green)",
        fontsize=11.5, y=0.995,
    )


os.makedirs(os.path.join(HERE, "..", "charts"), exist_ok=True)

for tag, rt, title in (
    ("i2t", "image_to_text", "I2T (image → text)"),
    ("s2t", "audio_to_text", "S2T (speech → text)"),
):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    a1, a2, a3, a4 = axes.flat
    for lab, c, ls, mk, name in SERIES:
        line(a1, [val(lab, rt, b, "text_token_throughput") for b in BATCHES], c, ls, mk, name)
        line(a2, [val(lab, rt, b, "request_throughput") for b in BATCHES], c, ls, mk, name)
        line(a3, [scaled_ttft(lab, rt, b) for b in BATCHES], c, ls, mk, name)
        line(a4, [scaled_itl(lab, rt, b) for b in BATCHES], c, ls, mk, name)
    finish(a1, "Text throughput (tok/s) — higher better", "tokens / s")
    finish(a2, "Request throughput (req/s) — higher better", "requests / s")
    finish(a3, "TTFT p50 (ms) — lower better", "ms")
    finish(a4, "ITL mean (ms) — lower better", "ms")
    suptitle(fig, title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = os.path.join(HERE, "..", "charts", f"final_{tag}.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)

for tag, rt, title in (
    ("i2s", "image_to_speech", "I2S (image → speech)"),
    ("s2s", "audio_to_speech", "S2S (speech → speech)"),
    ("t2s", "text_to_speech", "T2S (text → speech)"),
):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    a1, a2, a3, a4 = axes.flat
    for lab, c, ls, mk, name in SERIES:
        line(a1, [val(lab, rt, b, "audio_seconds_throughput") for b in BATCHES], c, ls, mk, name)
        line(a2, [val(lab, rt, b, "request_throughput") for b in BATCHES], c, ls, mk, name)
        line(
            a3,
            [val(lab, rt, b, "rtf", "p50") or val(lab, rt, b, "rtf", "mean") for b in BATCHES],
            c, ls, mk, name,
        )
    ratio = []
    for b in BATCHES:
        w = val("this-work", rt, b, "request_throughput")
        v = val("vllm024", rt, b, "request_throughput")
        ratio.append(w / v if (w and v) else None)
    a4.bar(
        [x for x, r in zip(xs, ratio, strict=True) if r is not None],
        [r for r in ratio if r is not None], color=BLUE, alpha=0.75,
    )
    a4.axhline(2.0, color="#d62728", ls="--", lw=1, label="2x")
    a4.axhline(3.0, color="#d62728", ls=":", lw=1, label="3x")
    for x, r in zip(xs, ratio, strict=True):
        if r is not None:
            a4.text(x, r + 0.05, f"{r:.1f}x", ha="center", fontsize=8, color=BLUE)
    finish(a1, "Audio throughput (audio-s/s) — higher better", "audio s / s")
    finish(a2, "Request throughput (req/s) — higher better", "requests / s")
    finish(a3, "RTF p50 — lower better", "RTF")
    a4.set_title("Speech: this-work req/s / vLLM-0.24")
    a4.set_ylabel("x faster")
    a4.set_xticks(xs)
    a4.set_xticklabels([str(b) for b in BATCHES])
    a4.set_xlabel("batch size")
    a4.legend(fontsize=7.5)
    suptitle(fig, title)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = os.path.join(HERE, "..", "charts", f"final_{tag}.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)
