#!/usr/bin/env python3
"""FINAL 2x2 charts, all 4 paths, all natural-EOS, FRESH data (this campaign).
Series: winning (blue solid) | encoders-impl baseline (blue dotted) |
        m-star main (GREY) | vLLM-Omni 0.24 (green).
Text panels: tok/s | req/s | TTFT p50 | ITL mean.
Speech panels: audio-s/s | req/s | RTF p50 | req/s ratio winning/vLLM024.
Reads exp_nateos_out/{winning,encoders_impl,main,vllm024}/<rt>_b<B>/results.json.
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
plt.style.use("chartstyle.mplstyle")
BLUE, GREEN, GREY = "#1f77b4", "#2ca02c", "#7f7f7f"
BATCHES = [1, 2, 4, 8, 16, 32]; xs = list(range(len(BATCHES)))
E = "/m-coriander/coriander/tim/exp_nateos_out"
# (dir, color, linestyle, marker, label)
SERIES = [
    ("winning",       BLUE,  "-",  "o", "M* winning (this work)"),
    ("encoders_impl", BLUE,  ":",  "D", "M* encoders-impl (baseline)"),
    ("main",          GREY,  "-",  "^", "m-star main"),
    ("vllm024",       GREEN, "-",  "s", "vLLM-Omni 0.24"),
]

def val(label, rt, B, *path):
    f = f"{E}/{label}/{rt}_b{B}/results.json"
    if not os.path.exists(f): return None
    d = json.load(open(f))
    for k in path:
        if not isinstance(d, dict) or d.get(k) is None: return None
        d = d[k]
    return d

def scaled_ttft(label, rt, B):
    v = val(label, rt, B, "ttft", "text", "p50")
    return v*1000 if (v is not None and v < 5) else v
def scaled_itl(label, rt, B):
    v = val(label, rt, B, "itl", "text", "mean")
    return v*1000 if (v is not None and v < 2) else v

def line(ax, ys, color, ls, mk, lbl):
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linestyle=ls,
                marker=mk, markersize=6.5, linewidth=2, markeredgecolor="white",
                markeredgewidth=0.7, label=lbl, zorder=3)

def finish(ax, title, ylab):
    ax.set_title(title); ax.set_ylabel(ylab); ax.set_xlabel("batch size (concurrency)")
    ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BATCHES])
    ax.margins(y=0.15); ax.set_ylim(bottom=0); ax.legend(loc="upper left", fontsize=7.5)

# ---------- TEXT (i2t, s2t) ----------
for tag, rt, title in (("i2t","image_to_text","I2T (image → text)"),
                       ("s2t","audio_to_text","S2T (speech → text)")):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    a1, a2, a3, a4 = axes.flat
    for lab, c, ls, mk, name in SERIES:
        line(a1, [val(lab, rt, B, "text_token_throughput") for B in BATCHES], c, ls, mk, name)
        line(a2, [val(lab, rt, B, "request_throughput") for B in BATCHES], c, ls, mk, name)
        line(a3, [scaled_ttft(lab, rt, B) for B in BATCHES], c, ls, mk, name)
        line(a4, [scaled_itl(lab, rt, B) for B in BATCHES], c, ls, mk, name)
    finish(a1, "Text throughput (tok/s) — higher better", "tokens / s")
    finish(a2, "Request throughput (req/s) — higher better", "requests / s")
    finish(a3, "TTFT p50 (ms) — lower better", "ms")
    finish(a4, "ITL mean (ms) — lower better", "ms")
    fig.suptitle(f"Qwen3-Omni  {title}  — natural-EOS: M* winning (blue) vs baseline (dotted) "
                 f"vs m-star main (grey) vs vLLM-Omni 0.24 (green)", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = f"charts/final_{tag}.png"; fig.savefig(out, dpi=140); print("wrote", out)

# ---------- SPEECH (i2s, s2s) ----------
for tag, rt, title in (("i2s","image_to_speech","I2S (image → speech)"),
                       ("s2s","audio_to_speech","S2S (speech → speech)")):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    a1, a2, a3, a4 = axes.flat
    for lab, c, ls, mk, name in SERIES:
        line(a1, [val(lab, rt, B, "audio_seconds_throughput") for B in BATCHES], c, ls, mk, name)
        line(a2, [val(lab, rt, B, "request_throughput") for B in BATCHES], c, ls, mk, name)
        rtf = [val(lab, rt, B, "rtf", "p50") or val(lab, rt, B, "rtf", "mean") for B in BATCHES]
        line(a3, rtf, c, ls, mk, name)
    # 4th panel: req/s ratio winning / vLLM024 with 2x,3x guides
    ratio = []
    for B in BATCHES:
        w = val("winning", rt, B, "request_throughput"); v = val("vllm024", rt, B, "request_throughput")
        ratio.append(w/v if (w and v) else None)
    a4.bar([x for x, r in zip(xs, ratio) if r is not None],
           [r for r in ratio if r is not None], color=BLUE, alpha=0.75)
    a4.axhline(2.0, color="#d62728", ls="--", lw=1, label="2x"); a4.axhline(3.0, color="#d62728", ls=":", lw=1, label="3x")
    for x, r in zip(xs, ratio):
        if r is not None: a4.text(x, r+0.05, f"{r:.1f}x", ha="center", fontsize=8, color=BLUE)
    finish(a1, "Audio throughput (audio-s/s) — higher better", "audio s / s")
    finish(a2, "Request throughput (req/s) — higher better", "requests / s")
    finish(a3, "RTF p50 — lower better", "RTF")
    a4.set_title("Speech goal: winning req/s / vLLM-0.24"); a4.set_ylabel("x faster")
    a4.set_xticks(xs); a4.set_xticklabels([str(b) for b in BATCHES]); a4.set_xlabel("batch size"); a4.legend(fontsize=7.5)
    fig.suptitle(f"Qwen3-Omni  {title}  — natural-EOS: M* winning (blue) vs baseline (dotted) "
                 f"vs m-star main (grey) vs vLLM-Omni 0.24 (green)", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = f"charts/final_{tag}.png"; fig.savefig(out, dpi=140); print("wrote", out)
