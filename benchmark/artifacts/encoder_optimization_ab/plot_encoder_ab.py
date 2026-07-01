#!/usr/bin/env python
"""Parse bench_encoder_fast.py output and plot a 2-panel (vision, audio)
varlen-attention A/B chart. Colors: dense=red, per_segment=blue, adaptive=green.
No em dashes (per styling). Usage: python plot_encoder_ab.py <bench_log>"""
import sys, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = sys.argv[1] if len(sys.argv) > 1 else "/workspace/autoresearch/logs/enc_final.log"
txt = open(log, errors="ignore").read()
# strip CUDA assert spam
txt = "\n".join(l for l in txt.splitlines() if "vectorized_gather" not in l and "Assertion" not in l)

COLORS = {"dense": ("#d62728", "x", ":"), "per_segment": ("#1f77b4", "^", "--"),
          "adaptive": ("#2ca02c", "o", "-")}
LABELS = {"dense": "dense (old baseline)", "per_segment": "per_segment",
          "adaptive": "adaptive (shipped)"}


def parse_section(name):
    m = re.search(rf"=====\s*{name}.*?=====\n(.*?)(?:=====|\Z)", txt, re.S)
    if not m:
        return None, []
    head = re.search(rf"=====\s*({name}.*?)=====", txt).group(1).strip()
    rows = []
    for line in m.group(1).splitlines():
        nums = re.findall(r"([0-9.]+)(?:±[0-9.]+)?", line)
        parts = line.split()
        if parts and parts[0].isdigit():
            bs = int(parts[0])
            vals = re.findall(r"([0-9.]+)±([0-9.]+)", line)
            if len(vals) == 3:
                rows.append((bs, [(float(m_), float(s_)) for m_, s_ in vals]))
    return head, rows


fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
for ax, (sec, ylab) in zip(axes, [("VISION", "native vision encoder (ms/img)"),
                                   ("AUDIO", "native audio encoder (ms/req)")]):
    head, rows = parse_section(sec)
    if not rows:
        ax.text(0.5, 0.5, f"no {sec} data", ha="center", transform=ax.transAxes); continue
    batches = [r[0] for r in rows]
    for i, bk in enumerate(["dense", "per_segment", "adaptive"]):
        means = [r[1][i][0] for r in rows]
        stds = [r[1][i][1] for r in rows]
        c, mk, ls = COLORS[bk]
        ax.errorbar(batches, means, yerr=stds, marker=mk, ls=ls, color=c, label=LABELS[bk],
                    markersize=8, linewidth=2, capsize=4, markerfacecolor="none", markeredgewidth=1.8)
    ax.set_xscale("log", base=2); ax.set_xticks(batches); ax.set_xticklabels(batches)
    ax.set_xlabel("batch size")
    ax.set_ylabel(ylab + "  (lower is better)")
    ax.set_title(f"{sec.title()} encoder varlen A/B")
    ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend(loc="upper left")
fig.suptitle("Qwen3-Omni native encoder: adaptive varlen attention (1x H200, no flash-attn, bf16, repeats=8, error bars 1 sigma)\n"
             "vision: adaptive picks per-segment (flat, fast).  audio: adaptive picks dense at low batch, per-segment at high batch.",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = "/workspace/autoresearch/bench_artifacts/optimization/encoder_adaptive_ab.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
