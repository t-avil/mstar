#!/usr/bin/env python
"""Stable vision-encoder A/B chart: native forward ms/img vs batch, per varlen
backend, with error bars. Numbers from bench_encoder_fast.py (repeats=8)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

batches = [1, 4, 8, 16, 32]
# (mean, std) ms/img
data = {
    "dense (old baseline, O(n²) mask)": ([15.89, 8.59, 11.95, 19.11, 34.14], [1.28, 0.18, 0.19, 0.16, 0.13], "#c44", "x", ":"),
    "per_segment (shipped)":            ([13.60, 5.97, 5.07, 4.65, 4.48], [1.36, 0.32, 0.06, 0.03, 0.03], "#d1791f", "o", "-"),
    "padded_batch":                     ([17.44, 7.10, 6.13, 5.62, 5.46], [1.00, 0.21, 0.10, 0.03, 0.02], "#1f77b4", "^", "--"),
}
fig, ax = plt.subplots(figsize=(8.5, 5.2))
for label, (m, s, c, mk, ls) in data.items():
    ax.errorbar(batches, m, yerr=s, marker=mk, ls=ls, color=c, label=label,
                markersize=8, linewidth=2, capsize=4, markerfacecolor="none", markeredgewidth=1.8)
    ax.annotate(f"{m[-1]:.1f}", (batches[-1], m[-1]), textcoords="offset points",
                xytext=(8, 0), fontsize=9, color=c, va="center")
ax.set_xscale("log", base=2); ax.set_xticks(batches); ax.set_xticklabels(batches)
ax.set_xlabel("batch size (images encoded together)")
ax.set_ylabel("native vision-encoder forward (ms/img) — lower is better")
ax.set_title("Qwen3-Omni native VISION encoder: varlen-attention A/B (1×H200, no flash-attn, bf16)\n"
             "per-segment is flat & 7.6× faster than the dense baseline at bs32", fontsize=11)
ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend(loc="upper left")
ax.text(0.5, -0.13, "MEASURED: bench_encoder_fast.py, repeats=8, error bars = ±1σ  |  dense = pre-optimization baseline",
        transform=ax.transAxes, ha="center", fontsize=8, color="#555")
fig.tight_layout()
out = "/workspace/autoresearch/bench_artifacts/optimization/vision_encoder_ab.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
