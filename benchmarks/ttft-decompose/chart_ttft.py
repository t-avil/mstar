#!/usr/bin/env python3
"""Chart the B=1 TTFT decomposition from raw.json. Uses the shared workspace
mplstyle. Regenerable from raw.json alone.

Usage: chart_ttft.py <raw.json> <out_dir>
"""
import sys, json, os
from statistics import median
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW, OUTDIR = sys.argv[1], sys.argv[2]
os.makedirs(OUTDIR, exist_ok=True)
style = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chartstyle.mplstyle")
if os.path.exists(style):
    plt.style.use(style)

raw = json.load(open(RAW))
dp = raw["datapoints"]

# Non-overlapping top-level stages that sum to total_server_ttft.
STAGES = ["api_to_preproc", "preprocess_cpu", "admission_total",
          "encoder_plus_prefill_wall", "emit_to_client_total"]
LABELS = {
    "api_to_preproc": "API->preproc enqueue",
    "preprocess_cpu": "Preprocess (CPU: mel/img+tokenize)",
    "admission_total": "Admission/dispatch (conductor)",
    "encoder_plus_prefill_wall": "Encoder + Thinker prefill (GPU)",
    "emit_to_client_total": "First-token emit -> client (IPC)",
}
# one fixed colour per stage, reused across both paths
COLORS = {
    "api_to_preproc": "#9ecae1",
    "preprocess_cpu": "#fdae6b",
    "admission_total": "#74c476",
    "encoder_plus_prefill_wall": "#6a51a3",
    "emit_to_client_total": "#e6550d",
}

def med_stage(path, stage):
    vals = [d["stages_ms"][stage] for d in dp
            if d["path"] == path and d["phase"] == "measure"
            and d["stages_ms"].get(stage) is not None]
    return median(vals) if vals else 0.0

paths = ["S2T", "I2T"]
fig, ax = plt.subplots(figsize=(7.8, 3.2))
ypos = {"S2T": 1, "I2T": 0}
for path in paths:
    left = 0.0
    for s in STAGES:
        w = med_stage(path, s)
        ax.barh(ypos[path], w, left=left, color=COLORS[s], edgecolor="white",
                linewidth=0.7, label=LABELS[s] if path == "S2T" else None)
        if w > 6:
            ax.text(left + w / 2, ypos[path], f"{w:.0f}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        left += w
    ax.text(left + 2, ypos[path], f"total {left:.0f} ms", va="center", fontsize=9)

# vLLM reference TTFT (FINDINGS: ~0.118 s = 118 ms)
ax.axvline(118, color="#444444", ls="--", lw=1.2)
ax.text(118, 1.62, "vLLM TTFT 118ms", color="#444444", fontsize=8, ha="center")

ax.set_yticks([0, 1]); ax.set_yticklabels(["I2T\n(image->text)", "S2T\n(audio->text)"])
ax.set_xlabel("median TTFT contribution (ms), B=1, isolated single server")
ax.set_title("Qwen3-Omni M* — B=1 TTFT decomposition (8xH200, GPUs 0,1)")
ax.set_ylim(-0.6, 1.8)
ax.grid(axis="x", alpha=0.5); ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.55), ncol=2, frameon=False, fontsize=8)
fig.tight_layout()
out = os.path.join(OUTDIR, "ttft_decomposition.png")
fig.savefig(out)
print("wrote", out)

# Second panel: isolated vs co-located(fair-b1) vs heavy-contention, S2T TTFT
fig2, ax2 = plt.subplots(figsize=(5.2, 3.2))
conds = ["isolated\n(this run)", "co-located\n(fair-b1)", "heavy CPU\ncontention"]
vals = [109, 317, 8107]  # client p50 ms
colors = ["#74c476", "#fdae6b", "#de2d26"]
bars = ax2.bar(conds, vals, color=colors, edgecolor="white")
ax2.set_yscale("log")
ax2.set_ylabel("S2T TTFT p50 (ms, log)")
ax2.set_title("M* B=1 S2T TTFT vs CPU co-residency")
for b, v in zip(bars, vals):
    ax2.text(b.get_x() + b.get_width() / 2, v * 1.1, f"{v}", ha="center", fontsize=9)
ax2.axhline(118, color="#444", ls="--", lw=1); ax2.text(2.0, 130, "vLLM 118", fontsize=8, color="#444")
fig2.tight_layout()
out2 = os.path.join(OUTDIR, "ttft_vs_coresidency.png")
fig2.savefig(out2)
print("wrote", out2)
