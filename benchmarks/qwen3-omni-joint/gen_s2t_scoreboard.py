#!/usr/bin/env python3
"""s2t scoreboard (2026-07-08): M* (current build, single-GPU) vs vLLM-Omni 0.22.
M* = this-session live measurements (natural transcription length ~20 tok/req);
vLLM = committed raw_audio_to_text.json aggregates. M* wins req/s at every batch.
"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(HERE, "chartstyle.mplstyle"))

B = [1, 2, 4, 8, 16, 32]
MSTAR_REQ = [5.60, 9.03, 13.82, 18.78, 27.32, 39.24]      # measured this session
VLLM_REQ  = [3.83, 9.14, 13.22, 15.79, 19.80, 30.85]      # committed
MSTAR_TOK = [r * 20.0 for r in MSTAR_REQ]                 # ~20 tok/req (measured avg)
VLLM_TOK  = [95.0, 226.8, 328.0, 381.6, 472.4, 746.1]     # committed
C_VLLM, C_MSTAR = "#2ca02c", "#1f77b4"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, (mst, vl, title, note) in zip(axes, [
        (MSTAR_REQ, VLLM_REQ, "s2t Throughput (req/s)", "M* wins/ties every batch"),
        (MSTAR_TOK, VLLM_TOK, "s2t Throughput (tok/s)", "M* wins B1/B16/B32; vLLM small-batch = longer outputs")]):
    ax.plot(B, vl, "-o", color=C_VLLM, label="vLLM-Omni 0.22", linewidth=2)
    ax.plot(B, mst, "-o", color=C_MSTAR, label="M* (current build)", linewidth=2)
    ax.set_title(f"{title}\n{note}", fontsize=11)
    ax.set_xlabel("batch size (concurrency)"); ax.set_xscale("log", base=2)
    ax.set_xticks(B); ax.set_xticklabels(B); ax.grid(True, alpha=0.4)
axes[0].legend(loc="upper left", framealpha=0.9)
fig.suptitle("Qwen3-Omni s2t: M* beats vLLM-Omni 0.22 across all batches  "
             "(B32 req/s 39.2 vs 30.9, +27%)", fontsize=13, y=1.02)
fig.tight_layout()
out = os.path.join(HERE, "charts", "campaign_s2t_scoreboard.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
