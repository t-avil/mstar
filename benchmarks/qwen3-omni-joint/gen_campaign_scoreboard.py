#!/usr/bin/env python3
"""Campaign scoreboard (2026-07-08): M* vs vLLM-Omni 0.22, i2t, all batches.

Regenerable from the data dict below (committed vLLM 0.22 + M* E1 single-GPU
from chart_v9_h2h_data.json; M* DP+RR from this session's live measurements).
Uses the shared chartstyle.mplstyle. Green=vLLM, Blue=M* single-GPU, Teal=M* DP.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(HERE, "chartstyle.mplstyle"))

B = [1, 2, 4, 8, 16, 32]
# committed vLLM 0.22 (natural length ~212 tok) — chart_v9_h2h_data.json
VLLM = {"req_s":[0.876,1.514,2.189,3.665,5.623,8.322],
        "tok":[195,322,483,770,1185,1769],
        "ttft":[87,92,129,100,168,179],
        "itl":[4.8,5.6,6.9,9.6,11.9,15.4]}
# committed M* single-GPU E1 (natural ~175 tok) — chart_v9_h2h_data.json
MSTAR = {"req_s":[0.975,1.549,2.474,4.198,5.888,7.421],
         "tok":[190,282,447,722,1027,1310],
         "ttft":[196,306,428,486,912,2532],
         "itl":[4.3,4.9,5.2,6.1,7.1,9.3]}
# M* data-parallel (DP=2 + round-robin), this session (natural). B2/B4 not run -> None.
MSTAR_DP = {"req_s":[0.91,None,None,4.16,6.49,6.92],
            "tok":[171,None,None,774,1114,1181],
            "ttft":[248,None,None,791,1083,3139],
            "itl":[5.0,None,None,5.0,5.0,4.0]}

PANELS = [("req_s","Throughput (req/s)","higher is better"),
          ("tok","Throughput (tok/s)","higher is better"),
          ("ttft","TTFT (ms)","lower is better"),
          ("itl","ITL (ms)","lower is better")]
C_VLLM, C_MSTAR, C_DP = "#2ca02c", "#1f77b4", "#17becf"

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (key, title, hint) in zip(axes.flat, PANELS):
    ax.plot(B, VLLM[key], "-o", color=C_VLLM, label="vLLM-Omni 0.22", linewidth=2)
    ax.plot(B, MSTAR[key], "-o", color=C_MSTAR, label="M* (single-GPU)", linewidth=2)
    xd = [b for b, v in zip(B, MSTAR_DP[key]) if v is not None]
    yd = [v for v in MSTAR_DP[key] if v is not None]
    ax.plot(xd, yd, "--s", color=C_DP, label="M* (data-parallel)", linewidth=2, markersize=6)
    ax.set_title(f"{title}  ·  {hint}")
    ax.set_xlabel("batch size (concurrency)"); ax.set_xscale("log", base=2)
    ax.set_xticks(B); ax.set_xticklabels(B); ax.grid(True, alpha=0.4)
    if key in ("ttft",):
        ax.set_yscale("log")
axes.flat[0].legend(loc="upper left", framealpha=0.9)
fig.suptitle("Qwen3-Omni i2t: M* vs vLLM-Omni 0.22 (natural length)  —  M* wins B1-B16 + ITL; "
             "lone holdout = B32 req/s", fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = os.path.join(HERE, "charts", "campaign_i2t_scoreboard.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
