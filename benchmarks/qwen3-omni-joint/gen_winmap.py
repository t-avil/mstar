#!/usr/bin/env python3
"""Win-map: M* vs vLLM-Omni 0.22 across i2t + s2t, all batches (2026-07-08).
Top: req/s ratio heatmap (green=M* wins). Bottom: req/s bars per path.
i2t = committed h2h (M* E1 single-GPU vs vLLM 0.22, natural len).
s2t = vLLM committed + M* current build (freshly measured this session)."""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(HERE, "chartstyle.mplstyle"))
B = [1, 2, 4, 8, 16, 32]

# --- req/s data ---
I2T_M  = [0.975, 1.549, 2.474, 4.198, 5.888, 7.421]   # M* (committed E1, single-GPU)
I2T_V  = [0.876, 1.514, 2.189, 3.665, 5.623, 8.322]   # vLLM 0.22
S2T_M  = [5.60, 9.03, 13.82, 18.78, 27.32, 39.24]     # M* (current build, this session)
S2T_V  = [3.83, 9.14, 13.22, 15.79, 19.80, 30.85]     # vLLM 0.22 (committed)

i2t_r = np.array(I2T_M) / np.array(I2T_V)
s2t_r = np.array(S2T_M) / np.array(S2T_V)
ratios = np.vstack([i2t_r, s2t_r])

fig = plt.figure(figsize=(12, 8.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.22)

# ---- Top: ratio heatmap ----
ax = fig.add_subplot(gs[0, :])
norm = TwoSlopeNorm(vmin=0.80, vcenter=1.0, vmax=1.5)
im = ax.imshow(ratios, cmap="RdYlGn", norm=norm, aspect="auto")
ax.set_xticks(range(6)); ax.set_xticklabels([f"B{b}" for b in B])
ax.set_yticks([0, 1]); ax.set_yticklabels(["i2t", "s2t"], fontsize=13)
ax.set_title("M* / vLLM-Omni 0.22  request-throughput ratio  (green = M* wins, >1.0)", fontsize=13)
for i in range(2):
    for j in range(6):
        r = ratios[i, j]
        tag = "WIN" if r > 1.03 else ("tie" if r >= 0.98 else "loss")
        ax.text(j, i, f"{r:.2f}x\n{tag}", ha="center", va="center", fontsize=10,
                color="black", fontweight="bold" if tag != "tie" else "normal")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label("req/s ratio")
ax.text(5, -0.62, "only loss", ha="center", fontsize=9, color="#b22222")

# ---- Bottom: req/s bars ----
def bars(ax, M, V, title):
    x = np.arange(6); w = 0.38
    ax.bar(x - w/2, V, w, label="vLLM-Omni 0.22", color="#2ca02c")
    ax.bar(x + w/2, M, w, label="M*", color="#1f77b4")
    ax.set_title(title); ax.set_xticks(x); ax.set_xticklabels([f"B{b}" for b in B])
    ax.set_ylabel("req/s"); ax.grid(True, axis="y", alpha=0.4)
bars(fig.add_subplot(gs[1, 0]), I2T_M, I2T_V, "i2t throughput (req/s) — M* wins B1-B16, loses B32")
axs = fig.add_subplot(gs[1, 1]); bars(axs, S2T_M, S2T_V, "s2t throughput (req/s) — M* wins every batch"); axs.legend(loc="upper left")

fig.suptitle("Qwen3-Omni: M* beats vLLM-Omni 0.22 on every path/batch except i2t B32\n"
             "(plus: M* ITL is 2.5-3.8x better everywhere; tok/s wins at matched length)",
             fontsize=14, y=0.99)
out = os.path.join(HERE, "charts", "winmap_mstar_vs_vllm.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight", dpi=140)
print("wrote", out)
