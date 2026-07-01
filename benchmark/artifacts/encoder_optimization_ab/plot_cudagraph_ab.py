import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
b = [1, 4, 8]
eager = [19.91, 76.62, 267.71]; graph = [8.76, 75.03, 265.78]
sp = [e/g for e, g in zip(eager, graph)]
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.bar([x-0.15 for x in range(len(b))], sp, 0.3, color=["#2ca02c","#9e9e9e","#9e9e9e"], label="CUDA-graph speedup (block-loop)")
ax.axhline(1.0, color="#d62728", ls="--", lw=1.5, label="no speedup (1.0x)")
for i, s in enumerate(sp):
    ax.annotate(f"{s:.2f}x", (i-0.15, s), textcoords="offset points", xytext=(0,4), ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(range(len(b))); ax.set_xticklabels([f"bs{x}" for x in b])
ax.set_ylabel("CUDA-graph speedup on encoder block-loop (x)")
ax.set_ylim(0, 2.6)
ax.set_title("MEASURED CUDA-graph A/B (vision encoder block-loop, fixed shape)\n"
             "Helps only at batch 1 (2.3x, launch-bound); vanishes at batch>=4 (compute-bound)", fontsize=11)
ax.legend(loc="upper right")
ax.text(0.5, -0.16, "Idealized best case: mask precomputed in static buffer (capture forbids the real data-dependent build); "
        "math SDPA for capture-safety. Real path needs the piecewise+FlashInfer rewrite; variable shapes add padding waste.",
        transform=ax.transAxes, ha="center", fontsize=7.5, color="#555", wrap=True)
fig.tight_layout()
out = "/workspace/autoresearch/bench_artifacts/optimization/cudagraph_ab.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
