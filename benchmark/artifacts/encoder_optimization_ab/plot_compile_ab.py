#!/usr/bin/env python
"""torch.compile A/B chart: eager vs compile(dynamic=True) for native encoders.
Colors: eager=blue, compiled=red. No em dashes."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
b = [1, 4, 8, 16, 32]
data = {
    "VISION (ms/img)": {"eager": [14.93, 8.54, 11.86, 19.10, 33.98],
                        "compiled": [23.57, 8.33, 11.41, 18.72, 33.55], "rec": 14},
    "AUDIO (ms/req)":  {"eager": [14.69, 4.88, 5.17, 7.12, 11.77],
                        "compiled": [28.95, 8.00, 5.73, 7.50, 12.16], "rec": 16},
}
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (title, d) in zip(axes, data.items()):
    ax.plot(b, d["eager"], marker="o", color="#1f77b4", lw=2, ms=8, mfc="none", label="eager (baseline)")
    ax.plot(b, d["compiled"], marker="s", ls="--", color="#d62728", lw=2, ms=8, mfc="none", label="torch.compile(dynamic=True)")
    for x, e, c in zip(b, d["eager"], d["compiled"]):
        ax.annotate(f"{e/c:.2f}x", (x, max(e, c)), textcoords="offset points", xytext=(0, 6),
                    fontsize=8, ha="center", color="#444")
    ax.set_xscale("log", base=2); ax.set_xticks(b); ax.set_xticklabels(b)
    ax.set_xlabel("batch size"); ax.set_ylabel(title + ", lower is better")
    ax.set_title(f"{title.split()[0]}: {d['rec']} recompiles across shapes")
    ax.grid(True, which="both", ls=":", alpha=0.4); ax.legend(loc="upper left")
fig.suptitle("torch.compile A/B on native encoders (1x H200, no flash-attn): compile does NOT win\n"
             "marginal-or-slower steady state + 14-16 recompiles (the issue #131 shape-variance pitfall)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = "/workspace/autoresearch/bench_artifacts/optimization/torch_compile_ab.png"
fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
