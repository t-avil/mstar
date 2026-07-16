#!/usr/bin/env python3
"""Speech gate charts (2026-07-16): does PD hurt audio generation?
2x2 per path (req/s | TTFT audio | ITL audio | RTF), house style/colors.
Series: colocated (green square, the incumbent for speech), PD-base (red x),
PD-speech variant = talker on rank1 (blue circle).
Data: rm_out/speech_{pd,pdspeech,colo,colo2}/<cell>/results.json (eiv2 branch).
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
plt.style.use("../chartstyle.mplstyle")
BLUE, GREEN, RED = "#1f77b4", "#2ca02c", "#d62728"
RM = "/m-coriander/coriander/tim/rm_out"

def load(d):
    f = os.path.join(RM, d, "results.json")
    if not os.path.exists(f): return None
    r = json.load(open(f))
    if (r.get("completed") or 0) <= 0: return None
    a = lambda k1, k2: ((r.get(k1) or {}).get("audio") or {}).get(k2)
    rtf = r.get("rtf"); rtf = rtf.get("mean") if isinstance(rtf, dict) else None
    return dict(rps=r.get("request_throughput"), ttft=(a("ttft","p50") or 0)*1e3,
                itl=(a("itl","mean") or 0)*1e3, rtf=rtf)

def series(dirfmt, batches):
    out = {}
    for b in batches:
        m = load(dirfmt.format(B=b))
        if m: out[b] = m
    return out

def panel(ax, title, sers, key, ylab, lower_better=False):
    for label, (data, color, marker, ls) in sers.items():
        bs = sorted(data); ys = [data[b][key] for b in bs]
        if not bs: continue
        ax.plot(bs, ys, marker=marker, color=color, ls=ls, label=label)
        for b, y in zip(bs, ys):
            ax.annotate(f"{y:.2f}" if key=="rtf" else f"{y:.0f}" if key!="rps" else f"{y:.2f}",
                        (b, y), textcoords="offset points", xytext=(0, 7),
                        fontsize=8, color=color, ha="center")
    ax.set_title(title + ("  — lower is better" if lower_better else ""))
    ax.set_xscale("log", base=2); ax.set_xticks(sorted({b for d,_,_,_ in sers.values() for b in d}));
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("batch size (concurrency)"); ax.set_ylabel(ylab)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

# ---- s2s ----
s2s = {
    "colocated":  (series("speech_colo/s2s_B{B}", [1,8,32]), GREEN, "s", "-"),
    "PD-base":    (series("speech_pd/s2s_B{B}", [1,8,32]), RED, "x", "--"),
    "PD-speech (talker@rank1)": (series("speech_pdspeech/s2s_B{B}", [1,8,32]), BLUE, "o", "-"),
}
fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
fig.suptitle("Qwen3-Omni S2S (speech→speech): PD topologies vs colocated — eiv2, GPUs 6,7")
panel(axs[0][0], "Request throughput (req/s)", s2s, "rps", "requests / s")
panel(axs[0][1], "TTFT audio p50 (ms)", s2s, "ttft", "ms", True)
panel(axs[1][0], "ITL audio mean (ms)", s2s, "itl", "ms", True)
panel(axs[1][1], "RTF mean", s2s, "rtf", "rtf", True)
fig.tight_layout()
fig.savefig("charts/speech_gate_s2s_2x2.png", dpi=150)
print("wrote charts/speech_gate_s2s_2x2.png")

# ---- t2s ----
t2s = {
    "colocated":  (series("speech_colo2/t2s_B{B}", [1,8]), GREEN, "s", "-"),
    "PD-speech (talker@rank1)": (series("speech_pdspeech/t2s_B{B}", [1,8]), BLUE, "o", "-"),
}
fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))
fig.suptitle("Qwen3-Omni T2S (text→speech): PD-speech vs colocated — eiv2, GPUs 6,7")
panel(axs[0][0], "Request throughput (req/s)", t2s, "rps", "requests / s")
panel(axs[0][1], "TTFT audio p50 (ms)", t2s, "ttft", "ms", True)
panel(axs[1][0], "ITL audio mean (ms)", t2s, "itl", "ms", True)
panel(axs[1][1], "RTF mean", t2s, "rtf", "rtf", True)
fig.tight_layout()
fig.savefig("charts/speech_gate_t2s_2x2.png", dpi=150)
print("wrote charts/speech_gate_t2s_2x2.png")
