#!/usr/bin/env python3
"""SPEECH 2x2 per path (i2s, s2s), ALL NATURAL-EOS. Shows the 2-3x speech story:
  M* current (this session)   solid  blue  circle  <- exp_nateos_out/<LABEL>/
  M* committed (2-3x era)      dashed blue  tri-down <- baselines.json mstar_new_speech
  vLLM-Omni 0.22 (committed)   solid  green square  <- baselines.json vllm022_speech
Panels: audio-s/s throughput | req/s | RTF p50 (lower=better) | req/s ratio annotation.
Usage: gen_speech_nateos.py <LABEL>   (default encoff)
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
plt.style.use("chartstyle.mplstyle")
GREEN, BLUE = "#2ca02c", "#1f77b4"
BATCHES = [1, 2, 4, 8, 16, 32]
xs = list(range(len(BATCHES)))
LABEL = sys.argv[1] if len(sys.argv) > 1 else "encoff"
SWEEP = f"/m-coriander/coriander/tim/exp_nateos_out/{LABEL}"
B = json.load(open("/m-coriander/coriander/tim/deliverable/baselines.json"))
RT = {"i2s": "image_to_speech", "s2s": "audio_to_speech"}

def med(xs):
    xs = sorted(v for v in xs if v is not None)
    return xs[len(xs)//2] if xs else None

def load_sweep(tag):
    out = {}
    for b in BATCHES:
        f = os.path.join(SWEEP, f"{RT[tag]}_b{b}", "results.json")
        if not os.path.exists(f): continue
        r = json.load(open(f))
        rtf = r.get("rtf")
        if isinstance(rtf, dict): rtf = rtf.get("p50") or rtf.get("mean")
        elif isinstance(rtf, list): rtf = med(rtf)
        out[str(b)] = {"aud": r.get("audio_seconds_throughput"),
                       "req_s": r.get("request_throughput"), "rtf": rtf}
    return out

def ref(tag, kind):  # kind: mstar_new / vllm022
    a = B["mstar_new_speech_natEOS" if kind=="mstar_new" else "vllm022_speech_natEOS"]
    r = B["rtf_p50_natEOS"][f"{tag}_{kind}"]
    return {str(b): {"aud": a[f"{tag}_audio_sps"][str(b)], "req_s": a[f"{tag}_req_s"][str(b)],
                     "rtf": r[str(b)]} for b in BATCHES}

NEW = {t: load_sweep(t) for t in ("i2s", "s2s")}
MC  = {t: ref(t, "mstar_new") for t in ("i2s", "s2s")}
V022= {t: ref(t, "vllm022")   for t in ("i2s", "s2s")}

PANELS = [("aud","Audio throughput (audio-s / s)  — higher is better","audio-s / s"),
          ("req_s","Request throughput (req/s)  — higher is better","requests / s"),
          ("rtf","RTF p50  — lower is better","RTF")]

def lines(tag):
    return [(NEW[tag], BLUE,"-","o",f"M* current ({LABEL}, this session)"),
            (MC[tag],  BLUE,"--","v","M* committed (2-3x era, natural-EOS)"),
            (V022[tag],GREEN,"-","s","vLLM-Omni 0.22 (committed)")]

def series(d, key):
    return [d.get(str(b), {}).get(key) for b in BATCHES]

for tag, title in (("i2s","I2S (image → speech)"), ("s2s","S2S (speech → speech)")):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    flat = axes.flatten()
    for ax, (key, ptitle, ylab) in zip(flat, PANELS):
        for d, color, ls, mk, lbl in lines(tag):
            ys = series(d, key)
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not pts: continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linestyle=ls,
                    marker=mk, markersize=6.5, linewidth=2, markeredgecolor="white",
                    markeredgewidth=0.7, label=lbl, zorder=3)
        ax.set_title(ptitle); ax.set_ylabel(ylab); ax.set_xlabel("batch size (concurrency)")
        ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BATCHES])
        ax.margins(y=0.15); ax.set_ylim(bottom=0); ax.legend(loc="upper left", fontsize=7.5)
    # 4th panel: req/s speedup ratio current-vs-vLLM
    ax = flat[3]
    rr = [ (NEW[tag].get(str(b),{}).get("req_s") or 0)/V022[tag][str(b)]["req_s"] for b in BATCHES]
    ax.bar(xs, rr, color=BLUE, alpha=0.75)
    ax.axhline(2.0, color="#d62728", ls="--", lw=1, label="2x line")
    ax.axhline(3.0, color="#d62728", ls=":", lw=1, label="3x line")
    for x,v in zip(xs,rr): ax.text(x, v+0.05, f"{v:.1f}x", ha="center", fontsize=8)
    ax.set_title("req/s speedup: M* current / vLLM-Omni 0.22"); ax.set_ylabel("x faster")
    ax.set_xlabel("batch size (concurrency)"); ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BATCHES])
    ax.legend(loc="upper left", fontsize=7.5); ax.set_ylim(bottom=0)
    fig.suptitle(f"Qwen3-Omni  {title}  —  natural-EOS: M* current (blue solid) vs "
                 f"committed 2-3x-era M* (dashed) vs vLLM-Omni 0.22 (green)", fontsize=12, y=0.995)
    fig.text(0.5, 0.945, "Speech goal: M* current req/s ≥ 2-3x vLLM (bottom-right). "
             "Blue-solid overlapping blue-dashed = the 2-3x era advantage preserved.",
             ha="center", fontsize=8, color="#888888")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    out = f"charts/speech_nateos_{tag}_{LABEL}.png"; fig.savefig(out); print("wrote", out)
