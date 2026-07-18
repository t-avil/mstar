#!/usr/bin/env python3
"""TEXT 2x2 per path (i2t, s2t), ALL NATURAL-EOS (one comparable regime, no
length confound). Shows the dual-goal text result:
  M* current (this session)   solid  blue  circle   <- exp_nateos_out/<LABEL>/
  M* E1/v9   (older ref)       dashed blue  tri-down <- committed chart_v9_h2h[_s2t]
  M* +encoders (older ref)     dotted blue  diamond  <- committed chart_encoders_impl
  vLLM-Omni 0.22 (committed)   solid  green square   <- committed chart_v9_h2h[_s2t]
Panels: tok/s | req/s | TTFT p50 [ms] | ITL mean [ms].
Usage: gen_text_nateos.py <LABEL>   (default encoff)
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
REF = "/m-coriander/coriander/tim/campaign_i2t/final_sweep"
RT = {"i2t": "image_to_text", "s2t": "audio_to_text"}

def g(dd, *path, default=None):
    for k in path:
        if not isinstance(dd, dict) or dd.get(k) is None:
            return default
        dd = dd[k]
    return dd

def load_sweep(tag):
    out = {}
    for B in BATCHES:
        f = os.path.join(SWEEP, f"{RT[tag]}_b{B}", "results.json")
        if not os.path.exists(f): continue
        r = json.load(open(f))
        ttft = g(r, "ttft", "text", "p50"); itl = g(r, "itl", "text", "mean")
        out[str(B)] = {"req_s": g(r, "request_throughput"), "tok": g(r, "text_token_throughput"),
                       "ttft": ttft*1000.0 if (ttft is not None and ttft < 5) else ttft,
                       "itl":  itl*1000.0 if (itl is not None and itl < 2) else itl}
    return out

NEW = {t: load_sweep(t) for t in ("i2t", "s2t")}
_v9  = json.load(open(f"{REF}/chart_v9_h2h_data.json"))
_v9s = json.load(open(f"{REF}/chart_v9_h2h_s2t_data.json"))
_enc = json.load(open(f"{REF}/chart_encoders_impl_data.json"))
E1   = {"i2t": _v9["mstar"],   "s2t": _v9s["mstar"]}
V022 = {"i2t": _v9["vllm022"], "s2t": _v9s["vllm022"]}
ENC  = {"i2t": _enc["i2t"],    "s2t": _enc["s2t"]}

PANELS = [("tok","Text throughput (tok/s)  — higher is better","tokens / s"),
          ("req_s","Request throughput (req/s)  — higher is better","requests / s"),
          ("ttft","TTFT p50 (ms)  — lower is better","ms"),
          ("itl","ITL mean (ms)  — lower is better","ms")]

def lines(tag):
    return [
        (NEW[tag],  BLUE,  "-",  "o", f"M* current ({LABEL}, this session)"),
        (E1[tag],   BLUE,  "--", "v", "M* E1/v9 (older ref)"),
        (ENC[tag],  BLUE,  ":",  "D", "M* +encoders (older ref)"),
        (V022[tag], GREEN, "-",  "s", "vLLM-Omni 0.22 (committed)"),
    ]

def series(d, key):
    return [d.get(str(b), {}).get(key) for b in BATCHES]

for tag, title in (("i2t","I2T (image → text)"), ("s2t","S2T (speech → text)")):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    for ax, (key, ptitle, ylab) in zip(axes.flat, PANELS):
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
    fig.suptitle(f"Qwen3-Omni  {title}  —  ALL natural-EOS (one regime): "
                 f"M* current (blue solid) vs older M* (blue dashed/dotted) vs vLLM-Omni 0.22 (green)",
                 fontsize=12, y=0.995)
    fig.text(0.5, 0.945, "Text tok/s parity goal: blue-solid ≥ blue dashed/dotted (older M*) at every batch. "
             "All series natural-EOS → directly comparable, no length confound.",
             ha="center", fontsize=8, color="#888888")
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    out = f"charts/text_nateos_{tag}_{LABEL}.png"; fig.savefig(out); print("wrote", out)
