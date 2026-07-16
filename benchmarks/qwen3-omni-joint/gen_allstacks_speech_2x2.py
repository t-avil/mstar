#!/usr/bin/env python3
"""All-stacks 2x2 for SPEECH-output paths (i2s, s2s) — same style family.
Speech metrics: req/s | audio throughput (audio-sec/s) | RTF p50 (lower better) | TTFT p50.
Lines: M* (committed mstar_new) solid blue ; vLLM-Omni 0.22 solid green (committed) ;
vLLM-Omni 0.24 dashed green (fresh full sweep).
NOTE: committed speech data has ONE clean M* build (mstar_new); mstar_old is degenerate,
so there is no clean encoders-vs-newest split for speech — the blue line is that build."""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
plt.style.use("../chartstyle.mplstyle")
GREEN, BLUE = "#2ca02c", "#1f77b4"
BATCHES = [1, 2, 4, 8, 16, 32]; xs = list(range(len(BATCHES)))
V024 = json.load(open("chart_vllm_024_data.json"))
ENC = json.load(open("chart_encoders_impl_data.json"))   # M* encoders-implemented (branch)
NEW = json.load(open("chart_mstar_best_speech_data.json"))  # M* newest/best (godv9 E1)
RAW = {"i2s": "raw_image_to_speech.json", "s2s": "raw_audio_to_speech.json"}

def load_raw(path, system):
    d = json.load(open(path))["aggregates"]; out = {}
    for B in BATCHES:
        s = d.get(f"B{B}", {}).get(system, {}); rc = s.get("recomputed", {}); hn = s.get("harness", {})
        # speech TTFT = time-to-first-AUDIO (ttft_audio), consistent across all systems
        ttft = (hn.get("ttft_audio") or {}).get("p50")
        out[str(B)] = {"req_s": rc.get("request_throughput"),
                       "audio_thr": rc.get("audio_throughput"),
                       "rtf": rc.get("rtf_p50"),
                       "ttft": ttft * 1000.0 if ttft is not None else None}
    return out

PANELS = [
    ("req_s",     "Request throughput (req/s)",      "requests / s",  "%.2f"),
    ("audio_thr", "Audio throughput (audio-sec/s)",  "audio sec / s", "%.1f"),
    ("rtf",       "RTF p50  — lower is better",       "rtf",           "%.2f"),
    ("ttft",      "TTFT-audio p50 (ms)  — lower is better", "ms",      "%.0f"),
]
def series(d, key): return [d.get(str(b), {}).get(key) for b in BATCHES]

for tag, title in (("i2s", "I2S (image → speech)"), ("s2s", "S2S (speech → speech)")):
    enc   = ENC.get(tag, {})               # M* encoders-implemented (branch NUMBERS.md)
    new   = NEW.get(tag, {})               # M* newest/best (godv9 E1)
    v022  = load_raw(RAW[tag], "vllm")
    v024  = V024.get(tag, {})
    lines = [(new, BLUE, "-", "o", "M* newest (best)"),
             (enc, BLUE, "--", "v", "M* encoders-implemented"),
             (v022, GREEN, "-", "s", "vLLM-Omni 0.22"),
             (v024, GREEN, "--", "^", "vLLM-Omni 0.24")]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    for ax, (key, ptitle, ylab, fmt) in zip(axes.flat, PANELS):
        for d, color, ls, mk, lbl in lines:
            ys = series(d, key); pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if not pts: continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linestyle=ls,
                    marker=mk, markersize=6.5, linewidth=2, markeredgecolor="white",
                    markeredgewidth=0.7, label=lbl, zorder=3)
        ax.set_title(ptitle); ax.set_ylabel(ylab); ax.set_xlabel("batch size (concurrency)")
        ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BATCHES])
        ax.margins(y=0.15); ax.set_ylim(bottom=0); ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle(f"Qwen3-Omni  {title}:  M* (blue) vs vLLM-Omni (green) — newest/0.22 solid, "
                 f"encoders/0.24 dashed", fontsize=12.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs("charts", exist_ok=True)
    out = f"charts/allstacks_{tag}_2x2.png"; fig.savefig(out); print("wrote", out)
