#!/usr/bin/env python3
"""make_charts.py — programmatic charts for the FINAL Qwen3-Omni harness (DRAFT).

Reuses the existing make_proof_charts.py logic. Reads ONLY raw_<path>.json
(whose `aggregates` are recomputed from `datapoints` by final_bench.py, so every
plotted point traces to recorded data) and renders one 2x2 panel per path using
the shared chartstyle.mplstyle. No hand-edited charts; regenerable from raw alone.

Difference from the committed make_proof_charts.py:
  - style path fixed to the canonical shared file
    /home/tim/bench-wt/benchmarks/chartstyle.mplstyle (the original pointed at a
    stale /home/tim/exp_3way/... path).
  - adds the fifth path T2S (text_to_speech), charted as a speech path.

Chart-style preference (unchanged from the project convention): ONE statistic per
panel (TTFT=p50, ITL=mean, RTF=p50, throughput=rate); NO error bars; missing or
anomalous points are silently omitted (not plotted), no markers/footnotes.
"""
import json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/tim/bench-wt/benchmarks/qwen3-omni-joint"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "charts")
STYLE = "/home/tim/bench-wt/benchmarks/chartstyle.mplstyle"   # canonical shared style
if os.path.exists(STYLE):
    try:
        plt.style.use(STYLE)
    except Exception:
        pass
os.makedirs(OUT, exist_ok=True)

BATCHES = [1, 2, 4, 8, 16, 32]
# Fixed per-system color/marker/label mapping — same everywhere for comparability.
SYS = [("mstar_new", "M*-new (integrated)", "#1f77b4", "o", "-"),
       ("mstar_old", "M*-old (HF)",         "#7f7f7f", "s", "--"),
       ("vllm",      "vLLM-Omni",           "#2ca02c", "^", "-.")]
# All FIVE paths (raw filename -> title, modality). T2S added vs make_proof_charts.
PATHS = {
    "audio_to_text":   ("S2T  (audio -> text)",   "text"),
    "image_to_text":   ("I2T  (image -> text)",   "text"),
    "image_to_speech": ("I2S  (image -> speech)", "speech"),
    "audio_to_speech": ("S2S  (audio -> speech)", "speech"),
    "text_to_speech":  ("T2S  (text -> speech)",  "speech"),
}
ITL_OUTLIER_FRAC = 0.40


def cell(agg, b, s):
    return agg.get(f"B{b}", {}).get(s)


def raw_value(c, kind, modality):
    if not c:
        return None
    rec = c.get("recomputed", {}); har = c.get("harness", {})
    if kind == "tok":  return har.get("text_tok_throughput_reported")
    if kind == "reqs": return har.get("req_throughput_reported")
    if kind == "aud":  return har.get("audio_seconds_throughput_reported")
    if kind == "rtf":  return rec.get("rtf_p50")
    if kind == "ttft":
        d = har.get("ttft_audio" if modality == "speech" else "ttft_text")
        return d.get("p50") if isinstance(d, dict) else None
    if kind == "itl":
        d = har.get("itl_audio" if modality == "speech" else "itl_text")
        return d.get("mean") if isinstance(d, dict) else None
    return None


def drop_itl_outliers(xs, ys):
    if len(ys) < 3:
        return xs, ys
    kx, ky = [], []
    for i, (x, y) in enumerate(zip(xs, ys)):
        prev = ys[i - 1] if i > 0 else None
        if prev is not None and prev > 0 and y < ITL_OUTLIER_FRAC * prev:
            continue
        kx.append(x); ky.append(y)
    return kx, ky


def panel(ax, agg, modality, kind, title, ylab, lower_better=False):
    for s, lbl, col, mk, ls in SYS:
        xs, ys = [], []
        for b in BATCHES:
            v = raw_value(cell(agg, b, s), kind, modality)
            if v is None or v != v:
                continue
            xs.append(b); ys.append(v)
        if kind == "itl" and xs:
            xs, ys = drop_itl_outliers(xs, ys)
        if xs:
            ax.plot(xs, ys, marker=mk, color=col, ls=ls, label=lbl, lw=2, ms=6)
    ax.set_xscale("log", base=2); ax.set_xticks(BATCHES); ax.set_xticklabels(BATCHES)
    ax.set_xlabel("batch size"); ax.set_ylabel(ylab); ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    if lower_better:
        ax.set_ylim(bottom=0)


def main():
    for path, (title, modality) in PATHS.items():
        fp = os.path.join(ROOT, f"raw_{path}.json")
        if not os.path.exists(fp):
            print("skip (no data):", path); continue
        raw = json.load(open(fp))
        if raw.get("status") != "complete":
            print("skip (not a complete run):", path); continue
        agg = raw["aggregates"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Qwen3-Omni {title} -- {' vs '.join(l for _, l, *_ in SYS)}, B=1..32",
                     fontsize=13, fontweight="bold")
        if modality == "text":
            panel(axes[0, 0], agg, modality, "tok",  "Throughput (text tokens/s) -- higher better", "tok/s")
            panel(axes[0, 1], agg, modality, "reqs", "Throughput (requests/s) -- higher better", "req/s")
        else:
            panel(axes[0, 0], agg, modality, "aud",  "Throughput (audio sec/s) -- higher better", "audio s/s")
            panel(axes[0, 1], agg, modality, "rtf",  "RTF p50 -- lower better (<1 = real-time)", "RTF", lower_better=True)
        panel(axes[1, 0], agg, modality, "ttft", "TTFT p50 (s) -- lower better", "s", lower_better=True)
        panel(axes[1, 1], agg, modality, "itl",  "ITL mean (s) -- lower better", "s", lower_better=True)
        axes[0, 0].legend(fontsize=9, loc="best")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        outp = os.path.join(OUT, f"{path}_4metric.png")
        fig.savefig(outp, dpi=200, bbox_inches="tight"); plt.close(fig)
        print("wrote", outp)
    print("DONE")


if __name__ == "__main__":
    main()
