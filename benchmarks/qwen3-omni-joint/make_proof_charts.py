#!/usr/bin/env python3
"""Comprehensive 3-way proof charts (M*-new vs M*-old vs vLLM) per path, all batches.
Reads the corrected raw_<path>.json (datapoints == aggregates). 2x2 panels:
 text  (S2T,I2T): throughput tok/s | throughput req/s | TTFT(text) p50 | ITL(text) mean
 speech(I2S,S2S): throughput audio-s/s | RTF p50 | TTFT(audio) p50 | ITL(audio) mean

A3 corrections:
 (a) ONE TTFT statistic only: p50 (the prior dual mean/p50 line is dropped).
 (b) error bars where a per-request distribution exists:
       - RTF panel: +/-1 std of the per-request RTF distribution (from datapoints) --
         these matter at small batch where n is tiny.
       - TTFT / ITL panels: p50->p95 spread from the harness percentiles (drawn as an
         upper whisker, annotated) -- the raw per-request latencies are not stored, so
         the distribution summary is used and labelled as such.
       - throughput panels: wall-clock rates (one value per cell, no per-request
         distribution) -> no error bar, annotated.
 (c) MISSING (system,batch) cells are marked distinctly: an open marker is drawn on the
     panel baseline AND the missing cells are listed in a footnote, so a gap is never
     silently absent.
 (d) anomalous-low ITL points (e.g. S2S M*-old B=32 ITL~0.021s, far below the trend) are
     detected, DROPPED from the line, and annotated "dropped outlier".
"""
import json, os, sys, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/tim/bench-wt/benchmarks/qwen3-omni-joint"
OUT  = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "charts")
STYLE = "/home/tim/exp_3way/chartstyle.mplstyle"
if os.path.exists(STYLE):
    try: plt.style.use(STYLE)
    except Exception: pass
os.makedirs(OUT, exist_ok=True)

BATCHES = [1, 2, 4, 8, 16, 32]
SYS = [("mstar_new", "M*-new (integrated)", "#1f77b4", "o", "-"),
       ("mstar_old", "M*-old (HF)",         "#7f7f7f", "s", "--"),
       ("vllm",      "vLLM-Omni",           "#2ca02c", "^", "-.")]
PATHS = {
 "audio_to_text":   ("S2T  (audio -> text)",   "text"),
 "image_to_text":   ("I2T  (image -> text)",   "text"),
 "image_to_speech": ("I2S  (image -> speech)", "speech"),
 "audio_to_speech": ("S2S  (audio -> speech)", "speech"),
}
# ITL outlier rule: a point below this fraction of the series' median (excluding
# itself) is treated as anomalous and dropped from the line (case (d)).
ITL_OUTLIER_FRAC = 0.40


def cell(agg, b, s):
    return agg.get(f"B{b}", {}).get(s)


def rtf_samples(dps, s, b):
    return [d["rtf"] for d in dps
            if d.get("system") == s and d.get("batch") == b and d.get("rtf") is not None]


def raw_value(c, kind, modality):
    """Plotted value for a cell (None if absent)."""
    if not c:
        return None
    rec = c.get("recomputed", {}); har = c.get("harness", {})
    if kind == "tok":  return rec.get("text_token_throughput")
    if kind == "reqs": return rec.get("request_throughput")
    if kind == "aud":  return rec.get("audio_throughput")
    if kind == "rtf":  return rec.get("rtf_p50")
    if kind == "ttft":
        d = har.get("ttft_audio" if modality == "speech" else "ttft_text")
        return d.get("p50") if isinstance(d, dict) else None
    if kind == "itl":
        d = har.get("itl_audio" if modality == "speech" else "itl_text")
        return d.get("mean") if isinstance(d, dict) else None
    return None


def err_value(c, kind, modality, dps, s, b):
    """(lower, upper) error bar for a cell, or None. Only where a distribution exists."""
    if not c:
        return None
    rec = c.get("recomputed", {}); har = c.get("harness", {})
    if kind == "rtf":
        samp = rtf_samples(dps, s, b)
        if len(samp) > 1:
            sd = rec.get("rtf_std")
            if sd is None:
                m = sum(samp) / len(samp)
                sd = math.sqrt(sum((x - m) ** 2 for x in samp) / len(samp))
            return (sd, sd)  # +/-1 std (symmetric)
    if kind in ("ttft", "itl"):
        d = har.get(("ttft_" if kind == "ttft" else "itl_") +
                    ("audio" if modality == "speech" else "text"))
        if isinstance(d, dict):
            center = d.get("p50") if kind == "ttft" else d.get("mean")
            p95 = d.get("p95")
            if center is not None and p95 is not None and p95 > center:
                return (0.0, p95 - center)  # upper whisker to p95 (asymmetric)
    return None


def drop_itl_outliers(xs, ys):
    """Return (kept_xs, kept_ys, dropped[(b,val)]) dropping anomalously low ITL points.

    ITL grows with batch, so the anomaly signature is a sharp DROP below the
    previous batch's value (e.g. S2S M*-old B32 ITL~0.021s after ~0.097s at B16).
    A point is dropped only if it falls below ITL_OUTLIER_FRAC * its predecessor --
    this never flags the naturally-smallest first point (B1 has no predecessor)."""
    if len(ys) < 3:
        return xs, ys, []
    keep_x, keep_y, dropped = [], [], []
    for i, (x, y) in enumerate(zip(xs, ys)):
        prev = ys[i - 1] if i > 0 else None
        if prev is not None and prev > 0 and y < ITL_OUTLIER_FRAC * prev:
            dropped.append((x, y))
        else:
            keep_x.append(x); keep_y.append(y)
    return keep_x, keep_y, dropped


def panel(ax, agg, dps, modality, kind, title, ylab, lower_better=False):
    missing, notes = [], []
    present_any = {b: any(cell(agg, b, s) for s, *_ in SYS) for b in BATCHES}
    for s, lbl, col, mk, ls in SYS:
        xs, ys, el, eu = [], [], [], []
        for b in BATCHES:
            c = cell(agg, b, s)
            v = raw_value(c, kind, modality)
            if v is None or v != v:
                if present_any[b] and c is None:
                    missing.append((lbl, b, col))
                continue
            e = err_value(c, kind, modality, dps, s, b)
            xs.append(b); ys.append(v)
            el.append(e[0] if e else 0.0); eu.append(e[1] if e else 0.0)
        # (d) drop anomalous-low ITL points from the line
        if kind == "itl" and xs:
            kx, ky, dropped = drop_itl_outliers(xs, ys)
            for db, dv in dropped:
                ax.plot([db], [dv], marker="x", color=col, ms=9, mew=2, ls="none", zorder=5)
                notes.append(f"dropped outlier: {lbl} B{db} ITL={dv:.3f}s")
            # rebuild error arrays for kept points
            keep = set(kx)
            el = [e for x, e in zip(xs, el) if x in keep]
            eu = [e for x, e in zip(xs, eu) if x in keep]
            xs, ys = kx, ky
        if xs:
            if any(a or b for a, b in zip(el, eu)):
                ax.errorbar(xs, ys, yerr=[el, eu], marker=mk, color=col, ls=ls,
                            label=lbl, lw=2, ms=6, capsize=3, elinewidth=1)
            else:
                ax.plot(xs, ys, marker=mk, color=col, ls=ls, label=lbl, lw=2, ms=6)
    # (c) mark missing cells distinctly: open marker on baseline + footnote
    if missing:
        ymin, ymax = ax.get_ylim()
        y0 = ymin + 0.02 * (ymax - ymin)
        for lbl, b, col in missing:
            ax.plot([b], [y0], marker="o", mfc="none", mec=col, ms=10, mew=1.5,
                    ls="none", zorder=4)
        uniq = sorted({f"{lbl} B{b}" for lbl, b, _ in missing})
        notes.append("missing (open marker): " + ", ".join(uniq))
    ax.set_xscale("log", base=2); ax.set_xticks(BATCHES); ax.set_xticklabels(BATCHES)
    ax.set_xlabel("batch size"); ax.set_ylabel(ylab); ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    if lower_better:
        ax.set_ylim(bottom=0)
    if notes:
        ax.annotate("\n".join(notes), xy=(0.5, -0.30), xycoords="axes fraction",
                    ha="center", va="top", fontsize=7, color="#555555")


for path, (title, modality) in PATHS.items():
    fp = os.path.join(ROOT, f"raw_{path}.json")
    if not os.path.exists(fp):
        print("skip (no data):", path); continue
    raw = json.load(open(fp))
    agg = raw["aggregates"]; dps = raw.get("datapoints", [])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    fig.suptitle(f"Qwen3-Omni {title} -- M*-new vs M*-old vs vLLM, B=1..32",
                 fontsize=13, fontweight="bold")
    if modality == "text":
        panel(axes[0, 0], agg, dps, modality, "tok",  "Throughput (text tokens/s) -- higher better\n(tok/s self-counted per server)", "tok/s")
        panel(axes[0, 1], agg, dps, modality, "reqs", "Throughput (requests/s) -- higher better", "req/s")
    else:
        panel(axes[0, 0], agg, dps, modality, "aud",  "Throughput (audio sec/s) -- higher better", "audio s/s")
        panel(axes[0, 1], agg, dps, modality, "rtf",  "RTF p50 -- lower better (<1 = real-time)\n(error bar = +/-1 std of per-request RTF)", "RTF", lower_better=True)
    panel(axes[1, 0], agg, dps, modality, "ttft", "TTFT p50 (s) -- lower better\n(whisker = p50->p95 spread)", "s", lower_better=True)
    panel(axes[1, 1], agg, dps, modality, "itl",  "ITL mean (s) -- lower better\n(whisker = mean->p95 spread)", "s", lower_better=True)
    axes[0, 0].legend(fontsize=9, loc="best")
    # global legend note for the missing-marker convention
    fig.legend(handles=[Line2D([0], [0], marker="o", mfc="none", mec="#555555", ls="none",
                               ms=9, label="missing cell (no run)")],
               loc="lower center", ncol=1, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    outp = os.path.join(OUT, f"{path}_4metric.png")
    fig.savefig(outp, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", outp)
print("DONE")
