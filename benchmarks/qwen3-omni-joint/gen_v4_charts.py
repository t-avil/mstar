#!/usr/bin/env python3
"""v4 charts: 2x2 4-metric panels per path (same layout as the original
*_4metric.png proof charts).

 text  (S2T,I2T): tok/s | req/s | TTFT(text) p50 | ITL(text) mean
 speech(I2S,S2S): audio s/s | RTF p50 | TTFT p50 | ITL mean

Line scheme (user-specified):
  M* new (current)      = SOLID BLUE  — best current cells, layered per-METRIC:
                          committed mstar_v2 aggregates, overridden by
                          sweep_mstar_v3 (final stack, 07-03), overridden by
                          sweep_mstar_v4 (07-05/06 iteration medians).
                          Overrides only replace metrics they actually carry,
                          so no committed point is ever dropped.
  M* new (previous)     = DOTTED BLUE — committed mstar_new (4c33b33).
  vLLM-Omni (current)   = SOLID GREEN — committed vllm 0.22 aggregates.
  vLLM-Omni (previous)  = DOTTED GREEN — raw_vllm_021.json (pre-refresh).
All committed data; nothing re-benchmarked.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("chartstyle.mplstyle")

BLUE, GREEN = "#1f77b4", "#2ca02c"
PATHS = ["audio_to_text", "image_to_text", "audio_to_speech", "image_to_speech"]
SHORT = {"audio_to_text": "s2t", "image_to_text": "i2t",
         "audio_to_speech": "s2s", "image_to_speech": "i2s"}
TITLE = {"audio_to_text": "S2T  (audio -> text)", "image_to_text": "I2T  (image -> text)",
         "audio_to_speech": "S2S  (audio -> speech)", "image_to_speech": "I2S  (image -> speech)"}
BATCHES = [1, 2, 4, 8, 16, 32]
V3 = "/m-coriander/coriander/tim/sweep_mstar_v3"
V4 = "/m-coriander/coriander/tim/sweep_mstar_v4"


def g(h, k, sub):
    x = (h or {}).get(k)
    return x.get(sub) if isinstance(x, dict) else x


def merge(cell, updates):
    """Per-metric merge: only non-None values override."""
    for k, v in updates.items():
        if v is not None:
            cell[k] = v


def from_results(res, modality="text"):
    """Extract all metrics a sweep-dir results.json carries."""
    t = ((res.get("ttft") or {}).get(modality) or {})
    i = res.get("itl") or {}
    itl = (i.get(modality) or {}).get("mean") if isinstance(i.get(modality), dict) else None
    per_req = res.get("per_request", [])
    rtfs = sorted(
        (p["jct_ms"] / 1000.0) / (p["output_bytes"]["audio"] / 48000.0)
        for p in per_req
        if p.get("output_bytes", {}).get("audio", 0) > 0 and p.get("jct_ms"))
    return {
        "req_s": res.get("request_throughput"),
        "tok": res.get("text_token_throughput"),
        "aud": res.get("audio_seconds_throughput"),
        "ttft": t.get("p50"), "itl": itl,
        "rtf": rtfs[len(rtfs) // 2] if rtfs else None,
    }


def load(path):
    """-> {series: {B: {req_s, tok, aud, ttft, itl, rtf}}}"""
    raw = json.load(open(f"raw_{path}.json"))
    old_vllm = json.load(open("raw_vllm_021.json"))[path]
    s = SHORT[path]
    # speech paths report audio-side latency (matches the original charts)
    mod = "audio" if path.endswith("speech") else "text"
    out = {"mnew_cur": {}, "mnew_prev": {}, "vllm_cur": {}, "vllm_prev": {}}
    for bkey, sysmap in raw.get("aggregates", {}).items():
        b = int(bkey[1:])
        for sysname, dest in (("mstar_v2", "mnew_cur"), ("mstar_new", "mnew_prev"),
                              ("vllm", "vllm_cur")):
            blk = sysmap.get(sysname) or {}
            rec, har = blk.get("recomputed") or {}, blk.get("harness") or {}
            out[dest][b] = {
                # speech-path mstar_v2 aggregates carry req/s in the harness
                # block only (recomputed has just RTF there)
                "req_s": rec.get("request_throughput")
                or har.get("request_throughput"),
                "tok": rec.get("text_token_throughput")
                or har.get("text_token_throughput"),
                "aud": rec.get("audio_throughput"),
                "ttft": g(har, f"ttft_{mod}", "p50"),
                "itl": g(har, f"itl_{mod}", "mean"),
                "rtf": rec.get("rtf_p50"),
            }
    for bkey, cell in old_vllm.items():
        out["vllm_prev"][int(bkey[1:])] = {
            "req_s": cell.get("req_s"), "tok": cell.get("tok_s"),
            "aud": cell.get("audio_s"), "ttft": cell.get("ttft_p50"),
            "itl": cell.get("itl_mean"), "rtf": cell.get("rtf_p50"),
        }
    # Committed v2 TEXT latency (harness fields absent from aggregates for
    # mstar_v2) — parse the NUMBERS_V2.md latency table (text columns only,
    # so don't apply it to the speech paths).
    for line in open("NUMBERS_V2.md") if not path.endswith("speech") else []:
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) == 8 and parts[0] == s and parts[1].isdigit():
            b = int(parts[1])
            cell = out["mnew_cur"].setdefault(b, {})
            try:
                if cell.get("ttft") is None:
                    cell["ttft"] = float(parts[2])
                if cell.get("itl") is None:
                    cell["itl"] = float(parts[5])
            except ValueError:
                pass
    # Committed v2 SPEECH req/s+rtf live only in the v2 sweep dirs (the
    # aggregates carry RTF-only recomputed blocks there) — same source
    # gen_v2_report.py used for NUMBERS_V2.
    if path.endswith("speech"):
        for bdir in sorted(glob.glob(f"/m-coriander/coriander/tim/sweep_mstar_v2_final/{s}/B*")):
            b = int(os.path.basename(bdir)[1:])
            try:
                res = json.load(open(f"{bdir}/results.json"))
            except FileNotFoundError:
                continue
            cell = out["mnew_cur"].setdefault(b, {})
            r = from_results(res, mod)
            merge(cell, {"req_s": r["req_s"], "aud": r["aud"]})
            if cell.get("ttft") is None:
                cell["ttft"] = r["ttft"]
            if cell.get("itl") is None:
                cell["itl"] = r["itl"]
        # the new stack doesn't touch audio: committed v2 stays "current"
        return out

    # v3 (final stack, 07-03) then v4 (07-05/06 iteration) overrides for
    # mnew_cur — TEXT paths only, per-METRIC merge so committed points that
    # a sweep cell doesn't carry (e.g. closed-loop TTFT) are never dropped.
    for root in (V3, V4):
        for bdir in sorted(glob.glob(f"{root}/{s}/B*")):
            b = int(os.path.basename(bdir)[1:])
            try:
                res = json.load(open(f"{bdir}/results.json"))
            except FileNotFoundError:
                continue
            merge(out["mnew_cur"].setdefault(b, {}), from_results(res))
    return out


SERIES = [
    ("mnew_cur", "M* new", BLUE, "-", 2.4),
    ("mnew_prev", "M* new (previous)", BLUE, ":", 1.6),
    ("vllm_cur", "vLLM-Omni", GREEN, "-", 2.4),
    ("vllm_prev", "vLLM-Omni (previous)", GREEN, ":", 1.6),
]
# (metric key, panel title, ylabel, lower_better) per modality — same panel
# arrangement as the original *_4metric.png proof charts.
PANELS_TEXT = [
    ("tok", "Throughput (text tokens/s) -- higher better", "tok/s", False),
    ("req_s", "Throughput (requests/s) -- higher better", "req/s", False),
    ("ttft", "TTFT p50 (s) -- lower better", "s", True),
    ("itl", "ITL mean (s) -- lower better", "s", True),
]
PANELS_SPEECH = [
    ("aud", "Throughput (audio sec/s) -- higher better", "audio s/s", False),
    ("rtf", "RTF p50 -- lower better (<1 = real-time)", "RTF", True),
    ("ttft", "TTFT p50 (s) -- lower better", "s", True),
    ("itl", "ITL mean (s) -- lower better", "s", True),
]


def main():
    for path in PATHS:
        data = load(path)
        panels = PANELS_SPEECH if path.endswith("speech") else PANELS_TEXT
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Qwen3-Omni {TITLE[path]} -- M* new vs vLLM-Omni, "
                     "current vs previous, B=1..32", fontsize=13, fontweight="bold")
        for ax, (mk, title, ylab, lower_better) in zip(axes.flat, panels):
            for key, label, color, ls, lw in SERIES:
                xs = [b for b in BATCHES if data[key].get(b, {}).get(mk) is not None]
                ys = [data[key][b][mk] for b in xs]
                if not xs:
                    continue
                ax.plot(xs, ys, ls, color=color, linewidth=lw, marker="o",
                        markersize=5, label=label)
            ax.set_xscale("log", base=2)
            ax.set_xticks(BATCHES)
            ax.set_xticklabels([str(b) for b in BATCHES])
            ax.set_xlabel("batch size")
            ax.set_ylabel(ylab)
            ax.set_title(title, fontsize=11)
            ax.grid(True, alpha=0.3)
            if lower_better:
                ax.set_ylim(bottom=0)
        axes[0, 0].legend(fontsize=9, loc="best")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fname = f"charts/{path}_v4_4metric.png"
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("wrote", fname)


if __name__ == "__main__":
    main()
