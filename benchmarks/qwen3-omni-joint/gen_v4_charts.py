#!/usr/bin/env python3
"""v4 charts: 4-metric panels (req/s, TTFT p50, ITL mean, RTF p50) per path.

Line scheme (user-specified):
  M* new (current)      = SOLID BLUE  — best current cells: sweep_mstar_v3
                          (final stack, 07-03) where present, else the
                          committed mstar_v2 aggregates.
  M* new (previous)     = DOTTED BLUE — committed mstar_new (4c33b33).
  vLLM-Omni (current)   = SOLID GREEN — committed vllm 0.22 aggregates.
  vLLM-Omni (previous)  = DOTTED GREEN — raw_vllm_021.json (pre-refresh
                          0.21-era, extracted from git history).
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
BATCHES = [1, 2, 4, 8, 16, 32]
V3 = "/m-coriander/coriander/tim/sweep_mstar_v3"
V4 = "/m-coriander/coriander/tim/sweep_mstar_v4"  # 07-05/06 iteration medians


def g(h, k, sub):
    x = (h or {}).get(k)
    return x.get(sub) if isinstance(x, dict) else x


def load(path):
    """-> {series: {B: {req_s, ttft, itl, rtf}}}"""
    raw = json.load(open(f"raw_{path}.json"))
    old_vllm = json.load(open("raw_vllm_021.json"))[path]
    s = SHORT[path]
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
                "ttft": g(har, "ttft_text", "p50"),
                "itl": g(har, "itl_text", "mean"),
                "rtf": rec.get("rtf_p50"),
            }
    for bkey, cell in old_vllm.items():
        out["vllm_prev"][int(bkey[1:])] = {
            "req_s": cell["req_s"], "ttft": cell["ttft_p50"],
            "itl": cell["itl_mean"], "rtf": cell["rtf_p50"],
        }
    # Committed v2 latency (harness fields absent from aggregates for
    # mstar_v2) — parse the NUMBERS_V2.md latency table.
    for line in open("NUMBERS_V2.md"):
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) == 8 and parts[0] == s and parts[1].isdigit():
            b = int(parts[1])
            cell = out["mnew_cur"].setdefault(b, {})
            try:
                cell.setdefault("ttft", None)
                cell["ttft"] = cell.get("ttft") or float(parts[2])
                cell["itl"] = cell.get("itl") or float(parts[5])
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
            cell["req_s"] = res.get("request_throughput")
            t = ((res.get("ttft") or {}).get("text") or {})
            cell["ttft"] = cell.get("ttft") or t.get("p50")

    # v3 (final stack) overrides for mnew_cur — TEXT paths only: the v3
    # preview ran on the cross-NUMA pair and the new stack doesn't touch
    # audio, so committed v2 remains the best "current" for speech.
    if path.endswith("speech"):
        return out
    for bdir in sorted(glob.glob(f"{V3}/{s}/B*")):
        b = int(os.path.basename(bdir)[1:])
        try:
            res = json.load(open(f"{bdir}/results.json"))
        except FileNotFoundError:
            continue
        t = ((res.get("ttft") or {}).get("text") or {})
        i = res.get("itl") or {}
        itl = (i.get("text") or {}).get("mean") if isinstance(i.get("text"), dict) else None
        per_req = res.get("per_request", [])
        rtfs = sorted(
            (p["jct_ms"] / 1000.0) / (p["output_bytes"]["audio"] / 48000.0)
            for p in per_req
            if p.get("output_bytes", {}).get("audio", 0) > 0 and p.get("jct_ms"))
        out["mnew_cur"][b] = {
            "req_s": res.get("request_throughput"), "ttft": t.get("p50"),
            "itl": itl, "rtf": rtfs[len(rtfs)//2] if rtfs else None,
        }
    # v4 (07-05/06 iteration: prep-h2d + cfgv2 + checkstop + CDT) — per-METRIC
    # merge on top of v3: closed-loop cells carry no TTFT, so only non-None
    # values override and the v3 latency points survive.
    for bdir in sorted(glob.glob(f"{V4}/{s}/B*")):
        b = int(os.path.basename(bdir)[1:])
        try:
            res = json.load(open(f"{bdir}/results.json"))
        except FileNotFoundError:
            continue
        t = ((res.get("ttft") or {}).get("text") or {})
        i = res.get("itl") or {}
        itl = (i.get("text") or {}).get("mean") if isinstance(i.get("text"), dict) else None
        cell = out["mnew_cur"].setdefault(b, {})
        for k, v in (("req_s", res.get("request_throughput")),
                     ("ttft", t.get("p50")), ("itl", itl)):
            if v is not None:
                cell[k] = v
    return out


SERIES = [
    ("mnew_cur", "M* new", BLUE, "-", 2.4),
    ("mnew_prev", "M* new (previous)", BLUE, ":", 1.6),
    ("vllm_cur", "vLLM-Omni", GREEN, "-", 2.4),
    ("vllm_prev", "vLLM-Omni (previous)", GREEN, ":", 1.6),
]
METRICS = [("req_s", "requests / s", False), ("ttft", "TTFT p50 (s)", True),
           ("itl", "ITL mean (s)", True), ("rtf", "RTF p50 (lower = faster)", True)]


def main():
    for path in PATHS:
        data = load(path)
        speech = path.endswith("speech")
        metrics = METRICS if speech else METRICS[:3]
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 3.6))
        for ax, (mk, ylabel, lower_better) in zip(axes, metrics):
            for key, label, color, ls, lw in SERIES:
                xs = [b for b in BATCHES if data[key].get(b, {}).get(mk) is not None]
                ys = [data[key][b][mk] for b in xs]
                if not xs:
                    continue
                ax.plot(xs, ys, ls, color=color, linewidth=lw, marker="o",
                        markersize=4.5, label=label)
            ax.set_xscale("log", base=2)
            ax.set_xticks(BATCHES)
            ax.set_xticklabels([str(b) for b in BATCHES])
            ax.set_xlabel("concurrency")
            ax.set_ylabel(ylabel)
            if lower_better:
                ax.set_title(ylabel.split(" (")[0] + "  (lower is better)", fontsize=10)
            else:
                ax.set_title("throughput", fontsize=10)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{SHORT[path]} ({path}) — M* new vs vLLM-Omni, current vs previous", y=1.02)
        fig.tight_layout()
        fname = f"charts/{path}_v4_4metric.png"
        fig.savefig(fname)
        plt.close(fig)
        print("wrote", fname)


if __name__ == "__main__":
    main()
