#!/usr/bin/env python3
"""Generate NUMBERS_V2.md + charts including system mstar_v2.

Reads the committed raw_<path>.json aggregates for mstar_new / mstar_old /
vllm, and the mstar_v2 sweep results (per-cell results.json) directly, then
emits a ratio table and per-path throughput charts using the shared style.
The stock aggregate.py/make_numbers.py have hardcoded system columns; this
generator adds mstar_v2 without touching their output.
"""
import glob
import json
import os

SWEEP = "/m-coriander/coriander/tim/sweep_mstar_v2"
PATHS = ["audio_to_text", "image_to_text", "audio_to_speech", "image_to_speech"]
SHORT = {"audio_to_text": "s2t", "image_to_text": "i2t",
         "audio_to_speech": "s2s", "image_to_speech": "i2s"}
BATCHES = [1, 2, 4, 8, 16, 32]
SYSTEMS = ["mstar_v2", "mstar_new", "mstar_old", "vllm"]
LABEL = {"mstar_v2": "M*-v2", "mstar_new": "M*-new", "mstar_old": "M*-old", "vllm": "vLLM"}


def load_cells():
    cells = {}  # (path, system, B) -> {req_s, tok_s, audio_s, rtf_p50}
    for path in PATHS:
        raw = json.load(open(f"raw_{path}.json"))
        for bkey, sysmap in raw.get("aggregates", {}).items():
            b = int(bkey[1:])
            for sysname, blocks in sysmap.items():
                rec = blocks.get("recomputed") or {}
                cells[(path, sysname, b)] = {
                    "req_s": rec.get("request_throughput"),
                    "tok_s": rec.get("text_token_throughput"),
                    "audio_s": rec.get("audio_throughput"),
                    "rtf_p50": rec.get("rtf_p50"),
                }
        # mstar_v2 straight from the sweep results
        s = SHORT[path]
        for bdir in sorted(glob.glob(f"{SWEEP}/{s}/B*")):
            b = int(os.path.basename(bdir)[1:])
            try:
                res = json.load(open(f"{bdir}/results.json"))
            except FileNotFoundError:
                continue
            per_req = res.get("per_request", [])
            wall = res.get("wall_time_s") or 0
            durs = [p.get("output_bytes", {}).get("audio", 0) / 48000.0
                    for p in per_req]
            durs = [d for d in durs if d > 0]
            rtfs = sorted(
                (p["jct_ms"] / 1000.0) / (p["output_bytes"]["audio"] / 48000.0)
                for p in per_req
                if p.get("output_bytes", {}).get("audio", 0) > 0 and p.get("jct_ms")
            )
            cells[(path, "mstar_v2", b)] = {
                "req_s": res.get("request_throughput"),
                "tok_s": res.get("text_token_throughput"),
                "audio_s": (sum(durs) / wall) if durs and wall else res.get("audio_seconds_throughput"),
                "rtf_p50": rtfs[len(rtfs) // 2] if rtfs else None,
            }
    return cells


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def ratio(a, b):
    return "-" if (a is None or not b) else f"{a / b:.2f}x"


def main():
    cells = load_cells()
    lines = [
        "# NUMBERS_V2.md — M*-v2 (opt/decode-v2 b9de820, fp8 MoE + fused topk +",
        "# worker CPU fixes + inline/batch emit + encoders-on-rank-0) vs recorded baselines",
        "",
        "Sweep: 2026-07-02, GPUs 6,7, WARMUP=5, N=max(50,10B), closed loop.",
        "Baselines (mstar_new/mstar_old/vllm) are the committed v2 rebenchmark",
        "aggregates — NOT re-run. Metric: req/s primary (cross-system;",
        "tok/s embeds output-length skew: vLLM generates ~20% longer text).",
        "",
    ]
    for path in PATHS:
        lines.append(f"## {SHORT[path]} — {path}")
        lines.append("")
        lines.append("| B | M*-v2 req/s | M*-new | M*-old | vLLM | v2/vLLM | v2/new |")
        lines.append("|---|---|---|---|---|---|---|")
        for b in BATCHES:
            v2 = cells.get((path, "mstar_v2", b), {})
            new = cells.get((path, "mstar_new", b), {})
            old = cells.get((path, "mstar_old", b), {})
            vll = cells.get((path, "vllm", b), {})
            lines.append(
                f"| {b} | {fmt(v2.get('req_s'))} | {fmt(new.get('req_s'))} | "
                f"{fmt(old.get('req_s'))} | {fmt(vll.get('req_s'))} | "
                f"{ratio(v2.get('req_s'), vll.get('req_s'))} | "
                f"{ratio(v2.get('req_s'), new.get('req_s'))} |")
        if path.endswith("speech"):
            lines.append("")
            lines.append("| B | M*-v2 audio_s/s | vLLM audio_s/s | v2/vLLM | M*-v2 RTF p50 | vLLM RTF p50 |")
            lines.append("|---|---|---|---|---|---|")
            for b in BATCHES:
                v2 = cells.get((path, "mstar_v2", b), {})
                vll = cells.get((path, "vllm", b), {})
                lines.append(
                    f"| {b} | {fmt(v2.get('audio_s'), 2)} | {fmt(vll.get('audio_s'), 2)} | "
                    f"{ratio(v2.get('audio_s'), vll.get('audio_s'))} | "
                    f"{fmt(v2.get('rtf_p50'))} | {fmt(vll.get('rtf_p50'))} |")
        lines.append("")
    open("NUMBERS_V2.md", "w").write("\n".join(lines))
    print("wrote NUMBERS_V2.md")

    # charts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if os.path.exists("chartstyle.mplstyle"):
        plt.style.use("chartstyle.mplstyle")
    colors = {"mstar_v2": "#d62728", "mstar_new": "#1f77b4",
              "mstar_old": "#7f7f7f", "vllm": "#2ca02c"}
    os.makedirs("charts", exist_ok=True)
    for path in PATHS:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for sysname in SYSTEMS:
            ys = [cells.get((path, sysname, b), {}).get("req_s") for b in BATCHES]
            if all(y is None for y in ys):
                continue
            ax.plot(BATCHES, ys, marker="o", label=LABEL[sysname],
                    color=colors[sysname],
                    linewidth=2.2 if sysname == "mstar_v2" else 1.4)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCHES)
        ax.set_xticklabels([str(b) for b in BATCHES])
        ax.set_xlabel("concurrency (closed loop)")
        ax.set_ylabel("requests / s")
        ax.set_title(f"{SHORT[path]} ({path}) — request throughput")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        out = f"charts/{path}_v2_reqs.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
