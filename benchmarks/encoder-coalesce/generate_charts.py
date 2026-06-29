#!/usr/bin/env python3
"""Generate A/B charts for encoder-coalesce benchmark.

Reads per-run results.json files from raw_<label>_s2t_b<B>/ dirs,
assembles the unified raw.json, and produces:
  1. TTFT p50 vs concurrency (control vs experiment)
  2. ITL p95 vs concurrency
  3. JCT p50 vs concurrency
"""
import json
import os
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCH_DIR = Path(__file__).parent
CHART_DIR = BENCH_DIR / "charts"
STYLE = BENCH_DIR.parent / "chartstyle.mplstyle"

# Color mapping: control=orange, experiment=blue (per user spec)
COLORS = {"control": "#E57A24", "experiment": "#2C7BB6"}
LABELS = {"control": "Coalescing OFF (control)", "experiment": "Coalescing ON (experiment)"}
CONCURRENCIES = [1, 4, 8]


def percentile(values, p):
    """Compute percentile from sorted values."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (k - lo)) + s[hi] * (k - lo)


def load_run(label, conc):
    """Load results.json for a given label/concurrency."""
    dirname = f"raw_{label}_s2t_b{conc}"
    path = BENCH_DIR / dirname / "results.json"
    if not path.exists():
        print(f"WARNING: {path} not found, skipping")
        return None
    with open(path) as f:
        return json.load(f)


def extract_metrics(data):
    """Extract per-request TTFT, ITL, and JCT from results.json."""
    ttfts = []
    itls_all = []
    jcts = []
    for req in data.get("per_request", []):
        if req.get("phase") != "measure":
            continue
        ttft = req.get("ttft_text")
        if ttft is not None:
            ttfts.append(ttft * 1000.0)  # s -> ms
        itl_tokens = req.get("itl_text_per_token")
        if itl_tokens:
            itls_all.extend([v * 1000.0 for v in itl_tokens])  # s -> ms
        jct = req.get("jct_ms")
        if jct is not None:
            jcts.append(jct)
    return ttfts, itls_all, jcts


def main():
    if STYLE.exists():
        plt.style.use(str(STYLE))

    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # Collect metrics per (label, concurrency)
    all_data = {}
    for label in ["control", "exp"]:
        for conc in CONCURRENCIES:
            data = load_run(label, conc)
            if data is None:
                continue
            ttfts, itls, jcts = extract_metrics(data)
            key = ("control" if label == "control" else "experiment", conc)
            all_data[key] = {"ttfts": ttfts, "itls": itls, "jcts": jcts}

    # --- Chart 1: TTFT p50 vs concurrency ---
    fig, ax = plt.subplots()
    for group in ["control", "experiment"]:
        xs, ys = [], []
        for c in CONCURRENCIES:
            d = all_data.get((group, c))
            if d and d["ttfts"]:
                xs.append(c)
                ys.append(percentile(d["ttfts"], 50))
        ax.plot(xs, ys, marker="o", color=COLORS[group], label=LABELS[group])
    ax.set_xlabel("Max Concurrency (B)")
    ax.set_ylabel("TTFT p50 (ms)")
    ax.set_title("Time to First Token (p50) — S2T")
    ax.set_xticks(CONCURRENCIES)
    ax.legend()
    fig.savefig(str(CHART_DIR / "ttft_p50_vs_concurrency.png"))
    plt.close(fig)
    print(f"Saved {CHART_DIR / 'ttft_p50_vs_concurrency.png'}")

    # --- Chart 2: ITL p95 vs concurrency ---
    fig, ax = plt.subplots()
    for group in ["control", "experiment"]:
        xs, ys = [], []
        for c in CONCURRENCIES:
            d = all_data.get((group, c))
            if d and d["itls"]:
                xs.append(c)
                ys.append(percentile(d["itls"], 95))
        ax.plot(xs, ys, marker="s", color=COLORS[group], label=LABELS[group])
    ax.set_xlabel("Max Concurrency (B)")
    ax.set_ylabel("ITL p95 (ms)")
    ax.set_title("Inter-Token Latency (p95) — S2T")
    ax.set_xticks(CONCURRENCIES)
    ax.legend()
    fig.savefig(str(CHART_DIR / "itl_p95_vs_concurrency.png"))
    plt.close(fig)
    print(f"Saved {CHART_DIR / 'itl_p95_vs_concurrency.png'}")

    # --- Chart 3: JCT p50 vs concurrency ---
    fig, ax = plt.subplots()
    for group in ["control", "experiment"]:
        xs, ys = [], []
        for c in CONCURRENCIES:
            d = all_data.get((group, c))
            if d and d["jcts"]:
                xs.append(c)
                ys.append(percentile(d["jcts"], 50))
        ax.plot(xs, ys, marker="^", color=COLORS[group], label=LABELS[group])
    ax.set_xlabel("Max Concurrency (B)")
    ax.set_ylabel("JCT p50 (ms)")
    ax.set_title("Job Completion Time (p50) — S2T")
    ax.set_xticks(CONCURRENCIES)
    ax.legend()
    fig.savefig(str(CHART_DIR / "jct_p50_vs_concurrency.png"))
    plt.close(fig)
    print(f"Saved {CHART_DIR / 'jct_p50_vs_concurrency.png'}")

    # --- Assemble raw.json ---
    import subprocess
    git_commit = subprocess.run(
        ["git", "-C", str(BENCH_DIR.parent.parent), "rev-parse", "HEAD"],
        capture_output=True, text=True
    ).stdout.strip()

    import datetime
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    datapoints = []
    iter_idx = 0
    for label in ["control", "exp"]:
        group = "control" if label == "control" else "experiment"
        for conc in CONCURRENCIES:
            d = all_data.get((group, conc))
            if d is None:
                continue
            for i, ttft in enumerate(d["ttfts"]):
                datapoints.append({
                    "iter": iter_idx,
                    "phase": "measure",
                    "group": group,
                    "concurrency": conc,
                    "metric": "ttft_text_ms",
                    "value": ttft,
                })
                iter_idx += 1
            for i, itl in enumerate(d["itls"]):
                datapoints.append({
                    "iter": iter_idx,
                    "phase": "measure",
                    "group": group,
                    "concurrency": conc,
                    "metric": "itl_text_per_token_ms",
                    "value": itl,
                })
                iter_idx += 1
            for i, jct in enumerate(d["jcts"]):
                datapoints.append({
                    "iter": iter_idx,
                    "phase": "measure",
                    "group": group,
                    "concurrency": conc,
                    "metric": "jct_ms",
                    "value": jct,
                })
                iter_idx += 1

    raw = {
        "benchmark": "encoder-coalesce",
        "timestamp_utc": ts,
        "git_commit": git_commit,
        "device": {
            "cuda_visible_devices": "control=0,1  experiment=4,5",
            "gpu_name": "NVIDIA H200",
        },
        "units": "ms",
        "warmup_iters": 5,
        "datapoints": datapoints,
    }
    raw_path = BENCH_DIR / "raw.json"
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"Saved {raw_path} ({len(datapoints)} datapoints)")

    # --- Print summary table ---
    print("\n" + "=" * 90)
    print(f"{'Group':<14} {'B':>3}  {'TTFT p50':>10} {'ITL p50':>10} {'ITL p95':>10} {'ITL p99':>10} {'JCT p50':>10}")
    print("-" * 90)
    for group in ["control", "experiment"]:
        for conc in CONCURRENCIES:
            d = all_data.get((group, conc))
            if d is None:
                print(f"{group:<14} {conc:>3}  {'n/a':>10} {'n/a':>10} {'n/a':>10} {'n/a':>10} {'n/a':>10}")
                continue
            ttft_p50 = f"{percentile(d['ttfts'], 50):.1f}" if d["ttfts"] else "n/a"
            itl_p50 = f"{percentile(d['itls'], 50):.1f}" if d["itls"] else "n/a"
            itl_p95 = f"{percentile(d['itls'], 95):.1f}" if d["itls"] else "n/a"
            itl_p99 = f"{percentile(d['itls'], 99):.1f}" if d["itls"] else "n/a"
            jct_p50 = f"{percentile(d['jcts'], 50):.1f}" if d["jcts"] else "n/a"
            print(f"{group:<14} {conc:>3}  {ttft_p50:>10} {itl_p50:>10} {itl_p95:>10} {itl_p99:>10} {jct_p50:>10}")
    print("=" * 90)


if __name__ == "__main__":
    main()
