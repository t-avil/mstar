#!/usr/bin/env python3
"""Build raw.json (every datapoint) + Figure-5 charts from the sweep outputs.

Reads runs/out_<system>/B<B>/results.json (per-request jct_ms + output_bytes,
written by benchmark.runner --output-dir) for each system and batch size,
reconstructs per-request RTF and audio-seconds (24kHz int16 mono PCM ->
seconds = audio_bytes/48000), recomputes the aggregates the chart plots, and
cross-checks against the harness-printed values in stdout.txt.

Charts (shared style benchmarks/chartstyle.mplstyle, fixed per-system colors):
  charts/fig5a_rtf.png         RTF vs batch (lower is better)
  charts/fig5b_throughput.png  audio throughput vs batch (higher is better)
"""
import json, os, re, glob, statistics, subprocess, sys

D = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(D, "runs")
CHARTS = os.path.join(D, "charts")
STYLE = os.path.join(D, "..", "chartstyle.mplstyle")
BYTES_PER_AUDIO_SEC = 24000 * 2 * 1  # 24kHz int16 mono

# inference-system key -> (display label, fixed color). Same mapping everywhere.
# SGLang-Omni omitted from the figure: the paper's pinned commit (4a3960) is gone
# from GitHub, V1 deadlocks at startup, and the closest V0 build over-generates
# audio (talker degeneration) + runs ~10x slow — see FINDINGS.md. Not representative.
SYS_META = [
    ("ours",        "M*",          "#6a3df0"),
    ("vllm_omni",   "vLLM-Omni",   "#ff7f0e"),
]
# Canonical protocol = closed-loop max-concurrency (matches the fork's
# benchmark scripts + the reference CSV). Data under runs/cl/out_<sys>/B<B>.
# Offline sized-waves data is retained under runs/out_<sys>/ as a cross-check.
BATCHES = [1, 2, 4, 8, 16, 32]
PROTOCOL = "closed_loop (max-concurrency continuous batching), num_warmup=5"


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def parse_harness_stdout(path):
    """Best-effort scrape of the harness-printed RTF / audio-throughput for cross-check."""
    out = {}
    try:
        t = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return out
    m = re.search(r"RTF\s*:?\s*mean=([\d.]+)\s+p50=([\d.]+)\s+p95=([\d.]+)", t)
    if m:
        out["rtf_mean"], out["rtf_p50"], out["rtf_p95"] = map(float, m.groups())
    m = re.search(r"([\d.]+)\s*audio sec/s", t)
    if m:
        out["audio_throughput"] = float(m.group(1))
    m = re.search(r"([\d.]+)\s*req/s", t)
    if m:
        out["request_throughput"] = float(m.group(1))
    return out


def load_point(system, B):
    outdir = os.path.join(RUNS, "cl", f"out_{system}", f"B{B}")
    rj = os.path.join(outdir, "results.json")
    if not os.path.isfile(rj):
        return None
    data = json.load(open(rj))
    wall = data.get("wall_time_s") or 0.0
    dps = []
    for r in data.get("per_request", []):
        ab = (r.get("output_bytes") or {}).get("audio", 0)
        if ab <= 0:
            continue
        asec = ab / BYTES_PER_AUDIO_SEC
        jct_ms = r.get("jct_ms") or 0.0
        rtf = (jct_ms / 1000.0) / asec if asec > 0 else None
        dps.append({"request_id": r.get("request_id"), "phase": "measure",
                    "jct_ms": jct_ms, "audio_seconds": asec, "rtf": rtf})
    rtfs = sorted(d["rtf"] for d in dps if d["rtf"] is not None)
    total_audio = sum(d["audio_seconds"] for d in dps)
    recomputed = {
        "n": len(dps),
        "rtf_mean": statistics.mean(rtfs) if rtfs else None,
        "rtf_p50": pct(rtfs, 50), "rtf_p95": pct(rtfs, 95),
        "audio_throughput": (total_audio / wall) if wall > 0 else None,
        "request_throughput": (len(dps) / wall) if wall > 0 else None,
    }
    return {
        "batch": B,
        "num_requests": data.get("num_requests"),
        "wall_time_s": wall,
        "recomputed": recomputed,
        "harness_reported": parse_harness_stdout(os.path.join(outdir, "stdout.txt")),
        "datapoints": dps,
    }


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", D, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


def build_raw():
    systems = {}
    for key, label, _ in SYS_META:
        pts = [p for B in BATCHES if (p := load_point(key, B)) is not None]
        if pts:
            systems[key] = {"label": label, "points": pts}
    raw = {
        "benchmark": "qwen3-omni-seedtts-2gpu",
        "figure": "Figure 5 (Qwen3-Omni Seed-TTS, 2-GPU)",
        "timestamp_utc": os.environ.get("RUN_TS_UTC", ""),
        "git_commit": git_commit(),
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "device": {"cuda_visible_devices": "6,7", "gpu_name": "NVIDIA H200", "n_gpus": 2},
        "protocol": {"profiling": "closed_loop", "warmup_iters": 5, "batch_sizes": BATCHES,
                     "max_tokens": 256, "thinker_temperature": 0.0,
                     "audio_pcm": "24kHz int16 mono", "dataset": "seed-tts-eval en",
                     "disaggregation": "Thinker on one GPU; Talker+Code2Wav on the other"},
        "units": {"rtf": "dimensionless e2e/audio_dur (lower=better)",
                  "audio_throughput": "synth audio sec per wall sec (higher=better)"},
        "systems": systems,
    }
    with open(os.path.join(D, "raw.json"), "w") as f:
        json.dump(raw, f, indent=2)
    return raw


def plot(raw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(STYLE)
    os.makedirs(CHARTS, exist_ok=True)
    xpos = list(range(len(BATCHES)))

    def series(key, metric):
        sysd = raw["systems"].get(key)
        if not sysd:
            return None
        by_b = {p["batch"]: p["recomputed"].get(metric) for p in sysd["points"]}
        return [by_b.get(B) for B in BATCHES]

    # (a) RTF — lower is better
    fig, ax = plt.subplots()
    for key, label, color in SYS_META:
        ys = series(key, "rtf_mean")
        if ys is None:
            continue
        xs = [x for x, y in zip(xpos, ys) if y is not None]
        yy = [y for y in ys if y is not None]
        ax.plot(xs, yy, marker="o", color=color, label=label)
    ax.set_xticks(xpos); ax.set_xticklabels([str(b) for b in BATCHES])
    ax.set_xlabel("Batch size (B)"); ax.set_ylabel("RTF  (lower is better)")
    ax.set_title("Qwen3-Omni Seed-TTS, 2-GPU — RTF")
    ax.axhline(1.0, color="#999999", ls="--", lw=0.8)
    ax.legend()
    fig.savefig(os.path.join(CHARTS, "fig5a_rtf.png")); plt.close(fig)

    # (b) Audio throughput — higher is better
    fig, ax = plt.subplots()
    for key, label, color in SYS_META:
        ys = series(key, "audio_throughput")
        if ys is None:
            continue
        xs = [x for x, y in zip(xpos, ys) if y is not None]
        yy = [y for y in ys if y is not None]
        ax.plot(xs, yy, marker="o", color=color, label=label)
    ax.set_xticks(xpos); ax.set_xticklabels([str(b) for b in BATCHES])
    ax.set_xlabel("Batch size (B)")
    ax.set_ylabel("Audio throughput (audio s / wall s)  (higher is better)")
    ax.set_title("Qwen3-Omni Seed-TTS, 2-GPU — Audio throughput")
    ax.legend()
    fig.savefig(os.path.join(CHARTS, "fig5b_throughput.png")); plt.close(fig)


def summarize(raw):
    print("\n=== Recomputed summary (mean RTF / audio throughput) ===")
    for key, label, _ in SYS_META:
        sysd = raw["systems"].get(key)
        if not sysd:
            print(f"{label:12s}: (no data)"); continue
        for p in sysd["points"]:
            r = p["recomputed"]
            print(f"{label:12s} B={p['batch']:<3d} n={r['n']:<3d} "
                  f"RTF mean={r['rtf_mean']!s:8.8} p95={r['rtf_p95']!s:8.8} "
                  f"audio_tput={r['audio_throughput']!s:8.8} "
                  f"harness={p['harness_reported']}")


if __name__ == "__main__":
    raw = build_raw()
    if not raw["systems"]:
        print("No results found yet under runs/out_*/B*/results.json", file=sys.stderr)
        sys.exit(1)
    summarize(raw)
    plot(raw)
    print("\nWrote raw.json + charts/fig5a_rtf.png + charts/fig5b_throughput.png")
