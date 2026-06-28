#!/usr/bin/env python3
"""Per-path benchmark analyzer: build raw.json (every datapoint) + RTF/throughput
charts from a single benchmark dir's runs/out_<system>/B<B>/results.json.

Auto-discovers whichever out_<system> dirs are present, so the SAME script serves
a single-system bench branch (e.g. only out_vllm_omni) and, once several systems'
dirs are merged on the aggregation branch, a multi-system comparison.

results.json carries per-request jct_ms + output_bytes.audio (written by
benchmark.runner). Per-request RTF = (jct_ms/1000) / (audio_bytes / 48000),
audio = 24kHz int16 mono. Aggregates are recomputed here (CLAUDE.md: store
datapoints, compute aggregates downstream) and cross-checked vs stdout.txt.

Usage: bench_analyze.py <bench_dir>
"""
import json, os, re, statistics, sys

BYTES_PER_AUDIO_SEC = 24000 * 2 * 1  # 24kHz int16 mono

# out-dir suffix -> (display label, fixed color). Fixed mapping everywhere so any
# chart is comparable at a glance. M*-old and M*-new share inference-system="ours"
# in results.json, so they are distinguished by their out_<dir> name, not the field.
SYS_META = [
    ("mstar_new",  "M*-new",     "#6a3df0"),
    ("mstar_old",  "M*-old (HF)", "#1f77b4"),
    ("vllm_omni",  "vLLM-Omni",  "#ff7f0e"),
]
ALL_BATCHES = [1, 2, 4, 8, 16, 32]


def pct(sv, p):
    if not sv:
        return None
    if len(sv) == 1:
        return sv[0]
    k = (len(sv) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (k - lo)


def parse_stdout(path):
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
    return out


def load_point(runs, key, B):
    outdir = os.path.join(runs, f"out_{key}", f"B{B}")
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
        dps.append({"request_id": r.get("request_id"), "phase": "measure",
                    "jct_ms": jct_ms, "audio_seconds": asec,
                    "rtf": (jct_ms / 1000.0) / asec if asec > 0 else None})
    rtfs = sorted(d["rtf"] for d in dps if d["rtf"] is not None)
    total_audio = sum(d["audio_seconds"] for d in dps)
    return {
        "batch": B, "num_requests": data.get("num_requests"), "wall_time_s": wall,
        "recomputed": {
            "n": len(dps),
            "rtf_mean": statistics.mean(rtfs) if rtfs else None,
            "rtf_p50": pct(rtfs, 50), "rtf_p95": pct(rtfs, 95), "rtf_p99": pct(rtfs, 99),
            "rtf_std": statistics.pstdev(rtfs) if len(rtfs) > 1 else 0.0,
            "audio_throughput": (total_audio / wall) if wall > 0 else None,
            "request_throughput": (len(dps) / wall) if wall > 0 else None,
        },
        "harness_reported": parse_stdout(os.path.join(outdir, "stdout.txt")),
        "datapoints": dps,
    }


def main(bench_dir):
    runs = os.path.join(bench_dir, "runs")
    charts = os.path.join(bench_dir, "charts")
    style = os.path.join(bench_dir, "..", "chartstyle.mplstyle")
    systems = {}
    for key, label, _ in SYS_META:
        pts = [p for B in ALL_BATCHES if (p := load_point(runs, key, B)) is not None]
        if pts:
            systems[key] = {"label": label, "points": pts}
    if not systems:
        print(f"No results under {runs}/out_*/B*/results.json", file=sys.stderr)
        sys.exit(1)
    # path label from any results.json
    req_type = "?"
    for key in systems:
        for B in ALL_BATCHES:
            rj = os.path.join(runs, f"out_{key}", f"B{B}", "results.json")
            if os.path.isfile(rj):
                req_type = json.load(open(rj)).get("request_type", "?"); break
        break
    batches = sorted({p["batch"] for s in systems.values() for p in s["points"]})
    raw = {
        "benchmark": os.path.basename(bench_dir.rstrip("/")),
        "request_type": req_type,
        "timestamp_utc": os.environ.get("RUN_TS_UTC", ""),
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "device": {"gpu_name": "NVIDIA H200", "n_gpus": 2},
        "protocol": {"profiling": "closed_loop", "warmup_iters": 5, "batch_sizes": batches,
                     "max_tokens": 256, "thinker_temperature": 0.0, "seed": 42,
                     "audio_pcm": "24kHz int16 mono",
                     "disaggregation": "Thinker on one GPU; Talker+Code2Wav on the other"},
        "units": {"rtf": "e2e/audio_dur (lower=better)",
                  "audio_throughput": "synth audio sec / wall sec (higher=better)"},
        "systems": systems,
    }
    json.dump(raw, open(os.path.join(bench_dir, "raw.json"), "w"), indent=2)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(style)
    os.makedirs(charts, exist_ok=True)
    xpos = list(range(len(batches)))
    pretty = {"image_to_speech": "I2S (image→speech)", "audio_to_speech": "S2S (audio→speech)"}.get(req_type, req_type)

    def series(key, metric):
        sysd = systems.get(key)
        if not sysd:
            return None
        by_b = {p["batch"]: p["recomputed"].get(metric) for p in sysd["points"]}
        return [by_b.get(B) for B in batches]

    for metric, fname, ylab, title, hline in [
        ("rtf_mean", "rtf.png", "RTF  (lower is better)", f"Qwen3-Omni {pretty}, 2-GPU — RTF", 1.0),
        ("audio_throughput", "throughput.png", "Audio throughput (audio s / wall s)  (higher is better)",
         f"Qwen3-Omni {pretty}, 2-GPU — Audio throughput", None),
    ]:
        fig, ax = plt.subplots()
        for key, label, color in SYS_META:
            ys = series(key, metric)
            if ys is None:
                continue
            xs = [x for x, y in zip(xpos, ys) if y is not None]
            yy = [y for y in ys if y is not None]
            ax.plot(xs, yy, marker="o", color=color, label=label)
        ax.set_xticks(xpos); ax.set_xticklabels([str(b) for b in batches])
        ax.set_xlabel("Batch size (B)"); ax.set_ylabel(ylab); ax.set_title(title)
        if hline is not None:
            ax.axhline(hline, color="#999999", ls="--", lw=0.8)
        ax.legend()
        fig.savefig(os.path.join(charts, fname)); plt.close(fig)

    print(f"Wrote raw.json + charts for {raw['benchmark']} ({req_type}); systems={list(systems)}")
    for key, label, _ in SYS_META:
        if key in systems:
            for p in systems[key]["points"]:
                r = p["recomputed"]
                print(f"  {label:11s} B={p['batch']:<3d} RTF mean={r['rtf_mean']:.3f} "
                      f"p50={r['rtf_p50']:.3f} p99={r['rtf_p99']:.3f} tput={r['audio_throughput']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
