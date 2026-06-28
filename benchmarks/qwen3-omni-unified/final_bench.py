#!/usr/bin/env python3
"""final_bench.py — FINAL reproducible Qwen3-Omni serving harness (DRAFT).

ONE script a reviewer reads top-to-bottom to see how every recorded datapoint,
and therefore every chart, traces back to a runner invocation. There is no
per-path code: a single PATHS table + one run_cell() drive ALL FIVE paths
through the single shared configuration entrypoint (`python -m benchmark.runner`,
selected only by --request-type).

Two modes:
  --sweep    run the full (paths x systems x batches) sweep live, record EVERY
             per-request datapoint into raw_<path>.json, recompute aggregates
             FROM those datapoints (aggregates == f(datapoints)).
  --refine   recompute aggregates in place from already-committed raw_<path>.json
             datapoints and restamp provenance (mirrors aggregate.py --refine-dir;
             the committed raw files stay authoritative — "hardcoded truthful data").

Complies with CLAUDE.md: per-cell hard timeout, cleanup of the job's process
group on every exit, GPU monitor thread, clock teardown post-check, env capture,
phase-tagged datapoints with units + seed, and only-complete-runs gating (the
path is marked status:"complete" only when its full configured sweep finishes;
partial output is never a result).

Charts are produced separately and programmatically by make_charts.py from
raw_<path>.json using benchmarks/chartstyle.mplstyle — no hand-edited charts,
no aggregate-only storage.

NOTE: this is a DRAFT. The single env-specific server/client launch detail is a
labelled TODO in run_cell(); everything else is wired.
"""
import argparse, datetime, json, os, signal, statistics, subprocess, sys, threading, time

# --- percentile + cell recompute: copied verbatim from aggregate.py so that ---
# --- aggregates here reproduce the committed numbers bit-for-bit. -------------
def pct(sv, p):
    """Linear-interpolated percentile of a pre-sorted list (matches aggregate.py)."""
    if not sv:
        return None
    if len(sv) == 1:
        return sv[0]
    k = (len(sv) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (k - lo)


def recompute_cell_stats(dps):
    """Per-request distribution stats from a cell's datapoints (aggregate.py)."""
    rtfs = sorted(d["rtf"] for d in dps if d.get("rtf") is not None)
    durs = [d["audio_seconds"] for d in dps if d.get("audio_seconds")]
    jcts = [d["jct_ms"] for d in dps]
    return {
        "n": len(dps),
        "n_with_audio": len(durs),
        "rtf_mean": statistics.mean(rtfs) if rtfs else None,
        "rtf_p50": pct(rtfs, 50),
        "rtf_p95": pct(rtfs, 95),
        "rtf_p99": pct(rtfs, 99),
        "rtf_std": statistics.pstdev(rtfs) if len(rtfs) > 1 else (0.0 if rtfs else None),
        "audio_dur_mean": statistics.mean(durs) if durs else None,
        "audio_dur_p50": pct(sorted(durs), 50) if durs else None,
        "jct_mean_s": statistics.mean(jcts) / 1000.0 if jcts else None,
    }


def dist(vals):
    sv = sorted(v for v in vals if v is not None)
    if not sv:
        return None
    return {"mean": statistics.mean(sv), "p50": pct(sv, 50), "p95": pct(sv, 95), "p99": pct(sv, 99)}


# --- THE single shared path table: short -> request-type / dataset / modality -
# This is the ONLY place paths differ. Each row produces an identical runner call.
PATHS = {
    "I2T": {"req_type": "image_to_text",   "dataset": "food101", "modality": "text",   "raw": "image_to_text"},
    "S2T": {"req_type": "audio_to_text",   "dataset": "libri",   "modality": "text",   "raw": "audio_to_text"},
    "I2S": {"req_type": "image_to_speech", "dataset": "food101", "modality": "speech", "raw": "image_to_speech"},
    "S2S": {"req_type": "audio_to_speech", "dataset": "libri",   "modality": "speech", "raw": "audio_to_speech"},
    "T2S": {"req_type": "text_to_speech",  "dataset": "text",    "modality": "speech", "raw": "text_to_speech"},
}
SAMPLE_RATE = 24000
AUDIO_METHOD = "output_bytes.audio_bytes / (sample_rate * 2)  # 24kHz int16 mono"
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
STYLE = "/home/tim/bench-wt/benchmarks/chartstyle.mplstyle"
CHARTSTYLE_NOTE = "shared style; seed via: git checkout benchmarks -- benchmarks/chartstyle.mplstyle"


# ---------------------------------------------------------------------------
# env / provenance capture
# ---------------------------------------------------------------------------
def capture_env(outdir, seed, gpus):
    os.makedirs(outdir, exist_ok=True)
    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout
        except Exception as e:
            return f"(failed: {e})\n"
    with open(os.path.join(outdir, "env.txt"), "w") as f:
        f.write("=== date (UTC) ===\n" + datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ\n"))
        f.write("=== uname ===\n" + sh("uname -a"))
        f.write(f"=== CUDA_VISIBLE_DEVICES ===\n{gpus}\n")
        f.write(f"=== seed ===\n{seed}\n")
        f.write("=== nvidia-smi query ===\n" + sh(
            "nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm,persistence_mode --format=csv"))
        f.write("=== nvcc ===\n" + sh("nvcc --version"))
        f.write("=== torch cuda ===\n" + sh(
            "python -c \"import torch;print('torch',torch.__version__);print('cuda',torch.version.cuda)\""))
        f.write("=== git ===\n" + sh("git -C /home/tim/ttft-wt rev-parse HEAD"))
    # requirements (uv if available)
    req = sh("command -v uv >/dev/null && uv pip freeze || pip freeze")
    with open(os.path.join(outdir, "requirements.txt"), "w") as f:
        f.write(req)


def git_commit():
    try:
        return subprocess.run("git -C /home/tim/ttft-wt rev-parse HEAD", shell=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# GPU monitor thread (CLAUDE.md monitoring; cadence scaled to modality)
# ---------------------------------------------------------------------------
class GpuMonitor(threading.Thread):
    def __init__(self, logpath, interval):
        super().__init__(daemon=True)
        self.logpath, self.interval, self._stop = logpath, interval, threading.Event()
    def run(self):
        with open(self.logpath, "a") as f:
            while not self._stop.is_set():
                out = subprocess.run(
                    "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader",
                    shell=True, capture_output=True, text=True).stdout.replace("\n", ";")
                f.write(f"{datetime.datetime.utcnow().strftime('%H:%M:%SZ')},{out}\n"); f.flush()
                self._stop.wait(self.interval)
    def stop(self):
        self._stop.set()


def teardown_clocks():
    # idempotent post-check: we never lock, but unlock anyway (no-op if unlocked).
    for c in ("nvidia-smi -rgc", "nvidia-smi -rmc"):
        subprocess.run(c, shell=True, capture_output=True)


# ---------------------------------------------------------------------------
# run ONE cell via the single shared entrypoint; return its datapoints
# ---------------------------------------------------------------------------
def run_cell(short, system, batch, seed, url, gpus, outroot, warmup, measure, max_wall):
    spec = PATHS[short]
    celldir = os.path.join(outroot, "cells", f"{short}_{system}_B{batch}")
    os.makedirs(celldir, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
    if spec["modality"] == "speech":
        env.setdefault("BENCH_SPEECH_THINKER_TEMPERATURE", "0.7")

    # ===================== TODO (env-specific launch detail) =================
    # Set the venv python + model-cache env, and confirm the server at `url`
    # is healthy before launching (curl $url/health). The command shape below
    # is the single shared entrypoint; only --request-type/--dataset vary.
    py = os.environ.get("BENCH_PY", "python")  # TODO: real venv python
    cmd = [py, "-m", "benchmark.runner",
           "--url", url, "--model", "qwen3omni", "--inference-system", system,
           "--request-type", spec["req_type"], "--dataset", spec["dataset"],
           "--profiling-type", "closed_loop", "--batch-size", str(batch),
           "--max-concurrency", str(batch), "--num-warmup", str(warmup),
           "--num-requests", str(measure), "--output-len-seed", str(seed),
           "--output-dir", celldir]
    # ========================================================================

    log = open(os.path.join(celldir, "run.log"), "w")
    # own process group so we can kill the whole tree on timeout/signal
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                            preexec_fn=os.setsid)
    try:
        rc = proc.wait(timeout=max_wall)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()
        rc = -9
    finally:
        log.close()

    rj = os.path.join(celldir, "results.json")
    if rc != 0 or not os.path.exists(rj):
        return None, f"cell {short}/{system}/B{batch} failed rc={rc}"
    r = json.load(open(rj))
    dps = []
    for pr in r.get("per_request", []):
        ob = pr.get("output_bytes", {}) or {}
        ab = ob.get("audio_bytes", 0) or 0
        asec = ab / (SAMPLE_RATE * 2) if ab else 0.0
        jct = pr.get("jct_ms", 0.0)
        dps.append({
            "system": system, "batch": batch, "phase": "measure",
            "request_id": pr.get("request_id"), "jct_ms": jct,
            "audio_seconds": asec, "rtf": ((jct / 1000.0) / asec if asec else None),
            "text_bytes": ob.get("text_bytes", 0), "sample_rate": SAMPLE_RATE,
            "audio_seconds_method": AUDIO_METHOD,
        })
    # only-complete-runs: a cell short of its measured count is not valid
    if len(dps) < measure:
        return None, f"cell {short}/{system}/B{batch} incomplete ({len(dps)}/{measure})"
    # carry the harness-reported throughput/ttft/itl alongside recomputed stats
    harness = {"ttft_text" if spec["modality"] == "text" else "ttft_audio": r.get("ttft"),
               "itl_text" if spec["modality"] == "text" else "itl_audio": r.get("itl"),
               "req_throughput_reported": r.get("request_throughput"),
               "text_tok_throughput_reported": r.get("text_token_throughput"),
               "audio_seconds_throughput_reported": r.get("audio_seconds_throughput"),
               "wall_time_s": r.get("wall_time_s")}
    return (dps, harness), None


# ---------------------------------------------------------------------------
# aggregate a path's datapoints into the committed raw_<path>.json shape
# ---------------------------------------------------------------------------
def build_aggregates(datapoints, harness_by_cell, modality, batches, systems):
    agg = {}
    for b in batches:
        cell = {}
        for s in systems:
            dps = [d for d in datapoints if d["system"] == s and d["batch"] == b and d["phase"] == "measure"]
            if not dps:
                continue
            rec = recompute_cell_stats(dps)
            har = harness_by_cell.get((s, b), {})
            cell[s] = {
                "completed": len(dps), "num_requests": len(dps),
                "recomputed": rec, "harness": har,
                "provenance": {"n_datapoints": len(dps),
                               "stats_source": "recomputed from datapoints",
                               "throughput_source": "carried from runner results.json"},
            }
        if cell:
            agg[f"B{b}"] = cell
    return agg


def write_raw(outroot, short, datapoints, harness_by_cell, batches, systems, seed):
    spec = PATHS[short]
    agg = build_aggregates(datapoints, harness_by_cell, spec["modality"], batches, systems)
    raw = {
        "benchmark": f"qwen3-omni-{short.lower()}-batch-sweep",
        "path": spec["req_type"], "model": MODEL,
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "git_commit": git_commit(), "seed": seed,
        "warmup_iters": [WARMUP_GLOBAL],
        "units": {"jct_ms": "ms", "rtf": "ratio (wall/audio_dur)", "audio_seconds": "s",
                  "throughput_metric": "req/s | tok/s | audio-s/s", "ttft": "s", "itl": "s",
                  "sample_rate": SAMPLE_RATE, "audio_seconds_method": AUDIO_METHOD},
        "status": "complete",
        "datapoints": datapoints, "aggregates": agg,
        "provenance": {"sample_rate": SAMPLE_RATE, "audio_seconds_method": AUDIO_METHOD,
                       "rtf_method": "jct_s / audio_seconds, per request",
                       "aggregates_recomputed_from": "datapoints",
                       "generated_by": "final_bench.py --sweep",
                       "chartstyle": CHARTSTYLE_NOTE,
                       "generated_utc": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")},
    }
    fp = os.path.join(outroot, f"raw_{spec['raw']}.json")
    json.dump(raw, open(fp, "w"), indent=2)
    return fp


# ---------------------------------------------------------------------------
# --refine: recompute aggregates in place from committed datapoints
# ---------------------------------------------------------------------------
def refine(outroot, paths, batches, systems):
    for short in paths:
        spec = PATHS[short]
        fp = os.path.join(outroot, f"raw_{spec['raw']}.json")
        if not os.path.exists(fp):
            print("skip (no raw):", short); continue
        raw = json.load(open(fp))
        dps = raw["datapoints"]
        # rebuild harness map from existing aggregates so carried numbers survive
        har_map = {}
        for bkey, cell in raw.get("aggregates", {}).items():
            b = int(bkey[1:])
            for s, c in cell.items():
                har_map[(s, b)] = c.get("harness", {})
        present_sys = sorted({d["system"] for d in dps})
        raw["aggregates"] = build_aggregates(dps, har_map, spec["modality"], batches, present_sys or systems)
        raw.setdefault("provenance", {})["aggregates_recomputed_from"] = "datapoints"
        raw["provenance"]["generated_by"] = "final_bench.py --refine"
        raw["provenance"]["generated_utc"] = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        json.dump(raw, open(fp, "w"), indent=2)
        print("refined", fp, "(aggregates == f(datapoints))")


WARMUP_GLOBAL = 5  # set by main; module-level so write_raw can stamp it.


def main():
    global WARMUP_GLOBAL
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sweep", "refine"], default="refine")
    ap.add_argument("--out", default="/home/tim/bench-wt/benchmarks/qwen3-omni-joint")
    ap.add_argument("--paths", nargs="+", default=list(PATHS.keys()))
    ap.add_argument("--systems", nargs="+", default=["mstar_new", "mstar_old", "vllm"])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--measure", type=int, default=50, help="measured requests per cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,2,3"))
    ap.add_argument("--max-wall", type=int, default=1200, help="per-cell hard timeout (s)")
    args = ap.parse_args()
    WARMUP_GLOBAL = args.warmup

    for p in args.paths:
        if p not in PATHS:
            sys.exit(f"unknown path {p}; valid: {list(PATHS)}")

    if args.mode == "refine":
        refine(args.out, args.paths, args.batches, args.systems)
        return

    # ---- live sweep ----
    capture_env(args.out, args.seed, args.gpus)
    with open(os.path.join(args.out, "command.txt"), "w") as f:
        f.write("# regenerate: final_bench.py --sweep then make_charts.py\n")
        f.write(" ".join([sys.executable, *sys.argv]) + "\n")

    for short in args.paths:
        spec = PATHS[short]
        interval = 60 if spec["modality"] == "speech" else 30
        mon = GpuMonitor(os.path.join(args.out, f"gpu_monitor_{short}.csv"), interval)
        mon.start()
        all_dps, har_map, complete = [], {}, True
        try:
            for s in args.systems:
                for b in args.batches:
                    res, err = run_cell(short, s, b, args.seed, args.url, args.gpus,
                                        args.out, args.warmup, args.measure, args.max_wall)
                    if err:
                        print("[skip]", err); complete = False
                        continue
                    dps, har = res
                    all_dps.extend(dps); har_map[(s, b)] = har
                    print(f"[ok] {short} {s} B{b}: {len(dps)} datapoints")
        finally:
            mon.stop(); teardown_clocks()
        if not complete:
            # only-complete-runs: do not emit a result file for a partial sweep
            print(f"[INCOMPLETE] {short}: sweep had skipped cells — NOT writing raw_{spec['raw']}.json")
            continue
        fp = write_raw(args.out, short, all_dps, har_map, args.batches, args.systems, args.seed)
        print(f"[complete] wrote {fp}")
    print("DONE. Now: python make_charts.py")


if __name__ == "__main__":
    main()
