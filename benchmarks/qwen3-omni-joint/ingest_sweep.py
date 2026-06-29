#!/usr/bin/env python3
"""Ingest sweep output into the joint raw_<path>.json files.

Usage:
  python ingest_sweep.py --sweep-dir /home/tim/tmp/sweep_mnew \
      --system mstar_new --paths s2t,i2t,s2s,i2s \
      [--raw-dir benchmarks/qwen3-omni-joint]

Reads each <sweep-dir>/<short>/B<n>/results.json produced by sweep.sh.
Upserts the system's datapoint into raw_<path>.json (both the datapoints
list and the aggregates dict).  Existing data for other systems is preserved.

After ingesting, run make_proof_charts.py to regenerate charts.
"""
import argparse, json, os, sys, statistics

SHORT_TO_PATH = {
    "s2t": "audio_to_text",
    "s2s": "audio_to_speech",
    "i2t": "image_to_text",
    "i2s": "image_to_speech",
}
BATCHES = [1, 2, 4, 8, 16, 32]


def load_or_init(fp):
    if os.path.exists(fp):
        return json.load(open(fp))
    return {"benchmark": os.path.basename(fp).replace("raw_", "").replace(".json", ""),
            "datapoints": [], "aggregates": {}}


PCM_BYTES_PER_SEC = 48000  # 16-bit 24 kHz mono


def _derive_audio_from_per_request(res):
    """Compute audio_throughput and RTF from per_request when top-level aggregates are absent."""
    per_req = res.get("per_request", [])
    wall = res.get("wall_time_s", 0)
    if not per_req or wall <= 0:
        return None, None
    durations, rtfs = [], []
    for p in per_req:
        ab = p.get("output_bytes", {}).get("audio", 0)
        if ab <= 0:
            continue
        dur = ab / PCM_BYTES_PER_SEC
        durations.append(dur)
        jct_s = p.get("jct_ms", 0) / 1000.0
        if jct_s > 0 and dur > 0:
            rtfs.append(jct_s / dur)
    audio_tp = sum(durations) / wall if durations else None
    rtf_p50 = sorted(rtfs)[len(rtfs) // 2] if rtfs else None
    return audio_tp, rtf_p50


def build_aggregates(res):
    """Build the recomputed + harness aggregate dicts from a results.json."""
    rec = {}
    har = {}

    rec["request_throughput"] = res.get("request_throughput")
    rec["text_token_throughput"] = res.get("text_token_throughput")
    rec["audio_throughput"] = res.get("audio_seconds_throughput")
    rec["rtf_p50"] = res.get("rtf", {}).get("p50") if isinstance(res.get("rtf"), dict) else None

    if rec["audio_throughput"] is None or rec["rtf_p50"] is None:
        derived_tp, derived_rtf = _derive_audio_from_per_request(res)
        if rec["audio_throughput"] is None:
            rec["audio_throughput"] = derived_tp
        if rec["rtf_p50"] is None:
            rec["rtf_p50"] = derived_rtf

    for domain in ["text", "audio"]:
        ttft_src = res.get("ttft", {}).get(domain)
        if isinstance(ttft_src, dict):
            har[f"ttft_{domain}"] = {
                "mean": ttft_src.get("mean"),
                "p50": ttft_src.get("p50"),
                "p95": ttft_src.get("p95"),
                "p99": ttft_src.get("p99"),
            }
        itl_src = res.get("itl", {}).get(domain)
        if isinstance(itl_src, dict):
            har[f"itl_{domain}"] = {
                "mean": itl_src.get("mean"),
                "p50": itl_src.get("p50"),
                "p95": itl_src.get("p95"),
                "p99": itl_src.get("p99"),
            }

    har["req_throughput_reported"] = res.get("request_throughput")
    har["text_tok_throughput_reported"] = res.get("text_token_throughput")
    har["audio_throughput_reported"] = res.get("audio_seconds_throughput")
    har["audio_dur_mean_reported"] = res.get("audio_duration_mean_s")

    return rec, har


def build_datapoint(res, system):
    """Build a single summary datapoint for the datapoints list."""
    dp = {
        "system": system,
        "batch": res.get("batch_size"),
        "phase": "measure",
        "request_throughput": res.get("request_throughput"),
        "completed": res.get("completed"),
        "failed": res.get("failed"),
    }
    for domain in ["text", "audio"]:
        ttft = res.get("ttft", {}).get(domain)
        if isinstance(ttft, dict):
            dp[f"ttft_{domain}_mean"] = ttft.get("mean")
            dp[f"ttft_{domain}_p50"] = ttft.get("p50")
            dp[f"ttft_{domain}_p99"] = ttft.get("p99")
        itl = res.get("itl", {}).get(domain)
        if isinstance(itl, dict):
            dp[f"itl_{domain}_mean"] = itl.get("mean")
    return dp


def ingest(sweep_dir, system, paths, raw_dir):
    for short in paths:
        full_path = SHORT_TO_PATH.get(short, short)
        raw_file = os.path.join(raw_dir, f"raw_{full_path}.json")
        data = load_or_init(raw_file)

        data["datapoints"] = [dp for dp in data["datapoints"]
                              if dp.get("system") != system]

        for b in BATCHES:
            res_file = os.path.join(sweep_dir, short, f"B{b}", "results.json")
            if not os.path.exists(res_file):
                print(f"  SKIP {short} B={b} (no results.json)")
                continue

            res = json.load(open(res_file))

            dp = build_datapoint(res, system)
            data["datapoints"].append(dp)

            bk = f"B{b}"
            data["aggregates"].setdefault(bk, {})
            rec, har = build_aggregates(res)
            data["aggregates"][bk][system] = {
                "recomputed": rec,
                "harness": har,
                "completed": res.get("completed"),
                "num_requests": res.get("num_requests"),
                "provenance": {
                    "git_commit": res.get("git_commit", "unknown"),
                    "build": res.get("build", system),
                    "flags": res.get("flags", "none"),
                },
            }

            req_s = res.get("request_throughput", 0)
            print(f"  {short} B={b}: req/s={req_s:.4f}")

        with open(raw_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  -> {raw_file}")


def main():
    p = argparse.ArgumentParser(description="Ingest sweep results into raw JSON")
    p.add_argument("--sweep-dir", required=True, help="Sweep output directory")
    p.add_argument("--system", required=True, help="System label (mstar_new, mstar_old, vllm)")
    p.add_argument("--paths", required=True, help="Comma-separated: s2t,i2t,s2s,i2s")
    p.add_argument("--raw-dir", default="benchmarks/qwen3-omni-joint",
                   help="Directory containing raw_*.json files")
    args = p.parse_args()

    paths = [x.strip() for x in args.paths.split(",")]
    print(f"Ingesting {args.system} from {args.sweep_dir}")
    print(f"  paths: {paths}")
    print(f"  raw_dir: {args.raw_dir}")
    ingest(args.sweep_dir, args.system, paths, args.raw_dir)
    print("DONE")


if __name__ == "__main__":
    main()
