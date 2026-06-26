"""Record + aggregate serving-benchmark results so EVERY value is captured,
including runs that crashed ("didn't run" becomes data, not a silent gap).

Two subcommands:

  fail   — write a status=failed results.json for a cell whose run produced no
           results.json (timeout / crash), capturing the log tail. So a failed
           (system, path, batch) is recorded explicitly instead of just missing.

  collect — walk the <series>/<path>/bs<N>/results.json tree and emit one flat
            table (CSV + JSON) with every metric and a status column, for ALL
            cells across all four runtimes and the full batch sweep.

Usage:
  python -m benchmark.serving_scripts.bench_record fail \
      --dir $OUT/$path/bs$bs --system ours_hf --path $path --bs $bs --log $od/run.log
  python -m benchmark.serving_scripts.bench_record collect \
      --data-root $OUT_ROOT --out benchmark/artifacts/serving/all_results
"""
from __future__ import annotations

import argparse
import csv
import json
import os


def _ls(stats, key="p50"):
    if not stats:
        return None
    for k in (key, "p50", "mean"):
        if isinstance(stats, dict) and stats.get(k) is not None:
            return stats[k]
    return None


def _modal(map_, key="p50"):
    if not map_:
        return None
    vals = [_ls(v, key) for v in map_.values()]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def cmd_fail(args):
    os.makedirs(args.dir, exist_ok=True)
    tail = ""
    if args.log and os.path.isfile(args.log):
        with open(args.log, errors="replace") as fh:
            tail = "".join(fh.readlines()[-20:])
    rec = {
        "system": args.system, "inference_system": args.system,
        "request_type": args.path, "batch_size": int(args.bs),
        "status": "failed", "error_tail": tail[-2000:],
    }
    with open(os.path.join(args.dir, "results.json"), "w") as fh:
        json.dump(rec, fh, indent=2)
    print(f"recorded FAILED: {args.system}/{args.path}/bs{args.bs}")


def cmd_collect(args):
    rows = []
    root = args.data_root
    for series in sorted(os.listdir(root)):
        sdir = os.path.join(root, series)
        if not os.path.isdir(sdir):
            continue
        for path in sorted(os.listdir(sdir)):
            pdir = os.path.join(sdir, path)
            if not os.path.isdir(pdir):
                continue
            for bsdir in sorted(os.listdir(pdir)):
                rj = os.path.join(pdir, bsdir, "results.json")
                if not os.path.isfile(rj):
                    continue
                with open(rj) as fh:
                    d = json.load(fh)
                rows.append({
                    "series": series,
                    "request_type": d.get("request_type", path),
                    "batch_size": d.get("batch_size") or bsdir.replace("bs", ""),
                    "status": d.get("status", "ok"),
                    "ttft_p50_ms": _scale(_modal(d.get("ttft"), "p50"), 1000),
                    "ttft_p95_ms": _scale(_modal(d.get("ttft"), "p95"), 1000),
                    "itl_p50_ms": _scale(_modal(d.get("itl"), "p50"), 1000),
                    "itl_p95_ms": _scale(_modal(d.get("itl"), "p95"), 1000),
                    "rtf_p50": _ls(d.get("rtf"), "p50"),
                    "rtf_p95": _ls(d.get("rtf"), "p95"),
                    "audio_sec_throughput": d.get("audio_seconds_throughput"),
                    "request_throughput": d.get("request_throughput"),
                    "text_token_throughput": d.get("text_token_throughput"),
                    "jct_p50_ms": d.get("jct_p50_ms"),
                })
    rows.sort(key=lambda r: (r["request_type"], str(r["batch_size"]), r["series"]))
    cols = list(rows[0].keys()) if rows else []
    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(args.out + ".json", "w") as fh:
        json.dump(rows, fh, indent=2)
    n_ok = sum(1 for r in rows if r["status"] == "ok")
    n_fail = sum(1 for r in rows if r["status"] == "failed")
    print(f"collected {len(rows)} cells ({n_ok} ok, {n_fail} failed) -> {args.out}.csv/.json")


def _scale(v, k):
    return None if v is None else round(v * k, 4)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fail")
    f.add_argument("--dir", required=True)
    f.add_argument("--system", required=True)
    f.add_argument("--path", required=True)
    f.add_argument("--bs", required=True)
    f.add_argument("--log", default=None)
    f.set_defaults(func=cmd_fail)
    c = sub.add_parser("collect")
    c.add_argument("--data-root", required=True)
    c.add_argument("--out", default="benchmark/artifacts/serving/all_results")
    c.set_defaults(func=cmd_collect)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
