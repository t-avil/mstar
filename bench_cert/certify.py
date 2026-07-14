"""Certification harness: makes M*-vs-vLLM comparisons trustworthy.

    python -m benchmark.certify --url http://localhost:8344 \\
        --request-type image_to_text --paired-flag MSTAR_COADMIT=0:1 \\
        --dynflags-path /tmp/dynflags.json --pairs 2

Every chronic false alarm in this project's recent history (see
LEARNINGS_TTFT.md / LEARNINGS_FIX20.md) was a *single loaded cell* mistaken
for a signal: budget "regression", ordered-emit's "30% cost", the
576-vs-2532ms encoders scare. All three were cross-load or single-sample
comparisons, not code effects. This module encodes the fix as a protocol,
not a reminder:

  1. Load-gate: refuse to start (or explicitly mark UNTRUSTWORTHY) a cell
     that starts under host contention.
  2. Length-matched cells: --ignore-eos + a fixed output length removes the
     ~1.20x M*-vs-vLLM raw-tok/s artifact caused by different natural EOS
     behavior (M* ~176 tok/req vs vLLM ~212 tok/req on food101).
  3. n >= protocol minimum (96 for i2t/i2s, 256 for a2t/a2s -- s2t B32 at
     n<96 is a measured wave lottery, 39.7 vs 24.6 rps on identical flags).
  4. Paired back-to-back OFF/ON/OFF/ON repeats (when --paired-flag is given,
     toggled live via MSTAR_DYNFLAGS -- no reboot, no time-separated noise)
     with an explicit sign-consistency check across pairs, so a single
     lucky/unlucky pair can't masquerade as a verdict.
  5. An explicit UNTRUSTWORTHY stamp when load drifted materially between
     the first and second half of the run, instead of silently trusting a
     number produced while the box was moving.

This module is pure client side: it drives benchmark/runner.py as a
subprocess and never touches a GPU, starts a server, or writes model code.
It has no import dependency on the `benchmark` or `mstar` packages beyond
invoking `python -m benchmark.runner` as a subprocess, so it can be dropped
into any checkout that has that entrypoint.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class CertAbort(Exception):
    """Raised to stop the protocol early (load-gate timeout, cell failure)."""


# ---------------------------------------------------------------------------
# Committed vLLM-Omni 0.22 reference bands (benchmarks branch, h2h_* results).
# Precise, per-run comparisons should prefer --vllm-json over these constants
# -- these exist as a documented fallback / sanity check, not ground truth.
# ---------------------------------------------------------------------------
REFERENCE_BANDS: dict = {
    "image_to_text": {
        "batch": 32,
        "request_throughput_rps": (8.03, 8.59),
        "ttft_text_ms": (170.0, 190.0),  # ~179ms point estimate, quiet-window
        "length_norm_factor": 1.20,
        "note": (
            "M* generates ~176 tok/req vs vLLM ~212 tok/req on food101 under "
            "natural (non-length-matched) sampling: raw tok/s comparisons carry "
            "a ~1.20x length artifact. Pass --output-len 212 (with --ignore-eos, "
            "which certify sets automatically) to remove it before trusting a "
            "tok/s delta."
        ),
    },
    "audio_to_text": {
        "batch": 32,
        "ttft_text_ms": (215.0, 282.0),
        "text_token_throughput_tok_s": (598.0, 664.0),
        "note": "s2t B32 needs n>=256 -- n<96 measured as a wave lottery (39.7 vs 24.6 rps, identical flags).",
    },
}

DEFAULT_N = {
    "image_to_text": 96,
    "image_to_speech": 96,
    "audio_to_text": 256,
    "audio_to_speech": 256,
}
DEFAULT_OUTPUT_LEN = {
    "image_to_text": 212,
    "audio_to_text": 512,
}
# Metrics compared pairwise / against reference bands. Keys match the flat
# dict produced by extract_metrics().
COMPARE_METRICS = [
    "request_throughput_rps",
    "ttft_text_p50_ms",
    "text_token_throughput_tok_s",
    "itl_text_p50_ms",
]

# Standalone choice lists (mirror benchmark.base enums) so this module never
# needs to import the mstar/benchmark package tree -- keeps it a pure,
# drop-in client-side tool.
REQUEST_TYPE_CHOICES = [
    "text_to_text", "text_to_image", "text_to_speech",
    "vision_language_action", "video_to_video",
    "image_to_text", "image_to_image", "image_to_speech",
    "audio_to_text", "audio_to_speech",
    "video_to_text", "video_to_speech",
]
MODEL_CHOICES = ["bagel", "orpheus", "qwen3omni", "pi05", "vjepa2ac"]
INFERENCE_SYSTEM_CHOICES = ["ours", "ours_openai", "vllm_omni", "vox_serve", "sglang_omni"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _f(v, nd=2) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def _s_to_ms(v: Optional[float]) -> Optional[float]:
    return None if v is None else v * 1000.0


# ---------------------------------------------------------------------------
# Load gate
# ---------------------------------------------------------------------------

def read_load1() -> float:
    return os.getloadavg()[0]


def load_gate(max_load: float, timeout_s: float, poll_s: float, strict: bool, log=print) -> float:
    """Block until 1-min loadavg <= max_load, or timeout.

    On timeout: raises CertAbort if strict, else logs a warning and returns
    the over-threshold reading (the caller is responsible for reflecting
    this in the untrustworthy stamp).
    """
    start = time.monotonic()
    while True:
        load1 = read_load1()
        if load1 <= max_load:
            return load1
        elapsed = time.monotonic() - start
        if elapsed >= timeout_s:
            msg = f"load-gate timeout: load1={load1:.1f} > max_load={max_load:.1f} after {elapsed:.0f}s"
            if strict:
                raise CertAbort(msg)
            log(f"[load-gate] WARNING: {msg}; proceeding anyway (--no-strict-load-gate)")
            return load1
        log(f"[load-gate] load1={load1:.1f} > {max_load:.1f}; waiting {poll_s:.0f}s ({elapsed:.0f}/{timeout_s:.0f}s)")
        time.sleep(poll_s)


# ---------------------------------------------------------------------------
# MSTAR_DYNFLAGS toggling
# ---------------------------------------------------------------------------

def write_dynflags(path: str, flag: str, value: str) -> None:
    """Atomically write {flag: value} to the dynflags JSON the server polls."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump({flag: value}, f)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Cell execution
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    index: int
    flag_state: str  # "OFF" / "ON" / "n/a"
    flag_value: Optional[str]
    load_before: float
    load_after: float
    duration_s: float
    results_path: str
    metrics: dict = field(default_factory=dict)


def extract_metrics(results: dict) -> dict:
    """Flatten a runner.py results.json into the scalars certify compares."""
    out = {
        "request_throughput_rps": results.get("request_throughput"),
        "text_token_throughput_tok_s": results.get("text_token_throughput"),
        "completed": results.get("completed"),
        "failed": results.get("failed"),
        "wall_time_s": results.get("wall_time_s"),
    }
    ttft_text = (results.get("ttft") or {}).get("text") or {}
    out["ttft_text_p50_ms"] = _s_to_ms(ttft_text.get("p50"))
    out["ttft_text_mean_ms"] = _s_to_ms(ttft_text.get("mean"))
    itl_text = (results.get("itl") or {}).get("text") or {}
    out["itl_text_p50_ms"] = _s_to_ms(itl_text.get("p50"))
    e2e = results.get("e2e_latency") or {}
    out["e2e_p50_ms"] = _s_to_ms(e2e.get("p50"))
    # Derived mean tokens/req -- this is what makes the 1.20x length artifact
    # visible instead of silently baked into a raw tok/s number.
    tput, wall, completed = out["text_token_throughput_tok_s"], out["wall_time_s"], out["completed"]
    out["mean_tokens_per_req"] = (tput * wall / completed) if (tput and wall and completed) else None
    return out


def build_runner_cmd(args, output_dir: str) -> list:
    cmd = [
        sys.executable, "-m", "benchmark.runner",
        "--url", args.url,
        "--model", args.model,
        "--request-type", args.request_type,
        "--num-requests", str(args.num_requests),
        "--num-warmup", str(args.num_warmup),
        "--profiling-type", "closed_loop",
        "--max-concurrency", str(args.max_concurrency),
        "--inference-system", args.inference_system,
        "--output-dir", output_dir,
    ]
    if args.output_len is not None:
        cmd += ["--output-len-min", str(args.output_len), "--output-len-max", str(args.output_len), "--ignore-eos"]
    if args.dataset:
        cmd += ["--dataset", args.dataset]
    return cmd


def run_cell(args, work_dir: Path, index: int, flag_state: str, flag_value: Optional[str], repo_dir: Path, log=print) -> CellResult:
    if args.paired_flag is not None and flag_value is not None:
        write_dynflags(args.dynflags_path, args.paired_flag_name, flag_value)
        log(f"[cell {index}] wrote dynflags {args.paired_flag_name}={flag_value} -> {args.dynflags_path}; settling {args.flag_settle_s:.0f}s")
        time.sleep(args.flag_settle_s)

    load_before = load_gate(args.max_load, args.load_gate_timeout_s, args.load_poll_s, args.strict_load_gate, log=log)

    cell_dir = work_dir / f"cell_{index}_{flag_state}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_runner_cmd(args, str(cell_dir))
    log(f"[cell {index}] ({flag_state}) load1={load_before:.1f} :: {' '.join(cmd)}")

    t0 = time.monotonic()
    proc = subprocess.run(
        cmd, cwd=str(repo_dir), timeout=args.cell_timeout_s,
        capture_output=not args.verbose,
    )
    duration = time.monotonic() - t0
    if proc.returncode != 0:
        tail = ""
        if proc.stderr:
            tail = proc.stderr.decode(errors="replace")[-2000:]
        raise CertAbort(f"cell {index} runner exited {proc.returncode}: {tail}")

    load_after = read_load1()
    results_path = cell_dir / "results.json"
    if not results_path.exists():
        raise CertAbort(f"cell {index}: runner did not write {results_path}")
    metrics = extract_metrics(json.loads(results_path.read_text()))
    return CellResult(
        index=index, flag_state=flag_state, flag_value=flag_value,
        load_before=load_before, load_after=load_after, duration_s=duration,
        results_path=str(results_path), metrics=metrics,
    )


def fabricate_dry_run_cells() -> list:
    """Two fake OFF/ON cells for --dry-run (no server, no subprocess)."""
    off = CellResult(
        index=0, flag_state="OFF", flag_value="0", load_before=12.3, load_after=13.1,
        duration_s=11.2, results_path="<dry-run>",
        metrics={
            "request_throughput_rps": 8.10, "text_token_throughput_tok_s": 1720.0,
            "ttft_text_p50_ms": 640.0, "itl_text_p50_ms": 12.4,
            "completed": 96, "failed": 0, "wall_time_s": 11.85,
            "mean_tokens_per_req": 212.0,
        },
    )
    on = CellResult(
        index=1, flag_state="ON", flag_value="1", load_before=13.5, load_after=14.0,
        duration_s=10.6, results_path="<dry-run>",
        metrics={
            "request_throughput_rps": 8.46, "text_token_throughput_tok_s": 1793.0,
            "ttft_text_p50_ms": 601.0, "itl_text_p50_ms": 12.1,
            "completed": 96, "failed": 0, "wall_time_s": 11.34,
            "mean_tokens_per_req": 212.0,
        },
    )
    return [off, on]


# ---------------------------------------------------------------------------
# Pairing / sign consistency / untrustworthy stamp
# ---------------------------------------------------------------------------

def build_pairs(cells: list) -> list:
    """Consecutive OFF, ON cells become one pair. Skips anything malformed
    rather than guessing, since a mispaired OFF/ON comparison is worse than
    no comparison."""
    pairs = []
    i = 0
    while i + 1 < len(cells):
        off, on = cells[i], cells[i + 1]
        if off.flag_state != "OFF" or on.flag_state != "ON":
            i += 1
            continue
        deltas = {}
        for m in COMPARE_METRICS:
            ov, nv = off.metrics.get(m), on.metrics.get(m)
            if ov is None or nv is None or ov == 0:
                deltas[m] = None
                continue
            sign = 1 if nv > ov else (-1 if nv < ov else 0)
            deltas[m] = {"off": ov, "on": nv, "delta_pct": (nv - ov) / abs(ov) * 100.0, "sign": sign}
        pairs.append({"pair_idx": len(pairs), "off_cell": off.index, "on_cell": on.index, "deltas": deltas})
        i += 2
    return pairs


def sign_consistency(pairs: list) -> dict:
    out = {}
    for m in COMPARE_METRICS:
        signs = [p["deltas"][m]["sign"] for p in pairs if p["deltas"].get(m) is not None]
        if not signs:
            out[m] = {"consistent": None, "signs": []}
            continue
        nonzero = [s for s in signs if s != 0]
        out[m] = {"consistent": (len(set(nonzero)) <= 1) if nonzero else True, "signs": signs}
    return out


def compute_untrustworthy(cells: list, max_load: float, drift_pct_threshold: float) -> tuple:
    """UNTRUSTWORTHY when load moved materially between the first and second
    half of the cells (chronological), or any cell finished well over the
    load ceiling. This is the direct fix for the encoders '576-vs-2532ms'
    false alarm, which was exactly a cross-load comparison mistaken for a
    code effect."""
    reasons = []
    loads = [(c.load_before + c.load_after) / 2.0 for c in cells]
    if len(loads) >= 2:
        mid = max(1, len(loads) // 2)
        first_half, second_half = loads[:mid], loads[mid:] or loads[-1:]
        m1, m2 = statistics.mean(first_half), statistics.mean(second_half)
        if m1 > 0:
            drift_pct = abs(m2 - m1) / m1 * 100.0
            if drift_pct > drift_pct_threshold:
                reasons.append(
                    f"load drifted {drift_pct:.0f}% between first-half cells (avg {m1:.1f}) "
                    f"and second-half cells (avg {m2:.1f}) -- exceeds {drift_pct_threshold:.0f}% threshold"
                )
    for c in cells:
        if c.load_after > max_load * 1.5:
            reasons.append(f"cell {c.index} ({c.flag_state}) ended at load {c.load_after:.1f}, >1.5x max_load {max_load:.1f}")
    return (len(reasons) > 0, reasons)


# ---------------------------------------------------------------------------
# Reference-band comparison
# ---------------------------------------------------------------------------

def compare_to_bands(candidate_metrics: dict, request_type: str) -> list:
    band = REFERENCE_BANDS.get(request_type)
    if band is None:
        return []
    rows = []
    for metric_key, band_key, unit in [
        ("request_throughput_rps", "request_throughput_rps", " rps"),
        ("ttft_text_p50_ms", "ttft_text_ms", " ms"),
        ("text_token_throughput_tok_s", "text_token_throughput_tok_s", " tok/s"),
    ]:
        rng = band.get(band_key)
        if rng is None:
            continue
        lo, hi = rng
        cand = candidate_metrics.get(metric_key)
        rows.append({
            "metric": metric_key, "candidate": cand, "band": f"{lo}-{hi}{unit}",
            "in_band": (lo <= cand <= hi) if cand is not None else None,
        })
    if "note" in band:
        rows.append({"metric": "note", "candidate": None, "band": band["note"], "in_band": None})
    return rows


def compare_jsons(candidate_path: str, vllm_path: str) -> dict:
    """Direct head-to-head against a committed vLLM results.json (e.g. a
    benchmarks-branch h2h_* path) instead of the hardcoded bands."""
    cand = extract_metrics(json.loads(Path(candidate_path).read_text()))
    ref = extract_metrics(json.loads(Path(vllm_path).read_text()))
    rows = []
    for m in COMPARE_METRICS:
        cv, rv = cand.get(m), ref.get(m)
        delta_pct = ((cv - rv) / abs(rv) * 100.0) if (cv is not None and rv not in (None, 0)) else None
        rows.append({"metric": m, "candidate": cv, "vllm_reference": rv, "delta_pct": delta_pct})
    rows.append({"metric": "mean_tokens_per_req", "candidate": cand.get("mean_tokens_per_req"),
                 "vllm_reference": ref.get("mean_tokens_per_req"), "delta_pct": None})
    return {"candidate_path": candidate_path, "vllm_path": vllm_path, "rows": rows}


# ---------------------------------------------------------------------------
# Verdict assembly + rendering
# ---------------------------------------------------------------------------

def build_verdict(args, cells: list) -> dict:
    pairs = build_pairs(cells) if args.paired_flag else []
    consistency = sign_consistency(pairs) if pairs else {}
    untrustworthy, reasons = compute_untrustworthy(cells, args.max_load, args.load_drift_pct)

    n_min = DEFAULT_N.get(args.request_type)
    if n_min is not None and args.num_requests < n_min:
        untrustworthy = True
        reasons.append(f"num_requests={args.num_requests} < protocol minimum {n_min} for {args.request_type}")

    verdict = {
        "generated_at": _now_iso(),
        "protocol": {
            "url": args.url, "model": args.model, "request_type": args.request_type,
            "num_requests": args.num_requests, "max_concurrency": args.max_concurrency,
            "output_len": args.output_len, "ignore_eos": args.output_len is not None,
            "paired_flag": args.paired_flag_name if args.paired_flag else None,
            "max_load": args.max_load, "load_drift_pct_threshold": args.load_drift_pct,
        },
        "cells": [
            {
                "index": c.index, "flag_state": c.flag_state, "flag_value": c.flag_value,
                "load_before": c.load_before, "load_after": c.load_after,
                "duration_s": c.duration_s, "results_path": c.results_path, "metrics": c.metrics,
            }
            for c in cells
        ],
        "pairs": pairs,
        "sign_consistency": consistency,
        "untrustworthy": untrustworthy,
        "untrustworthy_reasons": reasons,
    }
    if args.request_type in REFERENCE_BANDS and cells:
        verdict["reference_band_compare"] = compare_to_bands(cells[-1].metrics, args.request_type)
    if args.vllm_json:
        verdict["reference_json_compare"] = compare_jsons(cells[-1].results_path, args.vllm_json) if cells else None
    return verdict


def render_markdown(verdict: dict) -> str:
    p = verdict["protocol"]
    lines = [f"# Certification verdict -- {p['request_type']} B{p['max_concurrency']}", ""]
    lines.append(f"Generated: {verdict['generated_at']}  |  url: {p['url']}  |  n={p['num_requests']}")
    if p["paired_flag"]:
        lines.append(f"Paired flag: `{p['paired_flag']}`")
    lines.append("")
    if verdict["untrustworthy"]:
        lines.append("**STAMP: UNTRUSTWORTHY**")
        for r in verdict["untrustworthy_reasons"]:
            lines.append(f"- {r}")
    else:
        lines.append("**STAMP: trustworthy** (load-gated, n >= protocol minimum, sign-checked)")
    lines += ["", "## Cells", "",
              "| # | flag | rps | ttft p50 (ms) | tok/s | itl p50 (ms) | mean tok/req | load before->after |",
              "|---|---|---|---|---|---|---|---|"]
    for c in verdict["cells"]:
        m = c["metrics"]
        lines.append(
            f"| {c['index']} | {c['flag_state']} | {_f(m.get('request_throughput_rps'))} | "
            f"{_f(m.get('ttft_text_p50_ms'), 1)} | {_f(m.get('text_token_throughput_tok_s'), 1)} | "
            f"{_f(m.get('itl_text_p50_ms'), 2)} | {_f(m.get('mean_tokens_per_req'), 1)} | "
            f"{c['load_before']:.1f}->{c['load_after']:.1f} |"
        )
    if verdict["pairs"]:
        lines += ["", "## Paired deltas (ON vs OFF)", "",
                  "| pair | metric | off | on | delta % |", "|---|---|---|---|---|"]
        for pr in verdict["pairs"]:
            for m, d in pr["deltas"].items():
                if d is None:
                    continue
                lines.append(f"| {pr['pair_idx']} | {m} | {_f(d['off'])} | {_f(d['on'])} | {d['delta_pct']:+.1f}% |")
        lines += ["", "## Sign consistency across pairs", "",
                  "| metric | consistent | signs |", "|---|---|---|"]
        for m, s in verdict["sign_consistency"].items():
            lines.append(f"| {m} | {s['consistent']} | {s['signs']} |")
    if verdict.get("reference_band_compare"):
        lines += ["", "## vs committed reference band", "",
                  "| metric | candidate | band | in band? |", "|---|---|---|---|"]
        for row in verdict["reference_band_compare"]:
            lines.append(f"| {row['metric']} | {_f(row['candidate']) if row['candidate'] is not None else ''} | {row['band']} | {row['in_band']} |")
    if verdict.get("reference_json_compare"):
        rjc = verdict["reference_json_compare"]
        lines += ["", f"## vs vLLM json ({rjc['vllm_path']})", "",
                  "| metric | candidate | vllm | delta % |", "|---|---|---|---|"]
        for row in rjc["rows"]:
            dp = f"{row['delta_pct']:+.1f}%" if row["delta_pct"] is not None else ""
            lines.append(f"| {row['metric']} | {_f(row['candidate'])} | {_f(row['vllm_reference'])} | {dp} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_paired_flag(spec: str) -> tuple:
    """'MSTAR_COADMIT=0:1' -> ('MSTAR_COADMIT', '0', '1')."""
    name, _, values = spec.partition("=")
    off_val, _, on_val = values.partition(":")
    if not name or not off_val or not on_val:
        raise argparse.ArgumentTypeError(f"--paired-flag must look like NAME=OFF:ON, got {spec!r}")
    return name, off_val, on_val


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="Server URL (required unless --dry-run / --compare-only).")
    ap.add_argument("--model", default="qwen3omni", choices=MODEL_CHOICES)
    ap.add_argument("--inference-system", default="ours", choices=INFERENCE_SYSTEM_CHOICES)
    ap.add_argument("--request-type", choices=REQUEST_TYPE_CHOICES)
    ap.add_argument("--dataset", default=None, help="Override dataset (default: runner.py's per-request-type default).")
    ap.add_argument("--num-requests", type=int, default=None, help="Default: protocol minimum for --request-type (96 i2t/i2s, 256 a2t/a2s).")
    ap.add_argument("--num-warmup", type=int, default=4)
    ap.add_argument("--max-concurrency", type=int, default=32, help="Closed-loop concurrency (the 'B32' convention).")
    ap.add_argument("--output-len", type=int, default=None, help="Fixed output length (--ignore-eos, min=max=this). Default: protocol default per request-type (212 i2t, 512 a2t). Pass 0 to disable length-matching.")

    ap.add_argument("--paired-flag", type=str, default=None, metavar="NAME=OFF:ON", help="Toggle this MSTAR_DYNFLAGS var OFF/ON/OFF/... across --pairs paired cells.")
    ap.add_argument("--dynflags-path", type=str, default=None, help="Path the SERVER's MSTAR_DYNFLAGS env var points at (required with --paired-flag).")
    ap.add_argument("--pairs", type=int, default=2, help="Number of OFF/ON pairs (default 2 -> OFF,ON,OFF,ON).")
    ap.add_argument("--reps", type=int, default=1, help="Cells to run when --paired-flag is not given (no toggling).")
    ap.add_argument("--flag-settle-s", type=float, default=5.0, help="Sleep after writing dynflags before the next cell starts.")

    ap.add_argument("--max-load", type=float, default=25.0, help="1-min loadavg ceiling to start a cell.")
    ap.add_argument("--load-gate-timeout-s", type=float, default=600.0)
    ap.add_argument("--load-poll-s", type=float, default=15.0)
    ap.add_argument("--strict-load-gate", dest="strict_load_gate", action="store_true", default=True)
    ap.add_argument("--no-strict-load-gate", dest="strict_load_gate", action="store_false")
    ap.add_argument("--load-drift-pct", type=float, default=30.0, help="UNTRUSTWORTHY threshold for load drift between first/second half of the run.")

    ap.add_argument("--cell-timeout-s", type=float, default=900.0, help="Hard per-cell subprocess timeout.")
    ap.add_argument("--work-dir", type=str, default=None, help="Default: benchmark/.cert_runs/<timestamp>_<request_type>.")
    ap.add_argument("--repo-dir", type=str, default=None, help="cwd for the runner subprocess. Default: parent of this file.")
    ap.add_argument("--verbose", action="store_true")

    ap.add_argument("--compare-reference", action="store_true", help="After the run, compare the last cell against REFERENCE_BANDS / --vllm-json.")
    ap.add_argument("--vllm-json", type=str, default=None, help="A committed vLLM results.json (e.g. checked out from the benchmarks branch) for a direct head-to-head instead of the hardcoded bands.")
    ap.add_argument("--candidate-json", type=str, default=None, help="Standalone mode: compare an existing results.json against --vllm-json / bands without running anything.")

    ap.add_argument("--dry-run", action="store_true", help="Fabricate two fake OFF/ON cells and print the verdict, without touching the network or a GPU.")

    args = ap.parse_args(argv)

    if args.dry_run:
        args.request_type = args.request_type or "image_to_text"
        args.paired_flag_name = "DRY_RUN_FLAG"
        return args

    if args.candidate_json:
        return args  # standalone compare mode; validated in main()

    if not args.url:
        ap.error("--url is required (unless --dry-run or --candidate-json)")
    if not args.request_type:
        ap.error("--request-type is required")
    if args.num_requests is None:
        args.num_requests = DEFAULT_N.get(args.request_type, 64)
    if args.output_len is None:
        args.output_len = DEFAULT_OUTPUT_LEN.get(args.request_type)
    elif args.output_len == 0:
        args.output_len = None

    args.paired_flag_name = None
    args.paired_flag_off = None
    args.paired_flag_on = None
    if args.paired_flag:
        try:
            name, off_val, on_val = parse_paired_flag(args.paired_flag)
        except argparse.ArgumentTypeError as e:
            ap.error(str(e))
        args.paired_flag_name, args.paired_flag_off, args.paired_flag_on = name, off_val, on_val
        if not args.dynflags_path:
            ap.error("--paired-flag requires --dynflags-path")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.dry_run:
        cells = fabricate_dry_run_cells()
        args.request_type = args.request_type or "image_to_text"
        args.max_concurrency, args.num_requests = 32, 96
        args.paired_flag, args.paired_flag_name = "DRY_RUN_FLAG=0:1", "DRY_RUN_FLAG"
        args.url, args.model = "http://dry-run.invalid", "qwen3omni"
        args.output_len, args.vllm_json = 212, None
        verdict = build_verdict(args, cells)
        print(json.dumps(verdict, indent=2))
        print()
        print(render_markdown(verdict))
        return 0

    if args.candidate_json:
        cand = extract_metrics(json.loads(Path(args.candidate_json).read_text()))
        rows = []
        if args.vllm_json:
            rows = compare_jsons(args.candidate_json, args.vllm_json)["rows"]
        rt = args.request_type
        band_rows = compare_to_bands(cand, rt) if rt else []
        print(json.dumps({"candidate_metrics": cand, "vllm_json_compare": rows, "band_compare": band_rows}, indent=2))
        return 0

    repo_dir = Path(args.repo_dir) if args.repo_dir else Path(__file__).resolve().parents[1]
    work_dir = Path(args.work_dir) if args.work_dir else (
        repo_dir / "benchmark" / ".cert_runs" / f"{_now_iso()}_{args.request_type}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    cells = []
    try:
        if args.paired_flag:
            sequence = []
            for _ in range(args.pairs):
                sequence.append(("OFF", args.paired_flag_off))
                sequence.append(("ON", args.paired_flag_on))
            for idx, (state, val) in enumerate(sequence):
                cells.append(run_cell(args, work_dir, idx, state, val, repo_dir))
        else:
            for idx in range(args.reps):
                cells.append(run_cell(args, work_dir, idx, "n/a", None, repo_dir))
    except CertAbort as e:
        print(f"ABORTED: {e}", file=sys.stderr)
        if cells:
            partial = build_verdict(args, cells)
            partial["aborted"] = str(e)
            (work_dir / "verdict.json").write_text(json.dumps(partial, indent=2))
        return 1
    finally:
        # Leave the dynflags file OFF -- a stuck-ON shared flag file poisons
        # every later ad hoc check on this server, same spirit as the GPU
        # clock-lock teardown-is-mandatory rule.
        if args.paired_flag and args.dynflags_path and os.path.exists(args.dynflags_path):
            try:
                write_dynflags(args.dynflags_path, args.paired_flag_name, args.paired_flag_off)
            except OSError:
                pass

    verdict = build_verdict(args, cells)
    (work_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    md = render_markdown(verdict)
    (work_dir / "verdict.md").write_text(md)
    print(md)
    print(f"[certify] wrote {work_dir}/verdict.json and verdict.md")
    return 1 if verdict["untrustworthy"] else 0


if __name__ == "__main__":
    sys.exit(main())
