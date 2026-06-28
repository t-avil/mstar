#!/usr/bin/env python3
"""Generalized cross-agent aggregator + charting for the Qwen3-Omni rebench.

Generalizes exp_3way/analyze_3way.py (whose results.json recompute + stdout.txt
TTFT/ITL/throughput parsers are authoritative and reused here) to:

  PATHS    : audio_to_text (S2T), image_to_text (I2T),
             image_to_speech (I2S), audio_to_speech (S2S)
  BATCHES  : 1,2,4,8,16,32 (any subset; missing ones skipped)
  SYSTEMS  : mstar_new, mstar_old, vllm  (+ optional change-variant tags,
             e.g. mstar_new_spON --- all tags are configurable via --tags)

INPUT LAYOUT (searched under each --roots dir, in order):
    <root>/out_<tag>/<path>/B<n>/{results.json,stdout.txt}
Multiple roots let it pull from several agent dirs (exp_code2wav, exp_audioenc,
exp_imageenc, exp_throughput, exp_rebench). For a given (tag,path,batch) the
FIRST root that has results.json wins, so list roots in priority order.

WHAT IT DOES
  1. Recompute RTF / audio_dur / throughput from results.json per-request
     datapoints (authoritative). Parse TTFT/ITL/req-s/tok-s/audio-s/text from
     the stdout.txt agg block.
  2. Emit a CLAUDE.md-schema raw.json per path: every datapoint tagged by
     system + batch + phase, with units + warmup_iters.
  3. Produce the 4 joint comparison charts (one per path): two subplots ---
     throughput-vs-batch and (RTF-vs-batch for speech paths / TTFT-vs-batch for
     text paths) --- with a fixed per-system color/label mapping reused across
     every chart (chartstyle.mplstyle).
  4. Print a joint table per path x batch and a plain RATIO table: M*-new/vLLM and
     M*-new/M*-old per metric (lower-is-better metrics inverted so >1 = M*-new faster).
     No PASS/fail verdict, no >=10% rule.
  5. Be robust to missing tags/batches/paths --- skip gracefully, never crash.

Builds + self-tests only; runs no GPU jobs, launches no servers, commits nothing.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

SAMPLE_RATE = 24000              # output PCM sample rate (Hz)
BYTES_PER_AUDIO_SEC = SAMPLE_RATE * 2  # 24kHz int16 mono PCM

# Self-describing provenance stamped on every emitted raw_<path>.json (B1).
AUDIO_SECONDS_METHOD = "output_bytes.audio_bytes / (sample_rate * 2)  # 24kHz int16 mono"
RTF_METHOD = "jct_s / audio_seconds  (per request; lower=better)"
# Token throughput is NOT derived from a shared/neutral tokenizer: each server
# reports its own self-counted text_token_throughput in results.json. Recorded so
# tok/s comparisons across systems are read with that caveat in mind.
TOKEN_COUNT_SOURCE = (
    "server results.json field 'text_token_throughput' (each system self-counts "
    "tokens with its OWN tokenizer; NOT a shared/neutral tokenizer -- treat "
    "cross-system tok/s as indicative, req/s and audio-s/s are tokenizer-free)"
)

# Canonical path order + short labels.
PATHS = [
    ("audio_to_text", "S2T"),
    ("image_to_text", "I2T"),
    ("image_to_speech", "I2S"),
    ("audio_to_speech", "S2S"),
]
SPEECH_PATHS = {"image_to_speech", "audio_to_speech"}

DEFAULT_TAGS = ["mstar_new", "mstar_old", "vllm"]
DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32]
DEFAULT_STYLE = "/home/tim/exp_3way/chartstyle.mplstyle"

# Fixed per-tag style mapping (CLAUDE.md: same color + label everywhere).
# Baselines are solid lines; mstar_new_<flag> variants are dashed overlays in
# their own fixed colors so the win is visible against the baselines. Unknown
# variant tags get a color chosen DETERMINISTICALLY from the tag name (see
# _style_for) so a given tag keeps the same color across runs/charts.
SYSTEM_STYLE = {
    "mstar_new": {"label": "M*-new", "color": "#1f77b4", "marker": "o", "ls": "-"},
    "mstar_old": {"label": "M*-old (HF)", "color": "#7f7f7f", "marker": "s", "ls": "-"},
    "vllm": {"label": "vLLM-Omni", "color": "#2ca02c", "marker": "^", "ls": "-"},
    # known mstar_new_<flag> variants (stable colors + labels)
    "mstar_new_ccON": {"label": "M*-new +codec-chunk", "color": "#d62728", "marker": "D", "ls": "--"},
    "mstar_new_spON": {"label": "M*-new +Code2Wav-SP", "color": "#9467bd", "marker": "D", "ls": "--"},
    "mstar_new_gpuimgON": {"label": "M*-new +GPU-img", "color": "#ff7f0e", "marker": "D", "ls": "--"},
}
BASELINE_TAGS = ["mstar_new", "mstar_old", "vllm"]
_VARIANT_PALETTE = ["#d62728", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2", "#bcbd22"]


def _style_for(tag, _variant_idx=0):
    if tag in SYSTEM_STYLE:
        return SYSTEM_STYLE[tag]
    # deterministic color from the tag name (stable regardless of which subset
    # of variants is present in a given run)
    idx = sum(ord(c) for c in tag) % len(_VARIANT_PALETTE)
    label = tag.replace("mstar_new_", "M*-new +") if tag.startswith("mstar_new_") else tag
    return {"label": label, "color": _VARIANT_PALETTE[idx], "marker": "D", "ls": "--"}


def discover_tags(rootdirs):
    """Auto-discover system tags by globbing out_* across all roots.

    Returns baselines first (canonical order, if present), then any other
    discovered tags sorted alphabetically (so variants like mstar_new_ccON,
    mstar_new_spON appear in a stable order).
    """
    found = set()
    for root in rootdirs:
        try:
            for name in os.listdir(root):
                if name.startswith("out_") and os.path.isdir(os.path.join(root, name)):
                    found.add(name[len("out_"):])
        except OSError:
            continue
    ordered = [t for t in BASELINE_TAGS if t in found]
    ordered += sorted(t for t in found if t not in BASELINE_TAGS)
    return ordered


def pct(sv, p):
    """Linear-interpolated percentile of a pre-sorted list (matches analyze_3way)."""
    if not sv:
        return None
    if len(sv) == 1:
        return sv[0]
    k = (len(sv) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (k - lo)


def parse_stdout(path):
    """Pull TTFT/ITL/throughput/audio-dur from the harness agg block.

    Tolerant of the 1-vs-2-space spacing variants the harness emits for text-only
    runs ('ITL  (text)' in speech runs vs 'ITL  (text)' / 'TTFT (text)' in text runs).
    """
    out = {}
    try:
        t = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return out
    for mod in ("audio", "text"):
        m = re.search(
            rf"TTFT\s*\({mod}\)\s*:\s*mean=([\d.]+)s\s+p50=([\d.]+)s\s+p95=([\d.]+)s\s+p99=([\d.]+)s",
            t,
        )
        if m:
            out[f"ttft_{mod}"] = dict(zip(("mean", "p50", "p95", "p99"), map(float, m.groups())))
        m = re.search(
            rf"ITL\s*\({mod}\)\s*:\s*mean=([\d.]+)s\s+p50=([\d.]+)s\s+p95=([\d.]+)s\s+p99=([\d.]+)s",
            t,
        )
        if m:
            out[f"itl_{mod}"] = dict(zip(("mean", "p50", "p95", "p99"), map(float, m.groups())))
    m = re.search(r"RTF\s*:\s*mean=([\d.]+)\s+p50=([\d.]+)\s+p95=([\d.]+)\s+p99=([\d.]+)", t)
    if m:
        out["rtf_reported"] = dict(zip(("mean", "p50", "p95", "p99"), map(float, m.groups())))
    m = re.search(r"Audio dur\s*:\s*mean=([\d.]+)s", t)
    if m:
        out["audio_dur_reported"] = float(m.group(1))
    m = re.search(r"([\d.]+)\s*audio sec/s", t)
    if m:
        out["audio_throughput_reported"] = float(m.group(1))
    m = re.search(r"([\d.]+)\s*req/s", t)
    if m:
        out["req_throughput_reported"] = float(m.group(1))
    m = re.search(r"([\d.]+)\s*text tok/s", t)
    if m:
        out["text_tok_throughput_reported"] = float(m.group(1))
    m = re.search(r"Text tokens:\s*(\d+)\s*total\s*\(([\d.]+)\s*avg/req\)", t)
    if m:
        out["text_tokens_total"] = int(m.group(1))
        out["text_tokens_avg"] = float(m.group(2))
    return out


def load_cell(rootdirs, tag, path, batch):
    """Load one (tag,path,batch) cell from the first root that has it. None if absent."""
    rj = stdout = None
    for root in rootdirs:
        cand = os.path.join(root, f"out_{tag}", path, f"B{batch}")
        if os.path.isfile(os.path.join(cand, "results.json")):
            rj = os.path.join(cand, "results.json")
            stdout = os.path.join(cand, "stdout.txt")
            break
    if rj is None:
        return None
    try:
        data = json.load(open(rj))
    except (OSError, ValueError) as e:
        print(f"  [warn] unreadable {rj}: {e}", file=sys.stderr)
        return None

    wall = data.get("wall_time_s") or 0.0
    dps = []
    for r in data.get("per_request", []) or []:
        ob = r.get("output_bytes") or {}
        ab = ob.get("audio", 0) or 0
        jct = r.get("jct_ms") or 0.0
        asec = ab / BYTES_PER_AUDIO_SEC if ab else 0.0
        dps.append(
            {
                "request_id": r.get("request_id"),
                "phase": "measure",
                "jct_ms": jct,
                "audio_seconds": asec,
                "text_bytes": ob.get("text", 0) or 0,
                "rtf": (jct / 1000.0) / asec if asec > 0 else None,
            }
        )
    rtfs = sorted(x["rtf"] for x in dps if x["rtf"] is not None)
    durs = [x["audio_seconds"] for x in dps if x["audio_seconds"] > 0]
    total_audio = sum(durs)
    recomputed = {
        "n": len(dps),
        "n_with_audio": len(durs),
        "rtf_mean": statistics.mean(rtfs) if rtfs else None,
        "rtf_p50": pct(rtfs, 50),
        "rtf_p95": pct(rtfs, 95),
        "rtf_p99": pct(rtfs, 99),
        "rtf_std": statistics.pstdev(rtfs) if len(rtfs) > 1 else (0.0 if rtfs else None),
        "audio_dur_mean": statistics.mean(durs) if durs else None,
        "audio_dur_p50": pct(sorted(durs), 50) if durs else None,
        "jct_mean_s": statistics.mean([x["jct_ms"] for x in dps]) / 1000.0 if dps else None,
        "audio_throughput": (total_audio / wall) if wall > 0 else None,
        "request_throughput": (len(dps) / wall) if wall > 0 else None,
        "wall_time_s": wall,
    }
    # token throughput is NOT recomputable from per_request (only text *bytes*
    # are stored, not token counts) -- take the authoritative json field, then
    # fall back to the stdout agg block.
    harness = parse_stdout(stdout) if stdout else {}
    tok_tput = data.get("text_token_throughput")
    if tok_tput is None:
        tok_tput = harness.get("text_tok_throughput_reported")
    recomputed["text_token_throughput"] = tok_tput

    return {
        "source_dir": os.path.dirname(rj),
        "recomputed": recomputed,
        "harness": harness,
        "completed": data.get("completed"),
        "failed": data.get("failed"),
        "num_requests": data.get("num_requests"),
        "num_warmup": data.get("num_warmup"),
        "batch_size": data.get("batch_size", batch),
        "datapoints": dps,
    }


# --------------------------------------------------------------------------- #
# metric accessors used by table / charts / verdict
# --------------------------------------------------------------------------- #
def throughput_metric(path):
    """(key, label, unit) of the primary throughput metric for this path."""
    if path in SPEECH_PATHS:
        return ("audio_throughput", "Throughput (synth audio s / wall s)", "audio sec/s")
    return ("text_token_throughput", "Throughput (text tokens / wall s)", "tok/s")


def second_metric(path):
    """(getter, label, unit, lower_is_better) of the 2nd subplot metric."""
    if path in SPEECH_PATHS:
        return (lambda c: c["recomputed"].get("rtf_p50"), "RTF (p50, lower=better)", "RTF", True)
    return (
        lambda c: (c["harness"].get("ttft_text") or {}).get("mean"),
        "TTFT text (mean, lower=better)",
        "s",
        True,
    )


def get_tput(cell, path):
    return cell["recomputed"].get(throughput_metric(path)[0])


# --------------------------------------------------------------------------- #
def build(rootdirs, tags, paths, batches):
    """grid[path][batch][tag] = cell. Skips anything missing."""
    grid = {}
    for path, _ in paths:
        pg = {}
        for b in batches:
            bg = {}
            for tag in tags:
                cell = load_cell(rootdirs, tag, path, b)
                if cell is not None:
                    bg[tag] = cell
            if bg:
                pg[b] = bg
        if pg:
            grid[path] = pg
    return grid


def write_raw_json(grid, path, short, tags, outdir):
    """One CLAUDE.md-schema raw.json per path with every datapoint tagged."""
    tput_key, _, tput_unit = throughput_metric(path)
    datapoints = []
    warmups = set()
    for b in sorted(grid.get(path, {})):
        for tag in tags:
            cell = grid[path][b].get(tag)
            if not cell:
                continue
            if cell.get("num_warmup") is not None:
                warmups.add(cell["num_warmup"])
            for dp in cell["datapoints"]:
                datapoints.append(
                    {
                        "system": tag,
                        "batch": b,
                        "phase": dp["phase"],
                        "request_id": dp["request_id"],
                        "jct_ms": dp["jct_ms"],
                        "audio_seconds": dp["audio_seconds"],
                        "rtf": dp["rtf"],
                        "text_bytes": dp["text_bytes"],
                        # B1: self-describing units on every datapoint
                        "sample_rate": SAMPLE_RATE,
                        "audio_seconds_method": AUDIO_SECONDS_METHOD,
                    }
                )
    raw = {
        "benchmark": f"qwen3-omni-{short.lower()}-batch-sweep",
        "path": path,
        "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "timestamp_utc": os.environ.get("RUN_TS_UTC", ""),
        "warmup_iters": sorted(warmups),
        "units": {
            "jct_ms": "ms (per-request end-to-end)",
            "rtf": "jct/audio_dur (lower=better; speech paths only)",
            "audio_seconds": "synthesized audio seconds (per request)",
            "throughput_metric": f"{tput_key} ({tput_unit})",
            "ttft": "s",
            "itl": "s",
            "sample_rate": SAMPLE_RATE,
            "audio_seconds_method": AUDIO_SECONDS_METHOD,
            "rtf_method": RTF_METHOD,
        },
        "provenance": {
            "sample_rate": SAMPLE_RATE,
            "audio_seconds_method": AUDIO_SECONDS_METHOD,
            "rtf_method": RTF_METHOD,
            "token_count_source": TOKEN_COUNT_SOURCE,
            "aggregates_recomputed_from": (
                "per-request 'datapoints' for distribution stats (n, rtf_*, "
                "audio_dur_*, jct_mean_s); throughput (audio_throughput / "
                "request_throughput / text_token_throughput) and harness "
                "(ttft/itl) need wall-time/server fields and are carried from "
                "the source results.json, not derivable from datapoints"
            ),
            "generated_by": "aggregate.py",
            "generated_utc": os.environ.get("RUN_TS_UTC", ""),
        },
        "datapoints": datapoints,
        # per-cell aggregates so consumers don't have to recompute
        "aggregates": {
            f"B{b}": {
                tag: {
                    "recomputed": grid[path][b][tag]["recomputed"],
                    "harness": grid[path][b][tag]["harness"],
                    "completed": grid[path][b][tag]["completed"],
                    "num_requests": grid[path][b][tag]["num_requests"],
                }
                for tag in grid[path][b]
            }
            for b in sorted(grid.get(path, {}))
        },
    }
    fn = os.path.join(outdir, f"raw_{path}.json")
    json.dump(raw, open(fn, "w"), indent=2)
    return fn, len(datapoints)


def fmt(v, d=3):
    return f"{v:.{d}f}" if isinstance(v, (int, float)) else "  -  "


def print_tables(grid, tags, paths, batches):
    for path, short in paths:
        if path not in grid:
            print(f"\n[{short}] {path}: no data found, skipped.")
            continue
        tput_key, _, tput_unit = throughput_metric(path)
        getter2, lbl2, unit2, lower2 = second_metric(path)
        print("\n" + "=" * 100)
        print(f"[{short}] {path}   throughput unit = {tput_unit}   2nd metric = {unit2} ({lbl2})")
        print("=" * 100)
        hdr = (
            f"{'B':>3} {'system':12} | {'tput':>10} | {'2nd':>9} | "
            f"{'RTFp50':>7} {'RTFmean':>7} | {'TTFTt':>6} {'ITLt':>6} | "
            f"{'req/s':>6} | {'ok':>7}"
        )
        print(hdr)
        print("-" * len(hdr))
        for b in batches:
            if b not in grid[path]:
                continue
            for tag in tags:
                cell = grid[path][b].get(tag)
                if not cell:
                    continue
                rc, hs = cell["recomputed"], cell["harness"]
                lbl = SYSTEM_STYLE.get(tag, {}).get("label", tag)
                tput = rc.get(tput_key)
                m2 = getter2(cell)
                ttftt = (hs.get("ttft_text") or {}).get("mean")
                itlt = (hs.get("itl_text") or {}).get("mean")
                ok = cell.get("completed")
                nq = cell.get("num_requests")
                print(
                    f"{b:>3} {lbl:12} | {fmt(tput, 2):>10} | {fmt(m2):>9} | "
                    f"{fmt(rc.get('rtf_p50')):>7} {fmt(rc.get('rtf_mean')):>7} | "
                    f"{fmt(ttftt):>6} {fmt(itlt, 4):>6} | "
                    f"{fmt(rc.get('request_throughput'), 2):>6} | {ok}/{nq}"
                )
            print("-" * len(hdr))


# Metrics shown in the ratio table. Each is (key, getter, lower_is_better).
# For higher-is-better metrics the ratio is new/other (>1 = new faster/higher).
# For lower-is-better metrics the ratio is other/new, so >1 still reads
# unambiguously as "X.XX x faster" regardless of metric direction.
def _ratio_metrics(path):
    is_speech = path in SPEECH_PATHS
    tput_key = throughput_metric(path)[0]
    mets = [
        ("throughput", lambda c: c["recomputed"].get(tput_key), False),
    ]
    if is_speech:
        mets.append(("RTF_p50", lambda c: c["recomputed"].get("rtf_p50"), True))
        mets.append(("TTFT_audio_p50",
                     lambda c: (c["harness"].get("ttft_audio") or {}).get("p50"), True))
        mets.append(("ITL_audio_mean",
                     lambda c: (c["harness"].get("itl_audio") or {}).get("mean"), True))
    else:
        mets.append(("req_per_s", lambda c: c["recomputed"].get("request_throughput"), False))
        mets.append(("TTFT_text_p50",
                     lambda c: (c["harness"].get("ttft_text") or {}).get("p50"), True))
        mets.append(("ITL_text_mean",
                     lambda c: (c["harness"].get("itl_text") or {}).get("mean"), True))
    return mets


def _ratio(new_v, other_v, lower_is_better):
    """Plain ratio. Higher-is-better: new/other. Lower-is-better: other/new.
    Either way >1 means M*-new is faster/higher. None if not computable."""
    if new_v in (None, 0) or other_v in (None, 0):
        return None
    return (other_v / new_v) if lower_is_better else (new_v / other_v)


def print_ratio_table(grid, paths, batches, new="mstar_new", old="mstar_old", base="vllm"):
    """Plain ratio table (A4): per cell, M*-new/vLLM and M*-new/M*-old for every
    metric. No PASS/no verdict, no >=10% rule -- just the ratios."""
    print("\n" + "#" * 100)
    print(f"RATIOS: {new} vs {base}  and  {new} vs {old}   (>1.00x = M*-new faster/higher; "
          f"lower-is-better metrics are inverted so the direction is uniform)")
    print("#" * 100)
    for path, short in paths:
        if path not in grid:
            continue
        mets = _ratio_metrics(path)
        print(f"\n[{short}] {path}")
        hdr = f"  {'B':>3}  {'metric':16} | {'M*-new':>11} | {'vLLM':>11} | {'M*-old':>11} | "
        hdr += f"{'new/vLLM':>9} | {'new/old':>9}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for b in batches:
            if b not in grid[path]:
                continue
            cells = grid[path][b]
            cn = cells.get(new)
            if not cn:
                continue
            cv, co = cells.get(base), cells.get(old)
            for name, getter, lower in mets:
                nv = getter(cn)
                vv = getter(cv) if cv else None
                ov = getter(co) if co else None
                r_v = _ratio(nv, vv, lower)
                r_o = _ratio(nv, ov, lower)
                print(
                    f"  {b:>3}  {name:16} | {fmt(nv, 4):>11} | {fmt(vv, 4):>11} | {fmt(ov, 4):>11} | "
                    f"{(f'{r_v:.2f}x' if r_v is not None else '  -  '):>9} | "
                    f"{(f'{r_o:.2f}x' if r_o is not None else '  -  '):>9}"
                )
            print("  " + "-" * (len(hdr) - 2))


def make_charts(grid, tags, paths, batches, outdir, style):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # also makes matplotlib.style importable

    if os.path.isfile(style):
        try:
            plt.style.use(style)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] could not apply style {style}: {e}", file=sys.stderr)
    else:
        print(
            f"  [warn] style file {style} missing -- using matplotlib defaults. "
            f"(seed it from the benchmarks branch: "
            f"git checkout benchmarks -- benchmarks/chartstyle.mplstyle)",
            file=sys.stderr,
        )

    chartdir = os.path.join(outdir, "charts")
    os.makedirs(chartdir, exist_ok=True)

    written = []
    for path, short in paths:
        if path not in grid:
            continue
        _, tput_lbl, tput_unit = throughput_metric(path)
        getter2, lbl2, unit2, _ = second_metric(path)

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
        any_series = False
        for tag in tags:
            st = _style_for(tag)
            xs_t, ys_t, xs_2, ys_2 = [], [], [], []
            for b in batches:
                cell = grid[path].get(b, {}).get(tag)
                if not cell:
                    continue
                tv = get_tput(cell, path)
                if tv is not None:
                    xs_t.append(b)
                    ys_t.append(tv)
                m2 = getter2(cell)
                if m2 is not None:
                    xs_2.append(b)
                    ys_2.append(m2)
            if xs_t:
                any_series = True
                axL.plot(xs_t, ys_t, marker=st["marker"], color=st["color"],
                         linestyle=st.get("ls", "-"), label=st["label"])
            if xs_2:
                axR.plot(xs_2, ys_2, marker=st["marker"], color=st["color"],
                         linestyle=st.get("ls", "-"), label=st["label"])

        present_batches = sorted(grid[path].keys())
        for ax in (axL, axR):
            ax.set_xlabel("batch size")
            if len(present_batches) > 1:
                ax.set_xscale("log", base=2)
            ax.set_xticks(present_batches)
            ax.set_xticklabels([str(b) for b in present_batches])
            ax.margins(x=0.08)
        axL.set_ylabel(tput_unit)
        axL.set_title(f"{short}  {tput_lbl}")
        axR.set_ylabel(unit2)
        axR.set_title(f"{short}  {lbl2}")
        if any_series:
            axL.legend()
        fig.suptitle(f"{short} ({path}) -- M*-new vs M*-old vs vLLM", y=1.02)
        fig.tight_layout()
        fn = os.path.join(chartdir, f"{path}_throughput_rtf.png")
        fig.savefig(fn)
        plt.close(fig)
        written.append(fn)
    return written


# --------------------------------------------------------------------------- #
# REFINE MODE: operate on the committed raw_<path>.json (the canonical curated
# datapoints), not the heterogeneous source roots. Fold the I2S B1/B2 recheck
# per-request dumps INTO the datapoints, recompute every aggregate cell's
# distribution stats FROM the datapoints, and stamp provenance. Used because the
# committed deliverable was assembled from runs that the source roots no longer
# reproduce 1:1, so the raw files -- not the roots -- are authoritative.
# --------------------------------------------------------------------------- #
RECHECK_FOLD = {  # (path, system, batch) cells that get replaced from recheck reps
    ("image_to_speech", "mstar_new", 1),
    ("image_to_speech", "mstar_new", 2),
    ("image_to_speech", "mstar_old", 1),
    ("image_to_speech", "mstar_old", 2),
}


def recompute_cell_stats(dps):
    """Per-request distribution stats from a cell's datapoints. Mirrors the
    formulas in load_cell() so non-folded cells reproduce committed values."""
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


def _avg_dicts(dicts):
    """Field-wise mean over a list of {field: float} dicts (skip missing)."""
    out, keys = {}, set()
    for d in dicts:
        keys |= set(d)
    for k in keys:
        vals = [d[k] for d in dicts if d.get(k) is not None]
        if vals:
            out[k] = statistics.mean(vals)
    return out


def load_recheck_cell(recheck_dir, tag, path, batch):
    """Build a folded cell (datapoints + wall-derived throughput + harness) from
    all recheck reps for (tag, path, batch). None if no reps found."""
    reps = sorted(glob.glob(os.path.join(recheck_dir, tag, f"{path}_B{batch}_r*")))
    dps, walls, ttfts, itls, completed = [], [], [], [], 0
    for d in reps:
        rj = os.path.join(d, "results.json")
        if not os.path.isfile(rj):
            continue
        try:
            dd = json.load(open(rj))
        except (OSError, ValueError):
            continue
        m = re.search(r"_r(\d+)$", d)
        rep = int(m.group(1)) if m else 0
        wall = dd.get("wall_time_s") or 0.0
        for r in dd.get("per_request", []) or []:
            ob = r.get("output_bytes") or {}
            ab = ob.get("audio", 0) or 0
            if ab <= 0:
                continue
            jct = r.get("jct_ms") or 0.0
            asec = ab / BYTES_PER_AUDIO_SEC
            dps.append({
                "system": tag, "batch": batch, "phase": "measure",
                "request_id": r.get("request_id"), "rep": rep,
                "jct_ms": jct, "audio_seconds": asec,
                "rtf": (jct / 1000.0) / asec if asec > 0 else None,
                "text_bytes": ob.get("text", 0) or 0,
                "sample_rate": SAMPLE_RATE, "audio_seconds_method": AUDIO_SECONDS_METHOD,
            })
        if wall > 0:
            walls.append(wall)
        ho = parse_stdout(os.path.join(d, "out.txt"))
        if ho.get("ttft_audio"):
            ttfts.append(ho["ttft_audio"])
        if ho.get("itl_audio"):
            itls.append(ho["itl_audio"])
        completed += dd.get("completed") or 0
    if not dps:
        return None
    total_wall = sum(walls)
    total_aud = sum(d["audio_seconds"] for d in dps)
    recomputed = recompute_cell_stats(dps)
    recomputed.update({
        "audio_throughput": (total_aud / total_wall) if total_wall > 0 else None,
        "request_throughput": (len(dps) / total_wall) if total_wall > 0 else None,
        "wall_time_s": total_wall,
        "text_token_throughput": None,  # not derivable per-request; not used for I2S
    })
    harness = {}
    if ttfts:
        harness["ttft_audio"] = _avg_dicts(ttfts)
    if itls:
        harness["itl_audio"] = _avg_dicts(itls)
    return {
        "datapoints": dps,
        "recomputed": recomputed,
        "harness": harness,
        "completed": completed or len(dps),
        "num_requests": len(dps),
        "provenance": {
            "stats_source": "recomputed from datapoints",
            "throughput_source": f"recheck wall-time, {len(walls)} reps "
                                 f"(total audio {total_aud:.1f}s / total wall {total_wall:.1f}s)",
            "n_reps": len(walls), "n_datapoints": len(dps),
        },
    }


def refine_one_raw(rawpath, path_name, recheck_dir):
    """Fold recheck into datapoints (I2S B1/B2), recompute aggregates from the
    resulting datapoints, stamp provenance, write back. Returns a report dict."""
    raw = json.load(open(rawpath))
    old_dps = raw.get("datapoints", [])

    # carry-forward source for non-folded throughput/harness: the committed aggregates
    old_agg = raw.get("aggregates", {})

    def carried(system, batch):
        return (old_agg.get(f"B{batch}", {}) or {}).get(system)

    # 1) group existing datapoints by (system, batch), dropping folded cells
    groups = {}
    for dp in old_dps:
        key = (dp["system"], dp["batch"])
        if (path_name, dp["system"], dp["batch"]) in RECHECK_FOLD:
            continue
        groups.setdefault(key, []).append(dp)

    # 2) build folded cells from recheck
    folded_cells = {}
    for (p, system, batch) in sorted(RECHECK_FOLD):
        if p != path_name:
            continue
        cell = load_recheck_cell(recheck_dir, system, path_name, batch)
        if cell is None:
            continue
        folded_cells[(system, batch)] = cell
        groups[(system, batch)] = cell["datapoints"]

    # 3) rebuild datapoints array (sorted system then batch then rep/request) +
    #    stamp units on carried-over datapoints too
    new_dps = []
    for (system, batch) in sorted(groups, key=lambda k: (k[0], k[1])):
        for dp in groups[(system, batch)]:
            dp.setdefault("sample_rate", SAMPLE_RATE)
            dp.setdefault("audio_seconds_method", AUDIO_SECONDS_METHOD)
            new_dps.append(dp)

    # 4) rebuild aggregates strictly from the datapoint groups (no orphan cells)
    new_agg, mismatches = {}, []
    for (system, batch) in sorted(groups, key=lambda k: (k[1], k[0])):
        dps = groups[(system, batch)]
        bkey = f"B{batch}"
        if (system, batch) in folded_cells:
            fc = folded_cells[(system, batch)]
            cell = {
                "recomputed": fc["recomputed"],
                "harness": fc["harness"],
                "completed": fc["completed"],
                "num_requests": fc["num_requests"],
                "provenance": fc["provenance"],
            }
        else:
            stats = recompute_cell_stats(dps)
            src = carried(system, batch)
            src_rc = (src or {}).get("recomputed", {})
            # carry wall-derived throughput (not recomputable from datapoints)
            for k in ("audio_throughput", "request_throughput", "wall_time_s",
                      "text_token_throughput"):
                stats[k] = src_rc.get(k)
            # verify the carried committed stats matched a recompute (A1 invariant)
            for k, v in recompute_cell_stats(dps).items():
                cv = src_rc.get(k)
                if isinstance(v, float) and isinstance(cv, (int, float)):
                    if abs(v - cv) > 1e-6 * max(1.0, abs(v)):
                        mismatches.append(f"{path_name} {bkey}/{system} {k}: "
                                          f"recompute={v:.6g} committed={cv:.6g}")
            cell = {
                "recomputed": stats,
                "harness": (src or {}).get("harness", {}),
                "completed": (src or {}).get("completed"),
                "num_requests": (src or {}).get("num_requests"),
                "provenance": {
                    "stats_source": "recomputed from datapoints",
                    "throughput_source": "carried from committed source results.json",
                    "n_datapoints": len(dps),
                },
            }
        new_agg.setdefault(bkey, {})[system] = cell

    raw["datapoints"] = new_dps
    raw["aggregates"] = new_agg
    # refresh units / provenance (B1) -- mirror write_raw_json
    tput_key, _, tput_unit = throughput_metric(path_name)
    raw["units"] = {
        "jct_ms": "ms (per-request end-to-end)",
        "rtf": "jct/audio_dur (lower=better; speech paths only)",
        "audio_seconds": "synthesized audio seconds (per request)",
        "throughput_metric": f"{tput_key} ({tput_unit})",
        "ttft": "s", "itl": "s",
        "sample_rate": SAMPLE_RATE,
        "audio_seconds_method": AUDIO_SECONDS_METHOD,
        "rtf_method": RTF_METHOD,
    }
    raw["provenance"] = {
        "sample_rate": SAMPLE_RATE,
        "audio_seconds_method": AUDIO_SECONDS_METHOD,
        "rtf_method": RTF_METHOD,
        "token_count_source": TOKEN_COUNT_SOURCE,
        "aggregates_recomputed_from": (
            "per-request 'datapoints' for distribution stats (n, rtf_*, "
            "audio_dur_*, jct_mean_s); throughput + harness carried from source "
            "results.json (folded I2S B1/B2 cells: from recheck wall-time/out.txt)"
        ),
        "recheck_folded_cells": sorted(f"{p}:{s}:B{b}" for (p, s, b) in RECHECK_FOLD
                                       if p == path_name and (s, b) in folded_cells),
        "generated_by": "aggregate.py --refine-dir",
        "generated_utc": os.environ.get("RUN_TS_UTC", ""),
    }
    json.dump(raw, open(rawpath, "w"), indent=2)

    # invariant check: every aggregate cell has datapoints and vice-versa
    agg_cells = {(s, int(b[1:])) for b, row in new_agg.items() for s in row}
    dp_cells = set(groups)
    return {
        "path": path_name,
        "n_datapoints": len(new_dps),
        "folded": sorted(folded_cells),
        "mismatches": mismatches,
        "orphan_agg": sorted(agg_cells - dp_cells),
        "orphan_dp": sorted(dp_cells - agg_cells),
    }


def grid_from_raw(outdir, paths):
    """Reconstruct grid[path][batch][tag]=cell from refined raw_<path>.json so the
    ratio table / charts / NUMBERS.md all read one structure."""
    grid = {}
    for path, _ in paths:
        fp = os.path.join(outdir, f"raw_{path}.json")
        if not os.path.isfile(fp):
            continue
        raw = json.load(open(fp))
        dps_by = {}
        for dp in raw.get("datapoints", []):
            dps_by.setdefault((dp["system"], dp["batch"]), []).append(dp)
        pg = {}
        for bkey, row in raw.get("aggregates", {}).items():
            b = int(bkey[1:])
            for tag, cell in row.items():
                pg.setdefault(b, {})[tag] = {
                    "recomputed": cell.get("recomputed", {}),
                    "harness": cell.get("harness", {}),
                    "completed": cell.get("completed"),
                    "num_requests": cell.get("num_requests"),
                    "datapoints": dps_by.get((tag, b), []),
                    "batch_size": b,
                }
        if pg:
            grid[path] = pg
    return grid


def emit_numbers_md(grid, paths, batches, outpath, new="mstar_new", old="mstar_old", base="vllm"):
    """NUMBERS.md -- the single source of truth for every headline number,
    computed from the corrected raw. The docs agent fills docs from THIS file."""
    L = ["# NUMBERS.md -- headline numbers (auto-generated by aggregate.py --refine-dir)",
         "",
         "Single source of truth. Every value computed from the corrected "
         "`raw_<path>.json` (datapoints == aggregates). Ratios: `new/vLLM` and "
         "`new/old`; for lower-is-better metrics (RTF, TTFT, ITL) the ratio is "
         "inverted so **>1.00x always means M\\*-new is faster**.",
         "",
         f"- audio_seconds = {AUDIO_SECONDS_METHOD}",
         f"- token throughput source: {TOKEN_COUNT_SOURCE}",
         ""]

    def g(cell, getter):
        return getter(cell) if cell else None

    for path, short in paths:
        if path not in grid:
            continue
        is_speech = path in SPEECH_PATHS
        tput_key, _, tput_unit = throughput_metric(path)
        L.append(f"## {short} -- {path}")
        L.append("")
        L.append(f"throughput unit = **{tput_unit}**; "
                 + ("RTF p50 / TTFT(audio) p50 / ITL(audio) mean"
                    if is_speech else "req/s / TTFT(text) p50 / ITL(text) mean"))
        L.append("")
        mets = _ratio_metrics(path)
        hdr = "| B | metric | M*-new | M*-old | vLLM | new/vLLM | new/old |"
        L.append(hdr)
        L.append("|---|---|---|---|---|---|---|")
        for b in batches:
            if b not in grid[path]:
                continue
            cells = grid[path][b]
            cn, co, cv = cells.get(new), cells.get(old), cells.get(base)
            if not cn:
                continue
            for name, getter, lower in mets:
                nv, ov, vv = g(cn, getter), g(co, getter), g(cv, getter)
                rv = _ratio(nv, vv, lower)
                ro = _ratio(nv, ov, lower)
                L.append(
                    f"| {b} | {name} | {fmt(nv,4)} | {fmt(ov,4)} | {fmt(vv,4)} | "
                    f"{(f'{rv:.2f}x' if rv is not None else '-')} | "
                    f"{(f'{ro:.2f}x' if ro is not None else '-')} |"
                )
        L.append("")
        if is_speech:
            # audio-length (output duration) ratios -- fairness check
            L.append(f"### {short} audio-length (output duration) p50 -- fairness")
            L.append("")
            L.append("| B | M*-new (s) | M*-old (s) | vLLM (s) | new/vLLM | old/vLLM |")
            L.append("|---|---|---|---|---|---|")
            for b in batches:
                if b not in grid[path]:
                    continue
                cells = grid[path][b]
                def dur(t):
                    c = cells.get(t)
                    return c["recomputed"].get("audio_dur_p50") if c else None
                n, o, v = dur(new), dur(old), dur(base)
                rnv = f"{n/v:.3f}" if n and v else "-"
                rov = f"{o/v:.3f}" if o and v else "-"
                L.append(f"| {b} | {fmt(n,3)} | {fmt(o,3)} | {fmt(v,3)} | {rnv} | {rov} |")
            L.append("")
    open(outpath, "w").write("\n".join(L) + "\n")
    return outpath


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refine-dir", default=None,
                    help="REFINE MODE: directory holding the committed raw_<path>.json "
                         "(+ recheck/). Folds the I2S B1/B2 recheck per-request dumps into "
                         "datapoints, recomputes aggregates FROM datapoints, stamps "
                         "units/provenance, writes NUMBERS.md + charts. Use this for the "
                         "committed deliverable; --roots is for fresh source-root aggregation.")
    ap.add_argument("--recheck-dir", default=None,
                    help="recheck root for --refine-dir (default: <refine-dir>/recheck).")
    ap.add_argument("--roots", nargs="+", default=None,
                    help="exp_root dir(s) to search, in priority order (first hit wins). "
                         "Required unless --refine-dir is given.")
    ap.add_argument("--tags", nargs="+", default=None,
                    help="system tags to include. Default: auto-discover by globbing out_* "
                         "across roots (baselines mstar_new/mstar_old/vllm first, then variant "
                         "tags like mstar_new_ccON/mstar_new_spON sorted).")
    ap.add_argument("--batches", nargs="+", type=int, default=DEFAULT_BATCHES)
    ap.add_argument("--paths", nargs="+", default=[p for p, _ in PATHS],
                    help="path names to include (default: all 4).")
    ap.add_argument("--out", default=None,
                    help="output dir for raw_*.json + charts/ (default: <first root>/aggregate_out).")
    ap.add_argument("--style", default=DEFAULT_STYLE, help="matplotlib .mplstyle path.")
    ap.add_argument("--no-charts", action="store_true", help="skip chart generation.")
    args = ap.parse_args()

    # restrict canonical path list to requested paths, preserving order + labels
    paths = [(p, s) for p, s in PATHS if p in args.paths]
    for p in args.paths:
        if p not in {pp for pp, _ in PATHS}:
            paths.append((p, p[:4].upper()))

    # ----- REFINE MODE (operates on committed raw files) ---------------------
    if args.refine_dir:
        outdir = os.path.abspath(args.refine_dir)
        recheck_dir = os.path.abspath(args.recheck_dir) if args.recheck_dir \
            else os.path.join(outdir, "recheck")
        print(f"REFINE MODE: dir={outdir}  recheck={recheck_dir}")
        print("Refined raw.json (datapoints folded + aggregates recomputed):")
        all_ok = True
        for path, short in paths:
            fp = os.path.join(outdir, f"raw_{path}.json")
            if not os.path.isfile(fp):
                print(f"  [skip] {fp} (missing)")
                continue
            rep = refine_one_raw(fp, path, recheck_dir)
            issues = []
            if rep["mismatches"]:
                issues.append(f"{len(rep['mismatches'])} stat mismatch")
            if rep["orphan_agg"]:
                issues.append(f"orphan aggregates {rep['orphan_agg']}")
            if rep["orphan_dp"]:
                issues.append(f"orphan datapoints {rep['orphan_dp']}")
            ok = not issues
            all_ok = all_ok and ok
            folded = ",".join(f"{s}/B{b}" for s, b in rep["folded"]) or "(none)"
            print(f"  {short}: {rep['n_datapoints']} dps  folded[{folded}]  "
                  f"{'OK' if ok else 'ISSUES: ' + '; '.join(issues)}")
            for m in rep["mismatches"]:
                print(f"      mismatch: {m}", file=sys.stderr)

        grid = grid_from_raw(outdir, paths)
        print_ratio_table(grid, paths, args.batches)
        nums = emit_numbers_md(grid, paths, args.batches, os.path.join(outdir, "NUMBERS.md"))
        print(f"\nWrote {nums}")
        if not args.no_charts:
            seen = {t for pg in grid.values() for bg in pg.values() for t in bg}
            chart_tags = [t for t in BASELINE_TAGS if t in seen] + \
                         sorted(t for t in seen if t not in BASELINE_TAGS)
            charts = make_charts(grid, chart_tags, paths, args.batches, outdir, args.style)
            print("Wrote charts:")
            for c in charts:
                print(f"  {c}")
        print(f"\nREFINE {'OK -- datapoints == aggregates for every cell' if all_ok else 'FOUND ISSUES (see above)'}")
        return 0 if all_ok else 1

    if not args.roots:
        print("[error] --roots is required unless --refine-dir is given.", file=sys.stderr)
        return 2
    roots = [os.path.abspath(r) for r in args.roots]
    for r in roots:
        if not os.path.isdir(r):
            print(f"[warn] root not found (skipped): {r}", file=sys.stderr)
    roots = [r for r in roots if os.path.isdir(r)]
    if not roots:
        print("[error] no valid --roots", file=sys.stderr)
        return 2

    tags = args.tags if args.tags is not None else discover_tags(roots)
    if not tags:
        print("[error] no out_<tag> dirs found under roots (nothing to aggregate).", file=sys.stderr)
        return 1
    print(f"Tags ({'given' if args.tags is not None else 'auto-discovered'}): {tags}")

    outdir = args.out or os.path.join(roots[0], "aggregate_out")
    os.makedirs(outdir, exist_ok=True)

    grid = build(roots, tags, paths, args.batches)
    if not grid:
        print("[error] no (tag,path,batch) cells found under the given roots.", file=sys.stderr)
        return 1

    # raw.json per path
    print("Wrote raw.json:")
    for path, short in paths:
        if path in grid:
            fn, n = write_raw_json(grid, path, short, tags, outdir)
            print(f"  {fn}  ({n} datapoints)")

    print_tables(grid, tags, paths, args.batches)
    print_ratio_table(grid, paths, args.batches)

    if not args.no_charts:
        charts = make_charts(grid, tags, paths, args.batches, outdir, args.style)
        print("\nWrote charts:")
        for c in charts:
            print(f"  {c}")

    # coverage summary
    print("\nCoverage (cells found):")
    for path, short in paths:
        if path not in grid:
            print(f"  {short}: (none)")
            continue
        cov = []
        for b in args.batches:
            if b in grid[path]:
                cov.append(f"B{b}:{','.join(t for t in tags if t in grid[path][b])}")
        print(f"  {short}: " + " | ".join(cov))
    return 0


if __name__ == "__main__":
    sys.exit(main())
