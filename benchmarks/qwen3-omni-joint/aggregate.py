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
  4. Print a joint table per path x batch and the verdict: is M*-new >=10% over
     BOTH M*-old AND vLLM (throughput up / RTF down)? plus the M*-new-vs-vLLM ratio.
  5. Be robust to missing tags/batches/paths --- skip gracefully, never crash.

Builds + self-tests only; runs no GPU jobs, launches no servers, commits nothing.
"""
import argparse
import json
import os
import re
import statistics
import sys

BYTES_PER_AUDIO_SEC = 24000 * 2  # 24kHz int16 mono PCM

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


def print_verdict(grid, paths, batches, new="mstar_new", old="mstar_old", base="vllm"):
    print("\n" + "#" * 100)
    print(f"VERDICT: is {new} >=10% better than BOTH {old} AND {base}?  (throughput up / RTF down)")
    print("#" * 100)
    for path, short in paths:
        if path not in grid:
            continue
        tput_key, _, tput_unit = throughput_metric(path)
        is_speech = path in SPEECH_PATHS
        print(f"\n[{short}] {path}")
        for b in batches:
            if b not in grid[path]:
                continue
            cells = grid[path][b]
            cn = cells.get(new)
            if not cn:
                continue
            tn = get_tput(cn, path)
            verds = []
            # throughput: higher is better -> new >= 1.10 * other
            for other_tag in (old, base):
                co = cells.get(other_tag)
                to = get_tput(co, path) if co else None
                if tn is not None and to not in (None, 0):
                    ratio = tn / to
                    win = ratio >= 1.10
                    verds.append((other_tag, "tput", ratio, win))
            # RTF (speech only): lower is better -> new <= 0.90 * other
            if is_speech:
                rn = cn["recomputed"].get("rtf_p50")
                for other_tag in (old, base):
                    co = cells.get(other_tag)
                    ro = co["recomputed"].get("rtf_p50") if co else None
                    if rn not in (None, 0) and ro not in (None, 0):
                        ratio = ro / rn  # >1 means new is faster (lower RTF)
                        win = (rn <= 0.90 * ro)
                        verds.append((other_tag, "rtf", ratio, win))
            # summarize
            tput_wins = [v for v in verds if v[1] == "tput"]
            beats_both_tput = len(tput_wins) >= 2 and all(v[3] for v in tput_wins)
            vs_vllm = next((v for v in verds if v[0] == base and v[1] == "tput"), None)
            vs_vllm_rtf = next((v for v in verds if v[0] == base and v[1] == "rtf"), None)
            parts = []
            for other_tag, metric, ratio, win in verds:
                lo = SYSTEM_STYLE.get(other_tag, {}).get("label", other_tag)
                parts.append(f"{metric} vs {lo}: {ratio:.2f}x {'PASS' if win else 'no'}")
            tag_line = f"  B{b:<2}: " + "  |  ".join(parts) if parts else f"  B{b:<2}: (no comparators)"
            flag = ""
            if beats_both_tput:
                flag = "  <<< >=10% tput over BOTH"
            print(tag_line + flag)
            if vs_vllm:
                extra = f"        M*-new vs vLLM: throughput {vs_vllm[2]:.2f}x"
                if vs_vllm_rtf:
                    extra += f", RTF {vs_vllm_rtf[2]:.2f}x faster"
                print(extra)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True,
                    help="exp_root dir(s) to search, in priority order (first hit wins).")
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

    # restrict canonical path list to requested paths, preserving order + labels
    paths = [(p, s) for p, s in PATHS if p in args.paths]
    for p in args.paths:
        if p not in {pp for pp, _ in PATHS}:
            paths.append((p, p[:4].upper()))

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
    print_verdict(grid, paths, args.batches)

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
