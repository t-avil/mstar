"""Reproducible serving charts for the Qwen3-Omni figs-5/6 + I2T/S2T deliverable.

Reads the persisted ``results.json`` files written by ``benchmark.runner`` and
emits the four required charts. Nothing here re-runs the model: it is pure
plotting over saved metrics, so the charts are auditable and regenerable.

Expected directory layout (produced by the serving_scripts/*.sh harness)::

    <data-root>/
      ours_native/<path>/bs<N>/results.json    # M*-new (native encoders)
      ours_hf/<path>/bs<N>/results.json         # M*-old (HF-wrapper encoders)
      vllm_omni/<path>/bs<N>/results.json
      sglang_omni/<path>/bs<N>/results.json

where <path> in {image_to_text, audio_to_text, image_to_speech, text_to_speech}.
The series label is taken from the TOP-LEVEL dir name (both M* variants report
inference_system="ours" in the json, so the dir is what distinguishes them).

Charts emitted:
  * qwen3_omni_i2t_s2t_ttft.png  - TTFT vs batch, I2T & S2T, one line per runtime
  * qwen3_omni_i2t_s2t_itl.png   - ITL  vs batch, I2T & S2T, one line per runtime
  * qwen3_omni_i2s_rtf_throughput.png - figs-5/6 analog: RTF + audio throughput
        vs batch for I2S (and T2S if present), one line per runtime

Usage::

    python -m benchmark.serving_scripts.plot_serving \
        --data-root /mnt/storage/timchick/bench_artifacts \
        --out-dir benchmark/artifacts/serving \
        --topology "2-GPU disaggregated"      # title suffix (fig 5 vs fig 6)

    python -m benchmark.serving_scripts.plot_serving --selftest   # synthetic demo
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Pretty, stable legend names + a fixed (color, marker, linestyle) per series so
# every chart agrees AND overlapping lines stay distinguishable. M*-old vs M*-new
# share decode (ITL), so their ITL lines coincide exactly - distinct markers +
# a dashed style for HF keep both visible even when the values are identical.
# Color scheme: green (M*-new, the winner), red (M*-old/HF), blue (vLLM),
# purple (SGLang). Baseline stays gray as a reference line.
SERIES = {
    "ours_native": ("M*-new (native enc, opt)", "#2ca02c", "o", "-"),
    "ours_native_baseline": ("M*-new (native, dense-mask baseline)", "#9e9e9e", "x", ":"),
    "ours_hf": ("M*-old (HF enc)", "#d62728", "s", "--"),
    "vllm_omni": ("vLLM-Omni", "#1f77b4", "^", "-"),
    "sglang_omni": ("SGLang-Omni", "#7b3fa0", "D", "-"),
}
PATH_LABEL = {
    "image_to_text": "I2T",
    "audio_to_text": "S2T",
    "image_to_speech": "I2S",
    "text_to_speech": "T2S",
}

# Series to omit from a given render (set in main from --exclude). Used by both
# load_runs and the "not run / crashed" annotation so excluded series neither
# plot nor get flagged absent.
EXCLUDE: set[str] = set()


def _scalar_latency(stats: dict | None, stat: str = "p50") -> float | None:
    """Pull one number (seconds) out of a serialized LatencyStats dict."""
    if not stats:
        return None
    for key in (stat, "p50", "mean"):
        v = stats.get(key)
        if v is not None:
            return float(v)
    return None


def _ttft_or_itl(modality_map: dict | None, stat: str = "p50") -> float | None:
    """ttft/itl are {modality: LatencyStats}. Prefer a text modality, else the
    slowest modality (TTFT is gated by the slowest-to-first stream)."""
    if not modality_map:
        return None
    for pref in ("text", "audio", "image"):
        if pref in modality_map:
            v = _scalar_latency(modality_map[pref], stat)
            if v is not None:
                return v
    vals = [_scalar_latency(s, stat) for s in modality_map.values()]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def load_runs(data_root: str, exclude: set[str] | None = None) -> dict:
    """-> runs[series][path][bs] = results.json dict."""
    exclude = exclude or set()
    runs: dict[str, dict[str, dict[int, dict]]] = {}
    for series in sorted(os.listdir(data_root)):
        sdir = os.path.join(data_root, series)
        if not os.path.isdir(sdir) or series not in SERIES or series in exclude:
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
                    data = json.load(fh)
                bs = int(data.get("batch_size") or bsdir.replace("bs", "") or 1)
                runs.setdefault(series, {}).setdefault(path, {})[bs] = data
    return runs


def _series_xy(runs, path, extractor):
    """For each series with data on `path`, return (label, color, marker, ls, xs, ys)."""
    out = []
    for series, (label, color, marker, ls) in SERIES.items():
        per_bs = runs.get(series, {}).get(path, {})
        xs, ys = [], []
        for bs in sorted(per_bs):
            y = extractor(per_bs[bs])
            if y is not None:
                xs.append(bs)
                ys.append(y)
        if xs:
            out.append((label, color, marker, ls, xs, ys))
    return out


def _line_panel(ax, runs, path, extractor, ylabel, title, logx=True, logy=False):
    series = _series_xy(runs, path, extractor)
    for label, color, marker, ls, xs, ys in series:
        # markersize/linewidth + alpha so coincident lines (e.g. M*-old vs M*-new
        # ITL, which are identical) stay individually visible.
        ax.plot(xs, ys, marker=marker, ls=ls, color=color, label=label,
                markersize=8, linewidth=2, alpha=0.85, markerfacecolor="none",
                markeredgewidth=1.8)
        # Label the rightmost point's value so single-point (bs=1) charts are
        # readable and sweeps show the endpoint magnitude.
        vtxt = f"{ys[-1]:.3g}"
        ax.annotate(vtxt, (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, color=color)
    ax.set_xlabel("batch size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    if series:
        ax.legend()
        # Honest about absent runtimes, distinguishing CRASHED (a recorded
        # status=failed cell - e.g. competitor torch.cat crash on speech) from
        # NOT RUN (no cell at all - e.g. M*-HF serving never executed).
        present = {lbl for lbl, *_ in series}
        crashed, notrun = [], []
        for key, (label, *_rest) in SERIES.items():
            if label in present or key in EXCLUDE:
                continue
            cells = runs.get(key, {}).get(path, {})
            if any(c.get("status") == "failed" for c in cells.values()):
                crashed.append(label)
            else:
                notrun.append(label)
        notes = []
        if crashed:
            notes.append("crashed: " + ", ".join(crashed))
        if notrun:
            notes.append("not run: " + ", ".join(notrun))
        if notes:
            ax.text(0.02, 0.02, "  |  ".join(notes),
                    transform=ax.transAxes, fontsize=8, color="#a00", va="bottom",
                    bbox=dict(boxstyle="round", fc="#fff4f4", ec="#a00", alpha=0.75))
    else:
        ax.text(0.5, 0.5, "no data for any runtime", ha="center", va="center",
                transform=ax.transAxes)
    return bool(series)


def _slug(s):
    return "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_")


def _stamp(fig, provenance):
    """Stamp data provenance on every figure so a chart can never be mistaken
    for measured data (or vice versa)."""
    if not provenance:
        return
    fake = "SYNTHETIC" in provenance.upper()
    fig.text(0.5, 0.005, provenance, ha="center", va="bottom", fontsize=10,
             color="#b00" if fake else "#333",
             bbox=dict(boxstyle="round", fc="#fff0f0" if fake else "#f0f0f0",
                       ec="#b00" if fake else "#aaa", alpha=0.9))


def plot_text_paths(runs, out_dir, topology, dpi=200, provenance=""):
    """TTFT chart + ITL chart, each across runtimes for I2T and S2T."""
    paths = ["image_to_text", "audio_to_text"]
    outs = []
    # --- TTFT (ms) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, p in zip(axes, paths):
        _line_panel(ax, runs, p,
                    lambda d: (_ttft_or_itl(d.get("ttft"), "p50") or 0) * 1000 or None,
                    "TTFT p50 (ms, log) - lower is better", f"{PATH_LABEL[p]}  ({topology})",
                    logy=True)
    fig.suptitle("Qwen3-Omni TTFT vs batch across runtimes  (lower is better; encoder-bound path)")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); _stamp(fig, provenance)
    p = os.path.join(out_dir, f"qwen3_omni_i2t_s2t_ttft__{_slug(topology)}.png")
    fig.savefig(p, dpi=dpi); plt.close(fig); outs.append(p)
    # --- ITL (ms/token) ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, p_ in zip(axes, paths):
        _line_panel(ax, runs, p_,
                    lambda d: (_ttft_or_itl(d.get("itl"), "p50") or 0) * 1000 or None,
                    "ITL p50 (ms/token) - lower is better", f"{PATH_LABEL[p_]}  ({topology})")
    fig.suptitle("Qwen3-Omni ITL across runtimes  "
                 "(M*-native & M*-HF coincide - encoder swap doesn't touch decode)")
    fig.tight_layout(rect=(0, 0.03, 1, 1)); _stamp(fig, provenance)
    pp = os.path.join(out_dir, f"qwen3_omni_i2t_s2t_itl__{_slug(topology)}.png")
    fig.savefig(pp, dpi=dpi); plt.close(fig); outs.append(pp)
    return outs


def plot_speech_paths(runs, out_dir, topology, dpi=200, provenance=""):
    """Figs-5/6 analog: RTF + audio throughput vs batch, for BOTH speech paths
    (I2S = image->speech, T2S = text->speech). T2S is the paper's original TTS
    path; plotting both shows the optimization level on each."""
    speech = [("image_to_speech", "I2S"), ("text_to_speech", "T2S")]
    # 2 rows (RTF, throughput) x 2 cols (I2S, T2S)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    any_data = False
    for col, (path, plabel) in enumerate(speech):
        r = _line_panel(
            axes[0][col], runs, path,
            lambda d: _scalar_latency(d.get("rtf"), "p50"),
            "RTF p50 - lower=better (<1 = real-time)", f"{plabel} RTF  ({topology})")
        axes[0][col].axhline(1.0, color="red", ls="--", alpha=0.5)
        t = _line_panel(
            axes[1][col], runs, path,
            lambda d: d.get("audio_seconds_throughput"),
            "audio throughput (audio-sec/wall-sec) - higher=better",
            f"{plabel} throughput  ({topology})")
        any_data = any_data or r or t
    fig.suptitle("Qwen3-Omni speech paths - RTF & throughput across runtimes "
                 "(I2S = figs 5/6 analog; T2S = paper TTS path)")
    fig.tight_layout(rect=(0, 0.025, 1, 1)); _stamp(fig, provenance)
    p = os.path.join(out_dir, f"qwen3_omni_i2s_t2s_rtf_throughput__{_slug(topology)}.png")
    fig.savefig(p, dpi=dpi); plt.close(fig)
    if not any_data:
        print("  (warning: no I2S/T2S data found - chart is empty)")
    return [p]


def _write_synthetic_tree(root):
    """Generate a fake but schema-correct data tree so the plotter is testable
    without a GPU. Numbers are illustrative only."""
    def ls(mean, hib=False):
        return {"mean": mean, "p50": mean, "higher_is_better": hib,
                "p95": mean * 1.2, "p99": mean * 1.3}
    rng = [1, 4, 8, 16, 32]
    # rough, monotone-ish synthetic shapes per series
    # NB: in reality vLLM-Omni / SGLang-Omni currently crash on the speech path
    # (handoff #8), so their rtf/tp would be None and the chart would annotate
    # them as "no data". Here they are populated so the demo shows the intended
    # all-4-runtime I2S/T2S layout; swap to None to preview the crash case.
    profile = {
        "ours_native": dict(ttft=0.107, itl=0.0079, rtf=0.093, tp=11.4),
        "ours_hf":     dict(ttft=3.40,  itl=0.0079, rtf=0.50,  tp=2.1),
        "vllm_omni":   dict(ttft=0.153, itl=0.0052, rtf=0.30,  tp=3.5),
        "sglang_omni": dict(ttft=0.176, itl=0.0060, rtf=0.40,  tp=2.7),
    }
    for series, pr in profile.items():
        for path in ("image_to_text", "audio_to_text", "image_to_speech", "text_to_speech"):
            for i, bs in enumerate(rng):
                d = {"system": "ours" if series.startswith("ours") else series,
                     "inference_system": "ours" if series.startswith("ours") else series,
                     "request_type": path, "batch_size": bs}
                scale = 1 + 0.15 * i  # mild batch growth
                d["ttft"] = {"text": ls(pr["ttft"] * scale)}
                d["itl"] = {"text": ls(pr["itl"] * scale)}
                if path in ("image_to_speech", "text_to_speech"):
                    d["rtf"] = ls(pr["rtf"] * scale) if pr["rtf"] else None
                    d["audio_seconds_throughput"] = (pr["tp"] * (1 + 0.4 * i)) if pr["tp"] else None
                od = os.path.join(root, series, path, f"bs{bs}")
                os.makedirs(od, exist_ok=True)
                with open(os.path.join(od, "results.json"), "w") as fh:
                    json.dump(d, fh)


def _write_measured_tree(root):
    """Write the ONLY real measurements we have: the bs=1 smoke test (~8 req)
    from the handoff. A single point per series/path (NOT a batch sweep). Series
    with no measurement (M*-HF serving was never run; vLLM/SGLang crash on
    speech) get no file, so the chart annotates them "no data".

    Provenance: handoff_qwen3_omni_serving.md TL;DR + the original 3way chart.
    These numbers are themselves caveated (bs=1, ~8 req, single trial)."""
    def ls(v, hib=False):
        return {"mean": v, "p50": v, "higher_is_better": hib, "p95": v, "p99": v}

    # (ttft_s, itl_s, req_throughput) - None throughput where unmeasured
    text = {
        "ours_native": {"image_to_text": (0.107, 0.0079, 0.63),
                        "audio_to_text": (0.097, 0.0083, 4.67)},
        "vllm_omni":   {"image_to_text": (0.153, 0.0052, 0.97),
                        "audio_to_text": (0.089, 0.0054, 5.10)},
        "sglang_omni": {"image_to_text": (0.362, 0.0060, None),
                        "audio_to_text": (0.176, 0.0060, None)},
    }
    # (rtf, audio_seconds_throughput). Competitors crash -> no entry.
    speech = {
        "ours_native": {"image_to_speech": (0.086, 11.4),
                        "text_to_speech": (0.100, 10.7)},
    }

    def _emit(series, path, d):
        od = os.path.join(root, series, path, "bs1")
        os.makedirs(od, exist_ok=True)
        d.update({"system": "ours" if series.startswith("ours") else series,
                  "inference_system": "ours" if series.startswith("ours") else series,
                  "request_type": path, "batch_size": 1})
        with open(os.path.join(od, "results.json"), "w") as fh:
            json.dump(d, fh)

    for series, paths in text.items():
        for path, (ttft, itl, tp) in paths.items():
            d = {"ttft": {"text": ls(ttft)}, "itl": {"text": ls(itl)}}
            if tp is not None:
                d["request_throughput"] = tp
            _emit(series, path, d)
    for series, paths in speech.items():
        for path, (rtf, atp) in paths.items():
            _emit(series, path, {"rtf": ls(rtf), "audio_seconds_throughput": atp})
    # Competitors CRASH on speech (handoff #8): record an explicit failed cell so
    # the chart annotates "crashed" (not "not run"). This mirrors what
    # bench_record.py writes on a real failed run.
    for series in ("vllm_omni", "sglang_omni"):
        for path in ("image_to_speech", "text_to_speech"):
            _emit(series, path, {"status": "failed",
                                 "error_tail": "_thinker_to_talker_prefill -> torch.cat(): "
                                               "empty list (greedy Thinker yields no text)"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None,
                    help="dir with <series>/<path>/bs<N>/results.json")
    ap.add_argument("--out-dir", default="benchmark/artifacts/serving")
    ap.add_argument("--topology", default="2-GPU disaggregated",
                    help="title suffix; use to distinguish fig 5 vs fig 6")
    ap.add_argument("--selftest", action="store_true",
                    help="SYNTHETIC demo tree (fabricated numbers; for plumbing only)")
    ap.add_argument("--measured", action="store_true",
                    help="the real bs=1 smoke-test numbers from the handoff (single point)")
    ap.add_argument("--provenance", default=None,
                    help="override the provenance stamp shown on each chart")
    ap.add_argument("--dpi", type=int, default=200, help="chart resolution (default 200)")
    ap.add_argument("--exclude", default="", help="comma-sep series keys to omit "
                    "(e.g. ours_native_baseline for the clean 4-runtime chart)")
    args = ap.parse_args()
    global EXCLUDE
    EXCLUDE = {s.strip() for s in args.exclude.split(",") if s.strip()}

    data_root = args.data_root
    provenance = args.provenance
    if args.measured:
        data_root = data_root or os.path.join(args.out_dir, "_measured_data")
        _write_measured_tree(data_root)
        provenance = provenance or ("MEASURED - bs=1 smoke test (~8 req, single trial); "
                                    "NOT a batch sweep. M*-HF serving + competitor speech not yet run.")
        print(f"wrote measured data tree -> {data_root}")
    elif args.selftest:
        data_root = data_root or os.path.join(args.out_dir, "_selftest_data")
        _write_synthetic_tree(data_root)
        provenance = provenance or "SYNTHETIC DEMO DATA - FABRICATED, NOT MEASURED"
        print(f"wrote synthetic data tree -> {data_root}")
    if not data_root or not os.path.isdir(data_root):
        ap.error("need --data-root <dir> (or --measured / --selftest)")
    if provenance is None:
        provenance = f"data: {data_root}"

    os.makedirs(args.out_dir, exist_ok=True)
    runs = load_runs(data_root, exclude=EXCLUDE)
    found = {s: {p: sorted(bs) for p, bs in d.items()} for s, d in runs.items()}
    print("loaded series/paths/batches:", json.dumps(found, indent=2))

    outs = []
    outs += plot_text_paths(runs, args.out_dir, args.topology, dpi=args.dpi, provenance=provenance)
    outs += plot_speech_paths(runs, args.out_dir, args.topology, dpi=args.dpi, provenance=provenance)
    for o in outs:
        print("wrote", o)


if __name__ == "__main__":
    main()
