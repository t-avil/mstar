"""Batch-fill instrumentation for AR-decode CUDA-graph replays.

Goal: answer "does the Talker AR-decode CUDA graph actually FILL at batch
(B=32), or do requests desync so the engine runs small batched replays
back-to-back, wasting the batched-decode advantage?".

Every node forward (Talker decode, Thinker decode, encoder forwards, prefills)
goes through ``KVCacheEngine.execute_forward``. When enabled, we record the
*actual* batch size used per replay, keyed by (node_name, graph_walk, path),
where ``path`` is the dispatch path (cuda_graph / batched / sequential). At
shutdown we emit a per-key histogram over the capture buckets [1,2,4,8,16,32]
plus the mean fill ratio (actual_bs / capture_bucket).

Gating: the whole thing is behind the ``MSTAR_BATCHFILL_STATS`` env var, which
defaults to OFF. When OFF, the only cost on the hot path is a single module
attribute load + truthiness check (``if batchfill_stats.ENABLED``), so there is
effectively zero overhead when unset.

Testability: ``fill_bucket`` and ``aggregate_fill_stats`` /
``aggregate_fill_stats_from_counts`` are pure functions with no torch / CUDA
dependency, so the histogram + fill-ratio math is fully unit-testable on CPU.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

# Capture batch sizes the Talker / Thinker CUDA graphs are captured at
# (submodules.py: capture sizes [1,2,4,8,16,32]; MAX_BATCH_SIZE=32). Kept here
# as the default fill-accounting buckets; pass an explicit tuple to override.
DEFAULT_CAPTURE_BUCKETS = (1, 2, 4, 8, 16, 32)

_TRUTHY = ("1", "true", "yes", "on")


def _env_enabled() -> bool:
    return os.environ.get("MSTAR_BATCHFILL_STATS", "").strip().lower() in _TRUTHY


# Resolved once at import. Hot-path checks read this attribute. Tests can flip it
# via ``set_enabled`` to exercise the recorder without re-importing.
ENABLED = _env_enabled()


def set_enabled(value: bool) -> None:
    """Override the env-derived enabled flag (mainly for tests)."""
    global ENABLED
    ENABLED = bool(value)


# ---------------------------------------------------------------------------
# Pure aggregation (CPU-only, unit-testable, no torch).
# ---------------------------------------------------------------------------
def fill_bucket(batch_size: int, buckets=DEFAULT_CAPTURE_BUCKETS) -> int:
    """Smallest capture bucket >= ``batch_size``.

    A CUDA-graph replay pads the actual batch up to the next captured batch
    size, so the bucket is the batch size the graph *actually ran*. If the batch
    exceeds the largest bucket (the engine would normally split it) we attribute
    it to the largest bucket for fill accounting.
    """
    for b in buckets:
        if batch_size <= b:
            return b
    return buckets[-1]


def aggregate_fill_stats_from_counts(bs_counts, buckets=DEFAULT_CAPTURE_BUCKETS) -> dict:
    """Aggregate a {actual_batch_size: occurrences} mapping into fill stats.

    Returns a summary with the per-bucket histogram, the raw per-batch-size
    histogram, mean batch size, and the mean fill ratio (actual_bs / bucket).
    ``frac_at_max_bucket`` is the fraction of replays that landed at the top
    capture bucket — the single number that says "is the big batch filling?".
    """
    buckets = tuple(buckets)
    bucket_counts = {b: 0 for b in buckets}
    raw = {}
    n = 0
    total_bs = 0
    fill_weighted = 0.0
    min_bs = None
    max_bs = None
    for bs, c in bs_counts.items():
        bs = int(bs)
        c = int(c)
        if bs <= 0 or c <= 0:
            continue
        n += c
        total_bs += bs * c
        bucket = fill_bucket(bs, buckets)
        bucket_counts[bucket] += c
        fill_weighted += (bs / bucket) * c
        raw[bs] = raw.get(bs, 0) + c
        min_bs = bs if min_bs is None else min(min_bs, bs)
        max_bs = bs if max_bs is None else max(max_bs, bs)
    top = buckets[-1]
    return {
        "count": n,
        "buckets": list(buckets),
        "bucket_counts": bucket_counts,
        "raw_bs_counts": dict(sorted(raw.items())),
        "mean_bs": (total_bs / n) if n else 0.0,
        "min_bs": min_bs if min_bs is not None else 0,
        "max_bs": max_bs if max_bs is not None else 0,
        "mean_fill_ratio": (fill_weighted / n) if n else 0.0,
        "frac_at_max_bucket": (bucket_counts[top] / n) if n else 0.0,
    }


def aggregate_fill_stats(batch_sizes, buckets=DEFAULT_CAPTURE_BUCKETS) -> dict:
    """Aggregate a flat sequence of actual per-replay batch sizes into fill stats.

    Thin wrapper over :func:`aggregate_fill_stats_from_counts`; convenient for
    tests that pass a list like ``[32, 32, 1, 2, 16]``.
    """
    return aggregate_fill_stats_from_counts(Counter(int(b) for b in batch_sizes), buckets)


# ---------------------------------------------------------------------------
# Recorder (process-global, thread-safe, bounded memory).
# ---------------------------------------------------------------------------
class BatchFillRecorder:
    """Accumulates per-replay batch sizes keyed by (node_name, graph_walk, path).

    Memory is bounded: per key we keep a Counter over the (few) distinct batch
    sizes, not one entry per replay.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._counts: dict[tuple, Counter] = defaultdict(Counter)

    def record(self, node_name: str, graph_walk: str, batch_size: int, path: str) -> None:
        try:
            bs = int(batch_size)
        except (TypeError, ValueError):
            return
        if bs <= 0:
            return
        with self._lock:
            self._counts[(node_name, graph_walk, path)][bs] += 1

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()

    def snapshot_counts(self) -> dict:
        with self._lock:
            return {k: dict(v) for k, v in self._counts.items()}

    def summarize(self, buckets=DEFAULT_CAPTURE_BUCKETS) -> dict:
        """Per-key fill summary, keyed by ``node_name/graph_walk/path``."""
        out = {}
        for (node_name, graph_walk, path), counts in self.snapshot_counts().items():
            out[f"{node_name}/{graph_walk}/{path}"] = aggregate_fill_stats_from_counts(
                counts, buckets
            )
        return out

    def is_empty(self) -> bool:
        with self._lock:
            return not self._counts

    def format_summary(self, buckets=DEFAULT_CAPTURE_BUCKETS) -> str:
        summary = self.summarize(buckets)
        if not summary:
            return "[batchfill] no replays recorded"
        lines = ["[batchfill] per-stage batch-fill summary "
                 "(buckets=%s)" % (list(buckets),)]
        for key in sorted(summary):
            s = summary[key]
            hist = " ".join(
                f"{b}:{s['bucket_counts'][b]}" for b in s["buckets"]
            )
            lines.append(
                f"  {key}: n={s['count']} mean_bs={s['mean_bs']:.2f} "
                f"fill={s['mean_fill_ratio']:.2f} "
                f"frac@{s['buckets'][-1]}={s['frac_at_max_bucket']:.2f} "
                f"hist[{hist}]"
            )
        return "\n".join(lines)


# Process-global recorder used by the engine hook.
RECORDER = BatchFillRecorder()


def record(node_name: str, graph_walk: str, batch_size: int, path: str) -> None:
    """Hot-path entry. Callers must guard with ``if batchfill_stats.ENABLED``."""
    RECORDER.record(node_name, graph_walk, batch_size, path)


def dump_summary(buckets=DEFAULT_CAPTURE_BUCKETS) -> None:
    """Log the accumulated fill summary. Safe to call when disabled/empty."""
    if RECORDER.is_empty():
        return
    logger.info("%s", RECORDER.format_summary(buckets))
    out_path = os.environ.get("MSTAR_BATCHFILL_STATS_JSON")
    if out_path:
        try:
            with open(out_path, "w") as fh:
                json.dump(RECORDER.summarize(buckets), fh, indent=2)
            logger.info("[batchfill] wrote summary JSON to %s", out_path)
        except OSError as err:
            logger.warning("[batchfill] could not write summary JSON to %s: %s",
                           out_path, err)
