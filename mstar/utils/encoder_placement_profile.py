"""Per-request stage timing for Qwen3-Omni I2T encoder placement profiling.

Env-gated (``MSTAR_PROFILE_ENCODER_PLACEMENT=1``); default OFF -> all
entry points become no-ops and the hot path is unchanged.  When ON we
record monotonic-ns timestamps for each I2T stage and append one JSON
line per request-segment to ``MSTAR_PROFILE_ENCODER_PLACEMENT_PATH``
(default ``/home/tim/tmp/encoder_placement_profile.jsonl``).

This is a profiling experiment, NOT an optimization.  Stages captured:

    ts_arrived_ns           image arrives at the API server
    ts_preprocess_start_ns  image preprocess (CPU or GPU) begins
    ts_preprocess_end_ns    image preprocess finishes
    ts_vision_fwd_start_ns  vision encoder forward starts (CPU launch ts)
    ts_vision_fwd_end_ns    vision encoder forward returns (CPU side)
    ts_vision_delivered_ns  vision tokens / deepstack ready for Thinker
    ts_thinker_prefill_text_start_ns / _end_ns
                            Thinker prefill_text walk start / end
    ts_first_decode_ns      first thinker_decode call (first emitted token)
    ts_complete_ns          request_done in conductor

Multi-process model: mstar runs the API server, conductor and one or more
GPU workers as separate processes.  Each process accumulates its own
per-request stage map and flushes it as a JSON line on the per-stage
sentinel that ends that process's involvement (worker -> on
``record_request_done``; conductor -> on ``record_complete``; api_server
-> on ``record_complete``).  Multiple lines can therefore exist for the
same ``request_id``; the MVP analyser merges them by request_id by
unioning their stage timestamps (each stage is written by exactly one
process, so there are no conflicts).  Appending short JSON lines from
multiple writers to the same file with ``open(..., 'a')`` is atomic on
Linux when the line length stays below ``PIPE_BUF`` (4096 bytes), which
holds for our ~16-key records.

We avoid GPU sync stalls during the run: CPU timestamps wrap the
PyTorch *call* (which only enqueues kernels for async GPU stages, so
the GPU work itself may finish later), and a single
``torch.cuda.synchronize()`` is performed on the worker's
``record_request_done`` boundary -- this ensures the emitted stamps
reflect work that actually finished, while the on-line markers stay
sync-free.  This is consistent with the worker's NVTX range usage
(``synchronize=False`` by default).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Env / state
# ---------------------------------------------------------------------------


def _enabled() -> bool:
    return os.environ.get("MSTAR_PROFILE_ENCODER_PLACEMENT", "0") == "1"


def _output_path() -> str:
    return os.environ.get(
        "MSTAR_PROFILE_ENCODER_PLACEMENT_PATH",
        "/home/tim/tmp/encoder_placement_profile.jsonl",
    )


# Per-request stage timestamps.  Keyed by request_id -> stage_name -> ts_ns.
# Also holds aux fields (batch_size, path, ...).  The keys ``ts_*_ns`` and
# anything starting with ``meta_`` are serialized as-is.
_records: dict[str, dict[str, Any]] = {}
_records_lock = threading.Lock()

# Per-request guard so we only stamp ts_first_decode_ns once.
_first_decode_seen: set[str] = set()


def _now_ns() -> int:
    return time.monotonic_ns()


# ---------------------------------------------------------------------------
# Stage stamps -- one helper per stage so call sites stay short.
# ---------------------------------------------------------------------------


def _stamp(request_id: str | None, stage: str, **extra: Any) -> None:
    """Record one stage timestamp for ``request_id`` (no-op when OFF or no rid)."""
    if not _enabled() or request_id is None:
        return
    ts = _now_ns()
    with _records_lock:
        rec = _records.setdefault(request_id, {"request_id": request_id})
        # Last-write-wins per stage (cheap); first decode is guarded separately.
        rec[stage] = ts
        if extra:
            for k, v in extra.items():
                rec[k] = v


def record_arrived(request_id: str, *, path: str | None = None) -> None:
    """Image (request) just arrived at the API server."""
    _stamp(request_id, "ts_arrived_ns", path=path)


def record_preprocess_start(request_id: str | None) -> None:
    _stamp(request_id, "ts_preprocess_start_ns")


def record_preprocess_end(request_id: str | None) -> None:
    _stamp(request_id, "ts_preprocess_end_ns")


def record_vision_fwd_start(request_id: str | None, *, batch_size: int | None = None) -> None:
    if batch_size is not None:
        _stamp(request_id, "ts_vision_fwd_start_ns", batch_size=batch_size)
    else:
        _stamp(request_id, "ts_vision_fwd_start_ns")


def record_vision_fwd_end(request_id: str | None) -> None:
    _stamp(request_id, "ts_vision_fwd_end_ns")


def record_vision_delivered(request_id: str | None) -> None:
    """Vision embeddings + deepstack handed off to Thinker (post-slice)."""
    _stamp(request_id, "ts_vision_delivered_ns")


def record_prefill_text_start(request_id: str | None) -> None:
    _stamp(request_id, "ts_thinker_prefill_text_start_ns")


def record_prefill_text_end(request_id: str | None) -> None:
    _stamp(request_id, "ts_thinker_prefill_text_end_ns")


def record_first_decode(request_id: str | None) -> None:
    """Stamp the first thinker_decode for ``request_id`` (subsequent calls no-op)."""
    if not _enabled() or request_id is None:
        return
    with _records_lock:
        if request_id in _first_decode_seen:
            return
        _first_decode_seen.add(request_id)
        rec = _records.setdefault(request_id, {"request_id": request_id})
        rec["ts_first_decode_ns"] = _now_ns()


def set_meta(request_id: str | None, **fields: Any) -> None:
    """Attach metadata fields (e.g. batch_size, path) to a request's record."""
    if not _enabled() or request_id is None:
        return
    with _records_lock:
        rec = _records.setdefault(request_id, {"request_id": request_id})
        for k, v in fields.items():
            rec[k] = v


def record_complete(request_id: str | None, *, sync_cuda: bool = False) -> None:
    """Stamp ts_complete_ns (conductor side) and flush this process's record.

    Conductors run in their own process and observe the global "request
    done" event after the conductor has reaped all partitions.  Flushing
    here writes the conductor-side stamps to JSONL.
    """
    if not _enabled() or request_id is None:
        return
    _stamp(request_id, "ts_complete_ns")
    flush(request_id, sync_cuda=sync_cuda, proc="conductor")


def record_request_done(request_id: str | None, *, sync_cuda: bool = True) -> None:
    """Worker-side flush, called after the last node-batch for ``request_id``.

    The optional ``torch.cuda.synchronize()`` is the ONLY sync we do per
    request: it ensures the emitted CPU timestamps reflect GPU work that
    has actually finished (relevant for stages whose CPU return only
    enqueued kernels).  All in-run stamps stay sync-free so
    instrumentation never serializes the pipeline.
    """
    if not _enabled() or request_id is None:
        return

    if sync_cuda:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:  # pragma: no cover - never let profiling break a run
            logger.debug("encoder_placement_profile: cuda sync failed", exc_info=True)

    flush(request_id, sync_cuda=False, proc="worker")


def flush(
    request_id: str | None, *, sync_cuda: bool = False, proc: str | None = None
) -> None:
    """Append this process's accumulated record for ``request_id`` to JSONL.

    ``proc`` is an optional explicit tag for the calling process
    (``api_server`` / ``conductor`` / ``worker``).  Falls back to the
    ``MSTAR_PROFILE_PROC_TAG`` env var (handy when a wrapper script wants
    to override) and finally ``"unknown"``.
    """
    if not _enabled() or request_id is None:
        return

    if sync_cuda:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:  # pragma: no cover
            logger.debug("encoder_placement_profile: cuda sync failed", exc_info=True)

    with _records_lock:
        rec = _records.pop(request_id, None)
        _first_decode_seen.discard(request_id)

    if rec is None or len(rec) <= 1:
        # Nothing useful recorded in this process (e.g. text-only request,
        # or process never saw this rid) -- skip.
        return

    # Tag the process so downstream analysis can tell records apart.
    rec["proc"] = proc or os.environ.get("MSTAR_PROFILE_PROC_TAG") or "unknown"
    rec["pid"] = os.getpid()

    path = _output_path()
    try:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        # Atomic append on Linux for lines < PIPE_BUF (4096); our rows are
        # well under that.  One open() per record is fine -- this fires at
        # request-done frequency, not per-token.
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError as e:  # pragma: no cover
        logger.warning("encoder_placement_profile: failed to write %s: %s", path, e)


def drop(request_id: str | None) -> None:
    """Discard any partial state for ``request_id`` (e.g. on abort)."""
    if not _enabled() or request_id is None:
        return
    with _records_lock:
        _records.pop(request_id, None)
        _first_decode_seen.discard(request_id)
