# ---------------------------------------------------------------------------
# Encoder coalescer — chunk-boundary triggered batching of encoder dispatches.
# ---------------------------------------------------------------------------
#
# Background
# ----------
# A prior experiment (``exp/encoder-coalesce``, see LEARNINGS §3.2) tried
# *wall-clock* encoder coalescing with a fixed 10 ms window: hold incoming
# encoder requests for up to N ms in the hope that more requests arrive and
# the encoder can run on a fatter batch. The result was neutral-to-negative
# because the Qwen3-Omni encoders are only ~10-20 ms total — a 10 ms wait was
# proportionally too large to ever pay back.
#
# This module re-attempts coalescing with a *workload-driven* trigger instead
# of a timer. The Thinker is the long pole; when ``MSTAR_CHUNKED_PREFILL`` is
# on it yields the scheduler at chunk boundaries. Those yields are the natural
# "is there more encoder work I should batch first?" moments. Three triggers
# replace the old timer:
#
#   (a) size: queue length reaches ``MSTAR_ENCODER_COALESCE_SIZE`` (default 4).
#   (b) chunk-boundary: the Thinker has just finished a (chunked) prefill walk
#       and is about to swap to the next step — flush before that step so the
#       newly-released encoder embeds can ride along.
#   (c) idle: the Thinker has no in-flight prefill/decode work — there is no
#       reason to hold the encoder back.
#
# (a) bounds latency, (b) makes the coalesce window match the actual
# scheduler-yield cadence, (c) keeps low-load TTFT honest (B=1 never waits).
#
# The OFF path (default) is byte-identical to today: the coalescer is never
# constructed, MicroScheduler never consults it, encoder nodes dispatch the
# moment they become ready.
#
# Audit hooks
# -----------
# The chunk-boundary event is fired from ``worker.py`` in ``_postprocess_batch``
# right after a Thinker prefill walk completes (see ``CHUNK_BOUNDARY_HOOK``
# comment there). The "idle" check is folded into ``get_next_batch``: when the
# scheduler is about to return None (no work), we force-flush.
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


_ENCODER_NODE_NAMES: frozenset[str] = frozenset({"audio_encoder", "vision_encoder"})


def encoder_chunk_coalesce_enabled() -> bool:
    """``MSTAR_ENCODER_CHUNK_COALESCE`` — master switch. Default OFF.

    When OFF, ``MicroScheduler`` never builds an ``EncoderCoalescer`` and the
    encoder-dispatch path is unchanged.
    """
    raw = os.environ.get("MSTAR_ENCODER_CHUNK_COALESCE")
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def encoder_coalesce_size() -> int:
    """``MSTAR_ENCODER_COALESCE_SIZE`` — size cap that triggers a flush. Default 4."""
    raw = os.environ.get("MSTAR_ENCODER_COALESCE_SIZE", "4").strip()
    try:
        v = int(raw)
    except ValueError:
        logger.warning("MSTAR_ENCODER_COALESCE_SIZE=%r is not an int; using 4", raw)
        return 4
    return max(1, v)


@dataclass
class _CoalescerStats:
    """Lightweight counters so a NEUTRAL benchmark can be diagnosed.

    If ``flushed_by_chunk == 0``, the chunk-boundary trigger never fired and
    all flushes fell through to size or idle — likely meaning the Thinker
    never reached a coalesce-worthy point with requests pending in the
    queue, e.g. because chunked prefill is OFF or the workload happens to
    arrive synchronously."""
    enqueued_audio: int = 0
    enqueued_vision: int = 0
    flushed_by_size: int = 0
    flushed_by_chunk: int = 0
    flushed_by_idle: int = 0
    deferred_polls: int = 0


class EncoderCoalescer:
    """Per-worker pending-queue for ``audio_encoder`` / ``vision_encoder`` dispatches.

    The scheduler (``MicroScheduler.get_next_batch``) calls ``should_dispatch``
    *every* time it considers an encoder node. The coalescer decides:

      * Return True  → release the queue for this encoder (the scheduler then
        dispatches normally, possibly batching across requests via the
        ``forward_batched`` path).
      * Return False → defer; the scheduler skips encoder nodes this round and
        considers other partitions (Thinker prefill / decode) instead.

    Externally fired events:
      * ``on_chunk_boundary()`` after a Thinker prefill walk completes.
      * ``on_idle_tick()``      when the scheduler is about to return None.

    The class holds no tensors: it only tracks request IDs / counts. Actual
    encoder dispatch is done by the existing scheduler/engine path the moment
    we say "go".
    """

    def __init__(self, size_cap: int | None = None):
        self.size_cap = size_cap if size_cap is not None else encoder_coalesce_size()
        # OrderedDict[node_name, OrderedDict[request_id, None]] — preserves
        # arrival order so a forced flush picks the oldest requests first.
        # node_name is "audio_encoder" or "vision_encoder"; we keep them
        # separate so a backed-up audio queue never gates vision dispatch.
        self._pending: dict[str, "OrderedDict[str, None]"] = {
            n: OrderedDict() for n in _ENCODER_NODE_NAMES
        }
        # Each encoder is flushed when one of:
        #   * the queue hits ``size_cap``
        #   * a chunk-boundary event fires AND queue has >=1
        #   * an idle tick fires AND queue has >=1
        # ``_release_until_seen`` is the "go" latch — set when an event fires,
        # cleared after the scheduler observes one positive ``should_dispatch``.
        self._release_pending: dict[str, str | None] = {n: None for n in _ENCODER_NODE_NAMES}
        self.stats = _CoalescerStats()

    # -- enqueue / observation ---------------------------------------------

    def observe_ready(self, node_name: str, request_id: str) -> None:
        """Note that this encoder node is ready for ``request_id``.

        Idempotent. The scheduler calls this every iteration for every ready
        encoder entry; the OrderedDict dedups while preserving first-seen
        order.
        """
        if node_name not in self._pending:
            return
        q = self._pending[node_name]
        if request_id not in q:
            q[request_id] = None
            if node_name == "audio_encoder":
                self.stats.enqueued_audio += 1
            elif node_name == "vision_encoder":
                self.stats.enqueued_vision += 1

    def drop(self, node_name: str, request_id: str) -> None:
        """Drop a request from the pending queue (after the scheduler pops it
        or the request is aborted). Idempotent."""
        q = self._pending.get(node_name)
        if q is not None:
            q.pop(request_id, None)

    # -- triggers ----------------------------------------------------------

    def on_chunk_boundary(self) -> None:
        """Thinker just finished a (chunked) prefill walk; flush pending encoders.

        Fires "go" for every encoder that has at least one pending request.
        The scheduler still picks one encoder node per iteration; the latch
        survives across iterations so both audio and vision queues drain.
        """
        for n, q in self._pending.items():
            if q and self._release_pending[n] != "size":
                # Only mark "chunk" if not already latched by size (size flush
                # is one-shot; chunk is the dominant correctness trigger).
                self._release_pending[n] = "chunk"
        # Periodic stats dump so a NEUTRAL benchmark can be diagnosed from
        # the server log alone (see /home/tim/exp_encchunk/benchmark/
        # mvp_encoder_chunk_coalesce.sh "If NEUTRAL" note).
        total_flushes = (
            self.stats.flushed_by_size
            + self.stats.flushed_by_chunk
            + self.stats.flushed_by_idle
        )
        if total_flushes > 0 and total_flushes % 50 == 0:
            logger.info(
                "MSTAR_ENCODER_CHUNK_COALESCE stats: %s", self.summary()
            )

    def on_idle_tick(self) -> None:
        """Scheduler has no other work to do; flush whatever encoders are pending."""
        for n, q in self._pending.items():
            if q and self._release_pending[n] is None:
                self._release_pending[n] = "idle"

    # -- query -------------------------------------------------------------

    def should_dispatch(self, node_name: str) -> bool:
        """May the scheduler dispatch this encoder node now?

        Returns True when one of:
          (a) queue size ≥ ``size_cap``
          (b) a chunk-boundary event has set the release latch
          (c) an idle tick has set the release latch
        Else returns False (caller defers this encoder for now).

        The latch is consumed (cleared) only when the queue actually drains,
        not on every call — see ``mark_dispatched``.
        """
        q = self._pending.get(node_name)
        if not q:
            return False
        if len(q) >= self.size_cap:
            self._release_pending[node_name] = "size"
            return True
        if self._release_pending[node_name] is not None:
            return True
        self.stats.deferred_polls += 1
        return False

    def mark_dispatched(self, node_name: str, request_ids: list[str]) -> None:
        """Called by the scheduler after it actually pops + dispatches a batch.

        Removes the dispatched rids from the pending queue and counts the
        flush reason against stats so a NEUTRAL benchmark can be diagnosed.
        """
        q = self._pending.get(node_name)
        if q is None:
            return
        for rid in request_ids:
            q.pop(rid, None)
        reason = self._release_pending.get(node_name)
        if reason == "size":
            self.stats.flushed_by_size += 1
        elif reason == "chunk":
            self.stats.flushed_by_chunk += 1
        elif reason == "idle":
            self.stats.flushed_by_idle += 1
        # If the queue is now empty, clear the latch. Otherwise keep it set:
        # multiple encoders may have been ready and the next iteration should
        # still see "go".
        if not q:
            self._release_pending[node_name] = None

    def summary(self) -> dict[str, int]:
        """Stat dump for logs / inspection (called on server teardown)."""
        s = self.stats
        return {
            "enqueued_audio": s.enqueued_audio,
            "enqueued_vision": s.enqueued_vision,
            "flushed_by_size": s.flushed_by_size,
            "flushed_by_chunk": s.flushed_by_chunk,
            "flushed_by_idle": s.flushed_by_idle,
            "deferred_polls": s.deferred_polls,
            "queue_audio_now": len(self._pending["audio_encoder"]),
            "queue_vision_now": len(self._pending["vision_encoder"]),
            "size_cap": self.size_cap,
        }
