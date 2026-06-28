"""Env-gated cross-process TTFT boundary tracing.

Emits one log line per (request, event) with a system-wide monotonic clock
timestamp (``time.monotonic_ns``, CLOCK_MONOTONIC on Linux — comparable across
processes on the same host). A downstream parser reconciles the events of a
single request into a per-stage TTFT breakdown.

Enabled only when ``MSTAR_NODE_TIMING`` is a positive integer, the same flag
that gates per-node GPU timing in the worker. Default path is a no-op so the
baseline stays byte-identical.
"""
from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger("mstar.ttft")

ENABLED: bool = int(os.environ.get("MSTAR_NODE_TIMING", "0") or "0") > 0


def trace(request_id: str, event: str, **fields) -> None:
    """Log a TTFT boundary crossing for ``request_id``.

    No-op unless MSTAR_NODE_TIMING is set. ``fields`` are appended as
    ``key=value`` tokens (e.g. node=audio_encoder walk=prefill_audio).
    """
    if not ENABLED:
        return
    extra = ""
    if fields:
        extra = " " + " ".join(f"{k}={v}" for k, v in fields.items())
    logger.info(
        "TTFT_TRACE rid=%s ev=%s t_ns=%d%s",
        request_id, event, time.monotonic_ns(), extra,
    )
