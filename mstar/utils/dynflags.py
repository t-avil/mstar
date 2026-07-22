"""Dynamic env flags for fast A/B triage (MSTAR_DYNFLAGS).

Point MSTAR_DYNFLAGS at a JSON file of {"ENV_VAR": "value"}. Processes that
call maybe_refresh() pick up edits to that file at runtime (mtime-gated stat,
~1µs when unchanged) and apply them to os.environ, so flag helpers that read
the environment per call (mixed_single_chunk_enabled, _envflag, ...) flip
WITHOUT a server restart — one server, interleaved A/B cells, no cold capture
between configs and no time-separated noise.

ONLY flags that are read dynamically (per call / per step) respond. Flags
cached at init (e.g. Worker.mixed_single_chunk, the scheduler's
_mixed_min_decode cache) need their cache registered via register_cache_clear
from the owning module, or they keep their boot value. Triage-only: benchmarks
that produce committed numbers must still run with static env.
"""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

_path = os.environ.get("MSTAR_DYNFLAGS", "").strip() or None
_last_mtime: float = -1.0
_cache_clears: list = []


def register_cache_clear(fn) -> None:
    """Register a callable invoked after each applied refresh (clear caches
    derived from env). Safe to call from any module at import/init time."""
    _cache_clears.append(fn)


def enabled() -> bool:
    return _path is not None


def maybe_refresh() -> bool:
    """Stat the flags file; on mtime change, apply its keys to os.environ and
    run registered cache-clears. Returns True when a refresh was applied.
    Never raises: a missing/garbled file is skipped (logged once per change).
    """
    global _last_mtime
    if _path is None:
        return False
    try:
        mtime = os.stat(_path).st_mtime
    except OSError:
        return False
    if mtime == _last_mtime:
        return False
    _last_mtime = mtime
    try:
        with open(_path) as f:
            flags = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("MSTAR_DYNFLAGS: unreadable %s: %s", _path, e)
        return False
    applied = {}
    for k, v in flags.items():
        v = str(v)
        if os.environ.get(k) != v:
            os.environ[k] = v
            applied[k] = v
    for fn in _cache_clears:
        try:
            fn()
        except Exception:
            logger.exception("MSTAR_DYNFLAGS: cache clear failed")
    if applied:
        logger.warning("MSTAR_DYNFLAGS applied: %s", applied)
    return bool(applied)
