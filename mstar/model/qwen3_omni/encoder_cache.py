"""Process-local LRU cache for encoder outputs, keyed by input content hash.

Experiment E2 (MSTAR_ENCODER_CACHE). Workloads that repeatedly feed the same
audio clip (system prompt audio across many requests) or the same image
(reference image, product image) re-run the encoder on identical inputs every
time. This module memoizes the encoder forward by hashing the raw input bytes
and stashing the encoded tokens in an LRU.

Design notes:
- The cache is process-local. The encoder worker process is the only consumer
  that benefits; no cross-process sharing.
- Keys are ``(modality, sha256_of_raw_bytes)``. We deliberately hash the raw
  input bytes (pcm/image patch tensor + grid_thw or audio_seqlens) rather than
  any preprocessed/normalized form, so two identical clips map to one key.
- Values are the same tensor dicts the encoder forward would produce. They live
  on the encoder device. We approximate per-entry GPU bytes for eviction.
- LRU eviction triggers when bytes_used > capacity_mb * 1024**2. We use an
  OrderedDict (move_to_end on hit) as the LRU mechanism.
- When the flag is OFF (default), ``is_enabled()`` returns False and the
  submodules skip the cache call entirely — the path is byte-identical to the
  pre-experiment baseline.
- Concurrency: the cache is not protected by a lock. The encoder worker calls
  the cache from a single execution thread per submodule, so this is safe in
  practice. If multi-threaded forwards are ever introduced, wrap mutating ops
  in a Lock.

Hit-rate instrumentation:
- Every lookup updates ``hits`` / ``misses`` counters. ``log_summary()`` prints
  a one-line summary so a benchmark verdict is interpretable even when the
  dataset cycles distinct items (i.e. the verdict is NEUTRAL by design when
  hit rate is ~0).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from typing import Any

import torch

logger = logging.getLogger(__name__)

_ENV_FLAG = "MSTAR_ENCODER_CACHE"
_ENV_SIZE_MB = "MSTAR_ENCODER_CACHE_SIZE_MB"
_ENV_LOG_EVERY = "MSTAR_ENCODER_CACHE_LOG_EVERY"
_DEFAULT_SIZE_MB = 512
_DEFAULT_LOG_EVERY = 50


def is_enabled() -> bool:
    """True iff MSTAR_ENCODER_CACHE=1. Default OFF — byte-identical baseline."""
    return os.environ.get(_ENV_FLAG, "0") == "1"


def _capacity_bytes() -> int:
    """Cache capacity in bytes, from MSTAR_ENCODER_CACHE_SIZE_MB or default."""
    try:
        mb = int(os.environ.get(_ENV_SIZE_MB, str(_DEFAULT_SIZE_MB)))
    except ValueError:
        mb = _DEFAULT_SIZE_MB
    return max(1, mb) * 1024 * 1024


def _tensor_bytes(t: torch.Tensor) -> int:
    return int(t.element_size() * t.numel())


def _value_bytes(value: dict[str, list[torch.Tensor]]) -> int:
    """Approximate GPU bytes for a cached encoder output dict."""
    total = 0
    for _name, tensors in value.items():
        for t in tensors:
            if isinstance(t, torch.Tensor):
                total += _tensor_bytes(t)
    return total


def _bytes_for_hash(t: torch.Tensor) -> bytes:
    """Stable byte view of a tensor's raw storage.

    Move to CPU and make contiguous so two tensors with the same logical data
    but different strides/devices still hash identically. Cast to a known
    layout but DO NOT change dtype: we hash whatever the encoder will see
    after preprocessing on the host side (the data worker produced these on
    CPU already; for GPU mel/preprocess paths the input may live on GPU).
    """
    if t.device.type != "cpu":
        t = t.detach().cpu()
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def content_hash(*tensors: torch.Tensor | None, extra: bytes = b"") -> str:
    """SHA-256 of the concatenated raw bytes of ``tensors`` (skipping None).

    ``extra`` lets a caller fold in a small modality-specific salt (e.g.
    a grid_thw tuple) without needing a second tensor argument.
    """
    h = hashlib.sha256()
    for t in tensors:
        if t is None:
            continue
        # Fold in shape and dtype so two distinct tensors with overlapping
        # bytes never collide on a layout difference.
        h.update(repr(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(_bytes_for_hash(t))
    if extra:
        h.update(extra)
    return h.hexdigest()


class EncoderCache:
    """LRU cache keyed by (modality, content_hash) -> encoder output dict."""

    def __init__(self, capacity_bytes: int, log_every: int = _DEFAULT_LOG_EVERY):
        self.capacity_bytes = capacity_bytes
        self._store: "OrderedDict[tuple[str, str], dict[str, list[torch.Tensor]]]" = OrderedDict()
        self._bytes_used = 0
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        # Periodic hit-rate logging makes the cache verdict interpretable when
        # the benchmark dataset cycles distinct items (hit rate ~0 is the
        # expected outcome — and now visible in the log).
        self._log_every = max(0, log_every)

    def get(self, modality: str, key: str) -> dict[str, list[torch.Tensor]] | None:
        with self._lock:
            entry = self._store.get((modality, key))
            if entry is None:
                self.misses += 1
                hit = False
            else:
                self._store.move_to_end((modality, key))
                self.hits += 1
                hit = True
            total = self.hits + self.misses
            should_log = self._log_every > 0 and total > 0 and total % self._log_every == 0
            summary = self.summary() if should_log else None
        if summary:
            logger.info(summary)
        return entry if hit else None

    def put(self, modality: str, key: str, value: dict[str, list[torch.Tensor]]) -> None:
        with self._lock:
            tag = (modality, key)
            if tag in self._store:
                # Refresh LRU position; capacity bookkeeping unchanged.
                self._store.move_to_end(tag)
                return
            size = _value_bytes(value)
            # Evict until we fit (or the store is empty).
            while self._bytes_used + size > self.capacity_bytes and self._store:
                _ev_key, ev_val = self._store.popitem(last=False)
                self._bytes_used -= _value_bytes(ev_val)
            self._store[tag] = value
            self._bytes_used += size

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def summary(self) -> str:
        return (
            f"encoder_cache: entries={len(self._store)} "
            f"bytes_used={self._bytes_used} cap={self.capacity_bytes} "
            f"hits={self.hits} misses={self.misses} "
            f"hit_rate={self.hit_rate():.3f}"
        )

    def log_summary(self) -> None:
        logger.info(self.summary())


_singleton: EncoderCache | None = None
_singleton_lock = threading.Lock()


def _log_every() -> int:
    try:
        return int(os.environ.get(_ENV_LOG_EVERY, str(_DEFAULT_LOG_EVERY)))
    except ValueError:
        return _DEFAULT_LOG_EVERY


def get_cache() -> EncoderCache:
    """Return the process-local cache singleton. Lazily constructed."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = EncoderCache(_capacity_bytes(), log_every=_log_every())
                logger.info(
                    "Encoder cache initialized: capacity=%d MB log_every=%d",
                    _singleton.capacity_bytes // (1024 * 1024),
                    _singleton._log_every,
                )
    return _singleton


def reset_for_test() -> None:
    """Test helper — drop the singleton so the next get_cache() rebuilds."""
    global _singleton
    with _singleton_lock:
        _singleton = None
