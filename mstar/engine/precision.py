"""Env-gated fp32 / TF32 precision toggles for A/B benchmarking.

Two independent, low-risk toggles. Each defaults to the *current* behavior, so
leaving the env vars unset changes nothing.

1. ``MSTAR_FP32_MATMUL_PRECISION`` in ``{highest, high, medium}``
   Drives ``torch.set_float32_matmul_precision(...)`` (applied once at engine
   import). On Hopper this is the real "fp32 hack": ``high``/``medium`` route
   fp32 matmuls through TF32 tensor cores. The codebase historically hardcoded
   ``high``; this toggle just makes that choice measurable. Default: ``high``.

2. ``MSTAR_VOCODER_FP32_PRECISION`` in ``{ieee, tf32}``
   The Code2Wav vocoder runs in true fp32 (no autocast,
   ``force_float32_submodules=True``) and is the only meaningful fp32 surface
   left. This toggle scopes a TF32-vs-IEEE choice to the *vocoder forward only*
   via a context manager that saves/restores the backend flags, so it does not
   leak into the other (bf16) engines sharing the process.

   NOTE: TF32 keeps fp32 range but truncates the mantissa to 10 bits. Setting
   this to ``tf32`` trades vocoder audio fidelity for speed and MUST be cleared
   by an audio-quality gate (waveform byte / PSNR / MOS comparison against the
   ``ieee`` reference) before it is trusted. It exists for A/B measurement.

The vocoder backend flags are set with the PyTorch 2.9 string API
(``torch.backends.cuda.matmul.fp32_precision`` /
``torch.backends.cudnn.conv.fp32_precision``) when present, falling back to the
legacy ``allow_tf32`` booleans on older builds. All of this is wrapped in
try/except so a version/attribute mismatch degrades to a no-op rather than
crashing inference.
"""

from __future__ import annotations

import contextlib
import logging
import os

import torch

logger = logging.getLogger(__name__)

ENV_MATMUL_PRECISION = "MSTAR_FP32_MATMUL_PRECISION"
ENV_VOCODER_FP32_PRECISION = "MSTAR_VOCODER_FP32_PRECISION"

_VALID_MATMUL = ("highest", "high", "medium")
# Current hardcoded behavior at mstar/engine/__init__.py before this toggle.
_DEFAULT_MATMUL = "high"

_VALID_VOCODER = ("ieee", "tf32")


def resolve_matmul_precision() -> str:
    """Resolve ``MSTAR_FP32_MATMUL_PRECISION`` to a valid mode.

    Unset or invalid -> the default (``high``), preserving current behavior.
    """
    raw = os.environ.get(ENV_MATMUL_PRECISION)
    if raw is None:
        return _DEFAULT_MATMUL
    val = raw.strip().lower()
    if val not in _VALID_MATMUL:
        logger.warning(
            "%s=%r is not one of %s; falling back to %r",
            ENV_MATMUL_PRECISION,
            raw,
            _VALID_MATMUL,
            _DEFAULT_MATMUL,
        )
        return _DEFAULT_MATMUL
    return val


def resolve_vocoder_fp32_precision() -> str | None:
    """Resolve ``MSTAR_VOCODER_FP32_PRECISION``.

    Returns ``"ieee"`` or ``"tf32"`` when explicitly set, else ``None`` which
    means "do not force anything" (current behavior). Invalid -> ``None``.
    """
    raw = os.environ.get(ENV_VOCODER_FP32_PRECISION)
    if raw is None:
        return None
    val = raw.strip().lower()
    if val not in _VALID_VOCODER:
        logger.warning(
            "%s=%r is not one of %s; ignoring (vocoder precision unchanged)",
            ENV_VOCODER_FP32_PRECISION,
            raw,
            _VALID_VOCODER,
        )
        return None
    return val


def apply_matmul_precision() -> str:
    """Apply the resolved fp32 matmul precision globally and return it."""
    prec = resolve_matmul_precision()
    torch.set_float32_matmul_precision(prec)
    return prec


def _set_one_backend(
    obj,
    new_attr: str,
    new_value: str,
    legacy_attr: str,
    legacy_value: bool,
):
    """Set a single backend's fp32 precision, returning a restore thunk.

    Prefers the PyTorch 2.9 string attribute ``new_attr``; falls back to the
    legacy boolean ``legacy_attr`` on ``obj`` (or a sibling object passed in).
    Returns a zero-arg callable that restores the previous value, or ``None``
    if neither attribute is available / settable.
    """
    if obj is None:
        return None
    # New string API (PyTorch >= 2.9).
    if hasattr(obj, new_attr):
        try:
            old = getattr(obj, new_attr)
            setattr(obj, new_attr, new_value)
            return lambda: setattr(obj, new_attr, old)
        except Exception:  # pragma: no cover - defensive
            logger.warning("failed to set %s.%s", obj, new_attr, exc_info=True)
    # Legacy boolean API.
    if legacy_attr is not None and hasattr(obj, legacy_attr):
        try:
            old = getattr(obj, legacy_attr)
            setattr(obj, legacy_attr, legacy_value)
            return lambda: setattr(obj, legacy_attr, old)
        except Exception:  # pragma: no cover - defensive
            logger.warning("failed to set %s.%s", obj, legacy_attr, exc_info=True)
    return None


def _save_and_set_vocoder_precision(mode: str):
    """Set CUDA matmul + cuDNN conv fp32 precision to ``mode`` for the vocoder.

    ``mode`` is ``"tf32"`` or ``"ieee"``. Returns a list of restore thunks.
    """
    new_value = mode  # 'tf32' or 'ieee' for the string API
    bool_value = mode == "tf32"  # for the legacy allow_tf32 booleans
    restorers = []

    # CUDA matmul: new attr lives on torch.backends.cuda.matmul, as does the
    # legacy allow_tf32 boolean.
    try:
        matmul = torch.backends.cuda.matmul
        r = _set_one_backend(matmul, "fp32_precision", new_value, "allow_tf32", bool_value)
        if r is not None:
            restorers.append(r)
    except Exception:  # pragma: no cover - defensive
        logger.warning("vocoder precision: cuda.matmul unavailable", exc_info=True)

    # cuDNN conv: new attr lives on torch.backends.cudnn.conv (>=2.9); the
    # legacy allow_tf32 boolean lives on torch.backends.cudnn itself.
    try:
        cudnn = torch.backends.cudnn
        conv = getattr(cudnn, "conv", None)
        r = None
        if conv is not None and hasattr(conv, "fp32_precision"):
            r = _set_one_backend(conv, "fp32_precision", new_value, None, bool_value)
        if r is None:
            r = _set_one_backend(cudnn, "fp32_precision", new_value, "allow_tf32", bool_value)
        if r is not None:
            restorers.append(r)
    except Exception:  # pragma: no cover - defensive
        logger.warning("vocoder precision: cudnn unavailable", exc_info=True)

    return restorers


def _restore(restorers) -> None:
    for r in reversed(restorers):
        try:
            r()
        except Exception:  # pragma: no cover - defensive
            logger.warning("vocoder precision: restore failed", exc_info=True)


@contextlib.contextmanager
def vocoder_precision_context():
    """Scope the vocoder fp32-vs-TF32 choice to the wrapped forward.

    No-op (current behavior) unless ``MSTAR_VOCODER_FP32_PRECISION`` is set.
    Always restores the backend flags on exit, including on exception, so the
    setting never leaks into the bf16 engines sharing the process.
    """
    mode = resolve_vocoder_fp32_precision()
    if mode is None:
        yield
        return
    restorers = _save_and_set_vocoder_precision(mode)
    try:
        yield
    finally:
        _restore(restorers)


def log_precision_settings(matmul_precision: str | None = None) -> None:
    """Log the resolved precision toggles once at startup."""
    if matmul_precision is None:
        matmul_precision = resolve_matmul_precision()
    vocoder = resolve_vocoder_fp32_precision()
    logger.info(
        "precision toggles: %s=%s (set via %s), vocoder fp32 precision=%s (set via %s)",
        ENV_MATMUL_PRECISION,
        matmul_precision,
        "env" if os.environ.get(ENV_MATMUL_PRECISION) else "default",
        vocoder if vocoder is not None else "unchanged",
        "env" if os.environ.get(ENV_VOCODER_FP32_PRECISION) else "default",
    )
