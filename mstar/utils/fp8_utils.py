"""Central FP8 (e4m3 / e5m2) helpers for the M* engine.

Three independent, env-gated FP8 sub-levers live here. All are **default
OFF** — when the corresponding env var is unset the helpers return values
that make every caller fall back to the existing bf16 path byte-for-byte.

  * ``MSTAR_FP8_KV``      — store the paged KV cache in fp8 (e4m3 by
                            default). Halves KV memory -> larger batches.
                            Query stays bf16; FlashInfer dequantizes K/V
                            with per-tensor ``k_scale`` / ``v_scale``.
  * ``MSTAR_FP8_WEIGHTS`` — run decode/prefill Linear GEMMs (and, later,
                            the MoE experts) through ``torch._scaled_mm``
                            with fp8 weights + dynamic per-tensor act
                            quantization. ~2x GEMM throughput on Hopper.
  * ``MSTAR_FP8_ATTN``    — run the attention compute itself in fp8
                            (query also quantized). Highest risk; design
                            scaffold only.

Numeric constants
-----------------
``torch.float8_e4m3fn`` has max representable magnitude 448.0 and ~3 bits
of mantissa; ``torch.float8_e5m2`` reaches 57344.0 with ~2 bits. e4m3 is
the default for KV/weights (more precision, the dynamic range of post-
QK-norm K and of V comfortably fits once scaled). e5m2 is offered for KV
when a workload's V activations have heavier tails.

Everything here is import-safe and runnable on CPU (no CUDA / FlashInfer
dependency at module load) so the plumbing and the scale math can be
unit-tested without a GPU. The actual fp8 matmul / attention kernels only
fire on CUDA; on CPU the helpers degrade to the equivalent bf16 result.
"""
from __future__ import annotations

import os

import torch

# ── fp8 dtype range constants ──────────────────────────────────────────
# Max finite magnitude for each fp8 format (used to derive amax scales).
FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def is_fp8_dtype(dtype: torch.dtype | None) -> bool:
    """True iff ``dtype`` is one of torch's fp8 formats."""
    return dtype in _FP8_DTYPES


def fp8_dtype_max(dtype: torch.dtype) -> float:
    """Max finite magnitude representable by an fp8 dtype."""
    if dtype == torch.float8_e4m3fn:
        return FP8_E4M3_MAX
    if dtype == torch.float8_e5m2:
        return FP8_E5M2_MAX
    raise ValueError(f"not an fp8 dtype: {dtype}")


# ── env gates (read live so tests can toggle via monkeypatch) ──────────
def _flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in ("1", "true", "yes", "on")


def fp8_kv_enabled() -> bool:
    return _flag("MSTAR_FP8_KV")


def fp8_weights_enabled() -> bool:
    return _flag("MSTAR_FP8_WEIGHTS")


def fp8_attn_enabled() -> bool:
    return _flag("MSTAR_FP8_ATTN")


def _env_dtype(name: str, default: torch.dtype) -> torch.dtype:
    val = os.environ.get(name, "").lower()
    if val in ("e4m3", "e4m3fn", "float8_e4m3fn"):
        return torch.float8_e4m3fn
    if val in ("e5m2", "float8_e5m2"):
        return torch.float8_e5m2
    return default


def kv_cache_storage_dtype(default: torch.dtype) -> torch.dtype:
    """Resolve the dtype the paged KV cache tensor should be allocated in.

    Returns ``default`` (the model's autocast/compute dtype) unless
    ``MSTAR_FP8_KV`` is set, in which case it returns the configured fp8
    format (``MSTAR_FP8_KV_DTYPE``, default e4m3).
    """
    if not fp8_kv_enabled():
        return default
    return _env_dtype("MSTAR_FP8_KV_DTYPE", torch.float8_e4m3fn)


def fp8_compute_dtype(kv_dtype: torch.dtype, fallback: torch.dtype = torch.bfloat16) -> torch.dtype:
    """The dtype attention math (query) runs in given the KV storage dtype.

    For a real (non-fp8) KV cache the compute dtype is just the storage
    dtype — preserving today's behavior exactly. For an fp8 KV cache the
    query stays in ``fallback`` (bf16): FlashInfer reads bf16 Q against
    fp8 K/V and dequantizes K/V internally with the per-tensor scales.
    """
    return fallback if is_fp8_dtype(kv_dtype) else kv_dtype


def kv_scales() -> tuple[float, float]:
    """Per-tensor (k_scale, v_scale) calibration factors for fp8 KV.

    FlashInfer convention: the value stored in the cache is the real
    value divided by ``scale``; at attention time FlashInfer multiplies
    by ``scale`` to recover the bf16 magnitude. A scale of 1.0 stores the
    raw value (valid when |K|,|V| < fp8 max). Override per-deployment via
    ``MSTAR_FP8_KV_K_SCALE`` / ``MSTAR_FP8_KV_V_SCALE`` once a calibration
    pass has measured the activation amax.

    NOTE: 1.0 is a safe *plumbing* default, not a tuned value. The GPU
    parity gate is what decides the production scales (see DESIGN_fp8.md).
    """
    k = float(os.environ.get("MSTAR_FP8_KV_K_SCALE", "1.0"))
    v = float(os.environ.get("MSTAR_FP8_KV_V_SCALE", "1.0"))
    return k, v


# ── scale computation (CPU-safe) ───────────────────────────────────────
def amax_scale(tensor: torch.Tensor, fp8_dtype: torch.dtype = torch.float8_e4m3fn) -> torch.Tensor:
    """Per-tensor amax scale = max(|t|) / fp8_max.

    Returns a 0-dim float32 tensor. Clamped away from 0 so an all-zero
    input does not produce a 0 / NaN scale. Works on CPU and CUDA.
    """
    amax = tensor.detach().abs().max().to(torch.float32)
    scale = amax / fp8_dtype_max(fp8_dtype)
    return torch.clamp(scale, min=1e-12)


def quantize_to_fp8(
    tensor: torch.Tensor,
    scale: torch.Tensor | float,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Quantize ``tensor`` to fp8: ``round((t / scale))`` clamped to range.

    ``scale`` follows the dequant convention ``real ≈ stored * scale`` so
    the stored value is ``t / scale``. Values are clamped to the fp8
    representable range before the cast to avoid inf/NaN on overflow.
    """
    fmax = fp8_dtype_max(fp8_dtype)
    scaled = (tensor.to(torch.float32) / scale).clamp(-fmax, fmax)
    return scaled.to(fp8_dtype)


def dequantize_fp8(tensor_fp8: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    """Inverse of :func:`quantize_to_fp8` -> float32."""
    return tensor_fp8.to(torch.float32) * scale


# ── fp8 weight quantization + scaled_mm linear (MSTAR_FP8_WEIGHTS) ──────
def quantize_weight_fp8(
    weight: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor quantize a Linear weight ``[out, in]`` to fp8.

    Returns ``(weight_fp8, weight_scale)`` where ``weight_scale`` is a
    0-dim float32 tensor. Intended to run once at load time and be cached
    on the module (online/dynamic quantization — Qwen3-Omni ships bf16, so
    there is no fp8 checkpoint to load).
    """
    scale = amax_scale(weight, fp8_dtype)
    weight_fp8 = quantize_to_fp8(weight, scale, fp8_dtype)
    return weight_fp8, scale


def fp8_scaled_mm_supported() -> bool:
    """Whether the runtime can execute ``torch._scaled_mm`` on fp8.

    Requires CUDA and the ``_scaled_mm`` symbol. The actual SM>=89 (Ada /
    Hopper) check is left to the GPU validation run; this only gates the
    CPU import path so callers can fall back cleanly.
    """
    return torch.cuda.is_available() and hasattr(torch, "_scaled_mm")


def fp8_linear(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """fp8 ``x @ weight^T`` via ``torch._scaled_mm`` (dynamic act scale).

    ``x`` is quantized per-tensor on the fly; ``weight_fp8`` /
    ``weight_scale`` come from :func:`quantize_weight_fp8`. Falls back to a
    plain bf16 ``F.linear`` (dequantizing the weight) whenever scaled_mm is
    unavailable, so this is exercisable on CPU.

    Shapes: ``x`` is ``[..., in]``, ``weight_fp8`` is ``[out, in]``.
    """
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])

    if not fp8_scaled_mm_supported():
        # CPU / no-fp8-HW path: dequantize weight and do a normal matmul.
        w = dequantize_fp8(weight_fp8, weight_scale).to(x.dtype)
        out = torch.nn.functional.linear(x2d, w, bias)
        return out.reshape(*orig_shape[:-1], out.shape[-1])

    x_scale = amax_scale(x2d, fp8_dtype)
    x_fp8 = quantize_to_fp8(x2d, x_scale, fp8_dtype)
    # _scaled_mm: a [M,K] row-major fp8, b [K,N] column-major fp8.
    # weight is [out=N, in=K] row-major -> weight.t() is [K,N] col-major.
    out = torch._scaled_mm(
        x_fp8,
        weight_fp8.t(),
        scale_a=x_scale,
        scale_b=weight_scale,
        bias=bias,
        out_dtype=out_dtype,
    )
    return out.reshape(*orig_shape[:-1], out.shape[-1])
