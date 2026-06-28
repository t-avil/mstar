"""CPU-safe tests for the FP8 scaffolding (``mstar.utils.fp8_utils``).

These pin the parts of the FP8 levers that do *not* need a GPU:

  * the env gates default OFF (so bf16 behavior is unchanged);
  * KV storage-dtype selection flips to e4m3 only when MSTAR_FP8_KV is set;
  * amax scale + quantize/dequantize round-trips stay within the e4m3
    error bound;
  * ``fp8_linear`` on CPU (no scaled_mm) matches a bf16 reference matmul
    within the fp8 weight-quantization error;
  * ``fp8_compute_dtype`` keeps the query in bf16 for an fp8 cache and
    leaves a real cache untouched.

The actual fp8 kernels (FlashInfer fp8 KV, ``torch._scaled_mm``) are only
exercised on the GPU validation run — see DESIGN_fp8.md.
"""
import importlib

import pytest
import torch

import mstar.utils.fp8_utils as fp8


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (
        "MSTAR_FP8_KV", "MSTAR_FP8_WEIGHTS", "MSTAR_FP8_ATTN",
        "MSTAR_FP8_KV_DTYPE", "MSTAR_FP8_KV_K_SCALE", "MSTAR_FP8_KV_V_SCALE",
    ):
        monkeypatch.delenv(k, raising=False)
    importlib.reload(fp8)
    yield
    importlib.reload(fp8)


def test_gates_default_off():
    assert not fp8.fp8_kv_enabled()
    assert not fp8.fp8_weights_enabled()
    assert not fp8.fp8_attn_enabled()


def test_kv_storage_dtype_default_passthrough():
    # OFF -> the cache stays in the compute dtype, unchanged.
    assert fp8.kv_cache_storage_dtype(torch.bfloat16) == torch.bfloat16
    assert fp8.kv_cache_storage_dtype(torch.float32) == torch.float32


def test_kv_storage_dtype_fp8_when_enabled(monkeypatch):
    monkeypatch.setenv("MSTAR_FP8_KV", "1")
    assert fp8.kv_cache_storage_dtype(torch.bfloat16) == torch.float8_e4m3fn
    monkeypatch.setenv("MSTAR_FP8_KV_DTYPE", "e5m2")
    assert fp8.kv_cache_storage_dtype(torch.bfloat16) == torch.float8_e5m2


def test_compute_dtype_split():
    # Real cache: compute dtype == storage dtype (today's behavior).
    assert fp8.fp8_compute_dtype(torch.bfloat16) == torch.bfloat16
    assert fp8.fp8_compute_dtype(torch.float32) == torch.float32
    # fp8 cache: query stays bf16.
    assert fp8.fp8_compute_dtype(torch.float8_e4m3fn) == torch.bfloat16


def test_is_fp8_dtype():
    assert fp8.is_fp8_dtype(torch.float8_e4m3fn)
    assert fp8.is_fp8_dtype(torch.float8_e5m2)
    assert not fp8.is_fp8_dtype(torch.bfloat16)
    assert not fp8.is_fp8_dtype(None)


def test_amax_scale_and_roundtrip():
    torch.manual_seed(0)
    x = torch.randn(256, 128) * 7.0
    scale = fp8.amax_scale(x, torch.float8_e4m3fn)
    xq = fp8.quantize_to_fp8(x, scale, torch.float8_e4m3fn)
    assert xq.dtype == torch.float8_e4m3fn
    xdq = fp8.dequantize_fp8(xq, scale)
    # e4m3 has ~3 mantissa bits -> relative error a few percent. Compare on
    # the populated range via normalized RMS error.
    rms = (xdq - x).pow(2).mean().sqrt()
    denom = x.pow(2).mean().sqrt()
    assert (rms / denom).item() < 0.1


def test_amax_scale_all_zero_is_finite():
    z = torch.zeros(16, 16)
    s = fp8.amax_scale(z)
    assert torch.isfinite(s).all() and s.item() > 0


def test_quantize_clamps_overflow():
    # Values far above e4m3 max must clamp, not become inf.
    x = torch.full((8,), 1e6)
    xq = fp8.quantize_to_fp8(x, 1.0, torch.float8_e4m3fn)
    assert torch.isfinite(xq.to(torch.float32)).all()
    assert xq.to(torch.float32).max().item() <= fp8.FP8_E4M3_MAX


def test_fp8_linear_cpu_matches_bf16():
    torch.manual_seed(0)
    x = torch.randn(32, 64)
    w = torch.randn(48, 64)  # [out, in]
    bias = torch.randn(48)
    wq, ws = fp8.quantize_weight_fp8(w)
    # CPU path (no scaled_mm) dequantizes the weight and does F.linear.
    out = fp8.fp8_linear(x, wq, ws, bias, out_dtype=torch.float32)
    ref = torch.nn.functional.linear(x, w, bias)
    rms = (out - ref).pow(2).mean().sqrt()
    denom = ref.pow(2).mean().sqrt()
    assert (rms / denom).item() < 0.1
    assert out.shape == ref.shape


def test_fp8_linear_preserves_leading_dims():
    x = torch.randn(4, 7, 64)
    w = torch.randn(16, 64)
    wq, ws = fp8.quantize_weight_fp8(w)
    out = fp8.fp8_linear(x, wq, ws, None, out_dtype=torch.float32)
    assert out.shape == (4, 7, 16)


def test_kv_scales_env_override(monkeypatch):
    assert fp8.kv_scales() == (1.0, 1.0)
    monkeypatch.setenv("MSTAR_FP8_KV_K_SCALE", "0.5")
    monkeypatch.setenv("MSTAR_FP8_KV_V_SCALE", "2.0")
    assert fp8.kv_scales() == (0.5, 2.0)
