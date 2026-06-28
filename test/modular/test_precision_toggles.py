"""Pure-CPU tests for env-gated precision toggles (no GPU required).

Covers:
- env parsing for MSTAR_FP32_MATMUL_PRECISION / MSTAR_VOCODER_FP32_PRECISION
  (unset, valid, invalid),
- the vocoder precision context manager with the new PyTorch 2.9 string API,
- the fallback path when the new API is absent (legacy allow_tf32 booleans),
- that it stays a no-op (and never crashes) when the env var is unset.

All backend objects are monkeypatched so nothing touches a real GPU.
"""

import types

import pytest

from mstar.engine import precision


# ─── env parsing ──────────────────────────────────────────────────────────


def test_matmul_precision_unset_defaults_to_high(monkeypatch):
    monkeypatch.delenv(precision.ENV_MATMUL_PRECISION, raising=False)
    assert precision.resolve_matmul_precision() == "high"


@pytest.mark.parametrize("value", ["highest", "high", "medium"])
def test_matmul_precision_valid_values(monkeypatch, value):
    monkeypatch.setenv(precision.ENV_MATMUL_PRECISION, value)
    assert precision.resolve_matmul_precision() == value


def test_matmul_precision_case_and_whitespace_insensitive(monkeypatch):
    monkeypatch.setenv(precision.ENV_MATMUL_PRECISION, "  HIGHEST  ")
    assert precision.resolve_matmul_precision() == "highest"


def test_matmul_precision_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(precision.ENV_MATMUL_PRECISION, "bananas")
    assert precision.resolve_matmul_precision() == "high"


def test_vocoder_precision_unset_is_none(monkeypatch):
    monkeypatch.delenv(precision.ENV_VOCODER_FP32_PRECISION, raising=False)
    assert precision.resolve_vocoder_fp32_precision() is None


@pytest.mark.parametrize("value", ["ieee", "tf32"])
def test_vocoder_precision_valid_values(monkeypatch, value):
    monkeypatch.setenv(precision.ENV_VOCODER_FP32_PRECISION, value)
    assert precision.resolve_vocoder_fp32_precision() == value


def test_vocoder_precision_invalid_is_none(monkeypatch):
    monkeypatch.setenv(precision.ENV_VOCODER_FP32_PRECISION, "fp8")
    assert precision.resolve_vocoder_fp32_precision() is None


# ─── fake torch.backends for the context manager ──────────────────────────


def _install_fake_backends(monkeypatch, *, new_api: bool):
    """Replace torch.backends.cuda.matmul / cudnn(.conv) with fakes.

    new_api=True  -> objects expose the PyTorch 2.9 `fp32_precision` string attr.
    new_api=False -> objects only expose the legacy `allow_tf32` booleans.
    Returns the fake objects so tests can assert on them.
    """
    import torch

    if new_api:
        matmul = types.SimpleNamespace(fp32_precision="ieee", allow_tf32=False)
        conv = types.SimpleNamespace(fp32_precision="ieee")
        cudnn = types.SimpleNamespace(conv=conv, allow_tf32=False)
    else:
        # No fp32_precision anywhere; only the legacy booleans exist.
        matmul = types.SimpleNamespace(allow_tf32=False)
        conv = None
        cudnn = types.SimpleNamespace(allow_tf32=False)

    monkeypatch.setattr(torch.backends.cuda, "matmul", matmul, raising=False)
    monkeypatch.setattr(torch.backends, "cudnn", cudnn, raising=False)
    return matmul, conv, cudnn


def test_vocoder_context_noop_when_unset(monkeypatch):
    monkeypatch.delenv(precision.ENV_VOCODER_FP32_PRECISION, raising=False)
    matmul, _conv, cudnn = _install_fake_backends(monkeypatch, new_api=True)
    with precision.vocoder_precision_context():
        # Untouched: still the initial values.
        assert matmul.fp32_precision == "ieee"
        assert cudnn.conv.fp32_precision == "ieee"


def test_vocoder_context_new_api_sets_and_restores(monkeypatch):
    monkeypatch.setenv(precision.ENV_VOCODER_FP32_PRECISION, "tf32")
    matmul, conv, cudnn = _install_fake_backends(monkeypatch, new_api=True)
    with precision.vocoder_precision_context():
        assert matmul.fp32_precision == "tf32"
        assert conv.fp32_precision == "tf32"
    # Restored on exit.
    assert matmul.fp32_precision == "ieee"
    assert conv.fp32_precision == "ieee"


def test_vocoder_context_new_api_ieee(monkeypatch):
    monkeypatch.setenv(precision.ENV_VOCODER_FP32_PRECISION, "ieee")
    matmul, conv, _cudnn = _install_fake_backends(monkeypatch, new_api=True)
    # Pretend something left it on tf32 before us.
    matmul.fp32_precision = "tf32"
    conv.fp32_precision = "tf32"
    with precision.vocoder_precision_context():
        assert matmul.fp32_precision == "ieee"
        assert conv.fp32_precision == "ieee"
    # Restored to the pre-context value.
    assert matmul.fp32_precision == "tf32"
    assert conv.fp32_precision == "tf32"


def test_vocoder_context_fallback_to_allow_tf32(monkeypatch):
    """When the new fp32_precision API is absent, fall back to allow_tf32."""
    monkeypatch.setenv(precision.ENV_VOCODER_FP32_PRECISION, "tf32")
    matmul, _conv, cudnn = _install_fake_backends(monkeypatch, new_api=False)
    assert not hasattr(matmul, "fp32_precision")
    with precision.vocoder_precision_context():
        assert matmul.allow_tf32 is True
        assert cudnn.allow_tf32 is True
    # Restored.
    assert matmul.allow_tf32 is False
    assert cudnn.allow_tf32 is False


def test_vocoder_context_fallback_ieee(monkeypatch):
    monkeypatch.setenv(precision.ENV_VOCODER_FP32_PRECISION, "ieee")
    matmul, _conv, cudnn = _install_fake_backends(monkeypatch, new_api=False)
    matmul.allow_tf32 = True
    cudnn.allow_tf32 = True
    with precision.vocoder_precision_context():
        assert matmul.allow_tf32 is False
        assert cudnn.allow_tf32 is False
    assert matmul.allow_tf32 is True
    assert cudnn.allow_tf32 is True


def test_vocoder_context_restores_on_exception(monkeypatch):
    monkeypatch.setenv(precision.ENV_VOCODER_FP32_PRECISION, "tf32")
    matmul, conv, _cudnn = _install_fake_backends(monkeypatch, new_api=True)
    with pytest.raises(RuntimeError):
        with precision.vocoder_precision_context():
            assert matmul.fp32_precision == "tf32"
            raise RuntimeError("boom")
    # Still restored despite the exception.
    assert matmul.fp32_precision == "ieee"
    assert conv.fp32_precision == "ieee"


def test_log_precision_settings_does_not_crash(monkeypatch):
    monkeypatch.setenv(precision.ENV_MATMUL_PRECISION, "medium")
    monkeypatch.setenv(precision.ENV_VOCODER_FP32_PRECISION, "tf32")
    # Should log cleanly with no exception.
    precision.log_precision_settings()
    precision.log_precision_settings("highest")
