"""Parity for the opt-in GPU log-mel (MSTAR_GPU_MEL=1) vs HF's CPU
``WhisperFeatureExtractor``.

The native audio encoder is numerically parity-tested against HF, so the audio
features fed to it must stay identical regardless of where the mel spectrogram is
computed. This pins the GPU transform (mstar...qwen3_omni_model.gpu_log_mel) to HF's
output across clip lengths: same valid frame count and cos>=0.9999 / max-abs ~1e-5
(the production bf16 encoder tolerance is looser than this).

Requires CUDA + the Qwen3-Omni checkpoint (for the real mel filterbank); skips
otherwise. Point at a checkpoint with MSTAR_QWEN3_OMNI_DIR.
"""
import os
import numpy as np
import pytest
import torch


def _resolve_checkpoint():
    d = os.environ.get("MSTAR_QWEN3_OMNI_DIR")
    if d and os.path.isdir(d):
        return d
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download("Qwen/Qwen3-Omni-30B-A3B-Instruct",
                                 allow_patterns=["*.json", "*.txt"],
                                 local_files_only=True)
    except Exception:
        return None


CKPT = _resolve_checkpoint()
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(CKPT is None, reason="Qwen3-Omni checkpoint not available"),
]


@pytest.mark.parametrize("dur", [3.0, 7.3, 15.0, 30.0, 41.2], ids=lambda d: f"{d}s")
def test_gpu_mel_matches_hf(dur):
    from transformers import AutoProcessor
    from mstar.model.qwen3_omni.qwen3_omni_model import gpu_log_mel

    fe = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True).feature_extractor
    sr = fe.sampling_rate
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(int(dur * sr)) * 0.1).astype(np.float32)

    # HF reference (CPU numpy), cropped to valid frames exactly as the serving path does.
    hf = fe(audio, sampling_rate=sr, padding=True, truncation=False,
            return_attention_mask=True, return_tensors="pt")
    mask = hf["attention_mask"][0].bool()
    valid = int(mask.sum())
    ref = hf["input_features"][0][:, :valid].float()      # (n_mel, valid)

    # GPU transform under test.
    dev = torch.device("cuda")
    filters = torch.tensor(np.asarray(fe.mel_filters), dtype=torch.float32, device=dev)
    window = torch.hann_window(fe.n_fft, periodic=True, device=dev)
    wav = torch.tensor(audio, dtype=torch.float32, device=dev)
    out = gpu_log_mel(wav, filters, window, fe.n_fft, fe.hop_length).cpu()  # (n_mel, T)

    assert out.shape[1] == valid, f"frame count {out.shape[1]} != HF valid {valid}"
    a, b = out.flatten(), ref.flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    maxabs = (a - b).abs().max().item()
    assert cos > 0.9999 and maxabs < 1e-3, f"dur={dur}: cos={cos:.7f} maxabs={maxabs:.6f}"
