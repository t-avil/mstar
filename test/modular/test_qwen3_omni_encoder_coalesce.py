"""Cross-request encoder coalescing parity for Qwen3-Omni.

The scheduler-level coalescing window (``MSTAR_ENCODER_COALESCE`` in
``mstar/worker/micro_scheduler.py``) gathers several requests' audio/vision
encoder work into ONE ``forward_batched`` call. This test pins the correctness
contract that makes coalescing safe: dispatching N encoder requests through the
batched path (``preprocess`` + ``forward_batched``, exactly what the window
feeds the stateless engine) yields per-request outputs equal — within bf16
tolerance — to running each request through the single-request ``forward``.

If this holds, the coalescing window only changes *when/how many* requests share
a forward, never the per-request result.

Requires the real Qwen3-Omni checkpoint + CUDA + flash-attn (encoder weights
only, ~1-2 GB); skips automatically otherwise. Point at a checkpoint with
``MSTAR_QWEN3_OMNI_DIR`` or rely on the HF cache. Import-safe on CPU: every
heavy import is inside a test/fixture, so collection never needs CUDA.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, ".")

import numpy as np
import pytest
import torch

DEVICE = "cuda:0"
# bf16 batched-vs-single acceptance. The native encoders are varlen-packed, so
# batching is fp32-exact and ~0.9999 cosine in bf16 (see
# test_qwen3_omni_native_encoders.py). Pin the same bar here.
COS_MIN = 0.9999
RELL2_MAX = 0.05
N_REQS = 4


def _resolve_checkpoint() -> str | None:
    d = os.environ.get("MSTAR_QWEN3_OMNI_DIR")
    if d and os.path.isdir(d):
        return d
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            allow_patterns=["*.json", "*.safetensors", "*.txt"],
            local_files_only=True,
        )
    except Exception:
        return None


CKPT = _resolve_checkpoint()

try:
    import flash_attn  # noqa: F401
    _HAS_FA = True
except ImportError:
    _HAS_FA = False

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        CKPT is None,
        reason="Qwen3-Omni checkpoint not available (set MSTAR_QWEN3_OMNI_DIR)",
    ),
    pytest.mark.skipif(not _HAS_FA, reason="flash-attn required"),
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cmp(a, b):
    a, b = a.float(), b.float()
    rel = (a - b).norm() / b.norm().clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    return rel.item(), cos.item()


def _assert_close(a, b, what):
    assert a.shape == b.shape, f"{what}: shape {tuple(a.shape)} != {tuple(b.shape)}"
    rel, cos = _cmp(a, b)
    assert cos > COS_MIN and rel < RELL2_MAX, f"{what}: cos={cos:.5f} relL2={rel:.4f}"


def _engine_inputs(request_ids):
    from mstar.model.submodule_base import ModelInputsFromEngine
    return ModelInputsFromEngine(
        request_ids=list(request_ids),
        per_request_info={rid: None for rid in request_ids},
    )


@pytest.fixture(scope="module")
def cfg():
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(CKPT, trust_remote_code=True)


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)


def _load_native(native_cls, sub_cfg, prefix):
    from mstar.model.utils import ModuleAndPrefix, load_weights_from_hf_shards
    nat = native_cls(sub_cfg).to(DEVICE)
    load_weights_from_hf_shards(
        repo_dir=CKPT,
        modules=[ModuleAndPrefix(nat, prefix=prefix)],
        device=DEVICE,
    )
    return nat.eval()


def _audio_requests(processor, n):
    """n single-segment audio requests at different truncations (so the batch
    is heterogeneous, exercising the per-request varlen split)."""
    import soundfile as sf
    wav_path = os.path.join(os.path.dirname(__file__), "..", "qwen3-omni", "audio.wav")
    wav, sr0 = sf.read(wav_path)
    if wav.ndim > 1:
        wav = wav.mean(1)
    fe = processor.feature_extractor
    sr = getattr(fe, "sampling_rate", 16000)
    if sr0 != sr:
        m = int(len(wav) * sr / sr0)
        wav = np.interp(np.linspace(0, len(wav), m, endpoint=False), np.arange(len(wav)), wav)
    ao = fe(wav, sampling_rate=sr, padding=True, truncation=False,
            return_attention_mask=True, return_tensors="pt")
    feat = ao["input_features"].permute(0, 2, 1)[ao["attention_mask"].bool()].permute(1, 0)
    feat = feat.to(DEVICE).to(torch.bfloat16)            # (mel, T)
    total = feat.shape[1]
    reqs = []
    for i in range(n):
        frac = 1.0 - 0.15 * i
        t = max(64, int(total * frac))
        f = feat[:, :t].contiguous()
        lens = torch.tensor([t], device=DEVICE, dtype=torch.long)
        reqs.append((f, lens))
    return reqs


def _vision_requests(processor, n):
    """n single-image vision requests at different resolutions."""
    from PIL import Image
    img_path = os.path.join(os.path.dirname(__file__), "..", "bagel", "bagel.png")
    img = Image.open(img_path).convert("RGB")
    resolutions = [(448, 448), (672, 672), (336, 504), (504, 336)]
    reqs = []
    for i in range(n):
        h, w = resolutions[i % len(resolutions)]
        rimg = img.resize((w, h))
        vout = processor.image_processor(images=[np.array(rimg)], return_tensors="pt")
        g = vout["image_grid_thw"]
        if isinstance(g, list):
            g = torch.stack([torch.as_tensor(x) for x in g])
        pv = vout["pixel_values"].to(DEVICE).to(torch.bfloat16)
        reqs.append((pv, g.to(DEVICE)))
    return reqs


# --------------------------------------------------------------------------- #
# audio coalescing parity
# --------------------------------------------------------------------------- #
def test_audio_coalesce_parity(cfg, processor):
    from mstar.model.qwen3_omni.components.audio_encoder import NativeQwen3OmniAudioEncoder
    from mstar.model.qwen3_omni.submodules import NativeAudioEncoderSubmodule
    from mstar.model.submodule_base import NodeInputs

    enc = _load_native(NativeQwen3OmniAudioEncoder, cfg.thinker_config.audio_config,
                       "thinker.audio_tower")
    sub = NativeAudioEncoderSubmodule(enc, config=None)

    reqs = _audio_requests(processor, N_REQS)
    rids = [f"a{i}" for i in range(N_REQS)]
    node_inputs = [
        NodeInputs(tensor_inputs={"audio_features": f, "audio_seqlens": l})
        for f, l in reqs
    ]

    # batched dispatch (what the coalescing window feeds the stateless engine)
    batch_ei = _engine_inputs(rids)
    with torch.no_grad():
        # can_batch must accept this single-segment-per-request set
        from mstar.engine.base import NodeBatch
        nb = NodeBatch(node_name="audio_encoder", graph_walk="prefill_audio",
                       request_ids=rids, per_request_input_tensors={})
        assert sub.can_batch(nb, node_inputs), "single-segment reqs must be batchable"
        pre = sub.preprocess("prefill_audio", batch_ei, node_inputs)
        batched = sub.forward_batched("prefill_audio", batch_ei, **pre)

        # per-request single forwards
        for rid, (f, l) in zip(rids, reqs):
            single = sub.forward("prefill_audio", _engine_inputs([rid]),
                                 audio_features=f, audio_seqlens=l)
            _assert_close(batched[rid]["audio_embeds"][0],
                          single["audio_embeds"][0], f"audio.coalesce[{rid}]")


# --------------------------------------------------------------------------- #
# vision coalescing parity (embeds + every DeepStack level)
# --------------------------------------------------------------------------- #
def test_vision_coalesce_parity(cfg, processor):
    from mstar.model.qwen3_omni.components.vision_encoder import NativeQwen3OmniVisionEncoder
    from mstar.model.qwen3_omni.submodules import NativeVisionEncoderSubmodule
    from mstar.model.submodule_base import NodeInputs

    enc = _load_native(NativeQwen3OmniVisionEncoder, cfg.thinker_config.vision_config,
                       "thinker.visual")
    sub = NativeVisionEncoderSubmodule(enc, config=None)

    reqs = _vision_requests(processor, N_REQS)
    rids = [f"v{i}" for i in range(N_REQS)]
    node_inputs = [
        NodeInputs(tensor_inputs={"pixel_values": pv, "grid_thw": g})
        for pv, g in reqs
    ]

    batch_ei = _engine_inputs(rids)
    with torch.no_grad():
        from mstar.engine.base import NodeBatch
        nb = NodeBatch(node_name="vision_encoder", graph_walk="prefill_vision",
                       request_ids=rids, per_request_input_tensors={})
        assert sub.can_batch(nb, node_inputs)
        pre = sub.preprocess("prefill_vision", batch_ei, node_inputs)
        batched = sub.forward_batched("prefill_vision", batch_ei, **pre)

        for rid, (pv, g) in zip(rids, reqs):
            single = sub.forward("prefill_vision", _engine_inputs([rid]),
                                 pixel_values=pv, grid_thw=g)
            _assert_close(batched[rid]["vision_embeds"][0],
                          single["vision_embeds"][0], f"vision.coalesce[{rid}]")
            ds_b = batched[rid]["deepstack"]
            ds_s = single["deepstack"]
            assert len(ds_b) == len(ds_s), f"deepstack levels mismatch for {rid}"
            for i, (db, dsg) in enumerate(zip(ds_b, ds_s)):
                _assert_close(db, dsg, f"vision.coalesce.deepstack[{rid}][{i}]")
