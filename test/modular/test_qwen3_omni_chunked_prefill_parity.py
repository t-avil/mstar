"""Parity tests for resumable chunked Thinker prefill (MSTAR_CHUNKED_PREFILL).

Resumable chunked prefill splits a long audio/vision/text Thinker prefill into
token-budgeted chunks across scheduler steps so one long prefill does not
monopolize a step and stall other requests' first tokens (TTFT at batch). The
M-RoPE handling follows vLLM's recipe: precompute the full 3D position tensor
once, then index ``[:, computed:computed + chunk]`` per chunk -- no per-chunk
grid math.

State of the implementation (see DESIGN_chunked_prefill.md):
  * Model side (M-RoPE precompute + slice, chunk-capable prepare_inputs) is
    implemented; KV append is already resumable in the cache manager.
  * Scheduler/conductor re-enqueue is STUBBED behind the flag.

These tests therefore pin the load-bearing CORRECTNESS PROPERTIES that make
chunked prefill equivalent to single-shot, all of which are checkable on CPU:

  1. Slicing the precomputed 3D positions and concatenating reconstructs the
     full tensor bit-exactly (any chunk boundary).
  2. The per-token RoPE cos/sin computed on a sliced chunk equals the
     single-shot cos/sin for those tokens -- so every token receives identical
     rotary embeddings regardless of chunking. This is the heart of KV / first-
     token-logit parity: identical Q/K rotation -> identical KV writes ->
     identical attention -> identical logits.
  3. The unified per-request position advance equals the existing single-shot
     advance for text / audio.
  4. ``ThinkerSubmodule._maybe_chunk_prefill`` is byte-identical to single-shot
     when the flag is OFF and when the whole span fits in one chunk (flag ON).

A CUDA-gated test additionally checks the slicing path on-device and documents
the full engine-level KV/logit parity check (requires real weights + GPU).
"""
import os
from types import SimpleNamespace

import pytest
import torch

from mstar.model.qwen3_omni.components.rope import (
    compute_3d_cos_sin,
    compute_rope_freqs,
    get_rope_index_audio,
    get_rope_index_text,
    get_rope_index_vision,
    prefill_mrope_pos_advance,
    slice_mrope_positions,
)
from mstar.model.qwen3_omni.qwen3_omni_model import (
    DEFAULT_LONG_PREFILL_TOKEN_THRESHOLD,
    chunked_prefill_enabled,
    long_prefill_token_threshold,
)
from mstar.model.qwen3_omni.submodules import ThinkerSubmodule
from mstar.model.submodule_base import ARNodeInputs

HEAD_DIM = 128
MROPE_SECTION = [24, 20, 20]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_full_pos_ids(walk: str, start_pos: float = 7.0) -> torch.Tensor:
    """Build a representative full-span 3D position tensor for each walk."""
    if walk == "prefill_text":
        return get_rope_index_text(37, start_pos)
    if walk == "prefill_audio":
        audio_len = 50
        start = get_rope_index_text(1, start_pos)
        audio = get_rope_index_audio(audio_len, start_pos + 1)
        end = get_rope_index_text(1, start_pos + 1 + audio_len)
        return torch.cat([start, audio, end], dim=1)
    if walk == "prefill_vision":
        grid = torch.tensor([[1, 8, 12]], dtype=torch.long)  # T,H,W
        start = get_rope_index_text(1, start_pos)
        vision = get_rope_index_vision(
            grid, start_pos + 1, position_id_per_seconds=25.0,
            spatial_merge_size=2,
        )
        end_base = float(vision.max().item()) + 1
        end = get_rope_index_text(1, end_base)
        return torch.cat([start, vision, end], dim=1)
    raise ValueError(walk)


def _make_node_inputs(seq_len: int, pos_ids: torch.Tensor) -> ARNodeInputs:
    embeds = torch.randn(seq_len, 16)
    masks = torch.stack([
        torch.zeros(seq_len, dtype=torch.bool),
        torch.ones(seq_len, dtype=torch.bool),
    ])
    return ARNodeInputs(
        input_seq_len=seq_len,
        input_embeds=embeds,
        custom_pos_ids=pos_ids,
        tensor_inputs={"masks_for_talker": masks},
    )


# ---------------------------------------------------------------------------
# Env flag plumbing
# ---------------------------------------------------------------------------
def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("MSTAR_CHUNKED_PREFILL", raising=False)
    assert chunked_prefill_enabled() is False


def test_flag_on(monkeypatch):
    for v in ("1", "true", "YES", "on"):
        monkeypatch.setenv("MSTAR_CHUNKED_PREFILL", v)
        assert chunked_prefill_enabled() is True


def test_threshold_default(monkeypatch):
    monkeypatch.delenv("MSTAR_LONG_PREFILL_TOKEN_THRESHOLD", raising=False)
    assert long_prefill_token_threshold() == DEFAULT_LONG_PREFILL_TOKEN_THRESHOLD


def test_threshold_override(monkeypatch):
    monkeypatch.setenv("MSTAR_LONG_PREFILL_TOKEN_THRESHOLD", "256")
    assert long_prefill_token_threshold() == 256
    # Invalid / non-positive falls back to the default.
    monkeypatch.setenv("MSTAR_LONG_PREFILL_TOKEN_THRESHOLD", "0")
    assert long_prefill_token_threshold() == DEFAULT_LONG_PREFILL_TOKEN_THRESHOLD
    monkeypatch.setenv("MSTAR_LONG_PREFILL_TOKEN_THRESHOLD", "garbage")
    assert long_prefill_token_threshold() == DEFAULT_LONG_PREFILL_TOKEN_THRESHOLD


# ---------------------------------------------------------------------------
# Property 1: slicing reconstructs the full position tensor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("walk", ["prefill_text", "prefill_audio", "prefill_vision"])
def test_slice_reconstructs_full(walk):
    full = _build_full_pos_ids(walk)
    seq_len = full.shape[1]
    for split in (1, seq_len // 3, seq_len // 2, seq_len - 1):
        a = slice_mrope_positions(full, 0, split)
        b = slice_mrope_positions(full, split, seq_len - split)
        recon = torch.cat([a, b], dim=1)
        assert torch.equal(recon, full), f"{walk} split={split}"


def test_slice_bounds_validation():
    full = _build_full_pos_ids("prefill_text")
    with pytest.raises(ValueError):
        slice_mrope_positions(full, 0, full.shape[1] + 1)
    with pytest.raises(ValueError):
        slice_mrope_positions(full.unsqueeze(0), 0, 1)  # wrong rank


# ---------------------------------------------------------------------------
# Property 2: per-chunk cos/sin == single-shot cos/sin (KV parity heart)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("walk", ["prefill_text", "prefill_audio", "prefill_vision"])
def test_cos_sin_chunk_parity(walk):
    full = _build_full_pos_ids(walk)
    seq_len = full.shape[1]
    inv_freq = compute_rope_freqs(HEAD_DIM, rope_theta=1_000_000.0)
    cos_full, sin_full = compute_3d_cos_sin(full, inv_freq, MROPE_SECTION)

    split = seq_len // 2
    cos_parts, sin_parts = [], []
    for off, ln in ((0, split), (split, seq_len - split)):
        chunk_pos = slice_mrope_positions(full, off, ln)
        c, s = compute_3d_cos_sin(chunk_pos, inv_freq, MROPE_SECTION)
        cos_parts.append(c)
        sin_parts.append(s)
    cos_cat = torch.cat(cos_parts, dim=0)
    sin_cat = torch.cat(sin_parts, dim=0)
    # RoPE is position-wise, so this is exact.
    assert torch.equal(cos_cat, cos_full), f"{walk} cos"
    assert torch.equal(sin_cat, sin_full), f"{walk} sin"


# ---------------------------------------------------------------------------
# Property 3: unified advance == existing single-shot advance
# ---------------------------------------------------------------------------
def test_pos_advance_text_audio_equals_seq_len():
    start = 7.0
    text = _build_full_pos_ids("prefill_text", start)
    assert prefill_mrope_pos_advance(text, start) == text.shape[1]
    audio = _build_full_pos_ids("prefill_audio", start)
    assert prefill_mrope_pos_advance(audio, start) == audio.shape[1]


def test_pos_advance_vision_matches_grid_span():
    start = 7.0
    grid = torch.tensor([[1, 8, 12]], dtype=torch.long)
    vision = get_rope_index_vision(
        grid, start + 1, position_id_per_seconds=25.0, spatial_merge_size=2,
    )
    end_base = float(vision.max().item()) + 1
    full = _build_full_pos_ids("prefill_vision", start)
    # Mirror ThinkerSubmodule.prepare_inputs prefill_vision:
    expected = int(end_base + 1 - start)
    assert prefill_mrope_pos_advance(full, start) == expected


# ---------------------------------------------------------------------------
# Property 4: _maybe_chunk_prefill parity + slicing
# ---------------------------------------------------------------------------
def test_maybe_chunk_prefill_flag_off_identity(monkeypatch):
    monkeypatch.delenv("MSTAR_CHUNKED_PREFILL", raising=False)
    full = _build_full_pos_ids("prefill_text")
    ni = _make_node_inputs(full.shape[1], full)
    fwd = SimpleNamespace(step_metadata={})
    # _maybe_chunk_prefill does not use `self`; call unbound with a dummy.
    out = ThinkerSubmodule._maybe_chunk_prefill(object(), "prefill_text", fwd, ni)
    assert out is ni  # default OFF -> untouched single-shot input


def test_maybe_chunk_prefill_single_chunk_identity(monkeypatch):
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL", "1")
    full = _build_full_pos_ids("prefill_audio")
    ni = _make_node_inputs(full.shape[1], full)
    fwd = SimpleNamespace(step_metadata={})  # no chunk bounds -> single chunk
    out = ThinkerSubmodule._maybe_chunk_prefill(object(), "prefill_audio", fwd, ni)
    assert out is ni  # flag ON but whole span in one chunk -> byte-identical


def test_maybe_chunk_prefill_slices_and_reconstructs(monkeypatch):
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL", "1")
    full = _build_full_pos_ids("prefill_text")
    seq_len = full.shape[1]
    ni = _make_node_inputs(seq_len, full)
    split = seq_len // 2

    out_a = ThinkerSubmodule._maybe_chunk_prefill(
        object(), "prefill_text",
        SimpleNamespace(step_metadata={
            "prefill_chunk_offset": 0, "prefill_chunk_len": split}),
        ni,
    )
    out_b = ThinkerSubmodule._maybe_chunk_prefill(
        object(), "prefill_text",
        SimpleNamespace(step_metadata={
            "prefill_chunk_offset": split, "prefill_chunk_len": seq_len - split}),
        ni,
    )
    assert out_a.input_seq_len == split
    assert out_b.input_seq_len == seq_len - split
    # Concatenating the two chunks reconstructs the single-shot tensors exactly.
    assert torch.equal(
        torch.cat([out_a.input_embeds, out_b.input_embeds], dim=0),
        ni.input_embeds,
    )
    assert torch.equal(
        torch.cat([out_a.custom_pos_ids, out_b.custom_pos_ids], dim=1),
        ni.custom_pos_ids,
    )
    assert torch.equal(
        torch.cat([
            out_a.tensor_inputs["masks_for_talker"],
            out_b.tensor_inputs["masks_for_talker"],
        ], dim=1),
        ni.tensor_inputs["masks_for_talker"],
    )


def test_maybe_chunk_prefill_vision_not_implemented(monkeypatch):
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL", "1")
    full = _build_full_pos_ids("prefill_vision")
    seq_len = full.shape[1]
    ni = _make_node_inputs(seq_len, full)
    fwd = SimpleNamespace(step_metadata={
        "prefill_chunk_offset": 0, "prefill_chunk_len": seq_len // 2})
    with pytest.raises(NotImplementedError):
        ThinkerSubmodule._maybe_chunk_prefill(object(), "prefill_vision", fwd, ni)


# ---------------------------------------------------------------------------
# CUDA-gated: on-device slicing + full engine parity placeholder
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chunk_slicing_on_device():
    dev = torch.device("cuda")
    full = _build_full_pos_ids("prefill_audio").to(dev)
    inv_freq = compute_rope_freqs(HEAD_DIM, device=dev)
    cos_full, sin_full = compute_3d_cos_sin(full, inv_freq, MROPE_SECTION)
    split = full.shape[1] // 2
    parts_c, parts_s = [], []
    for off, ln in ((0, split), (split, full.shape[1] - split)):
        cp = slice_mrope_positions(full, off, ln)
        c, s = compute_3d_cos_sin(cp, inv_freq, MROPE_SECTION)
        parts_c.append(c)
        parts_s.append(s)
    assert torch.equal(torch.cat(parts_c, dim=0), cos_full)
    assert torch.equal(torch.cat(parts_s, dim=0), sin_full)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not os.environ.get("MSTAR_QWEN3_OMNI_DIR"),
    reason="needs CUDA + Qwen3-Omni checkpoint (MSTAR_QWEN3_OMNI_DIR)",
)
def test_full_engine_chunk_vs_singleshot_parity():
    """Full engine-level parity: 2-chunk prefill KV + first-token logits ==
    single-shot, within tolerance.

    This requires the scheduler/conductor re-enqueue path that is currently
    STUBBED (see DESIGN_chunked_prefill.md), so it is skipped pending that
    work. The driver, once the stub is implemented, must:
      1. Run a single-shot prefill of a >threshold span; capture the paged KV
         pages and the sampled first-token logits.
      2. Reset the request, set MSTAR_CHUNKED_PREFILL=1 with a threshold that
         forces >=2 chunks, run the same span; capture KV + first-token logits.
      3. Assert KV tensors match (exact in fp32, atol~1e-2 in bf16) and the
         argmax first token is identical (logits allclose within bf16 tol).
    """
    pytest.skip(
        "scheduler/conductor chunked re-enqueue is stubbed; see "
        "DESIGN_chunked_prefill.md and MicroScheduler."
        "_maybe_reenqueue_prefill_remainder"
    )
