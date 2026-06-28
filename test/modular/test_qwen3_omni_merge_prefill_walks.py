"""Parity for the merged Thinker prefill walks (``MSTAR_MERGE_PREFILL_WALKS``).

The merge optimization fuses the encoder + modality-prefill + text-prefill into
ONE Thinker walk so the encoder output flows straight into a single Thinker
prefill forward, eliminating the ~60 ms per-walk conductor round-trip between
the modality-prefill walk and the text-prefill walk (experiments B1 / B5 / B6).

Correctness invariant: a single merged Thinker forward is mathematically
identical to the default two-walk path *iff* its packed inputs — token/feature
embeddings, 3D-MRoPE position ids, and the Talker masks — are the concatenation
(in the original schedule/KV order) of what the two separate walks produce. Both
paths then run the same causal attention over the same KV layout, so the prefill
logits and first sampled token are identical. (Single-forward attention over a
contiguous causal sequence == sequential prefill that appends to the KV cache.)

This file pins that invariant at two levels, all CPU / tiny-shape (no GPU, no
30B weights):

  1. ``ThinkerSubmodule.prepare_inputs`` for ``prefill_audio_text`` /
     ``prefill_vision_text`` exactly equals ``cat(sub-walk prepare_inputs)`` with
     the 3D-MRoPE start position advanced across sub-walks as the per-walk cache
     state would advance it — proving embeds/positions/masks (and the merged
     full-length deepstack) are byte-identical to the two-walk path.
  2. The conductor-side plumbing: schedule collapse, talker prefill-step count,
     walk-graph registration, and input→node routing are all correct and remain
     byte-identical to the baseline when the flag is OFF.
"""
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
from mstar.model.qwen3_omni.submodules import ThinkerSubmodule
from mstar.engine.kv_store import PositionInfo


# --------------------------------------------------------------------------- #
# Tiny CPU stub of the Thinker submodule: prepare_inputs only needs
# ``embed_tokens`` + config + get_device (no transformer, no GPU).
# --------------------------------------------------------------------------- #
EMBED_DIM = 16
VOCAB = 256


class _StubThinker(ThinkerSubmodule):
    def __init__(self, config):
        nn.Module.__init__(self)
        embed = nn.Embedding(VOCAB, EMBED_DIM)
        torch.manual_seed(0)
        with torch.no_grad():
            embed.weight.normal_()
        self.model = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed))
        self.config = config
        self._inv_freq = None
        self._decode_thinker_mask = None
        self._audio_bos_embed = None
        self._audio_eos_embed = None
        self._vision_bos_embed = None
        self._vision_eos_embed = None

    def get_device(self):
        return torch.device("cpu")


def _config():
    cfg = Qwen3OmniModelConfig()
    # Shrink the sentinel/special token ids into the tiny test vocab so
    # ``embed_tokens`` doesn't need a 150k-row table.
    cfg.thinker.audio_start_token_id = 1
    cfg.thinker.audio_end_token_id = 2
    cfg.thinker.vision_start_token_id = 3
    cfg.thinker.vision_end_token_id = 4
    # Keep im_start/system/assistant out of the synthetic text so the talker
    # text-mask is the simple all-ones case (mask concat parity is what we test;
    # actual mask content is exercised separately by the model parity tests).
    cfg.im_start_token_id = 250
    cfg.system_token_id = 251
    cfg.assistant_token_id = 252
    return cfg


def _fwd_info(merged_sub_walks):
    return SimpleNamespace(
        step_metadata={"merged_sub_walks": merged_sub_walks},
    )


def _seen():
    return SimpleNamespace(add_tokens=lambda *_a, **_k: None)


def _assert_arnode_equal(merged, parts):
    """merged == concatenation of the per-sub-walk ARNodeInputs in ``parts``."""
    exp_embeds = torch.cat([p.input_embeds for p in parts], dim=0)
    exp_pos = torch.cat([p.custom_pos_ids for p in parts], dim=1)
    exp_mask = torch.cat(
        [p.tensor_inputs["masks_for_talker"] for p in parts], dim=1
    )
    exp_len = sum(p.input_seq_len for p in parts)

    assert merged.input_seq_len == exp_len
    assert merged.input_embeds.shape == exp_embeds.shape
    torch.testing.assert_close(merged.input_embeds, exp_embeds, rtol=0, atol=0)
    torch.testing.assert_close(merged.custom_pos_ids, exp_pos, rtol=0, atol=0)
    torch.testing.assert_close(
        merged.tensor_inputs["masks_for_talker"], exp_mask, rtol=0, atol=0
    )


# --------------------------------------------------------------------------- #
# 1. prepare_inputs parity: merged == cat(sub-walks), positions chained.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text_len,audio_len", [(7, 5), (1, 1), (20, 50)])
def test_merged_audio_text_prepare_inputs_parity(text_len, audio_len):
    cfg = _config()
    sub = _StubThinker(cfg)

    torch.manual_seed(text_len * 100 + audio_len)
    text_ids = torch.randint(10, 240, (text_len,), dtype=torch.long)
    audio_embeds = torch.randn(audio_len, EMBED_DIM)
    inputs = {"text_inputs": [text_ids], "audio_embeds": [audio_embeds]}

    # Default order for input_modalities == [audio, text]: audio prefilled
    # first (KV positions 0..A), text second.
    sub_order = ["prefill_audio", "prefill_text"]

    # Reference: run the two sub-walks, threading start_pos exactly as the
    # per-walk cache state advances (audio advances by its seq_len = A+2).
    audio_ref = sub.prepare_inputs(
        "prefill_audio", _fwd_info(None), inputs, _seen(),
        pos_info={"main": PositionInfo(position_id_start=0)},
    )
    text_ref = sub.prepare_inputs(
        "prefill_text", _fwd_info(None), inputs, _seen(),
        pos_info={"main": PositionInfo(position_id_start=audio_ref.input_seq_len)},
    )

    merged = sub.prepare_inputs(
        "prefill_audio_text", _fwd_info(sub_order), inputs, _seen(),
        pos_info={"main": PositionInfo(position_id_start=0)},
    )
    _assert_arnode_equal(merged, [audio_ref, text_ref])


def test_merged_audio_text_text_first_order():
    """If the schedule order is [text, audio], the merge must honour it."""
    cfg = _config()
    sub = _StubThinker(cfg)
    torch.manual_seed(7)
    text_ids = torch.randint(10, 240, (9,), dtype=torch.long)
    audio_embeds = torch.randn(11, EMBED_DIM)
    inputs = {"text_inputs": [text_ids], "audio_embeds": [audio_embeds]}

    text_ref = sub.prepare_inputs(
        "prefill_text", _fwd_info(None), inputs, _seen(),
        pos_info={"main": PositionInfo(position_id_start=0)},
    )
    audio_ref = sub.prepare_inputs(
        "prefill_audio", _fwd_info(None), inputs, _seen(),
        pos_info={"main": PositionInfo(position_id_start=text_ref.input_seq_len)},
    )
    merged = sub.prepare_inputs(
        "prefill_audio_text", _fwd_info(["prefill_text", "prefill_audio"]),
        inputs, _seen(), pos_info={"main": PositionInfo(position_id_start=0)},
    )
    _assert_arnode_equal(merged, [text_ref, audio_ref])


def test_merged_vision_text_prepare_inputs_parity():
    cfg = _config()
    sub = _StubThinker(cfg)
    torch.manual_seed(3)

    # grid_thw [1,4,4] -> merged vision tokens = (1*4*4) / spatial_merge_size^2.
    grid_thw = torch.tensor([1, 4, 4], dtype=torch.long)
    merge_sq = cfg.vision.spatial_merge_size ** 2
    vision_len = int((grid_thw.prod()) // merge_sq)
    vision_embeds = torch.randn(vision_len, EMBED_DIM)
    num_deepstack = len(cfg.vision.deepstack_visual_indexes)
    deepstack = [torch.randn(vision_len, EMBED_DIM) for _ in range(num_deepstack)]
    text_ids = torch.randint(10, 240, (6,), dtype=torch.long)
    inputs = {
        "text_inputs": [text_ids],
        "vision_embeds": [vision_embeds],
        "deepstack": deepstack,
        "image_grid_thw": [grid_thw],
    }

    # Default order [image, text]: vision first.
    vision_ref = sub.prepare_inputs(
        "prefill_vision", _fwd_info(None), inputs, _seen(),
        pos_info={"main": PositionInfo(position_id_start=0)},
    )
    vis_advance = vision_ref.tensor_inputs["mrope_pos_advance"]
    text_ref = sub.prepare_inputs(
        "prefill_text", _fwd_info(None), inputs, _seen(),
        pos_info={"main": PositionInfo(position_id_start=vis_advance)},
    )
    merged = sub.prepare_inputs(
        "prefill_vision_text", _fwd_info(["prefill_vision", "prefill_text"]),
        inputs, _seen(), pos_info={"main": PositionInfo(position_id_start=0)},
    )
    _assert_arnode_equal(merged, [vision_ref, text_ref])

    # Merged deepstack: full merged length, vision slot == the per-walk vision
    # deepstack, zeros over the text span (so _deepstack_process adds nothing
    # to text positions).
    total_len = merged.input_seq_len
    merged_ds = merged.tensor_inputs["deepstack"]
    assert len(merged_ds) == num_deepstack
    vis_seg = vision_ref.input_seq_len
    for i, fd in enumerate(merged_ds):
        assert fd.shape == (total_len, EMBED_DIM)
        # vision is first -> slot [0:vis_seg]
        torch.testing.assert_close(
            fd[:vis_seg], vision_ref.tensor_inputs["deepstack"][i], rtol=0, atol=0
        )
        assert torch.count_nonzero(fd[vis_seg:]) == 0
    # Combined MRoPE advance == vision span + text span.
    assert merged.tensor_inputs["mrope_pos_advance"] == vis_advance + text_ref.input_seq_len


# --------------------------------------------------------------------------- #
# 2. Conductor-side plumbing (pure python, no torch model).
# --------------------------------------------------------------------------- #
def _model():
    """A Qwen3OmniModel shell with just enough state to call the schedule /
    walk-graph helpers (no weights loaded)."""
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
    m = Qwen3OmniModel.__new__(Qwen3OmniModel)
    m.config = _config()
    return m


def test_merged_walks_registered():
    walks = _model().get_graph_walk_graphs()
    assert "prefill_audio_text" in walks
    assert "prefill_vision_text" in walks
    # Each merged walk = encoder -> single Thinker node (one conductor dispatch).
    for name, enc in [("prefill_audio_text", "audio_encoder"),
                      ("prefill_vision_text", "vision_encoder")]:
        nodes = walks[name].get_nodes()
        assert set(nodes) == {enc, "Thinker"}
    # Thinker partition advertises the merged walks.
    thinker = next(p for p in _model().get_partitions() if p.name == "Thinker")
    assert {"prefill_audio_text", "prefill_vision_text"} <= thinker.graph_walks


def test_schedule_collapse_flag_on(monkeypatch):
    monkeypatch.setenv("MSTAR_MERGE_PREFILL_WALKS", "1")
    m = _model()
    # Two-walk schedule (audio first, text second) collapses to one merged walk.
    schedule = [("prefill_audio", {"audio_features": "AF"}),
                ("prefill_text", {"text_inputs": "TI"})]
    merged = m._try_merge_schedule(schedule)
    assert merged is not None
    name, tdict, order = merged
    assert name == "prefill_audio_text"
    assert order == ["prefill_audio", "prefill_text"]
    assert tdict == {"audio_features": "AF", "text_inputs": "TI"}

    # Vision.
    name2, _, _ = m._try_merge_schedule(
        [("prefill_vision", {"pixel_values": "PV"}),
         ("prefill_text", {"text_inputs": "TI"})]
    )
    assert name2 == "prefill_vision_text"


def test_schedule_not_merged_when_unsupported():
    m = _model()
    # Single walk (text only) -> nothing to merge.
    assert m._try_merge_schedule([("prefill_text", {})]) is None
    # Three walks (audio + vision + text) -> unsupported, fall back.
    assert m._try_merge_schedule([
        ("prefill_audio", {}), ("prefill_vision", {}), ("prefill_text", {})
    ]) is None


def test_num_thinker_prefill_steps(monkeypatch):
    m = _model()
    monkeypatch.delenv("MSTAR_MERGE_PREFILL_WALKS", raising=False)
    # Flag OFF: one step per modality (baseline).
    assert m._num_thinker_prefill_steps(["audio", "text"]) == 2
    assert m._num_thinker_prefill_steps(["image", "text"]) == 2

    monkeypatch.setenv("MSTAR_MERGE_PREFILL_WALKS", "1")
    # Flag ON: supported text+single-modality collapses to one talker chunk.
    assert m._num_thinker_prefill_steps(["audio", "text"]) == 1
    assert m._num_thinker_prefill_steps(["image", "text"]) == 1
    # Text-only or multi-modality keep the default count.
    assert m._num_thinker_prefill_steps(["text"]) == 1
    assert m._num_thinker_prefill_steps(["audio", "image", "text"]) == 3


def test_default_off_is_byte_identical(monkeypatch):
    monkeypatch.delenv("MSTAR_MERGE_PREFILL_WALKS", raising=False)
    m = _model()
    assert m._merge_prefill_walks_enabled() is False
    # With the flag off, a supported 2-walk schedule is NOT collapsed by the
    # initial-args path (the merge helper is only invoked when enabled). The
    # helper itself is order-preserving and side-effect-free, so default
    # scheduling/behaviour is unchanged.
    assert m._num_thinker_prefill_steps(["audio", "text"]) == 2


def test_merged_input_routing():
    """text -> Thinker, audio features -> encoder, image grid -> both."""
    m = _model()
    meta = SimpleNamespace(kwargs={
        "prefill_schedule": [(
            "prefill_audio_text",
            {"text_inputs": "TI", "audio_features": "AF", "audio_seqlens": "AS"},
        )],
        "prefill_step": 0,
    })
    edges = m._get_thinker_prefill_inputs(meta, {})
    routed = {(e.name, e.next_node) for e in edges}
    assert ("text_inputs", "Thinker") in routed
    assert ("audio_features", "audio_encoder") in routed
    assert ("audio_seqlens", "audio_encoder") in routed
    assert all(e.next_node != "vision_encoder" for e in edges)

    # Vision: image_grid_thw fans out to encoder AND Thinker; the Thinker also
    # gets the video_second_per_grid placeholder even when absent.
    meta_v = SimpleNamespace(kwargs={
        "prefill_schedule": [(
            "prefill_vision_text",
            {"text_inputs": "TI", "pixel_values": "PV", "image_grid_thw": "GT"},
        )],
        "prefill_step": 0,
    })
    edges_v = m._get_thinker_prefill_inputs(meta_v, {})
    routed_v = {(e.name, e.next_node) for e in edges_v}
    assert ("pixel_values", "vision_encoder") in routed_v
    assert ("image_grid_thw", "vision_encoder") in routed_v
    assert ("image_grid_thw", "Thinker") in routed_v
    assert ("text_inputs", "Thinker") in routed_v
    assert ("video_second_per_grid", "Thinker") in routed_v
