"""CPU tests for merged multimodal prefill (MSTAR_MERGED_PREFILL, plan B1/B5).

Exercises the conductor-side schedule collapse + input routing and the
submodule-side span concatenation WITHOUT model weights or a GPU:

  * ``_maybe_merge_prefill_schedule`` collapses exactly one text + one vision
    walk (either order) into a single ``prefill_multimodal`` entry, and is a
    byte-identical no-op (returns the schedule unchanged, order=None) whenever
    the flag is off / output is audio / vision is chunked / the schedule is not
    the exact one-text+one-vision shape.
  * ``_get_thinker_prefill_inputs`` routes the merged walk's inputs: encoder
    gets pixel_values + image_grid_thw; Thinker gets text_inputs + image_grid_thw
    + video_second_per_grid (always emitted, empty payload when absent).
  * ``_get_thinker_forward`` runs the single merged walk then transitions to
    thinker_decode with is_last_prefill on the merged step.
  * ``_build_merged_multimodal_inputs`` threads the MRoPE start position across
    spans EXACTLY as the separate walks would (text advances by seq_len, vision
    by its 3D-grid mrope_pos_advance) and concatenates in modality order — the
    hard invariant, checked here with stubbed span builders (real numerics are a
    GPU parity check, see SMOKE.md).
"""
import pytest

torch = pytest.importorskip("torch")

from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.graph.base import TensorPointerInfo
from mstar.model.qwen3_omni import qwen3_omni_model as qm
from mstar.model.qwen3_omni import submodules as sm
from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
from mstar.model.qwen3_omni.submodules import ThinkerSubmodule, _VisionPrefillStage


# --- conductor-side shim: borrow just the schedule/routing/state methods -----
class _CondShim:
    _maybe_merge_prefill_schedule = Qwen3OmniModel._maybe_merge_prefill_schedule
    _get_thinker_prefill_inputs = Qwen3OmniModel._get_thinker_prefill_inputs
    _get_thinker_forward = Qwen3OmniModel._get_thinker_forward
    _chunk_bounds = Qwen3OmniModel._chunk_bounds
    _text_chunk_bounds = Qwen3OmniModel._text_chunk_bounds
    _vision_chunk_bounds = Qwen3OmniModel._vision_chunk_bounds
    _assert_walk_span = Qwen3OmniModel._assert_walk_span
    _walk_span_tokens = Qwen3OmniModel._walk_span_tokens
    _log_first_chunk = Qwen3OmniModel._log_first_chunk


def _tpi(uuid: str, n: int = 4) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[n], dtype="int64", nbytes=n * 8, address=0,
        stride=[1], uuid=uuid, source_session_id="s", source_entity="w",
    )


TEXT = ("prefill_text", {"text_inputs": _tpi("txt")})
VISION = ("prefill_vision", {
    "pixel_values": _tpi("pix"),
    "image_grid_thw": _tpi("grid"),
})


@pytest.fixture(autouse=True)
def _clear_flags(monkeypatch):
    for f in ("MSTAR_MERGED_PREFILL", "MSTAR_CHUNKED_PREFILL_V2_VISION",
              "MSTAR_CHUNKED_PREFILL_V2"):
        monkeypatch.delenv(f, raising=False)


# --- _maybe_merge_prefill_schedule -------------------------------------------
def test_merge_off_is_noop():
    shim = _CondShim()
    sched = [TEXT, VISION]
    out, order = shim._maybe_merge_prefill_schedule(sched, audio_output=False)
    assert out is sched and order is None  # unchanged object, no merge


def test_merge_text_then_vision(monkeypatch):
    monkeypatch.setenv("MSTAR_MERGED_PREFILL", "1")
    shim = _CondShim()
    out, order = shim._maybe_merge_prefill_schedule([TEXT, VISION], audio_output=False)
    assert len(out) == 1
    assert out[0][0] == "prefill_multimodal"
    assert order is False  # text first
    # merged entry is the union of both entries' tensor dicts
    assert set(out[0][1]) == {"text_inputs", "pixel_values", "image_grid_thw"}


def test_merge_vision_then_text(monkeypatch):
    monkeypatch.setenv("MSTAR_MERGED_PREFILL", "1")
    shim = _CondShim()
    out, order = shim._maybe_merge_prefill_schedule([VISION, TEXT], audio_output=False)
    assert len(out) == 1 and out[0][0] == "prefill_multimodal"
    assert order is True  # vision first


def test_merge_gated_off_by_audio_output(monkeypatch):
    monkeypatch.setenv("MSTAR_MERGED_PREFILL", "1")
    shim = _CondShim()
    out, order = shim._maybe_merge_prefill_schedule([TEXT, VISION], audio_output=True)
    assert order is None and len(out) == 2


def test_merge_gated_off_by_chunked_vision(monkeypatch):
    monkeypatch.setenv("MSTAR_MERGED_PREFILL", "1")
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL_V2_VISION", "1")
    shim = _CondShim()
    out, order = shim._maybe_merge_prefill_schedule([TEXT, VISION], audio_output=False)
    assert order is None and len(out) == 2


@pytest.mark.parametrize("sched", [
    [TEXT],                                   # text only
    [VISION],                                 # vision only
    [TEXT, VISION, TEXT],                     # extra text span
    [TEXT, ("prefill_audio", {"audio_features": _tpi("a")})],  # audio present
    [TEXT, TEXT],                             # two text, no vision
])
def test_merge_only_exact_text_plus_vision(monkeypatch, sched):
    monkeypatch.setenv("MSTAR_MERGED_PREFILL", "1")
    shim = _CondShim()
    out, order = shim._maybe_merge_prefill_schedule(sched, audio_output=False)
    assert order is None and out is sched


# --- _get_thinker_prefill_inputs routing for the merged walk -----------------
def _merged_meta(vision_first: bool):
    entry = {**VISION[1], **TEXT[1]}
    return CurrentForwardConductorMetadata(
        graph_walk="prefill_multimodal",
        is_prefill=True,
        kwargs={
            "prefill_schedule": [("prefill_multimodal", entry)],
            "prefill_step": 0,
            "audio_output": False,
            "prefill_chunk_offset": 0,
        },
    )


def test_merged_input_routing():
    shim = _CondShim()
    edges = shim._get_thinker_prefill_inputs(_merged_meta(False), {})
    by_dest = {}
    for e in edges:
        by_dest.setdefault(e.next_node, {})[e.name] = e
    # encoder gets pixel_values + image_grid_thw with real payloads
    assert set(by_dest["vision_encoder"]) == {"pixel_values", "image_grid_thw"}
    assert by_dest["vision_encoder"]["pixel_values"].tensor_info
    # Thinker gets text_inputs + image_grid_thw + video_second_per_grid; the
    # absent video_second_per_grid is emitted with an EMPTY payload (still marks
    # the declared input ready).
    assert set(by_dest["Thinker"]) == {
        "text_inputs", "image_grid_thw", "video_second_per_grid",
    }
    assert by_dest["Thinker"]["text_inputs"].tensor_info
    assert by_dest["Thinker"]["video_second_per_grid"].tensor_info == []


# --- state machine: merged walk -> decode ------------------------------------
def test_merged_walk_transitions_to_decode():
    shim = _CondShim()
    meta = _merged_meta(False)
    fwd = shim._get_thinker_forward(meta, {})  # completing the merged step
    meta = fwd.full_metadata
    meta.kwargs.update(fwd.step_metadata)
    assert meta.is_prefill is False
    assert meta.graph_walk == "thinker_decode"


# --- submodule: MRoPE position threading across spans ------------------------
NUM_DEEPSTACK = 2
HIDDEN = 4
VLEN = 5           # vision total_len (V + 2 sentinels)
VADV = 100         # vision 3D-grid mrope_pos_advance (> VLEN, like a real grid)
TLEN = 4           # text span length


class _StubModelInner:
    def embed_tokens(self, text_ids):
        return torch.zeros((text_ids.shape[0], HIDDEN))


class _StubModel:
    def __init__(self):
        self.model = _StubModelInner()


class _StubVision:
    deepstack_visual_indexes = [0, 1]


class _StubConfig:
    thinker_hidden_size = HIDDEN
    vision = _StubVision()


class _Recorder:
    def __init__(self):
        self.added = None

    def add_tokens(self, ids):
        self.added = ids


class _ThinkerShim:
    _build_merged_multimodal_inputs = ThinkerSubmodule._build_merged_multimodal_inputs

    def __init__(self):
        self.config = _StubConfig()
        self.model = _StubModel()
        self.vision_start_pos_seen = None

    def _get_talker_text_mask(self, text_ids):
        return torch.zeros(text_ids.shape[0], dtype=torch.bool)

    def _build_vision_full(self, inputs, start_pos, device):
        self.vision_start_pos_seen = start_pos
        return _VisionPrefillStage(
            wrapped_embeds=torch.full((VLEN, HIDDEN), 2.0),
            pos_ids=torch.zeros((3, VLEN)),
            deepstack=[torch.full((VLEN, HIDDEN), 3.0) for _ in range(NUM_DEEPSTACK)],
            mm_mask=torch.ones(VLEN, dtype=torch.bool),
            total_len=VLEN,
            mrope_pos_advance=VADV,
            start_pos=start_pos,
        )


class _FwdInfo:
    def __init__(self, vision_first):
        self.step_metadata = {"merged_vision_first": vision_first}


def _run_merged(monkeypatch, vision_first, start_pos):
    text_start_seen = {}

    def _fake_rope_text(seq_len, start, device):
        text_start_seen["seq_len"] = seq_len
        text_start_seen["start"] = start
        return torch.zeros((3, seq_len))

    monkeypatch.setattr(sm, "get_rope_index_text", _fake_rope_text)
    shim = _ThinkerShim()
    rec = _Recorder()
    inputs = {"text_inputs": [torch.zeros(TLEN, dtype=torch.long)]}
    out = shim._build_merged_multimodal_inputs(
        _FwdInfo(vision_first), inputs, start_pos, torch.device("cpu"), rec,
    )
    return shim, out, text_start_seen, rec


def test_thread_positions_vision_first(monkeypatch):
    S = 7.0
    shim, out, text_seen, rec = _run_merged(monkeypatch, True, S)
    # vision starts at S; text continues from S + vision advance.
    assert shim.vision_start_pos_seen == S
    assert text_seen["start"] == S + VADV
    assert text_seen["seq_len"] == TLEN
    # single custom advance = vision advance + text len (lands position_id_start
    # exactly where the two separate walks would).
    assert out.kwargs["mrope_pos_advance"] == VADV + TLEN
    # concatenation: vision rows first, then text rows.
    assert out.input_seq_len == VLEN + TLEN
    assert out.input_embeds.shape == (VLEN + TLEN, HIDDEN)
    assert out.custom_pos_ids.shape == (3, VLEN + TLEN)
    assert torch.equal(out.input_embeds[:VLEN], torch.full((VLEN, HIDDEN), 2.0))
    assert torch.equal(out.input_embeds[VLEN:], torch.zeros((TLEN, HIDDEN)))
    # per-layer deepstack: vision slice nonzero, text slice zero-filled.
    for i in range(NUM_DEEPSTACK):
        ds = out.tensor_inputs[f"deepstack_{i}"]
        assert ds.shape == (VLEN + TLEN, HIDDEN)
        assert torch.equal(ds[:VLEN], torch.full((VLEN, HIDDEN), 3.0))
        assert torch.equal(ds[VLEN:], torch.zeros((TLEN, HIDDEN)))
    assert out.tensor_inputs["masks_for_talker"].shape == (2, VLEN + TLEN)
    assert rec.added is not None  # seen_token_mask updated for the text span


def test_thread_positions_text_first(monkeypatch):
    S = 7.0
    shim, out, text_seen, rec = _run_merged(monkeypatch, False, S)
    # text starts at S; vision continues from S + text len.
    assert text_seen["start"] == S
    assert shim.vision_start_pos_seen == S + TLEN
    assert out.kwargs["mrope_pos_advance"] == TLEN + VADV
    # concatenation: text rows first, then vision rows.
    assert out.input_embeds.shape == (TLEN + VLEN, HIDDEN)
    assert torch.equal(out.input_embeds[:TLEN], torch.zeros((TLEN, HIDDEN)))
    assert torch.equal(out.input_embeds[TLEN:], torch.full((VLEN, HIDDEN), 2.0))
    for i in range(NUM_DEEPSTACK):
        ds = out.tensor_inputs[f"deepstack_{i}"]
        assert torch.equal(ds[:TLEN], torch.zeros((TLEN, HIDDEN)))
        assert torch.equal(ds[TLEN:], torch.full((VLEN, HIDDEN), 3.0))
