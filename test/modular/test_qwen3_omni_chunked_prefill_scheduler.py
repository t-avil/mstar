"""Conductor-driven re-enqueue tests for resumable chunked Thinker prefill.

These complement ``test_qwen3_omni_chunked_prefill_parity.py`` (which pins the
model-side M-RoPE slice/cos-sin properties). Here we exercise the SCHEDULER /
CONDUCTOR loop that was previously stubbed: the Thinker state machine must
re-emit the SAME ``prefill_text`` walk with an advanced
``prefill_chunk_offset`` until a long text span is consumed, set
``is_last_prefill`` only on the final chunk, and stay byte-identical when the
flag is OFF / the span fits one chunk. All CPU, no GPU, no model weights.
"""
import pytest

from mstar.conductor.request_info import CurrentForwardConductorMetadata
from mstar.graph.base import TensorPointerInfo
from mstar.model.qwen3_omni.qwen3_omni_model import (
    Qwen3OmniModel,
    plan_text_prefill_chunk,
)


# --- pure planner ----------------------------------------------------------
def test_plan_chunk_no_chunk_when_short():
    assert plan_text_prefill_chunk(500, 0, 512) is None
    assert plan_text_prefill_chunk(512, 0, 512) is None
    assert plan_text_prefill_chunk(None, 0, 512) is None
    assert plan_text_prefill_chunk(0, 0, 512) is None


def test_plan_chunk_sequence():
    # 1100 tokens / threshold 512 -> 512, 512, 76
    assert plan_text_prefill_chunk(1100, 0, 512) == (512, False)
    assert plan_text_prefill_chunk(1100, 512, 512) == (512, False)
    assert plan_text_prefill_chunk(1100, 1024, 512) == (76, True)
    # exact multiple closes on the boundary
    assert plan_text_prefill_chunk(1024, 512, 512) == (512, True)
    # offset past the end is a no-op guard
    assert plan_text_prefill_chunk(1100, 1100, 512) is None


# --- conductor state-machine loop (no model weights) -----------------------
class _Shim:
    """Borrow just the Thinker state-machine methods from Qwen3OmniModel so we
    can drive the prefill loop without constructing the heavyweight model."""

    _text_chunk_bounds = Qwen3OmniModel._text_chunk_bounds
    _get_thinker_forward = Qwen3OmniModel._get_thinker_forward
    _get_thinker_prefill_inputs = Qwen3OmniModel._get_thinker_prefill_inputs


def _tpi(n_tokens: int, uuid: str) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[n_tokens], dtype="int64", nbytes=n_tokens * 8, address=0,
        stride=[1], uuid=uuid, source_session_id="s", source_entity="w",
    )


def _make_meta(schedule, audio_output=False):
    return CurrentForwardConductorMetadata(
        graph_walk=schedule[0][0],
        is_prefill=True,
        kwargs={
            "prefill_schedule": schedule,
            "prefill_step": 0,
            "audio_output": audio_output,
            "prefill_chunk_offset": 0,
        },
    )


def _drive(shim, meta, persist, max_steps=50):
    """Replay the conductor loop: build step 0 from current meta, then call
    _get_thinker_forward repeatedly (as _process_done_forward does) until the
    Thinker leaves prefill. Returns the list of per-step records."""
    steps = []
    # step 0 is described by the initial metadata; emulate by building inputs
    # for the current (walk, offset) directly.
    schedule = meta.kwargs["prefill_schedule"]
    # Build the first chunk's step_metadata the way _get_thinker_initial_args
    # would (offset already 0 in kwargs).
    bounds = shim._text_chunk_bounds(meta, schedule, 0)
    is_last = (len(schedule) == 1)
    if bounds is not None:
        off, ln, done = bounds
        is_last = is_last and done
        steps.append({"walk": meta.graph_walk, "offset": off, "len": ln,
                      "is_last_prefill": is_last})
    else:
        steps.append({"walk": meta.graph_walk, "offset": 0, "len": None,
                      "is_last_prefill": is_last})

    for _ in range(max_steps):
        fwd = shim._get_thinker_forward(meta, persist)
        meta = fwd.full_metadata
        meta.kwargs.update(fwd.step_metadata)
        if not meta.is_prefill:
            steps.append({"walk": meta.graph_walk, "decode": True})
            break
        sm = fwd.step_metadata
        steps.append({
            "walk": meta.graph_walk,
            "offset": sm.get("prefill_chunk_offset"),
            "len": sm.get("prefill_chunk_len"),
            "is_last_prefill": sm.get("is_last_prefill"),
            "unpersist": len(fwd.unpersist_tensors),
        })
    return steps


def test_long_text_single_walk_chunks(monkeypatch):
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL", "1")
    monkeypatch.setenv("MSTAR_LONG_PREFILL_TOKEN_THRESHOLD", "512")
    schedule = [("prefill_text", {"text_inputs": _tpi(1100, "t0")})]
    meta = _make_meta(schedule, audio_output=False)
    persist = {"text_inputs": [schedule[0][1]["text_inputs"]]}
    steps = _drive(_Shim(), meta, persist)

    prefill = [s for s in steps if not s.get("decode")]
    # Three chunks: [0:512], [512:1024], [1024:1100]
    assert [(s["offset"], s["len"]) for s in prefill] == [
        (0, 512), (512, 512), (1024, 76)
    ]
    # is_last_prefill ONLY on the final chunk
    assert [s["is_last_prefill"] for s in prefill] == [False, False, True]
    # non-final chunks hold the input tensor (unpersist=0); final releases it
    assert prefill[1]["unpersist"] == 0 and prefill[2]["unpersist"] == 1
    # ends in decode
    assert steps[-1].get("decode") is True


def test_short_text_single_chunk_no_chunking(monkeypatch):
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL", "1")
    monkeypatch.setenv("MSTAR_LONG_PREFILL_TOKEN_THRESHOLD", "512")
    schedule = [("prefill_text", {"text_inputs": _tpi(300, "t0")})]
    meta = _make_meta(schedule, audio_output=False)
    persist = {"text_inputs": [schedule[0][1]["text_inputs"]]}
    steps = _drive(_Shim(), meta, persist)
    prefill = [s for s in steps if not s.get("decode")]
    assert len(prefill) == 1
    assert prefill[0]["len"] is None  # no chunk metadata -> single-shot
    assert prefill[0]["is_last_prefill"] is True


def test_audio_output_disables_chunking(monkeypatch):
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL", "1")
    monkeypatch.setenv("MSTAR_LONG_PREFILL_TOKEN_THRESHOLD", "512")
    schedule = [("prefill_text", {"text_inputs": _tpi(1100, "t0")})]
    meta = _make_meta(schedule, audio_output=True)  # Talker active -> no chunk
    persist = {"text_inputs": [schedule[0][1]["text_inputs"]]}
    steps = _drive(_Shim(), meta, persist)
    prefill = [s for s in steps if not s.get("decode")]
    assert len(prefill) == 1
    assert prefill[0]["len"] is None
    assert prefill[0]["is_last_prefill"] is True


def test_flag_off_is_single_shot(monkeypatch):
    monkeypatch.delenv("MSTAR_CHUNKED_PREFILL", raising=False)
    schedule = [("prefill_text", {"text_inputs": _tpi(5000, "t0")})]
    meta = _make_meta(schedule, audio_output=False)
    persist = {"text_inputs": [schedule[0][1]["text_inputs"]]}
    steps = _drive(_Shim(), meta, persist)
    prefill = [s for s in steps if not s.get("decode")]
    assert len(prefill) == 1  # flag OFF -> no chunking even for a huge span
    assert prefill[0]["len"] is None


def test_audio_walk_in_schedule_not_chunked(monkeypatch):
    monkeypatch.setenv("MSTAR_CHUNKED_PREFILL", "1")
    monkeypatch.setenv("MSTAR_LONG_PREFILL_TOKEN_THRESHOLD", "512")
    # text(1100) then audio: text chunks, audio runs single-shot, then decode.
    schedule = [
        ("prefill_text", {"text_inputs": _tpi(1100, "t0")}),
        ("prefill_audio", {"audio_features": _tpi(999, "a0")}),
    ]
    meta = _make_meta(schedule, audio_output=False)
    persist = {
        "text_inputs": [schedule[0][1]["text_inputs"]],
        "audio_features": [schedule[1][1]["audio_features"]],
    }
    steps = _drive(_Shim(), meta, persist)
    walks = [s["walk"] for s in steps]
    # 3 text chunks + 1 audio + decode
    assert walks == [
        "prefill_text", "prefill_text", "prefill_text",
        "prefill_audio", "thinker_decode",
    ]
    audio_step = [s for s in steps if s["walk"] == "prefill_audio"][0]
    assert audio_step["len"] is None  # audio not chunked
    # last text chunk is NOT is_last_prefill (audio walk follows it)
    text_steps = [s for s in steps if s["walk"] == "prefill_text"]
    assert text_steps[-1]["is_last_prefill"] is False
    # audio (final walk) IS last prefill
    assert audio_step["is_last_prefill"] is True
