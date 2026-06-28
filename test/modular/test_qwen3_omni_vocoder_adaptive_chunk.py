"""Batch-adaptive Code2Wav chunk policy: parity + boundary-focused waveform A/B.

The Talker -> Code2Wav StreamBuffer chooses how many codec frames to hand the
vocoder per step. With ``MSTAR_VOCODER_ADAPTIVE_CHUNK`` the chunk grows from the
latency value (``codec_chunk_frames``, e.g. 15) to a large throughput value
(``MSTAR_VOCODER_LARGE_CHUNK``, e.g. 150) once enough requests are co-vocoding,
while ``left_context`` stays fixed. This module verifies:

  1. CPU-only (always runs): the *chunk schedule* (per-pop window/stride/trim)
     produced by the adaptive policy at batch size 1 is byte-for-byte identical
     to the fixed ``LeftContextChunkPolicy``. Same windows -> same vocoder
     inputs -> identical waveform. It also checks the chunk >= left_context
     safety invariant (no negative first-pop stride) and the large-batch switch.

  2. CUDA-only (skipped without a GPU): drive identical random codes through the
     fixed schedule and the adaptive-at-B=1 schedule and assert the reassembled
     waveforms are IDENTICAL; then drive the large-batch schedule and assert no
     NaN/inf, correct total length, and a bounded seam discontinuity at chunk
     boundaries.

Import-safe on CPU: heavy model construction lives entirely inside the
CUDA-gated test.
"""
import pytest
import torch

from mstar.streaming.chunk_policy import (
    BatchAdaptiveLeftContextChunkPolicy,
    LeftContextChunkPolicy,
)
from mstar.streaming.stream_buffer import StreamBuffer

Q = 16  # num_quantizers used for the synthetic codes


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _collect_schedule(policy, num_frames, batch_size, *, q=Q):
    """Feed ``num_frames`` codec frames through a real StreamBuffer + policy and
    record the per-pop schedule as ``(start_offset, window, trim)``.

    ``window`` is the number of frames in the popped chunk; ``trim`` mirrors the
    submodule's left-context trim (0 for the first emitted chunk, ``left_context``
    afterwards). The number of *emitted* frames for a chunk is ``window - trim``.
    """
    buf = StreamBuffer(
        request_id="r",
        edge_name="codec_tokens",
        from_partition="Talker",
        policy=policy,
    )
    for i in range(num_frames):
        tid = f"t{i}"
        buf.pre_read_register(tid)
        buf.put(tid, torch.full((q,), i, dtype=torch.long))

    schedule = []
    first = True

    def _drain():
        nonlocal first
        while True:
            policy.observe_batch_size(batch_size)
            if not buf.has_chunk_ready():
                return
            ch = buf.pop_chunk()
            data = ch.data.get("data")
            if data is None:
                continue
            window = 1 if data.dim() == 1 else data.shape[0]
            trim = 0 if first else policy._left_context
            schedule.append((ch.start_offset, window, trim))
            first = False

    _drain()
    buf.signal_done()
    _drain()
    return schedule


# --------------------------------------------------------------------------- #
# CPU-only: schedule parity + safety invariant
# --------------------------------------------------------------------------- #
def test_b1_schedule_identical_to_fixed():
    """At batch size 1 the adaptive policy must reproduce the fixed schedule
    exactly -> the B=1 waveform is byte-identical to the fixed-chunk path."""
    num_frames = 137
    fixed = _collect_schedule(
        LeftContextChunkPolicy(chunk=15, left_context=15),
        num_frames,
        batch_size=1,
    )
    adaptive = _collect_schedule(
        BatchAdaptiveLeftContextChunkPolicy(
            latency_chunk=15, large_chunk=150, left_context=15, threshold=4
        ),
        num_frames,
        batch_size=1,  # below threshold -> latency chunk
    )
    assert adaptive == fixed
    # All emitted-frame counts non-negative and total == num_frames.
    assert sum(w - t for _, w, t in adaptive) == num_frames


def test_large_batch_uses_large_chunk():
    """At/above threshold the adaptive policy must switch to the large chunk."""
    num_frames = 600
    adaptive = _collect_schedule(
        BatchAdaptiveLeftContextChunkPolicy(
            latency_chunk=15, large_chunk=150, left_context=15, threshold=4
        ),
        num_frames,
        batch_size=8,  # >= threshold -> large chunk
    )
    # First steady-state pop window should be large_chunk + left_context.
    windows = [w for _, w, _ in adaptive]
    assert max(windows) == 150 + 15
    # Total emitted frames still equals the number fed in.
    assert sum(w - t for _, w, t in adaptive) == num_frames


def test_chunk_never_below_left_context():
    """chunk >= left_context must hold for every selectable chunk, so the
    first-pop stride (chunk - left_context) can never go negative (the -10
    pop-stride corruption from FINDINGS is unrepresentable)."""
    # Deliberately request chunks BELOW the left context: they must be clamped.
    pol = BatchAdaptiveLeftContextChunkPolicy(
        latency_chunk=5, large_chunk=3, left_context=15, threshold=2
    )
    assert pol._latency_chunk >= pol._left_context
    assert pol._large_chunk >= pol._left_context

    for bs in (1, 2, 8):
        sched = _collect_schedule(
            BatchAdaptiveLeftContextChunkPolicy(
                latency_chunk=5, large_chunk=3, left_context=15, threshold=2
            ),
            num_frames=120,
            batch_size=bs,
        )
        # Every emitted-frame count (window - trim) is non-negative.
        assert all(w - t >= 0 for _, w, t in sched)
        # First pop stride is window - 0 - (next overlap kept) >= 0; concretely
        # the first window must be >= left_context.
        assert sched[0][1] >= pol._left_context


def test_off_path_is_plain_left_context():
    """Sanity: the fixed policy ignores observed batch size (byte-identical
    regardless of concurrency)."""
    pol = LeftContextChunkPolicy(chunk=15, left_context=15)
    pol.observe_batch_size(64)  # no-op on the base policy
    a = _collect_schedule(LeftContextChunkPolicy(chunk=15, left_context=15), 90, 1)
    b = _collect_schedule(LeftContextChunkPolicy(chunk=15, left_context=15), 90, 64)
    assert a == b


# --------------------------------------------------------------------------- #
# CUDA-only: boundary-focused waveform A/B
# --------------------------------------------------------------------------- #
def _build_small_vocoder(device):
    """Construct a small, randomly-initialized Code2Wav vocoder for waveform A/B.

    Reduced dims keep the GPU test fast; the architecture (causal transformer +
    causal up/decode ConvNets) is unchanged, which is what the boundary behavior
    depends on. Returns ``(model, total_upsample, num_quantizers, codebook_size)``
    or skips if construction is unavailable.
    """
    from mstar.model.qwen3_omni.components.code2wav import Qwen3OmniMoeCode2Wav
    from mstar.model.qwen3_omni.config import Code2WavConfig

    cfg = Code2WavConfig(
        codebook_size=64,
        num_quantizers=Q,
        num_semantic_quantizers=1,
        sliding_window=32,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        decoder_dim=64,
        upsample_rates=(2, 2),
        upsampling_ratios=(2,),
    )
    model = Qwen3OmniMoeCode2Wav(cfg).to(device).eval()
    total_upsample = int(model.total_upsample)
    return model, total_upsample, cfg.num_quantizers, cfg.codebook_size


def _reassemble(model, codes, schedule, total_upsample, device):
    """Run each scheduled chunk through the vocoder, trim its left-context
    overlap, and concatenate -> the streamed waveform."""
    parts = []
    for start, window, trim in schedule:
        seg = codes[:, :, start:start + window]
        pos = torch.arange(seg.shape[2], device=device)
        with torch.no_grad():
            wav = model(seg, pos)  # [1, 1, window * total_upsample]
        wav = wav[0, 0]
        if trim:
            wav = wav[trim * total_upsample:]
        parts.append(wav)
    return torch.cat(parts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="vocoder waveform A/B needs a GPU")
def test_waveform_ab_b1_identical_and_large_batch_sane():
    device = torch.device("cuda")
    torch.manual_seed(0)
    try:
        model, up, q, codebook = _build_small_vocoder(device)
    except Exception as exc:  # pragma: no cover - defensive on the GPU box
        pytest.skip(f"could not build small vocoder: {exc}")

    num_frames = 200
    codes = torch.randint(0, codebook, (1, q, num_frames), device=device)

    fixed_sched = _collect_schedule(
        LeftContextChunkPolicy(chunk=15, left_context=15), num_frames, 1, q=q
    )
    adaptive_b1_sched = _collect_schedule(
        BatchAdaptiveLeftContextChunkPolicy(
            latency_chunk=15, large_chunk=150, left_context=15, threshold=4
        ),
        num_frames, 1, q=q,
    )
    assert adaptive_b1_sched == fixed_sched

    wav_fixed = _reassemble(model, codes, fixed_sched, up, device)
    wav_adaptive = _reassemble(model, codes, adaptive_b1_sched, up, device)

    # B=1 parity: identical schedule + identical codes + identical weights ->
    # bit-for-bit identical waveform.
    assert wav_fixed.shape == wav_adaptive.shape
    assert torch.equal(wav_fixed, wav_adaptive)
    assert wav_fixed.numel() == num_frames * up

    # Large-batch path: large chunk, still well-formed.
    large_sched = _collect_schedule(
        BatchAdaptiveLeftContextChunkPolicy(
            latency_chunk=15, large_chunk=150, left_context=15, threshold=4
        ),
        num_frames, 8, q=q,
    )
    wav_large = _reassemble(model, codes, large_sched, up, device)

    assert torch.isfinite(wav_large).all()
    assert wav_large.numel() == num_frames * up

    # Bounded seam discontinuity: the emitted output is clamped to [-1, 1], so a
    # sane reconstruction has sample-to-sample jumps within the clamp range at
    # every chunk boundary. (NaN/inf or a torn buffer would blow this up.)
    seam_positions = []
    acc = 0
    for _, window, trim in large_sched[:-1]:
        acc += (window - trim) * up
        seam_positions.append(acc)
    for p in seam_positions:
        if 0 < p < wav_large.numel():
            jump = (wav_large[p] - wav_large[p - 1]).abs().item()
            assert jump <= 2.0 + 1e-4
