"""Tests for the Talker/Thinker batch-fill instrumentation.

The aggregation math (``fill_bucket`` / ``aggregate_fill_stats`` /
``aggregate_fill_stats_from_counts``) and the ``BatchFillRecorder`` are pure
CPU logic and are unit-tested here without CUDA. The single CUDA-only check
(that the engine hook is wired) is skipped when no GPU is present.
"""

import importlib.util

import pytest

from mstar.engine.batchfill_stats import (
    DEFAULT_CAPTURE_BUCKETS,
    BatchFillRecorder,
    aggregate_fill_stats,
    aggregate_fill_stats_from_counts,
    fill_bucket,
)

try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover - torch optional in some CI
    _HAS_CUDA = False


# --- fill_bucket -------------------------------------------------------------
@pytest.mark.parametrize(
    "bs,expected",
    [
        (1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8),
        (9, 16), (16, 16), (17, 32), (32, 32),
        # above the largest bucket -> attributed to the top bucket
        (33, 32), (100, 32),
    ],
)
def test_fill_bucket_rounds_up_to_capture_size(bs, expected):
    assert fill_bucket(bs) == expected


def test_fill_bucket_custom_buckets():
    assert fill_bucket(3, buckets=(1, 4, 16)) == 4
    assert fill_bucket(5, buckets=(1, 4, 16)) == 16


# --- aggregate_fill_stats ----------------------------------------------------
def test_aggregate_empty():
    s = aggregate_fill_stats([])
    assert s["count"] == 0
    assert s["mean_bs"] == 0.0
    assert s["mean_fill_ratio"] == 0.0
    assert s["frac_at_max_bucket"] == 0.0
    assert s["bucket_counts"] == {b: 0 for b in DEFAULT_CAPTURE_BUCKETS}


def test_aggregate_perfect_fill_all_at_32():
    # Good fill: every replay is the full batch -> fill ratio 1.0.
    s = aggregate_fill_stats([32] * 10)
    assert s["count"] == 10
    assert s["mean_bs"] == 32.0
    assert s["mean_fill_ratio"] == 1.0
    assert s["frac_at_max_bucket"] == 1.0
    assert s["bucket_counts"][32] == 10
    assert s["min_bs"] == 32 and s["max_bs"] == 32


def test_aggregate_total_desync_all_bs1():
    # Bad fill: every replay is a single request padded to the 1-bucket. Note
    # that bs=1 pads to bucket 1 (fill ratio 1.0), but the histogram makes the
    # desync obvious: nothing reaches the top bucket.
    s = aggregate_fill_stats([1] * 8)
    assert s["count"] == 8
    assert s["mean_bs"] == 1.0
    assert s["frac_at_max_bucket"] == 0.0
    assert s["bucket_counts"][1] == 8


def test_aggregate_partial_fill_ratio():
    # bs=3 pads to bucket 4 -> fill 0.75; bs=5 pads to bucket 8 -> fill 0.625.
    s = aggregate_fill_stats([3, 5])
    assert s["bucket_counts"][4] == 1
    assert s["bucket_counts"][8] == 1
    assert s["mean_bs"] == 4.0
    assert s["mean_fill_ratio"] == pytest.approx((0.75 + 0.625) / 2)
    assert s["raw_bs_counts"] == {3: 1, 5: 1}


def test_aggregate_mixed_distribution_and_frac():
    sizes = [32, 32, 32, 16, 1, 1]
    s = aggregate_fill_stats(sizes)
    assert s["count"] == 6
    assert s["bucket_counts"][32] == 3
    assert s["bucket_counts"][16] == 1
    assert s["bucket_counts"][1] == 2
    assert s["frac_at_max_bucket"] == pytest.approx(3 / 6)
    assert s["mean_bs"] == pytest.approx((32 * 3 + 16 + 1 + 1) / 6)


def test_counts_and_list_paths_agree():
    sizes = [1, 1, 2, 4, 4, 4, 32]
    from collections import Counter

    a = aggregate_fill_stats(sizes)
    b = aggregate_fill_stats_from_counts(Counter(sizes))
    assert a == b


def test_aggregate_ignores_nonpositive():
    s = aggregate_fill_stats_from_counts({0: 5, -3: 2, 8: 4})
    assert s["count"] == 4
    assert s["bucket_counts"][8] == 4


# --- BatchFillRecorder -------------------------------------------------------
def test_recorder_keys_by_node_walk_path():
    r = BatchFillRecorder()
    for _ in range(3):
        r.record("talker", "talker_decode", 32, "cuda_graph")
    r.record("talker", "talker_decode", 1, "cuda_graph")
    r.record("thinker", "thinker_decode", 16, "cuda_graph")
    r.record("audio_encoder", "prefill_audio", 4, "batched")

    summary = r.summarize()
    assert set(summary) == {
        "talker/talker_decode/cuda_graph",
        "thinker/thinker_decode/cuda_graph",
        "audio_encoder/prefill_audio/batched",
    }
    talker = summary["talker/talker_decode/cuda_graph"]
    assert talker["count"] == 4
    assert talker["bucket_counts"][32] == 3
    assert talker["bucket_counts"][1] == 1
    assert talker["frac_at_max_bucket"] == pytest.approx(3 / 4)


def test_recorder_reset_and_empty():
    r = BatchFillRecorder()
    assert r.is_empty()
    r.record("talker", "talker_decode", 8, "cuda_graph")
    assert not r.is_empty()
    r.reset()
    assert r.is_empty()
    assert r.summarize() == {}


def test_recorder_ignores_bad_batch_size():
    r = BatchFillRecorder()
    r.record("talker", "talker_decode", 0, "cuda_graph")
    r.record("talker", "talker_decode", None, "cuda_graph")
    assert r.is_empty()


def test_format_summary_contains_key_metrics():
    r = BatchFillRecorder()
    r.record("talker", "talker_decode", 32, "cuda_graph")
    r.record("talker", "talker_decode", 32, "cuda_graph")
    text = r.format_summary()
    assert "talker/talker_decode/cuda_graph" in text
    assert "fill=1.00" in text
    assert "frac@32=1.00" in text


# --- engine hook (CUDA-only smoke) ------------------------------------------
@pytest.mark.skipif(not _HAS_CUDA, reason="requires CUDA")
def test_engine_hook_is_wired():
    # The hook lives in KVCacheEngine.execute_forward; just confirm the module
    # imports the recorder so the wiring exists. Pure logic is covered above.
    spec = importlib.util.find_spec("mstar.engine.kv_cache_engine")
    assert spec is not None
    import mstar.engine.kv_cache_engine as kce

    assert hasattr(kce, "batchfill_stats")
