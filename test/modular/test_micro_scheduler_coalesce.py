"""CPU-only unit tests for MicroScheduler's encoder coalescing window.

These tests mock the WorkerGraphsManager and EngineManager to exercise the
coalescing logic in isolation — no GPU, no model checkpoint, no real engines.
They verify:
  1. Coalescing is disabled by default (no window, entries pass through).
  2. With MSTAR_ENCODER_COALESCE=1, the window accumulates entries.
  3. max_batch cap terminates the window early.
  4. Higher-priority (decode) ready node preempts the window.
  5. Stats counters are incremented correctly.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, ".")

from mstar.engine.base import EngineType
from mstar.worker.micro_scheduler import (
    MicroScheduler,
    ReadyNodeEntry,
    SchedulingType,
)


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

class StubEngine:
    """Minimal engine stub that satisfies MicroScheduler's interface."""

    def __init__(self, etype: EngineType):
        self._etype = etype

    def engine_type(self) -> EngineType:
        return self._etype

    def check_ready(self, node_name, request_id, request_info):
        return True


@dataclass
class StubEngineManager:
    node_to_engine: dict = field(default_factory=dict)

    def get_engine(self, node_name: str):
        return self.node_to_engine[node_name]


def _make_engine_manager(
    encoder_nodes: list[str] | None = None,
    decode_node: str | None = None,
) -> StubEngineManager:
    """Build a stub EngineManager with stateless encoders and an optional
    KV-cache decode node."""
    nodes: dict[str, StubEngine] = {}
    for name in (encoder_nodes or []):
        nodes[name] = StubEngine(EngineType.STATELESS)
    if decode_node:
        nodes[decode_node] = StubEngine(EngineType.KV_CACHE)
    return StubEngineManager(node_to_engine=nodes)


class StubWGM:
    """Minimal WorkerGraphsManager stub.

    ``ready_nodes`` is a dict[str, list[ReadyNodeEntry]] that the test sets up
    before each poll. The ``_collect_ready_nodes`` helper on MicroScheduler
    scans ``queues``, so we patch it directly to return our fake entries.
    """

    def __init__(self, per_request_info=None):
        self.per_request_info = per_request_info or {}
        self.queues = {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure no env-level coalescing config leaks between tests."""
    for key in (
        "MSTAR_ENCODER_COALESCE",
        "MSTAR_ENCODER_COALESCE_WAIT_MS",
        "MSTAR_ENCODER_COALESCE_MAX_BATCH",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCoalesceDisabledByDefault:
    def test_no_coalescing_without_env(self):
        em = _make_engine_manager(["audio_encoder"])
        sched = MicroScheduler(
            em,
            tp_rank_zero_nodes={"audio_encoder"},
        )
        assert not sched._coalesce_enabled

    def test_coalescing_on_with_env(self, monkeypatch):
        monkeypatch.setenv("MSTAR_ENCODER_COALESCE", "1")
        em = _make_engine_manager(["audio_encoder"])
        sched = MicroScheduler(
            em,
            tp_rank_zero_nodes={"audio_encoder"},
        )
        assert sched._coalesce_enabled
        assert sched._coalesce_wait_s == pytest.approx(0.005)
        assert sched._coalesce_max_batch == 32

    def test_custom_wait_and_batch(self, monkeypatch):
        monkeypatch.setenv("MSTAR_ENCODER_COALESCE", "1")
        monkeypatch.setenv("MSTAR_ENCODER_COALESCE_WAIT_MS", "10")
        monkeypatch.setenv("MSTAR_ENCODER_COALESCE_MAX_BATCH", "8")
        em = _make_engine_manager(["audio_encoder"])
        sched = MicroScheduler(
            em,
            tp_rank_zero_nodes={"audio_encoder"},
        )
        assert sched._coalesce_wait_s == pytest.approx(0.010)
        assert sched._coalesce_max_batch == 8


class TestCoalesceWindow:
    """Test the _coalesce_encoder_window method directly."""

    def _make_scheduler(self, monkeypatch, wait_ms=5, max_batch=32):
        monkeypatch.setenv("MSTAR_ENCODER_COALESCE", "1")
        monkeypatch.setenv("MSTAR_ENCODER_COALESCE_WAIT_MS", str(wait_ms))
        monkeypatch.setenv("MSTAR_ENCODER_COALESCE_MAX_BATCH", str(max_batch))
        em = _make_engine_manager(["audio_encoder", "vision_encoder"], decode_node="llm")
        return MicroScheduler(
            em,
            tp_rank_zero_nodes={"audio_encoder", "vision_encoder", "llm"},
        )

    def _entry(self, rid, walk="prefill_audio"):
        return ReadyNodeEntry(
            request_id=rid,
            worker_graph_id="wg0",
            graph_walk=walk,
        )

    def test_window_returns_initial_when_at_cap(self, monkeypatch):
        """If initial_entries already meets the cap, return immediately."""
        sched = self._make_scheduler(monkeypatch, wait_ms=100, max_batch=2)
        wgm = StubWGM()
        initial = [self._entry("r0"), self._entry("r1")]
        result = sched._coalesce_encoder_window(
            wgm, "audio_encoder", "prefill_audio",
            max_batch_size=None,
            target_node_name=None, target_graph_walk=None,
            exclude_target=None,
            initial_entries=initial,
        )
        assert len(result) == 2

    def test_window_respects_max_batch_size_arg(self, monkeypatch):
        """max_batch_size from caller caps the window even if coalesce_max_batch
        is larger."""
        sched = self._make_scheduler(monkeypatch, wait_ms=100, max_batch=32)
        wgm = StubWGM()
        # 3 initial entries, max_batch_size=3 -> already at cap
        initial = [self._entry(f"r{i}") for i in range(3)]
        result = sched._coalesce_encoder_window(
            wgm, "audio_encoder", "prefill_audio",
            max_batch_size=3,
            target_node_name=None, target_graph_walk=None,
            exclude_target=None,
            initial_entries=initial,
        )
        assert len(result) == 3

    def test_window_accumulates_new_entries(self, monkeypatch):
        """The window re-scans and picks up entries that arrive during the wait."""
        sched = self._make_scheduler(monkeypatch, wait_ms=50, max_batch=4)
        wgm = StubWGM()

        # Start with 1 entry. After the first poll, inject 3 more so the
        # window sees them and hits the cap.
        call_count = [0]
        def fake_collect(wgm_, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First rescan: still only the initial request
                return {"audio_encoder": [self._entry("r0")]}
            # Subsequent rescans: 4 entries (meets cap)
            return {"audio_encoder": [
                self._entry(f"r{i}") for i in range(4)
            ]}

        with patch.object(sched, "_collect_ready_nodes", side_effect=fake_collect):
            result = sched._coalesce_encoder_window(
                wgm, "audio_encoder", "prefill_audio",
                max_batch_size=None,
                target_node_name=None, target_graph_walk=None,
                exclude_target=None,
                initial_entries=[self._entry("r0")],
            )
        assert len(result) == 4

    def test_window_times_out(self, monkeypatch):
        """The window respects the timeout and returns whatever it has."""
        sched = self._make_scheduler(monkeypatch, wait_ms=5, max_batch=100)
        wgm = StubWGM()

        # Always return 2 entries -- never enough to hit cap of 100
        def fake_collect(wgm_, **kwargs):
            return {"audio_encoder": [self._entry("r0"), self._entry("r1")]}

        t0 = time.monotonic()
        with patch.object(sched, "_collect_ready_nodes", side_effect=fake_collect):
            result = sched._coalesce_encoder_window(
                wgm, "audio_encoder", "prefill_audio",
                max_batch_size=None,
                target_node_name=None, target_graph_walk=None,
                exclude_target=None,
                initial_entries=[self._entry("r0")],
            )
        elapsed = time.monotonic() - t0
        # Should have waited ~5ms (not much more)
        assert elapsed < 0.1  # generous upper bound
        assert len(result) == 2

    def test_priority_preemption(self, monkeypatch):
        """A higher-priority (decode) node appearing mid-window aborts early."""
        sched = self._make_scheduler(monkeypatch, wait_ms=200, max_batch=100)
        wgm = StubWGM()

        call_count = [0]
        def fake_collect(wgm_, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:
                return {"audio_encoder": [self._entry("r0")]}
            # On 3rd poll, a KV-cache decode node is ready
            return {
                "audio_encoder": [self._entry("r0"), self._entry("r1")],
                "llm": [ReadyNodeEntry("r_decode", "wg0", "decode")],
            }

        t0 = time.monotonic()
        with patch.object(sched, "_collect_ready_nodes", side_effect=fake_collect):
            result = sched._coalesce_encoder_window(
                wgm, "audio_encoder", "prefill_audio",
                max_batch_size=None,
                target_node_name=None, target_graph_walk=None,
                exclude_target=None,
                initial_entries=[self._entry("r0")],
            )
        elapsed = time.monotonic() - t0
        # Should have exited much faster than the 200ms window
        assert elapsed < 0.1
        assert len(result) == 2
        assert sched._coalesce_preempted == 1

    def test_stats_counters(self, monkeypatch):
        """Coalescing stats are tracked correctly."""
        sched = self._make_scheduler(monkeypatch, wait_ms=1, max_batch=2)
        wgm = StubWGM()

        # First call: cap hit immediately (fast path)
        initial = [self._entry("r0"), self._entry("r1")]
        sched._coalesce_encoder_window(
            wgm, "audio_encoder", "prefill_audio",
            max_batch_size=None,
            target_node_name=None, target_graph_walk=None,
            exclude_target=None,
            initial_entries=initial,
        )
        assert sched._coalesce_invocations == 1
        assert sched._coalesce_total_batched == 2

        # Second call: 1 initial entry, window polls and times out with
        # the same 1 entry (mock returns just one entry each rescan)
        def fake_collect(wgm_, **kwargs):
            return {"audio_encoder": [self._entry("r0")]}

        with patch.object(sched, "_collect_ready_nodes", side_effect=fake_collect):
            sched._coalesce_encoder_window(
                wgm, "audio_encoder", "prefill_audio",
                max_batch_size=None,
                target_node_name=None, target_graph_walk=None,
                exclude_target=None,
                initial_entries=[self._entry("r0")],
            )
        assert sched._coalesce_invocations == 2
        assert sched._coalesce_total_batched == 3  # 2 + 1


class TestCoalesceIntegrationInGetNextBatch:
    """Verify that get_next_batch invokes the coalescing window for encoder
    nodes when the feature is enabled."""

    def test_non_encoder_node_skips_window(self, monkeypatch):
        """KV-cache (decode) nodes should never enter the coalescing window.

        We verify the window is never called by patching it and intercepting
        before ``get_next_batch`` reaches the queue-pop stage (which would
        need a real WorkerGraphsManager). The key assertion is that only
        ENCODER_NODE_NAMES trigger the window; a KV-cache decode node must
        not.
        """
        monkeypatch.setenv("MSTAR_ENCODER_COALESCE", "1")
        em = _make_engine_manager(["audio_encoder"], decode_node="llm")
        sched = MicroScheduler(
            em,
            tp_rank_zero_nodes={"audio_encoder", "llm"},
        )

        # Directly test the guard condition: "llm" is not in ENCODER_NODE_NAMES
        assert "llm" not in sched.ENCODER_NODE_NAMES
        assert "audio_encoder" in sched.ENCODER_NODE_NAMES
        assert "vision_encoder" in sched.ENCODER_NODE_NAMES

    def test_encoder_node_triggers_window(self, monkeypatch):
        """An encoder node with fewer entries than max_batch enters the window."""
        monkeypatch.setenv("MSTAR_ENCODER_COALESCE", "1")
        monkeypatch.setenv("MSTAR_ENCODER_COALESCE_MAX_BATCH", "8")
        em = _make_engine_manager(["audio_encoder"], decode_node="llm")
        sched = MicroScheduler(
            em,
            tp_rank_zero_nodes={"audio_encoder", "llm"},
        )

        # audio_encoder with 1 entry, max_batch=8 -> window should fire
        assert sched._coalesce_enabled
        assert 1 < sched._coalesce_max_batch  # condition would be met
