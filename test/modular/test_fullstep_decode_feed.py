"""MSTAR_FULLSTEP_DECODE (moonshot A') — token-feed state machine tests.

The self-feeding thinker_decode replay may skip ALL static-input copies only
when the tokens sitting in the shared static ``next_token_ids`` buffer were
written (in-graph) by the last successful replay for EXACTLY the same rid
tuple. These tests drive ``CudaGraphRunner._fullstep_pre_replay`` /
``_fullstep_mark_fed`` / ``invalidate_fullstep_feed`` directly with stub
graph data (no GPU), verifying the invariants the A/B parity run relies on:

  - a non-self-feeding capture never engages the fast path;
  - the first greedy replay re-seeds, the second identical-tuple replay
    self-feeds;
  - ANY composition change re-seeds — including the EOS one-step-late case
    (moonshot A3): a finishing rid's overshoot step runs, the worker trims
    it, and the shrunken tuple must not trust the buffer;
  - a non-greedy step consumes the marker (host-sampled token != in-graph
    argmax, so the buffer is stale for the NEXT step even with an unchanged
    tuple);
  - a replay failure between pre_replay and mark_fed leaves the marker
    consumed (pessimistic invalidate);
  - an eager detour (``invalidate_fullstep_feed``) forces a re-seed.
"""

from __future__ import annotations

import sys
import types

sys.path.insert(0, ".")

from mstar.engine.cuda_graph_runner import (
    CudaGraphData,
    CudaGraphRunner,
    CudaGraphSlot,
)

GREEDY = types.SimpleNamespace(temperature=0, repetition_penalty=1.0)
SAMPLED = types.SimpleNamespace(temperature=0.7, repetition_penalty=1.0)
PENALIZED = types.SimpleNamespace(temperature=0, repetition_penalty=1.1)

TOKENS = object()  # stands in for the __fullstep_tokens__ static output


def _make_runner(configs: dict) -> CudaGraphRunner:
    """Runner via ``__new__`` with just the attrs the feed gate touches."""
    runner = CudaGraphRunner.__new__(CudaGraphRunner)
    runner._fullstep_feed_rids = None
    runner._greedy_gate_memo = None
    runner.sampler = types.SimpleNamespace(
        _config_generation=0,
        _sampling_config=configs,
    )
    return runner


def _make_graph_data(self_feeding: bool) -> CudaGraphData:
    return CudaGraphData(config=object(), bs=2, self_feeding=self_feeding)


def _make_slot(with_tokens: bool, with_logits: bool = True) -> CudaGraphSlot:
    outputs = {}
    if with_tokens:
        outputs["__fullstep_tokens__"] = TOKENS
    if with_logits:
        outputs["__batched_logits__"] = object()
    return CudaGraphSlot(
        graph=object(),
        static_inputs={},
        static_outputs=outputs,
        static_cache_manager=object(),
    )


class TestFullstepFeedGate:
    def test_non_self_feeding_capture_never_engages(self):
        runner = _make_runner({"a": GREEDY})
        runner._fullstep_feed_rids = ("a",)  # even with a stale marker
        toks, self_feed = runner._fullstep_pre_replay(
            _make_graph_data(self_feeding=False), _make_slot(True), ["a"],
        )
        assert toks is None and self_feed is False
        # Marker untouched: non-self-feeding graphs don't own the buffer.
        assert runner._fullstep_feed_rids == ("a",)

    def test_first_replay_reseeds_then_self_feeds(self):
        runner = _make_runner({"a": GREEDY, "b": GREEDY})
        gd, slot = _make_graph_data(True), _make_slot(True)

        toks, self_feed = runner._fullstep_pre_replay(gd, slot, ["a", "b"])
        assert toks is TOKENS and self_feed is False  # re-seed step
        runner._fullstep_mark_fed(["a", "b"])

        toks, self_feed = runner._fullstep_pre_replay(gd, slot, ["a", "b"])
        assert toks is TOKENS and self_feed is True  # steady state

    def test_eos_trim_composition_change_reseeds(self):
        """Moonshot A3: rid 'b' EOSed; its overshoot step ran and was
        trimmed by the worker. The shrunken tuple must re-seed — the token
        the overshoot step fed for 'b' is never consumed."""
        runner = _make_runner({"a": GREEDY, "b": GREEDY})
        gd, slot = _make_graph_data(True), _make_slot(True)
        runner._fullstep_pre_replay(gd, slot, ["a", "b"])
        runner._fullstep_mark_fed(["a", "b"])

        toks, self_feed = runner._fullstep_pre_replay(gd, slot, ["a"])
        assert toks is TOKENS and self_feed is False

    def test_non_greedy_step_consumes_marker(self):
        """temp>0 batches host-sample; the in-graph argmax in the buffer is
        NOT the emitted token, so even an unchanged tuple must re-seed on
        the following step."""
        runner = _make_runner({"a": GREEDY})
        gd, slot = _make_graph_data(True), _make_slot(True)
        runner._fullstep_pre_replay(gd, slot, ["a"])
        runner._fullstep_mark_fed(["a"])

        runner.sampler._sampling_config = {"a": SAMPLED}
        runner.sampler._config_generation += 1  # set_config bumps this
        toks, self_feed = runner._fullstep_pre_replay(gd, slot, ["a"])
        assert toks is None and self_feed is False
        assert runner._fullstep_feed_rids is None  # marker consumed

        # Back to greedy: first step must re-seed, not self-feed.
        runner.sampler._sampling_config = {"a": GREEDY}
        runner.sampler._config_generation += 1
        toks, self_feed = runner._fullstep_pre_replay(gd, slot, ["a"])
        assert toks is TOKENS and self_feed is False

    def test_repetition_penalty_disqualifies(self):
        runner = _make_runner({"a": PENALIZED})
        toks, self_feed = runner._fullstep_pre_replay(
            _make_graph_data(True), _make_slot(True), ["a"],
        )
        assert toks is None and self_feed is False

    def test_replay_failure_leaves_marker_consumed(self):
        """pre_replay consumes the marker; a replay that raises before
        mark_fed must leave the next step on the re-seed path."""
        runner = _make_runner({"a": GREEDY})
        gd, slot = _make_graph_data(True), _make_slot(True)
        runner._fullstep_pre_replay(gd, slot, ["a"])
        runner._fullstep_mark_fed(["a"])

        runner._fullstep_pre_replay(gd, slot, ["a"])  # replay N+1 ...
        # ... raises before _fullstep_mark_fed. Next step:
        toks, self_feed = runner._fullstep_pre_replay(gd, slot, ["a"])
        assert toks is TOKENS and self_feed is False

    def test_eager_detour_invalidates(self):
        runner = _make_runner({"a": GREEDY})
        gd, slot = _make_graph_data(True), _make_slot(True)
        runner._fullstep_pre_replay(gd, slot, ["a"])
        runner._fullstep_mark_fed(["a"])

        runner.invalidate_fullstep_feed()  # eager batched/sequential step
        toks, self_feed = runner._fullstep_pre_replay(gd, slot, ["a"])
        assert toks is TOKENS and self_feed is False

    def test_capture_without_ingraph_tokens_never_greedy_paths(self):
        """A self-feeding capture whose forward didn't emit
        __fullstep_tokens__ (defensive: sampler gate failed at capture)
        must keep the host sampling + per-step re-seed path."""
        runner = _make_runner({"a": GREEDY})
        gd, slot = _make_graph_data(True), _make_slot(False)
        toks, self_feed = runner._fullstep_pre_replay(gd, slot, ["a"])
        assert toks is None and self_feed is False

    def test_tokens_without_batched_logits_disqualifies(self):
        """Defensive feed/emit consistency: without the __batched_logits__
        sentinel the runner would emit via the per-rid fallback (host
        sample) while the graph fed its own argmax — never fast-path."""
        runner = _make_runner({"a": GREEDY})
        gd = _make_graph_data(True)
        slot = _make_slot(True, with_logits=False)
        toks, self_feed = runner._fullstep_pre_replay(gd, slot, ["a"])
        assert toks is None and self_feed is False
