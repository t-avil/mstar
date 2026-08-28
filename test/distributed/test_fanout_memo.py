"""ShardingConfig.fanout_graph_edges memoizes replicated routing.

Where a signal is replicated, the routing decision is a function of the
sharding config and the (signal, nodes, walks, tp rank) asked about — none of
which change while a request runs. These cover that the replay is the same
answer, that it is taken only where that holds, and that it does not outlive
the topology it came from.
"""
from mstar.distributed.base import ShardDestination, ShardingConfig, ShardingGroup
from mstar.graph.base import GraphEdge, NodeAndGraphWalk, TensorPointerInfo


def _info(rows: int) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[rows, 4], dtype="float32", nbytes=rows * 16, address=0,
        stride=[4, 1], uuid="u", source_session_id="s", source_entity="w0",
    )


def _edge(name: str, rows: int = 8, next_node: str = "B") -> GraphEdge:
    return GraphEdge(next_node=next_node, name=name, tensor_info=[_info(rows)])


def _config(shard_dim=None):
    cfg = ShardingConfig(
        groups=[], tp_enabled_nodes=set(), shard_dim=shard_dim or {},
    )
    cfg.setup({
        NodeAndGraphWalk("A", "decode"): ["w0"],
        NodeAndGraphWalk("B", "decode"): ["w1"],
        NodeAndGraphWalk("C", "decode"): ["w2"],
    })
    return cfg


def _fanout(cfg, edge):
    return cfg.fanout_graph_edges(
        edge, source_node="A", source_graph_walk="decode",
        dest_graph_walk="decode",
    )


def test_the_replayed_route_matches_the_computed_one():
    cfg = _config()
    first = _fanout(cfg, _edge("hidden"))
    second = _fanout(cfg, _edge("hidden"))
    assert cfg._fanout_memo, "a replicated fanout should have been memoized"
    assert first.keys() == second.keys() == {"w1"}
    for worker in first:
        assert second[worker]._shard_dim == first[worker]._shard_dim
        assert second[worker]._total_fanin == first[worker]._total_fanin
        assert second[worker].next_node == first[worker].next_node


def test_the_replay_carries_this_step_tensors_not_the_cached_ones():
    """The route is reused; the payload is always the edge passed in."""
    cfg = _config()
    _fanout(cfg, _edge("hidden", rows=8))
    later = _edge("hidden", rows=32)
    for edge in _fanout(cfg, later).values():
        assert edge.tensor_info is later.tensor_info


def test_a_sharded_signal_is_never_memoized(monkeypatch):
    """Its route depends on the shapes, so a replay could be wrong."""
    cfg = _config(shard_dim={"hidden": 0})
    monkeypatch.setattr(cfg, "compute_fanout", lambda **kw: [
        ShardDestination(worker="w1", full_tensor=True, tp_rank=0),
    ])
    _fanout(cfg, _edge("hidden"))
    assert cfg._fanout_memo == {}


def test_a_partial_fanout_is_never_memoized(monkeypatch):
    """A sliced destination carries offsets computed from this step's dims."""
    cfg = _config()
    monkeypatch.setattr(cfg, "compute_fanout", lambda **kw: [
        ShardDestination(
            worker="w1", full_tensor=False, tp_rank=0,
            start_idxs=[0], end_idxs=[4],
        ),
    ])
    _fanout(cfg, _edge("hidden"))
    assert cfg._fanout_memo == {}


def test_two_routes_for_one_signal_keep_separate_entries():
    cfg = _config()
    to_b = _fanout(cfg, _edge("hidden", next_node="B"))
    to_c = _fanout(cfg, _edge("hidden", next_node="C"))
    assert len(cfg._fanout_memo) == 2
    assert set(to_b) == {"w1"} and set(to_c) == {"w2"}
    assert set(_fanout(cfg, _edge("hidden", next_node="C"))) == {"w2"}


def test_setup_drops_a_memo_from_the_previous_topology():
    cfg = _config()
    _fanout(cfg, _edge("hidden"))
    assert cfg._fanout_memo
    cfg.setup({NodeAndGraphWalk("A", "decode"): ["w0"]})
    assert cfg._fanout_memo == {}


def test_a_cloned_config_starts_empty():
    cfg = _config()
    _fanout(cfg, _edge("hidden"))
    assert cfg.clone_empty()._fanout_memo == {}
