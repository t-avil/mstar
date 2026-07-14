"""CPU-only reasoning tests for the EAGER FOLD scheduler logic (idea o1).

No GPU, no server. Binds the real MicroScheduler methods to a fake `self` so we
can exercise the chunk-cap gate and the arrival scan with pure Python fakes.
Run: PYTHONPATH=<worktree> <mstar-new venv>/python test_eager_fold_cpu.py
"""
import os
import types

from mstar.worker.micro_scheduler import MicroScheduler, ReadyNodeEntry


class FakeStepMeta(dict):
    pass


class FakeFwdInfo:
    def __init__(self, chunk_len, fwd_index=0, rep_penalty=1.0):
        self.step_metadata = {} if chunk_len is None else {"prefill_chunk_len": chunk_len}
        self.fwd_index = fwd_index
        self.sampling_config = {"decode": types.SimpleNamespace(repetition_penalty=rep_penalty)}


class FakeQueue:
    def __init__(self, ready):
        self._ready = ready  # {rid: [node_name,...]}

    def get_ready_node_names(self):
        return self._ready


class FakeEngine:
    def __init__(self, ready=True):
        self._ready = ready

    def check_ready(self, node, rid, fwd_info):
        return self._ready


class FakeEngineMgr:
    def __init__(self, engine):
        self._e = engine

    def get_engine(self, node):
        return self._e


class FakeWGM:
    def __init__(self, fwd_by_rid, ready):
        self._fwd = fwd_by_rid
        self.per_request_info = {rid: object() for rid in fwd_by_rid}
        self.queues = {"wg0": FakeQueue(ready)}
        self._walk = {rid: "prefill_text" for rid in fwd_by_rid}

    def get_partition_for_node(self, node):
        return "part0"

    def get_fwd_info(self, rid, part):
        return self._fwd[rid]

    def get_graph_walk(self, rid, part):
        return self._walk[rid]


def make_self(fwd_by_rid, ready, engine_ready=True):
    s = types.SimpleNamespace()
    s._MIXED_MAX_CHUNK_TOKENS = MicroScheduler._MIXED_MAX_CHUNK_TOKENS
    s._MIXED_DECODE_WALK = MicroScheduler._MIXED_DECODE_WALK
    s._MIXED_CHUNK_WALK = MicroScheduler._MIXED_CHUNK_WALK
    s._MIXED_VISION_CHUNK_WALK = MicroScheduler._MIXED_VISION_CHUNK_WALK
    s.tp_nodes = set()
    s.tp_rank_zero_nodes = {"decode"}
    s.pending_removes = set()
    s.held_until = {}
    s.engine_manager = FakeEngineMgr(FakeEngine(engine_ready))
    # bind real methods
    s._chunk_entry_passes_gates = MicroScheduler._chunk_entry_passes_gates.__get__(s)
    s._mixed_chunk_walks = MicroScheduler._mixed_chunk_walks.__get__(s)
    s.has_eager_fold_opportunity = MicroScheduler.has_eager_fold_opportunity.__get__(s)
    s._wgm = FakeWGM(fwd_by_rid, ready)
    return s


def test_chunk_cap_gate():
    # A 1024-token chunk: rejected under the captured cap, accepted with eager_ok.
    fwd = {"r1": FakeFwdInfo(chunk_len=1024, fwd_index=0)}
    s = make_self(fwd, {"r1": ["decode"]})
    entry = ReadyNodeEntry("r1", "wg0", "prefill_text")
    assert s._chunk_entry_passes_gates(s._wgm, "decode", "part0", entry) is False
    assert s._chunk_entry_passes_gates(s._wgm, "decode", "part0", entry, eager_ok=True) is True

    # 4096 > default eager cap (2048) -> rejected even with eager_ok.
    fwd2 = {"r1": FakeFwdInfo(chunk_len=4096, fwd_index=0)}
    s2 = make_self(fwd2, {"r1": ["decode"]})
    assert s2._chunk_entry_passes_gates(s2._wgm, "decode", "part0", entry, eager_ok=True) is False

    # rep_penalty != 1.0 -> rejected regardless.
    fwd3 = {"r1": FakeFwdInfo(chunk_len=1024, fwd_index=0, rep_penalty=1.1)}
    s3 = make_self(fwd3, {"r1": ["decode"]})
    assert s3._chunk_entry_passes_gates(s3._wgm, "decode", "part0", entry, eager_ok=True) is False
    print("test_chunk_cap_gate PASS")


def test_eager_peek_flag_off_is_false():
    os.environ.pop("MSTAR_EAGER_FOLD", None)
    fwd = {"r1": FakeFwdInfo(chunk_len=1024, fwd_index=0)}
    s = make_self(fwd, {"r1": ["decode"]})
    assert s.has_eager_fold_opportunity(s._wgm, ("decode", "thinker_decode")) is False
    print("test_eager_peek_flag_off_is_false PASS")


def test_eager_peek_hits_large_new_chunk():
    os.environ["MSTAR_EAGER_FOLD"] = "1"
    os.environ["MSTAR_MIXED_BATCH"] = "1"  # eager_fold_enabled ANDs this
    os.environ.pop("MSTAR_EAGER_FOLD_MAX_CHUNK", None)
    try:
        # 1024-token brand-new prefill_text chunk -> opportunity.
        fwd = {"r1": FakeFwdInfo(chunk_len=1024, fwd_index=0)}
        s = make_self(fwd, {"r1": ["decode"]})
        assert s.has_eager_fold_opportunity(s._wgm, ("decode", "thinker_decode")) is True

        # C=512 fits the captured fold -> NOT an eager opportunity.
        fwd_small = {"r1": FakeFwdInfo(chunk_len=512, fwd_index=0)}
        s_small = make_self(fwd_small, {"r1": ["decode"]})
        assert s_small.has_eager_fold_opportunity(s_small._wgm, ("decode", "thinker_decode")) is False

        # fwd_index != 0 (not a brand-new request) -> rejected.
        fwd_old = {"r1": FakeFwdInfo(chunk_len=1024, fwd_index=3)}
        s_old = make_self(fwd_old, {"r1": ["decode"]})
        assert s_old.has_eager_fold_opportunity(s_old._wgm, ("decode", "thinker_decode")) is False

        # engine not ready -> rejected.
        s_nr = make_self({"r1": FakeFwdInfo(chunk_len=1024, fwd_index=0)},
                         {"r1": ["decode"]}, engine_ready=False)
        assert s_nr.has_eager_fold_opportunity(s_nr._wgm, ("decode", "thinker_decode")) is False

        # wrong decode walk -> rejected.
        s2 = make_self({"r1": FakeFwdInfo(chunk_len=1024, fwd_index=0)}, {"r1": ["decode"]})
        assert s2.has_eager_fold_opportunity(s2._wgm, ("decode", "prefill_text")) is False
    finally:
        os.environ.pop("MSTAR_EAGER_FOLD", None)
        os.environ.pop("MSTAR_MIXED_BATCH", None)
    print("test_eager_peek_hits_large_new_chunk PASS")


if __name__ == "__main__":
    test_chunk_cap_gate()
    test_eager_peek_flag_off_is_false()
    test_eager_peek_hits_large_new_chunk()
    print("ALL EAGER FOLD CPU TESTS PASS")
