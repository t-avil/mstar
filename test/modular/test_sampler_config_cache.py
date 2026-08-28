"""Sampler reuses its per-batch device config tensors across decode steps.

The cache has to be invisible: every step must see the values it would have
uploaded fresh. These cover the two ways that could break — a config edited
under a live batch, and ``rand_offset``, the one row that is meant to change
every step — plus that a batch keeps its identity across steps and loses it
when membership changes.
"""
import torch

from mstar.utils.sampling import Sampler


def _sampler(rids, **cfg):
    s = Sampler(device=torch.device("cpu"))
    for rid in rids:
        s.add_request(rid)
        s.set_config(rid, vocab_size=16, **cfg)
    return s


def _tensors(sampler, rids):
    configs = [sampler._sampling_config[rid] for rid in rids]
    return sampler._batch_config_tensors(rids, configs, torch.device("cpu"))


def test_a_repeated_batch_reuses_the_same_tensors():
    s = _sampler(["a", "b"])
    first = _tensors(s, ["a", "b"])
    second = _tensors(s, ["a", "b"])
    assert all(x is y for x, y in zip(first, second, strict=True))


def test_rand_offset_advances_once_per_step():
    s = _sampler(["a", "b"])
    _tensors(s, ["a", "b"])
    for step in range(1, 4):
        offsets = _tensors(s, ["a", "b"])[5]
        assert offsets.tolist() == [step, step]


def test_a_rebuilt_batch_starts_from_the_recorded_offsets():
    """A rid's offset is bookkept in _step_offset; a rebuild reads it back."""
    s = _sampler(["a", "b"])
    s._step_offset["a"] = 7
    s._step_offset["b"] = 7
    assert _tensors(s, ["a", "b"])[5].tolist() == [7, 7]


def test_changing_membership_builds_a_new_entry():
    s = _sampler(["a", "b", "c"])
    two = _tensors(s, ["a", "b"])
    three = _tensors(s, ["a", "b", "c"])
    assert three[0].shape[0] == 3
    assert two[0] is not three[0]
    assert _tensors(s, ["a", "b"])[0] is two[0]


def test_order_is_part_of_the_batch_identity():
    s = _sampler(["a", "b"])
    forward = _tensors(s, ["a", "b"])
    assert _tensors(s, ["b", "a"])[0] is not forward[0]


def test_set_config_is_seen_by_the_next_step():
    s = _sampler(["a", "b"], temperature=1.0)
    assert _tensors(s, ["a", "b"])[0].tolist() == [1.0, 1.0]
    s.set_config("a", temperature=0.25)
    assert _tensors(s, ["a", "b"])[0].tolist() == [0.25, 1.0]


def test_the_cached_values_are_what_an_uncached_build_would_upload():
    s = _sampler(["a", "b"], temperature=0.7, top_k=5, top_p=0.9,
                 repetition_penalty=1.2, seed=11)
    cached = _tensors(s, ["a", "b"])
    _tensors(s, ["a", "b"])
    s._batch_cfg_cache.clear()
    fresh = _tensors(s, ["a", "b"])
    for got, want in zip(cached[:5], fresh[:5], strict=True):
        assert torch.equal(got, want)
        assert got.dtype == want.dtype


def test_the_cache_is_bounded():
    s = _sampler([f"r{i}" for i in range(3)])
    for i in range(Sampler._BATCH_CFG_CACHE_MAX + 5):
        rid = f"r{i % 3}"
        s._batch_cfg_cache[(f"synthetic{i}", rid)] = ()
        if len(s._batch_cfg_cache) >= Sampler._BATCH_CFG_CACHE_MAX:
            _tensors(s, ["r0"])
    assert len(s._batch_cfg_cache) <= Sampler._BATCH_CFG_CACHE_MAX
