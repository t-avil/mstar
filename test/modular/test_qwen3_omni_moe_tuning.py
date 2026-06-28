"""CPU-safe tests for the env-gated MoE optimizations.

These run without CUDA or sgl-kernel. They cover:

* ``MSTAR_FUSED_MOE`` mode parsing (auto / on / off).
* ``MSTAR_MOE_EXPERT_PARALLEL`` design-only recognition.
* The ``MSTAR_FUSED_MOE_TUNED`` tuned-tile-config loader: default-off,
  device + shape lookup, and nearest-batch-bucket selection.
* A pure-Python reference of the token->expert permutation that the fused
  kernel's ``moe_align_block_size`` performs, checked for equivalence
  against a naive per-expert grouping. This validates the routing /
  permutation logic that the GPU kernel relies on, on CPU.

The numerical fused-vs-naive parity (cos-sim) gate requires CUDA and lives
in ``test_qwen3_omni_fused_moe.py``; this file only covers the CPU-safe
logic so it can run in CI without a GPU.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import torch

from mstar.model.components import moe as moe_mod
from mstar.utils.fused_moe import tuning


# ------------------------------------------------------------------
# MSTAR_FUSED_MOE mode parsing
# ------------------------------------------------------------------


def test_fused_moe_mode_default_auto(monkeypatch):
    monkeypatch.delenv("MSTAR_FUSED_MOE", raising=False)
    assert moe_mod._fused_moe_mode() == "auto"


def test_fused_moe_mode_on_off(monkeypatch):
    for val in ("1", "true", "YES", "on"):
        monkeypatch.setenv("MSTAR_FUSED_MOE", val)
        assert moe_mod._fused_moe_mode() == "on"
    for val in ("0", "false", "No", "off"):
        monkeypatch.setenv("MSTAR_FUSED_MOE", val)
        assert moe_mod._fused_moe_mode() == "off"


def test_fused_moe_mode_garbage_is_auto(monkeypatch):
    monkeypatch.setenv("MSTAR_FUSED_MOE", "banana")
    assert moe_mod._fused_moe_mode() == "auto"


def test_fused_moe_on_without_cuda_raises(monkeypatch):
    """mode=on must fail loud on CPU rather than silently run the slow loop."""
    monkeypatch.setenv("MSTAR_FUSED_MOE", "on")
    hidden = torch.randn(4, 16)  # CPU tensor
    w1 = torch.randn(2, 8, 16)
    w2 = torch.randn(2, 16, 4)
    sel = torch.zeros(4, 1, dtype=torch.int64)
    rw = torch.ones(4, 1)
    try:
        moe_mod._dispatch(hidden, w1, w2, 2, sel, rw)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_fused_moe_off_runs_naive_on_cpu(monkeypatch):
    monkeypatch.setenv("MSTAR_FUSED_MOE", "off")
    hidden = torch.randn(4, 16)
    w1 = torch.randn(2, 8, 16) * 0.02
    w2 = torch.randn(2, 16, 4) * 0.02
    sel = torch.tensor([[0], [1], [0], [1]], dtype=torch.int64)
    rw = torch.ones(4, 1)
    out = moe_mod._dispatch(hidden, w1, w2, 2, sel, rw)
    assert out.shape == hidden.shape
    assert torch.isfinite(out).all()


# ------------------------------------------------------------------
# MSTAR_MOE_EXPERT_PARALLEL design-only recognition
# ------------------------------------------------------------------


def test_expert_parallel_recognized(monkeypatch):
    moe_mod._EP_WARNED = False
    monkeypatch.setenv("MSTAR_MOE_EXPERT_PARALLEL", "1")
    assert moe_mod._expert_parallel_requested() is True
    monkeypatch.delenv("MSTAR_MOE_EXPERT_PARALLEL", raising=False)
    assert moe_mod._expert_parallel_requested() is False


# ------------------------------------------------------------------
# MSTAR_FUSED_MOE_TUNED config loader
# ------------------------------------------------------------------


def test_tuned_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MSTAR_FUSED_MOE_TUNED", raising=False)
    assert tuning.tuned_configs_enabled() is False
    # device_name forced so the result depends only on the flag.
    assert tuning.load_tuned_config(M=1, E=128, n_inter=768,
                                    device_name="NVIDIA_H200") is None


def test_tuned_thinker_h200_lookup(monkeypatch):
    monkeypatch.setenv("MSTAR_FUSED_MOE_TUNED", "1")
    cfg = tuning.load_tuned_config(M=1, E=128, n_inter=768,
                                   device_name="NVIDIA_H200")
    assert cfg is not None
    for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M"):
        assert key in cfg


def test_tuned_talker_h200_lookup(monkeypatch):
    monkeypatch.setenv("MSTAR_FUSED_MOE_TUNED", "1")
    cfg = tuning.load_tuned_config(M=4, E=128, n_inter=384,
                                   device_name="NVIDIA_H200")
    assert cfg is not None
    assert cfg["BLOCK_SIZE_M"] >= 1


def test_tuned_nearest_bucket(monkeypatch):
    """M=3 should round up to the '4' bucket; M=1e9 clamps to the max bucket."""
    monkeypatch.setenv("MSTAR_FUSED_MOE_TUNED", "1")
    c3 = tuning.load_tuned_config(M=3, E=128, n_inter=768,
                                  device_name="NVIDIA_H200")
    c4 = tuning.load_tuned_config(M=4, E=128, n_inter=768,
                                  device_name="NVIDIA_H200")
    assert c3 == c4
    big = tuning.load_tuned_config(M=10**9, E=128, n_inter=768,
                                   device_name="NVIDIA_H200")
    assert big is not None  # clamped, not crashed


def test_tuned_unknown_device_falls_back(monkeypatch):
    monkeypatch.setenv("MSTAR_FUSED_MOE_TUNED", "1")
    assert tuning.load_tuned_config(M=1, E=128, n_inter=768,
                                    device_name="NVIDIA_MADE_UP") is None


# ------------------------------------------------------------------
# Routing / permutation equivalence (CPU reference for moe_align)
# ------------------------------------------------------------------


def _reference_permute(topk_ids: torch.Tensor, num_experts: int):
    """Group (token, slot) pairs by expert id, stable within an expert.

    Mirrors what ``moe_align_block_size`` does (minus the block padding):
    every (token, top-k slot) is assigned to its expert and the slots are
    laid out contiguously per expert. Returns, per expert, the list of
    flat slot indices ``token * top_k + slot``.
    """
    num_tokens, top_k = topk_ids.shape
    groups = {e: [] for e in range(num_experts)}
    for t in range(num_tokens):
        for s in range(top_k):
            e = int(topk_ids[t, s])
            groups[e].append(t * top_k + s)
    return groups


def test_permutation_partitions_all_slots():
    """Every (token, slot) lands in exactly one expert group; counts match."""
    torch.manual_seed(0)
    num_tokens, top_k, num_experts = 7, 8, 16
    logits = torch.randn(num_tokens, num_experts)
    _, topk_ids = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)

    groups = _reference_permute(topk_ids, num_experts)

    all_slots = sorted(s for g in groups.values() for s in g)
    assert all_slots == list(range(num_tokens * top_k))
    # Per-expert counts equal the number of times that expert appears.
    for e in range(num_experts):
        assert len(groups[e]) == int((topk_ids == e).sum())


def test_topk_router_matches_manual(monkeypatch):
    """TopKRouter softmax+topk+renorm equals a manual reference (CPU)."""
    torch.manual_seed(1)
    hidden, num_experts, top_k = 32, 16, 4
    router = moe_mod.TopKRouter(hidden, num_experts, top_k, norm_topk_prob=True)
    with torch.no_grad():
        router.weight.normal_(std=0.02)
    x = torch.randn(5, hidden)

    _, rw, sel = router(x)

    probs = torch.softmax(x @ router.weight.t(), dim=-1, dtype=torch.float)
    ref_w, ref_i = torch.topk(probs, top_k, dim=-1)
    ref_w = (ref_w / ref_w.sum(dim=-1, keepdim=True)).to(x.dtype)
    assert torch.equal(sel, ref_i)
    torch.testing.assert_close(rw, ref_w)
