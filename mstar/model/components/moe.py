"""Mixture-of-Experts blocks.

Top-K router + dispatch path for the standard fused-expert checkpoint
layout (``experts.gate_up_proj`` and ``experts.down_proj`` packed as
``(num_experts, ...)`` parameters). Two block flavors:

* :class:`SparseMoeBlock` — Top-K MoE with no shared expert (e.g. the
  Qwen3-Omni Thinker text backbone).
* :class:`SparseMoeBlockWithSharedExpert` — Top-K MoE plus a shared
  expert with sigmoid gating (e.g. the Qwen3-Omni Talker text backbone).
  The shared expert is passed in as an ``nn.Module`` so callers can pick
  any MLP shape they need.

Parallel (TP-aware) variants:

* :class:`ParallelSparseMoeBlock`
* :class:`ParallelSparseMoeBlockWithSharedExpert`

When the optional ``sgl-kernel`` dependency is installed and inputs are
on CUDA, dispatch goes through the Triton fused-MoE kernel in
:mod:`mstar.utils.fused_moe`; otherwise it falls back to the naive
per-expert loop in :func:`dispatch_experts_fused`.
"""
from __future__ import annotations

import logging
import os

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import TPCommGroup
from mstar.distributed.utils import divide

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Env-gated MoE kernel selection
# ---------------------------------------------------------------------------
#
# MSTAR_FUSED_MOE -- tri-state override of the fused-vs-naive expert dispatch:
#   unset / "auto" : use the fused Triton grouped-GEMM kernel whenever it is
#                    importable AND inputs are on CUDA (current default), else
#                    the naive per-expert loop. Behavior is unchanged.
#   "0"/"off"/...  : force the naive per-expert loop even on CUDA. Used to
#                    produce the parity baseline and to bisect kernel issues.
#   "1"/"on"/...   : force the fused kernel; raises if it is unavailable
#                    (e.g. sgl-kernel missing or input on CPU) so a
#                    misconfigured fast-path fails loud instead of silently
#                    running the slow loop.
#
# MSTAR_MOE_EXPERT_PARALLEL -- DESIGN ONLY (default OFF). Recognized here so the
#   flag is discoverable and logged, but expert parallelism across GPUs is not
#   yet wired; the current multi-GPU path is tensor-parallel (see
#   ParallelSparseMoeBlock._dispatch_tp and DESIGN_moe.md). Setting it logs a
#   one-time warning and otherwise has no effect.

_TRUTHY = ("1", "true", "yes", "on")
_FALSEY = ("0", "false", "no", "off")


def _fused_moe_mode() -> str:
    """Return 'auto' (default), 'on', or 'off' from ``MSTAR_FUSED_MOE``."""
    raw = os.environ.get("MSTAR_FUSED_MOE")
    if raw is None:
        return "auto"
    v = raw.strip().lower()
    if v in _TRUTHY:
        return "on"
    if v in _FALSEY:
        return "off"
    if v == "auto":
        return "auto"
    logger.warning("Unrecognized MSTAR_FUSED_MOE=%r; treating as 'auto'.", raw)
    return "auto"


_EP_WARNED = False


def _expert_parallel_requested() -> bool:
    """Whether ``MSTAR_MOE_EXPERT_PARALLEL`` is set (design-only; warns once)."""
    global _EP_WARNED
    raw = os.environ.get("MSTAR_MOE_EXPERT_PARALLEL")
    requested = raw is not None and raw.strip().lower() in _TRUTHY
    if requested and not _EP_WARNED:
        _EP_WARNED = True
        logger.warning(
            "MSTAR_MOE_EXPERT_PARALLEL is set but expert parallelism is not yet "
            "implemented; the multi-GPU path remains tensor-parallel. See "
            "DESIGN_moe.md for the proposed EP design.",
        )
    return requested


# Optional fused Triton MoE path. Imports succeed only on CUDA boxes
# with sgl-kernel installed; any import failure (including the final
# moe_align_block_size call) is treated as "fused path unavailable".
try:
    from mstar.utils.fused_moe import fused_experts as _fused_experts
    from mstar.utils.fused_moe.align import has_sgl_kernel

    _HAS_FUSED = has_sgl_kernel()
except Exception as e:  # pragma: no cover -- exercised only when optional dep missing
    _fused_experts = None
    _HAS_FUSED = False
    logger.warning(f"Could not load fused MoE kernel: {e}")


class TopKRouter(nn.Module):
    """Softmax top-k router shared by all MoE blocks.

    Args:
        hidden_size: input hidden dimension.
        num_experts: total number of routed experts.
        num_experts_per_tok: number of experts each token is dispatched
            to (top-k).
        norm_topk_prob: if True, renormalize the top-k probabilities so
            they sum to 1.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.top_k = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.weight = nn.Parameter(torch.zeros(num_experts, hidden_size))

    def forward(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            router_logits: ``(tokens, num_experts)`` softmax distribution.
            routing_weights: ``(tokens, top_k)`` top-k probabilities
                (optionally renormalized).
            selected_experts: ``(tokens, top_k)`` int64 indices.
        """
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_logits = F.linear(hidden_states, self.weight)
        router_logits = F.softmax(router_logits, dtype=torch.float, dim=-1)

        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)

        routing_weights = router_top_value.to(router_logits.dtype)
        return router_logits, routing_weights, router_indices


def dispatch_experts_fused(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """Naive per-expert dispatch using the fused HF checkpoint layout.

    Used as a fallback when the Triton fused-MoE kernel isn't available.
    Loops over the experts that received any tokens and runs SwiGLU per
    expert.

    Args:
        hidden_states: ``(tokens, hidden_size)``.
        gate_up_proj: ``(num_experts, 2 * moe_intermediate_size, hidden_size)``.
        down_proj: ``(num_experts, hidden_size, moe_intermediate_size)``.
        selected_experts: ``(tokens, top_k)`` int64.
        routing_weights: ``(tokens, top_k)`` float.
    """
    final_hidden_states = torch.zeros_like(hidden_states)

    with torch.no_grad():
        # one-hot mask over expert dim: (num_experts, top_k, tokens)
        expert_mask = F.one_hot(selected_experts, num_classes=num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx_t in expert_hit:
        expert_idx = expert_idx_t[0]
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]

        gate_up = torch.mm(current_state, gate_up_proj[expert_idx].T)
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden_states = torch.mm(F.silu(gate) * up, down_proj[expert_idx].T)
        current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype),
        )

    return final_hidden_states


def _dispatch(
    hidden_states: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_experts: int,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """Pick fused-Triton or the naive loop, honoring ``MSTAR_FUSED_MOE``.

    Default ('auto'): fused when importable and inputs are on CUDA, else
    naive. 'off' forces naive; 'on' forces fused and raises if unavailable.
    """
    mode = _fused_moe_mode()
    can_fuse = _HAS_FUSED and hidden_states.is_cuda

    if mode == "on" and not can_fuse:
        raise RuntimeError(
            "MSTAR_FUSED_MOE=on but the fused MoE kernel is unavailable "
            f"(_HAS_FUSED={_HAS_FUSED}, is_cuda={hidden_states.is_cuda}). "
            "Install sgl-kernel and run on CUDA, or unset the flag."
        )

    use_fused = can_fuse and mode != "off"
    if use_fused:
        return _fused_experts(
            hidden_states, gate_up_proj, down_proj,
            routing_weights, selected_experts,
        )
    return dispatch_experts_fused(
        hidden_states, gate_up_proj, down_proj,
        num_experts, selected_experts, routing_weights,
    )


class SparseMoeBlock(nn.Module):
    """Top-K sparse MoE with fused expert weights, no shared expert.

    Expert weights match the HF fused checkpoint layout:
      - ``experts.gate_up_proj``: ``(num_experts, 2 * moe_intermediate_size, hidden_size)``
      - ``experts.down_proj``: ``(num_experts, hidden_size, moe_intermediate_size)``
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        self.gate = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
        )
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size)
        )
        self.experts.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, moe_intermediate_size)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat = hidden_states.view(-1, hidden_dim).contiguous()
        _, routing_weights, selected_experts = self.gate(flat)
        out = _dispatch(
            flat, self.experts.gate_up_proj, self.experts.down_proj,
            self.num_experts, selected_experts, routing_weights,
        )
        return out.view(input_shape)


class SparseMoeBlockWithSharedExpert(nn.Module):
    """Top-K sparse MoE with a shared expert + sigmoid gating.

    Final output is::

        out = routed(x) + sigmoid(shared_gate(x)) * shared_expert(x)

    The shared expert is supplied by the caller (any ``nn.Module``
    matching the ``hidden_size → hidden_size`` interface). The routed
    path uses the same fused checkpoint layout as :class:`SparseMoeBlock`.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        shared_expert: nn.Module,
        norm_topk_prob: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        self.gate = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
        )
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * moe_intermediate_size, hidden_size)
        )
        self.experts.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, moe_intermediate_size)
        )
        self.shared_expert = shared_expert
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat = hidden_states.view(-1, hidden_dim).contiguous()

        shared = self.shared_expert(flat)

        _, routing_weights, selected_experts = self.gate(flat)
        routed = _dispatch(
            flat, self.experts.gate_up_proj, self.experts.down_proj,
            self.num_experts, selected_experts, routing_weights,
        )
        shared_gate = torch.sigmoid(self.shared_expert_gate(flat))
        return (routed + shared_gate * shared).view(input_shape)


# ---------------------------------------------------------------------------
# TP-aware MoE blocks
# ---------------------------------------------------------------------------


def _gate_up_weight_loader(
    tp_rank: int, tp_size: int, full_inter: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: int | str | None = None,
):
    """Load one expert's gate_proj or up_proj into the fused gate_up_proj param.

    ``loaded_shard_id`` is ``"gate:N"`` or ``"up:N"`` where N is the
    expert index.  ``loaded_weight`` shape is ``(inter, hidden)`` — a
    single expert's projection.  The TP rank's slice is taken and copied
    into the correct position in ``param`` which has shape
    ``(E, 2*shard_inter, hidden)``.
    """
    assert loaded_shard_id is not None
    kind, expert_str = loaded_shard_id.split(":")
    expert_idx = int(expert_str)
    shard_inter = divide(full_inter, tp_size)
    start = tp_rank * shard_inter
    tp_slice = loaded_weight[start:start + shard_inter, :]
    if kind == "gate":
        param.data[expert_idx, :shard_inter, :] = tp_slice
    else:
        param.data[expert_idx, shard_inter:, :] = tp_slice


def _down_proj_weight_loader(
    tp_rank: int, tp_size: int, full_inter: int,
    param: nn.Parameter, loaded_weight: torch.Tensor,
    loaded_shard_id: int | str | None = None,
):
    """Load one expert's down_proj into the fused down_proj param.

    ``loaded_shard_id`` is ``"down:N"``.  ``loaded_weight`` shape is
    ``(hidden, inter)``.  The TP rank's column slice is taken.
    """
    assert loaded_shard_id is not None
    expert_idx = int(loaded_shard_id.split(":")[1])
    shard_inter = divide(full_inter, tp_size)
    start = tp_rank * shard_inter
    param.data[expert_idx, :, :] = loaded_weight[:, start:start + shard_inter]


class ParallelSparseMoeBlock(nn.Module):
    """TP-aware Top-K sparse MoE.

    When ``tp_size == 1``, the forward is identical to
    :class:`SparseMoeBlock` (full fused kernel, no communication).
    When ``tp_size > 1``, expert weights are sharded along the
    intermediate dimension and an all-reduce is inserted between the
    down-projection GEMM and the top-k sum-reduce.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        norm_topk_prob: bool = True,
        comm_group: TPCommGroup | None = None,
    ) -> None:
        super().__init__()
        if comm_group is None:
            comm_group = TPCommGroup.trivial()
        self.comm_group = comm_group
        tp_size = comm_group.world_size
        tp_rank = comm_group.rank
        if tp_size > 1:
            _expert_parallel_requested()  # design-only: warns if EP requested

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        self.gate = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
        )

        shard_inter = divide(moe_intermediate_size, tp_size)
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * shard_inter, hidden_size)
        )
        self.experts.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, shard_inter)
        )
        self._attach_weight_loaders(tp_rank, tp_size, moe_intermediate_size)

    def _attach_weight_loaders(self, tp_rank: int, tp_size: int, full_inter: int):
        from functools import partial

        self.experts.gate_up_proj.weight_loader = partial(
            _gate_up_weight_loader, tp_rank, tp_size, full_inter,
        )
        self.experts.down_proj.weight_loader = partial(
            _down_proj_weight_loader, tp_rank, tp_size, full_inter,
        )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders(
            self.comm_group.rank, self.comm_group.world_size,
            self.moe_intermediate_size,
        )
        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat = hidden_states.view(-1, hidden_dim).contiguous()
        _, routing_weights, selected_experts = self.gate(flat)

        if self.comm_group.world_size == 1:
            out = _dispatch(
                flat, self.experts.gate_up_proj, self.experts.down_proj,
                self.num_experts, selected_experts, routing_weights,
            )
        else:
            out = self._dispatch_tp(flat, routing_weights, selected_experts)
        return out.view(input_shape)

    def _dispatch_tp(
        self, flat: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        from mstar.utils.fused_moe import fused_experts, moe_sum_reduce_triton

        # (tokens, top_k, hidden) — partial results before reduce
        cache3 = fused_experts(
            flat, self.experts.gate_up_proj, self.experts.down_proj,
            routing_weights, selected_experts, reduce_results=False,
        )
        self.comm_group.all_reduce(cache3)
        output = torch.empty_like(flat)
        moe_sum_reduce_triton(cache3, output, routed_scaling_factor=1.0)
        return output


class ParallelSparseMoeBlockWithSharedExpert(nn.Module):
    """TP-aware Top-K sparse MoE with a shared expert + sigmoid gating.

    The shared expert should be a ``ParallelGatedMLP`` constructed with
    the same ``comm_group`` so its all-reduce is handled internally.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        shared_expert: nn.Module,
        norm_topk_prob: bool = False,
        comm_group: TPCommGroup | None = None,
    ) -> None:
        super().__init__()
        if comm_group is None:
            comm_group = TPCommGroup.trivial()
        self.comm_group = comm_group
        tp_size = comm_group.world_size
        tp_rank = comm_group.rank
        if tp_size > 1:
            _expert_parallel_requested()  # design-only: warns if EP requested

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        self.gate = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            norm_topk_prob=norm_topk_prob,
        )

        shard_inter = divide(moe_intermediate_size, tp_size)
        self.experts = nn.Module()
        self.experts.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * shard_inter, hidden_size)
        )
        self.experts.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, shard_inter)
        )
        self.shared_expert = shared_expert
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)
        self._attach_weight_loaders(tp_rank, tp_size, moe_intermediate_size)

    def _attach_weight_loaders(self, tp_rank: int, tp_size: int, full_inter: int):
        from functools import partial

        self.experts.gate_up_proj.weight_loader = partial(
            _gate_up_weight_loader, tp_rank, tp_size, full_inter,
        )
        self.experts.down_proj.weight_loader = partial(
            _down_proj_weight_loader, tp_rank, tp_size, full_inter,
        )

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        self._attach_weight_loaders(
            self.comm_group.rank, self.comm_group.world_size,
            self.moe_intermediate_size,
        )
        return result

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        flat = hidden_states.view(-1, hidden_dim).contiguous()

        shared = self.shared_expert(flat)

        _, routing_weights, selected_experts = self.gate(flat)
        if self.comm_group.world_size == 1:
            routed = _dispatch(
                flat, self.experts.gate_up_proj, self.experts.down_proj,
                self.num_experts, selected_experts, routing_weights,
            )
        else:
            routed = self._dispatch_tp(flat, routing_weights, selected_experts)
        shared_gate = torch.sigmoid(self.shared_expert_gate(flat))
        return (routed + shared_gate * shared).view(input_shape)

    def _dispatch_tp(
        self, flat: torch.Tensor,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        from mstar.utils.fused_moe import fused_experts, moe_sum_reduce_triton

        cache3 = fused_experts(
            flat, self.experts.gate_up_proj, self.experts.down_proj,
            routing_weights, selected_experts, reduce_results=False,
        )
        self.comm_group.all_reduce(cache3)
        output = torch.empty_like(flat)
        moe_sum_reduce_triton(cache3, output, routed_scaling_factor=1.0)
        return output
