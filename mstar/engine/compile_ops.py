"""torch.library custom ops that make otherwise-untraceable engine calls into
opaque-but-traceable graph nodes, so ``torch.compile(fullgraph=False)`` stops
graph-breaking on them.

Background. The compiled Thinker (``forward_batched`` under
``torch.compile(mode="max-autotune-no-cudagraphs", fullgraph=False)`` then a
full manual CUDA-graph capture) calls several ``@torch.compiler.disable``
methods on :class:`BatchedCacheManager` per layer -- most importantly
``run_attention`` (the FlashInfer paged-attention wrapper). Each disabled call
forces a graph break, which is a fusion boundary: Inductor cannot fuse the
projection / norm / residual chains on either side of it, and the break also
fragments neighbouring traces (the same mechanism the pure-torch RMSNorm fix
exploited to remove ~800 breaks for +4.5%).

A registered custom op is the opposite of ``@torch.compiler.disable``: dynamo
keeps tracing the surrounding code into ONE graph and inserts the op as an
opaque node it never looks inside. Same runtime kernels, but the fusion
boundary at the call disappears. This mirrors vLLM's ``unified_attention``
custom op, which fetches its KV cache + attention metadata from a global
forward-context set OUTSIDE the compiled region and takes only plain tensors +
a layer identifier across the op boundary.

The op cannot receive the stateful ``BatchedCacheManager`` as an argument
(dynamo would trace into it, defeating the point, and custom-op schemas only
admit tensors / scalars). So the manager is published to a module global by the
non-compiled driver just before it invokes the compiled forward -- exactly
vLLM's forward-context pattern.

Gated by ``MSTAR_CUSTOM_OPS=1``; default-off, so the baseline boot path is
untouched.
"""
from __future__ import annotations

import os

import torch

# --------------------------------------------------------------------------
# Active-manager registry (the "forward context").
#
# Set by the non-compiled driver right before it calls a compiled forward
# (see cuda_graph_runner capture loop and BatchedCacheManager.plan_attention).
# The op body reads it at eager/capture time only: during CUDA-graph replay the
# op body does not re-run (its kernels were recorded), so the global matters
# solely on the warmup / capture / eager-serve paths, each of which sets it
# just-in-time.
# --------------------------------------------------------------------------
_ACTIVE_MANAGER = None


def set_active_manager(mgr) -> None:
    global _ACTIVE_MANAGER
    _ACTIVE_MANAGER = mgr


def get_active_manager():
    mgr = _ACTIVE_MANAGER
    assert mgr is not None, (
        "compile_ops: no active BatchedCacheManager. A custom op ran without a "
        "manager published by the driver -- set_active_manager was not called "
        "on this forward path."
    )
    return mgr


def custom_ops_enabled() -> bool:
    return os.environ.get("MSTAR_CUSTOM_OPS", "0") == "1"


# --------------------------------------------------------------------------
# mstar::run_attention -- FlashInfer paged attention + KV write.
#
# Wraps BatchedCacheManager.run_attention. The KV-cache write is a side effect
# on a tensor fetched from the active manager (not an op argument), so it is
# invisible to dynamo -- which is fine: the op's OUTPUT feeds o_proj (no DCE),
# and adjacent layers are strictly data-ordered through the residual stream, so
# there is no reorder hazard. Under the manual CUDA-graph capture, execution
# order is program order regardless. q/k/v are not mutated in place (set_kv_cache
# copies out), so mutates_args is empty.
# --------------------------------------------------------------------------
@torch.library.custom_op("mstar::run_attention", mutates_args=())
def run_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_idx: int
) -> torch.Tensor:
    return get_active_manager().run_attention(q=q, k=k, v=v, layer_idx=layer_idx)


@run_attention.register_fake
def _run_attention_fake(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_idx: int
) -> torch.Tensor:
    # Output shape == q shape: [total_tokens, num_q_heads, head_dim].
    return torch.empty_like(q)
