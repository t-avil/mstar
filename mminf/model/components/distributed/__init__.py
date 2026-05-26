"""Tensor-parallel building blocks.

Parallel linears (``ColumnParallelLinear``, ``RowParallelLinear``,
``MergedColumnParallelLinear``, ``QKVParallelLinear``) and the composed
parallel ``Attention`` / ``GatedMLP`` blocks. Each parallel parameter
carries a ``weight_loader`` attribute used by the model-level weight
loader to slice checkpoint tensors per-rank on load.
"""
from mminf.model.components.distributed.attention import ParallelAttention
from mminf.model.components.distributed.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from mminf.model.components.distributed.mlp import ParallelGatedMLP

__all__ = [
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "RowParallelLinear",
    "ParallelAttention",
    "ParallelGatedMLP",
]
