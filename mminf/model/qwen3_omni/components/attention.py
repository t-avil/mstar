"""Qwen3-Omni attention: shared ``Attention`` + 3D MRoPE override.

Reuses the shared ``Attention`` (which already supports per-head QK-norm
via ``qk_norm=True``). The Qwen3-specific piece is the 3D MRoPE path
used by the Thinker — for ``use_mrope=True`` the RoPE call goes through
``apply_interleaved_mrope`` with externally provided ``cos_sin_3d``
instead of the cache handle. Talker uses standard 1D RoPE
(``use_mrope=False``) and inherits the parent's ``_apply_rope`` as-is.

Follows the same shape conventions as the shared attention:
  q: [tokens, num_heads, head_dim]
  k: [tokens, num_kv_heads, head_dim]
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from mminf.engine.cache_manager import BatchedCacheManager
from mminf.model.components import Attention


class Qwen3OmniAttention(Attention):
    """Multi-head attention with QK-norm and pluggable 1D / 3D RoPE.

    When ``use_mrope=True`` (Thinker) the forward expects a
    ``cos_sin_3d`` tuple of ``(cos, sin)`` tensors and applies
    ``apply_interleaved_mrope``. When False (Talker), the parent's
    standard cache-handle RoPE is used.

    Args mirror the original Qwen3-Omni attention so existing call sites
    don't change.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1e-6,
        use_mrope: bool = False,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_bias=False,
            o_bias=False,
            qk_norm=True,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
        )
        self.use_mrope = use_mrope

    def set_qkv_proj_weight(self) -> None:
        """Back-compat alias for ``consolidate_qkv_weight``."""
        self.consolidate_qkv_weight()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
        cos_sin_3d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mrope_section: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [tokens, hidden_size]
            cache_handle: BatchedCacheManager with pre-planned attention.
            cos_sin_3d: (cos, sin) tuple each [tokens, head_dim] for 3D
                MRoPE. Required when ``use_mrope=True``.
            mrope_section: section sizes for interleaved 3D MRoPE
                (consumed externally before this call to build
                ``cos_sin_3d``); accepted for API compatibility.
        """
        num_tokens = hidden_states.shape[0]

        # Cast hidden_states to the weights' dtype so F.linear/matmul
        # don't complain on mixed-precision inputs. Restore on return.
        orig_dtype = hidden_states.dtype
        if self.q_proj is not None:
            target_dtype = self.q_proj.weight.dtype
        else:
            target_dtype = self.qkv_proj_weight.dtype
        hidden_states = hidden_states.to(target_dtype)

        q, k, v = self._project_qkv(hidden_states)
        q, k = self._apply_qk_norm(q, k)

        if self.use_mrope and cos_sin_3d is not None:
            from mminf.model.qwen3_omni.components.rope import apply_interleaved_mrope
            cos, sin = cos_sin_3d
            q, k = apply_interleaved_mrope(q, k, cos, sin)
        else:
            q, k = self._apply_rope(q, k, cache_handle)

        attn_output = cache_handle.run_attention(q=q, k=k, v=v)
        attn_output = attn_output.reshape(num_tokens, self.num_heads * self.head_dim)
        return self.o_proj(attn_output).to(orig_dtype)
