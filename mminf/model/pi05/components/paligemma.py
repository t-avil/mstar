"""PaliGemma transformer expert for Pi0.5 (prefix processing).

A Gemma-style transformer that integrates with mminf's BatchedCacheManager
for paged KV cache. Used for the prefill graph walk where it processes
the prefix tokens (image + language + state) and writes the KV cache
that the action expert later reads during action generation.

Composed entirely from ``mminf.model.components`` — Gemma RMSNorm
(``gemma_mode=True``), GELU-tanh GatedMLP, and the standard Attention
block. The model-specific piece is just the stack assembly + final norm
+ ``consolidate_fused_weights`` post-load hook.
"""
from __future__ import annotations

import torch
from torch import nn

from mminf.engine.cache_manager import BatchedCacheManager
from mminf.model.components import (
    Attention,
    DecoderLayer,
    GatedMLP,
    RMSNorm,
)
from mminf.model.pi05.config import Pi05Config


def _build_paligemma_layer(
    config: Pi05Config,
    input_hidden_size: int | None = None,
    intermediate_size: int | None = None,
) -> DecoderLayer:
    """One Gemma decoder layer for PaliGemma's prefix expert.

    ``input_hidden_size`` and ``intermediate_size`` are overridable so the
    same layer construction can be reused by the action expert (which has
    a different width but shares K/V dims with PaliGemma).
    """
    h = input_hidden_size if input_hidden_size is not None else config.hidden_size
    inter = intermediate_size if intermediate_size is not None else config.pali_intermediate_size
    return DecoderLayer(
        self_attn=Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            input_hidden_size=h,
            rope_theta=config.rope_theta,
        ),
        mlp=GatedMLP(h, inter, activation="gelu_tanh"),
        input_layernorm=RMSNorm(h, eps=config.rms_norm_eps, gemma_mode=True),
        post_attention_layernorm=RMSNorm(h, eps=config.rms_norm_eps, gemma_mode=True),
    )


class Pi05PaliGemmaExpert(nn.Module):
    """Stack of PaliGemma transformer layers.

    The submodule's input embeddings (image tokens + language tokens +
    state tokens) are passed in directly. This module owns only the
    transformer blocks plus a final Gemma RMSNorm; the embedding table is
    held by the parent submodule and shared with the action expert.
    """

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [_build_paligemma_layer(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma_mode=True)

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
        write_cache: bool = True,
    ) -> torch.Tensor:
        for layer_idx, layer in enumerate(self.layers):
            cache_handle.set_layer_idx(layer_idx)
            query_sequence = layer(
                hidden_states=query_sequence, cache_handle=cache_handle,
            )

        if write_cache:
            cache_handle.advance_seq_lens()

        return self.norm(query_sequence)

    def consolidate_fused_weights(self) -> None:
        """Fuse separate q/k/v and gate/up Linears into single buffers.
        Call after weight loading.
        """
        for layer in self.layers:
            layer.self_attn.consolidate_qkv_weight()
            layer.mlp.consolidate_gate_up_weight()
