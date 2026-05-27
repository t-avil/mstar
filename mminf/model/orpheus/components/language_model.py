"""Orpheus language model: Llama 3.2 3B-style transformer.

Built from the shared transformer components in ``mminf.model.components``.
Orpheus uses standard Llama-style RMSNorm (Llama mode, not Gemma),
SwiGLU MLP, GQA self-attention with Llama-3 RoPE scaling, and no
QK-norm.
"""
from __future__ import annotations

import torch
from torch import nn

from mminf.engine.kv_cache_engine import BatchedCacheManager
from mminf.model.components import (
    Attention,
    DecoderLayer,
    GatedMLP,
    RMSNorm,
)
from mminf.model.orpheus.config import OrpheusModelConfig


def _build_decoder_layer(config: OrpheusModelConfig) -> DecoderLayer:
    return DecoderLayer(
        self_attn=Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            rope_scale=config.rope_scaling["factor"],
            rope_low_freq_factor=config.rope_scaling["low_freq_factor"],
            rope_high_freq_factor=config.rope_scaling["high_freq_factor"],
            rope_old_context_len=config.rope_scaling["original_max_position_embeddings"],
        ),
        mlp=GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation="silu",
        ),
        input_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
        post_attention_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
    )


class OrpheusLanguageModel(nn.Module):
    def __init__(self, config: OrpheusModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [_build_decoder_layer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        for layer_idx, decoder_layer in enumerate(self.layers):
            cache_handle.set_layer_idx(layer_idx)
            query_sequence = decoder_layer(
                hidden_states=query_sequence, cache_handle=cache_handle,
            )
        cache_handle.advance_seq_lens()
        return self.norm(query_sequence)

    def consolidate_fused_weights(self) -> None:
        for layer in self.layers:
            layer.self_attn.consolidate_qkv_weight()
            layer.mlp.consolidate_gate_up_weight()


class OrpheusForCausalLM(nn.Module):
    def __init__(self, config: OrpheusModelConfig):
        super().__init__()
        self.model = OrpheusLanguageModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(query_sequence=query_sequence, cache_handle=cache_handle)

    def consolidate_fused_weights(self) -> None:
        self.model.consolidate_fused_weights()

    def load_weights(self, weights):
        """Load weights from a ``(name, tensor)`` iterable into this model.

        Orpheus's HF Llama checkpoint key naming already matches this
        module's parameter paths, so no name remap or stacked-shard
        routing is needed; each tensor is dispatched to the matching
        parameter via the default copy. After load, the separate
        ``q/k/v_proj`` and ``gate/up_proj`` Linears are fused into
        single buffers for the fast forward path.
        """
        from mminf.model.loader import load_weights_into

        loaded = load_weights_into(
            self, weights,
            skip_predicate=lambda n: "rotary_emb" in n,
        )
        self.consolidate_fused_weights()
        return loaded
