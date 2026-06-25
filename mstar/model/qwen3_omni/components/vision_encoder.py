"""Native Qwen3-Omni vision encoder (SigLIP2-style ViT + spatial merge + DeepStack).

From-scratch mstar reimplementation of HF's ``Qwen3OmniMoeVisionEncoder`` whose
submodule names mirror HF exactly, so
``load_weights_from_hf_shards(..., prefix="thinker.visual")`` loads it unchanged.

Why this exists: HF's vision encoder routes attention through the transformers
FA2 wrapper, which is ~3.3 s/call for a single 728-patch image on H100 — the
encoder is the TTFT bottleneck. The native blocks call ``flash_attn_varlen_func``
directly (mirroring ``mstar/model/bagel/components/vit_encoder.py``), collapsing
that to tens of ms, and multiple images batch naturally through ``cu_seqlens``.

The deterministic, cheap (~10 ms) frontend index/position/cu_seqlens helpers are
reused from transformers to guarantee identical patch ordering; the parity test
checks pooler_output, every DeepStack level, and the post-merge token count.

Output contract matches the HF wrapper's consumer (VisionEncoderSubmodule):
``forward(...) -> (vision_embeds[merged_tokens, out_hidden], deepstack[list])``.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
)

from mstar.model.qwen3_omni.components.audio_encoder import varlen_attention


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    orig_q, orig_k = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_q), k_embed.to(orig_k)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids):
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


class VisionPatchEmbed(nn.Module):
    """Patch embedding. Weight is stored as a Conv3d (matching HF's
    ``patch_embed.proj`` so checkpoints load unchanged), but because
    kernel==stride==patch size the convolution is exactly a per-patch linear
    projection. We compute it as an ``F.linear`` matmul instead of nn.Conv3d:
    cuDNN's bf16 Conv3d for this shape is pathologically slow (~3.5 s/image on
    H100) and is the dominant cost of the HF vision encoder. The matmul is
    bit-identical and ~0.1 ms.
    """

    def __init__(self, config):
        super().__init__()
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.embed_dim = config.hidden_size
        ks = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=ks, stride=ks, bias=True)

    def forward(self, x):
        # x: (num_patches, C*tT*pH*pW) already flattened in [C, tT, pH, pW] order,
        # which matches Conv3d weight flattening exactly.
        w = self.proj.weight.reshape(self.embed_dim, -1)
        x = x.reshape(-1, w.shape[1]).to(w.dtype)
        return torch.nn.functional.linear(x, w, self.proj.bias)


class VisionPatchMerger(nn.Module):
    """Mirrors HF Qwen3OmniMoeVisionPatchMerger (incl. use_postshuffle_norm)."""

    def __init__(self, config, use_postshuffle_norm=False):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.ln_q = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.mlp = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, config.out_hidden_size),
        ])

    def forward(self, hidden):
        hidden = self.ln_q(hidden.view(-1, self.hidden_size) if self.use_postshuffle_norm else hidden).view(
            -1, self.hidden_size)
        for layer in self.mlp:
            hidden = layer(hidden)
        return hidden


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scaling = self.head_dim ** -0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, cu_seqlens, max_seqlen, position_embeddings):
        s = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(s, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        o = varlen_attention(q, k, v, cu_seqlens, max_seqlen, self.scaling)
        return self.proj(o.reshape(s, -1))


class VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(self, hidden_states, cu_seqlens, max_seqlen, position_embeddings):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens, max_seqlen, position_embeddings)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class NativeQwen3OmniVisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.num_grid_per_side = int(config.num_position_embeddings ** 0.5)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes

        self.patch_embed = VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.depth)])
        self.merger = VisionPatchMerger(config, use_postshuffle_norm=False)
        self.merger_list = nn.ModuleList([
            VisionPatchMerger(config, use_postshuffle_norm=True)
            for _ in range(len(config.deepstack_visual_indexes))
        ])

    @torch.no_grad()
    def forward(self, pixel_values, grid_thw):
        dtype = self.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.to(dtype)

        bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
            grid_thw, num_grid_per_side=self.num_grid_per_side,
            spatial_merge_size=self.config.spatial_merge_size)
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw)

        hidden_states = self.patch_embed(pixel_values)
        pos_embeds = (self.pos_embed(bilinear_indices) * bilinear_weights[:, :, None]).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        rotary_pos_emb = self.rotary_pos_emb(position_ids)
        seq_len = hidden_states.shape[0]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())

        deepstack_features = []
        for i, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu_seqlens, max_seqlen, position_embeddings)
            if i in self.deepstack_visual_indexes:
                k = self.deepstack_visual_indexes.index(i)
                deepstack_features.append(self.merger_list[k](hidden_states))

        merged = self.merger(hidden_states)
        return merged, deepstack_features
