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

import logging
import os
from dataclasses import dataclass

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
)

from mstar.model.qwen3_omni.components.audio_encoder import varlen_attention

logger = logging.getLogger(__name__)


@dataclass
class _VisionGraph:
    """One captured block-loop graph for a fixed grid layout. ``cu_seqlens`` and
    ``position_embeddings`` are retained so the storage the captured kernels read
    outlives the graph; ``fi_state`` keeps the dedicated FlashInfer wrapper alive
    (and never re-planned)."""
    graph: "torch.cuda.CUDAGraph"
    static_x: torch.Tensor
    out_merged: torch.Tensor
    out_deepstack: list
    cu_seqlens: torch.Tensor
    position_embeddings: tuple
    fi_state: dict


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
    cuDNN's *low-precision* Conv3d for this (kernel==stride) shape is
    pathologically slow — measured ~3.2 s/image bf16 on H100, vs ~0.2 ms for the
    same conv in fp32 and ~40 us for this matmul — and is the dominant cost of
    the HF vision encoder. The matmul is exact in fp32; in bf16 it differs from
    the Conv3d only by accumulation-order rounding (<=~1.6e-2 per element), which
    the parity test bounds end-to-end (see qwen3_omni_encoder_parity.py for the
    per-layer profile).
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

        # CUDA-graph capture state for the block-loop tail. One captured graph
        # per distinct grid layout (keyed on seq_len + cu_seqlens). Opt-in via
        # MSTAR_ENCODER_CUDA_GRAPH=1; only legal with the FlashInfer varlen
        # backend (SDPA's data-dependent mask build is not capture-safe).
        self._cg_cache: dict = {}
        self._cg_pool = None
        self._cg_max_keys = int(os.environ.get("MSTAR_ENCODER_CG_MAX_KEYS", "16"))
        self._cg_warmed = False

    def _cuda_graph_enabled(self) -> bool:
        if os.environ.get("MSTAR_ENCODER_CUDA_GRAPH", "1") not in ("1", "true", "True"):
            return False
        import mstar.model.qwen3_omni.components.audio_encoder as AE
        return AE._FLASHINFER_AVAILABLE and AE._VARLEN_BACKEND == "flashinfer"

    def _block_loop_tail(self, hidden_states, cu_seqlens, max_seqlen, position_embeddings):
        """The expensive, capture-legal region: block loop + DeepStack mergers +
        final merger. ``cu_seqlens``/``max_seqlen``/``position_embeddings`` depend
        only on ``grid_thw`` (the spatial layout), so for a fixed layout they are
        constants and this whole region is replayable as one CUDA graph."""
        deepstack_features = []
        for i, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu_seqlens, max_seqlen, position_embeddings)
            if i in self.deepstack_visual_indexes:
                k = self.deepstack_visual_indexes.index(i)
                deepstack_features.append(self.merger_list[k](hidden_states))
        merged = self.merger(hidden_states)
        return merged, deepstack_features

    def _capture_graph(self, key, hidden_states, cu_seqlens, max_seqlen, position_embeddings):
        """Capture ``_block_loop_tail`` for one grid-layout ``key`` with a
        dedicated FlashInfer wrapper (planned once, never re-planned). Returns a
        ``_VisionGraph`` or None if capture failed (caller then runs eager)."""
        import mstar.model.qwen3_omni.components.audio_encoder as AE
        dev = hidden_states.device
        fi_state = AE.make_fi_state(dev)
        if fi_state is None:
            return None
        # Each captured key gets its OWN private graph pool. Sharing one pool
        # across lazily-captured keys trips the caching allocator's
        # ``use_count > 0`` assert because the earlier key's static buffers are
        # still live in the pool when the next capture_begin runs.
        static_x = hidden_states.clone()
        AE.set_fi_override(fi_state)
        graph = torch.cuda.CUDAGraph()
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._block_loop_tail(static_x, cu_seqlens, max_seqlen, position_embeddings)
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graph):
                merged, deepstack = self._block_loop_tail(
                    static_x, cu_seqlens, max_seqlen, position_embeddings)
        except Exception:
            logger.warning("vision encoder CUDA-graph capture failed for key=%s; "
                           "falling back to eager", key, exc_info=True)
            # A capture that throws inside capture_begin/capture can leave the
            # current stream stuck in capture mode (so later eager ops fail with
            # "Offset increment outside graph capture"). Reset the device's
            # capture state defensively.
            try:
                torch.cuda.synchronize(dev)
            except Exception:
                pass
            return None
        finally:
            AE.set_fi_override(None)
        # Keep cu_seqlens / position_embeddings alive: the captured kernels read
        # them, so their storage must outlive the graph.
        return _VisionGraph(
            graph=graph, static_x=static_x, out_merged=merged, out_deepstack=deepstack,
            cu_seqlens=cu_seqlens, position_embeddings=position_embeddings, fi_state=fi_state)

    def _maybe_cg_warmup(self, pixel_values, grid_thw):
        """Pre-capture graphs for all configured batch sizes BEFORE serving, so
        lazy capture never lands inside a measured request. Triggered once, on
        the first forward, when MSTAR_ENCODER_CG_WARMUP is a comma list of batch
        sizes (e.g. "1,4,8,16,32"). Uses the first image of the current request
        as the per-image template and replicates it to each batch size -- valid
        because the graph key is the grid layout, identical across images. The
        cost (a handful of encoder forwards) is paid once at startup instead of
        repeatedly in the hot path."""
        if self._cg_warmed:
            return
        self._cg_warmed = True
        spec = os.environ.get("MSTAR_ENCODER_CG_WARMUP", "1,2,4,8")
        if not spec or not self._cuda_graph_enabled():
            return
        try:
            batch_sizes = sorted({int(b) for b in spec.split(",") if b.strip()})
        except ValueError:
            return
        g = grid_thw if grid_thw.dim() == 2 else grid_thw.unsqueeze(0)
        g0 = g[:1]
        n0 = int(g0[0, 0] * g0[0, 1] * g0[0, 2])    # patch rows for image 0
        pv0 = pixel_values[:n0]
        logger.info("vision encoder CUDA-graph warmup: pre-capturing batch sizes %s", batch_sizes)
        for k in batch_sizes:
            try:
                self.forward(pv0.repeat(k, 1), g0.repeat(k, 1))   # triggers capture for key_k
            except Exception:
                logger.warning("vision CG warmup failed for bs=%d", k, exc_info=True)

    @torch.no_grad()
    def forward(self, pixel_values, grid_thw):
        self._maybe_cg_warmup(pixel_values, grid_thw)
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

        if self._cuda_graph_enabled():
            # Key on the exact varlen layout: same key => identical cu_seqlens /
            # position_embeddings, so the captured graph is valid. Pixel values
            # only enter via ``hidden_states``, which we copy into the static buf.
            key = (int(seq_len), tuple(cu_seqlens.tolist()))
            vg = self._cg_cache.get(key)
            if vg is None and len(self._cg_cache) < self._cg_max_keys:
                vg = self._capture_graph(
                    key, hidden_states, cu_seqlens, max_seqlen, position_embeddings)
                if vg is not None:
                    self._cg_cache[key] = vg
            if vg is not None:
                vg.static_x.copy_(hidden_states)
                vg.graph.replay()
                return (vg.out_merged.clone(),
                        [d.clone() for d in vg.out_deepstack])

        return self._block_loop_tail(hidden_states, cu_seqlens, max_seqlen, position_embeddings)
