"""Native Qwen3-Omni audio encoder (AuT, Whisper-style).

A from-scratch mstar reimplementation of HF's ``Qwen3OmniMoeAudioEncoder`` that
is decoupled from ``transformers`` at inference time. Submodule attribute names
mirror the HF module exactly so the existing
``load_weights_from_hf_shards(..., prefix="thinker.audio_tower")`` loads it with
no remapping.

Attention is NOT written from scratch: it goes through ``varlen_attention`` (a
flash-attn varlen call with an SDPA fallback) mirroring
``mstar/model/bagel/components/vit_encoder.py``.

The deterministic frontend helpers (chunking / valid-index / cu_seqlens / CNN
output-length) replicate HF's logic bit-for-bit; the parity test guards them.
"""
from __future__ import annotations

import logging
import math
import os
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

logger = logging.getLogger(__name__)

# Output container mirroring the field of HF's BaseModelOutput that downstream
# code reads (``.last_hidden_state``). Defined once at module scope rather than
# as an ad-hoc class re-created per forward call.
AudioEncoderOutput = namedtuple("AudioEncoderOutput", ["last_hidden_state"])

try:
    from flash_attn import flash_attn_varlen_func
    _FLASH_ATTN_AVAILABLE = True
except ImportError:  # pragma: no cover
    flash_attn_varlen_func = None
    _FLASH_ATTN_AVAILABLE = False
    logger.warning("flash_attn unavailable; native AuT falls back to SDPA varlen (slow).")


# --------------------------------------------------------------------------- #
# varlen attention primitive (mirrors bagel vit_encoder.run_attention)
# --------------------------------------------------------------------------- #
def _sdpa_varlen_dense(q, k, v, cu_seqlens, scale):
    # ORIGINAL fallback: builds a dense (total_len, total_len) block-diagonal
    # mask, i.e. O(total_tokens^2) memory/compute. At serving batch sizes the
    # cross-segment terms make encoder TTFT regress (README_qwen3_omni_encoders.md:
    # native large-batch numbers are SDPA-pessimistic). Kept for A/B + parity.
    total_len = q.shape[0]
    seg_ids = torch.zeros(total_len, dtype=torch.int32, device=q.device)
    seg_ids[cu_seqlens[1:-1].long()] = 1
    seg_ids = torch.cumsum(seg_ids, dim=0)
    attn_mask = seg_ids[:, None] == seg_ids[None, :]
    q_b = q.transpose(0, 1).unsqueeze(0)
    k_b = k.transpose(0, 1).unsqueeze(0)
    v_b = v.transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(q_b, k_b, v_b, attn_mask=attn_mask, scale=scale)
    return out.squeeze(0).transpose(0, 1).contiguous()


def _sdpa_varlen_per_segment(q, k, v, cu_seqlens, scale):
    # No-flash-attn varlen path that is LINEAR in batch: run a dense SDPA per
    # image/audio segment (O(sum L_i^2)) instead of one (sum L_i)^2 masked call.
    # Mathematically identical to the block-diagonal mask (attention never
    # crosses segments), but removes the quadratic cross-segment compute/memory
    # that made native large-batch TTFT explode. This is the production varlen
    # path on hardware without flash-attn (this H200).
    cu = cu_seqlens.tolist()
    out = torch.empty_like(q)
    for a, b in zip(cu[:-1], cu[1:]):
        qs = q[a:b].transpose(0, 1).unsqueeze(0)
        ks = k[a:b].transpose(0, 1).unsqueeze(0)
        vs = v[a:b].transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(qs, ks, vs, scale=scale)
        out[a:b] = o.squeeze(0).transpose(0, 1)
    return out


def _sdpa_varlen_padded(q, k, v, cu_seqlens, scale):
    # No-flash varlen via pad-to-max + batched SDPA with a key-padding mask: a
    # SINGLE attention kernel over (n_seg, heads, max_len, head_dim). Best when
    # segments are similar length (e.g. audio windows, which are mostly equal),
    # where per-segment's many tiny kernels lose to one batched call. Wastes work
    # on padding when lengths are very uneven (e.g. mixed-resolution images).
    cu = cu_seqlens.tolist()
    lens = [b - a for a, b in zip(cu[:-1], cu[1:])]
    nseg, max_len = len(lens), max(lens)
    h, d = q.shape[1], q.shape[2]
    qb = q.new_zeros(nseg, max_len, h, d)
    kb = q.new_zeros(nseg, max_len, h, d)
    vb = q.new_zeros(nseg, max_len, h, d)
    mask = torch.zeros(nseg, 1, 1, max_len, device=q.device, dtype=torch.bool)
    for i, (a, b) in enumerate(zip(cu[:-1], cu[1:])):
        n = b - a
        qb[i, :n] = q[a:b]; kb[i, :n] = k[a:b]; vb[i, :n] = v[a:b]
        mask[i, 0, 0, :n] = True
    qb, kb, vb = (t.permute(0, 2, 1, 3) for t in (qb, kb, vb))
    o = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask, scale=scale).permute(0, 2, 1, 3)
    out = torch.empty_like(q)
    for i, (a, b) in enumerate(zip(cu[:-1], cu[1:])):
        out[a:b] = o[i, : b - a]
    return out


def _sdpa_varlen_adaptive(q, k, v, cu_seqlens, scale):
    # Pick dense vs per-segment from segment STRUCTURE (no GPU sync — uses tensor
    # shapes only). Benchmarked on H200 (bench_encoder_fast.py):
    #   * FEW BIG segments (vision: ~728-tok images) -> per-segment always wins
    #     (cross-segment compute saved >> launch overhead of a few kernels).
    #   * MANY TINY segments (audio: ~50-tok windows) -> the dense single kernel
    #     beats hundreds of tiny per-segment kernels (launch-bound) UNTIL the
    #     total grows enough that dense's O(total^2) mask explodes (~bs32).
    # Decision metric M = total^2 / n_seg (= total * mean_seg), a proxy for the
    # dense/per-segment compute ratio: dense ~ total^2, per-segment ~ Σ L_i^2
    # (≈ total*mean_seg for similar lengths), so per-segment wins when M exceeds a
    # launch-cost threshold. Calibrated on H200 (bench_encoder_fast.py): τ≈5e5
    # cleanly separates vision (few big segments → M high → per_segment at every
    # batch) from audio (many tiny windows → M low at small batch → dense, then
    # crosses to per_segment ~bs32 as the dense O(n²) mask explodes). Shapes only,
    # no GPU sync.
    total = q.shape[0]
    n_seg = max(cu_seqlens.shape[0] - 1, 1)
    if (total * total) / n_seg > 500_000:
        return _sdpa_varlen_per_segment(q, k, v, cu_seqlens, scale)
    return _sdpa_varlen_dense(q, k, v, cu_seqlens, scale)


# Backend selectable via env for A/B benchmarking. Default = adaptive: picks
# dense vs per-segment per call from segment structure, so vision (few big
# segments) and audio (many tiny windows) each get their best path.
# MSTAR_VARLEN_BACKEND in {adaptive, per_segment, dense, padded}.
_VARLEN_BACKEND = os.environ.get("MSTAR_VARLEN_BACKEND", "adaptive")
_VARLEN_FALLBACKS = {"adaptive": _sdpa_varlen_adaptive,
                     "per_segment": _sdpa_varlen_per_segment, "dense": _sdpa_varlen_dense,
                     "padded": _sdpa_varlen_padded}


def _sdpa_varlen(q, k, v, cu_seqlens, scale):
    return _VARLEN_FALLBACKS.get(_VARLEN_BACKEND, _sdpa_varlen_per_segment)(
        q, k, v, cu_seqlens, scale)


@torch.compiler.disable
def varlen_attention(q, k, v, cu_seqlens, max_seqlen, scale):
    """q/k/v: (total_tokens, num_heads, head_dim). Bidirectional, packed by cu_seqlens."""
    if _FLASH_ATTN_AVAILABLE:
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=False, softmax_scale=scale,
        )
    return _sdpa_varlen(q, k, v, cu_seqlens, scale)


# --------------------------------------------------------------------------- #
# deterministic frontend helpers (replicate HF exactly)
# --------------------------------------------------------------------------- #
def _feat_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Post-CNN length per HF ``_get_feat_extract_output_lengths`` (module-level)."""
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


def chunk_and_pad_features(input_features, feature_lens, n_window):
    chunk_num = torch.ceil(feature_lens / (n_window * 2)).long()
    chunk_lengths = torch.full((chunk_num.sum(),), n_window * 2, dtype=torch.long,
                               device=feature_lens.device)
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % (n_window * 2)
    chunk_lengths = torch.where(chunk_lengths == 0, n_window * 2, chunk_lengths)
    chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    return padded_feature, chunk_lengths


def get_valid_indices(chunk_lengths: torch.Tensor) -> torch.Tensor:
    feature_lens_after_cnn = _feat_extract_output_lengths(chunk_lengths)
    max_len_after_cnn = feature_lens_after_cnn.max().item()
    mask = torch.arange(max_len_after_cnn, device=chunk_lengths.device) < feature_lens_after_cnn.unsqueeze(1)
    return mask.flatten().nonzero().squeeze(-1)


def get_audio_cu_seqlens(chunk_lengths, feature_lens, n_window_infer, n_window):
    aftercnn_lens = _feat_extract_output_lengths(feature_lens)
    feature_lens_after_cnn = _feat_extract_output_lengths(chunk_lengths)
    max_len_after_cnn = feature_lens_after_cnn.max().item()
    n_window_ratio = n_window_infer // (n_window * 2)
    window_aftercnn = max_len_after_cnn * n_window_ratio
    cu_chunk_lens = [0]
    for cnn_len in aftercnn_lens:
        cnn_len = int(cnn_len)
        cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
        remainder = cnn_len % window_aftercnn
        if remainder != 0:
            cu_chunk_lens += [remainder]
    return torch.tensor(cu_chunk_lens, device=feature_lens.device).cumsum(-1, dtype=torch.int32)


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_inc = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_inc * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )


# --------------------------------------------------------------------------- #
# native modules (weight names == HF)
# --------------------------------------------------------------------------- #
class NativeAudioAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, hidden_states, cu_seqlens, max_seqlen):
        s = hidden_states.shape[0]
        q = self.q_proj(hidden_states).reshape(s, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(s, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(s, self.num_heads, self.head_dim)
        o = varlen_attention(q, k, v, cu_seqlens, max_seqlen, self.scaling)
        return self.out_proj(o.reshape(s, -1))


class NativeAudioEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, activation):
        super().__init__()
        self.self_attn = NativeAudioAttention(embed_dim, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.activation_fn = ACT2FN[activation]

    def forward(self, hidden_states, cu_seqlens, max_seqlen):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens, max_seqlen)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return residual + hidden_states


class NativeQwen3OmniAudioEncoder(nn.Module):
    """Native AuT. Same I/O contract as HF: forward(input_features, feature_lens)
    -> object with ``.last_hidden_state`` of shape (num_audio_tokens, output_dim)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize
        self.num_heads = config.encoder_attention_heads

        self.positional_embedding = SinusoidsPositionEmbedding(config.max_source_positions, d_model)
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        mel_reduced = (((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(config.downsample_hidden_size * mel_reduced, d_model, bias=False)
        self.layers = nn.ModuleList([
            NativeAudioEncoderLayer(d_model, config.encoder_attention_heads,
                                    config.encoder_ffn_dim, config.activation_function)
            for _ in range(config.encoder_layers)
        ])
        self.ln_post = nn.LayerNorm(d_model)
        self.proj1 = nn.Linear(d_model, d_model)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(d_model, config.output_dim)

    @torch.no_grad()
    def forward(self, input_features, feature_lens=None, return_dict=True, **kwargs):
        # ``return_dict``/``**kwargs`` accepted for signature-compatibility with
        # HF's Qwen3OmniMoeAudioEncoder (some callers pass them); the native path
        # always returns the same AudioEncoderOutput regardless.
        assert feature_lens is not None, "native AuT requires feature_lens"
        param_dtype = self.conv2d1.weight.dtype
        input_features = input_features.to(param_dtype)
        padded_feature, chunk_lengths = chunk_and_pad_features(input_features, feature_lens, self.n_window)
        valid_indices = get_valid_indices(chunk_lengths)
        cu_seqlens = get_audio_cu_seqlens(chunk_lengths, feature_lens, self.n_window_infer, self.n_window)

        padded_feature = padded_feature.unsqueeze(1).to(param_dtype)
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            x = F.gelu(self.conv2d1(chunk))
            x = F.gelu(self.conv2d2(x))
            x = F.gelu(self.conv2d3(x))
            padded_embeds.append(x)
        padded_embed = torch.cat(padded_embeds, dim=0)

        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        pos = self.positional_embedding.positional_embedding[: padded_embed.shape[1], :].unsqueeze(0).to(padded_embed.dtype)
        padded_embed = padded_embed + pos
        hidden_states = torch.index_select(padded_embed.reshape(-1, padded_embed.shape[-1]), 0, valid_indices)

        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens, max_seqlen)

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)

        return AudioEncoderOutput(last_hidden_state=hidden_states)
