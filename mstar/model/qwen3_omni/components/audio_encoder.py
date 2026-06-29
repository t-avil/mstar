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
from dataclasses import dataclass

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
    # SDPA-only fallback selector (reached only when flash_attn is unavailable;
    # when flash_attn or the flashinfer path is live they are both ~2-7x faster than
    # any SDPA variant and are used instead — see varlen_attention / the backend
    # matrix in exp_audioenc/raw.json). Picks dense vs per-segment from segment
    # STRUCTURE alone (no GPU sync — tensor shapes only).
    #
    # The discriminator is MEAN SEGMENT LENGTH, i.e. the per-segment SDPA kernel
    # regime, NOT total size (the old M = total^2/n_seg metric was inverted: audio
    # at large batch has HIGH M yet still wants dense, while vision at batch 1 has
    # LOWER M yet wants per_segment — a single scalar on M cannot separate them):
    #   * SMALL segments (audio: ~100-tok windows) -> each per-segment SDPA is
    #     launch-bound, so n_seg tiny kernels lose to ONE dense masked kernel even
    #     though dense does n_seg x more attention work. Dense wins audio across
    #     b1..b32 (bench_audio_backend_matrix.py).
    #   * BIG segments (vision: ~728-tok images) -> each per-segment SDPA is
    #     compute-bound and efficient, so per_segment wins and dense wastes the
    #     O(total^2) cross-segment block.
    # Crossover measured on H200 at fixed total~6k: dense wins seg<=~300, per_segment
    # wins seg>=~400 -> threshold 350 sits in the gap and cleanly splits audio (104)
    # from vision (728). A total cap keeps dense's O(total^2) mask from blowing up at
    # extreme batch (small-seg + huge total -> per_segment is the memory-safe choice).
    _DENSE_MEAN_SEG = 350
    _DENSE_TOTAL_CAP = 16384
    total = q.shape[0]
    n_seg = max(cu_seqlens.shape[0] - 1, 1)
    mean_seg = total / n_seg
    if mean_seg < _DENSE_MEAN_SEG and total <= _DENSE_TOTAL_CAP:
        return _sdpa_varlen_dense(q, k, v, cu_seqlens, scale)
    return _sdpa_varlen_per_segment(q, k, v, cu_seqlens, scale)


# --------------------------------------------------------------------------- #
# FlashInfer ragged (no-KV-cache) varlen self-attention.
# This is the intended fast path that is also CUDA-graph-friendly: FlashInfer
# splits attention into a host-side ``plan`` (absorbs the variable-length layout
# from cu_seqlens) and a single fused, branch-free ``run`` kernel — so unlike the
# SDPA fallbacks there is no data-dependent mask build or per-segment loop, and
# the cost is ragged-native (~linear), not the O(n^2) dense mask. Mirrors how the
# rest of M* (Thinker/Talker) already use FlashInfer.
# --------------------------------------------------------------------------- #
try:
    import flashinfer as _flashinfer
    _FLASHINFER_AVAILABLE = True
except Exception:  # pragma: no cover
    _flashinfer = None
    _FLASHINFER_AVAILABLE = False

# Per-device {workspace, wrapper, last cu_seqlens object} so we plan ONCE per
# forward (all encoder layers share the same cu_seqlens object) and re-plan only
# when a new forward arrives.
_FI_STATE: dict = {}


def _fi_pad_hd(t, target):
    # Zero-pad the head_dim. FlashInfer's Hopper prefill kernel hard-asserts
    # head_dim in {64,128,256}; the Qwen3-Omni encoders use 72 (vision) which is
    # unsupported. Zero-padding q/k/v to 128 is EXACT: the padded dims contribute
    # 0 to Q.K^T (zero*zero) and 0 to the output (zero V), so slicing the output
    # back to the real head_dim recovers identical attention. Cost: ~128/72 more
    # attention FLOPs, paid to get the fused, graph-capturable FlashInfer path.
    return t if t.shape[-1] == target else F.pad(t, (0, target - t.shape[-1]))


def make_fi_state(device):
    """Build a fresh, isolated FlashInfer ragged-prefill wrapper + state dict.

    Each CUDA-graph capture key owns one of these so its wrapper is planned
    exactly once and never re-planned — re-planning a shared wrapper mutates
    the buffers a previously-captured graph recorded, silently corrupting that
    graph's replay. The shared per-device ``_FI_STATE`` is fine for eager use
    (it re-plans on every layout change) but unsafe to capture against twice.
    """
    if not _FLASHINFER_AVAILABLE:
        return None
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = _flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, kv_layout="NHD")
    return {"ws": ws, "wrapper": wrapper, "cu_obj": None}


# When set (by the vision encoder's CUDA-graph capture), _flashinfer_varlen
# routes through THIS state instead of the shared per-device _FI_STATE so the
# capture plans/records a dedicated wrapper. Cleared after capture. Replay never
# re-enters Python, so this only matters during warmup+capture.
_fi_override: dict | None = None


def set_fi_override(state):
    global _fi_override
    _fi_override = state


def _flashinfer_varlen(q, k, v, cu_seqlens, scale):
    """q/k/v: (total_tokens, num_heads, head_dim), packed/segmented by cu_seqlens.
    Bidirectional self-attention (qo_indptr == kv_indptr == cu_seqlens)."""
    if not _FLASHINFER_AVAILABLE:
        return _sdpa_varlen_adaptive(q, k, v, cu_seqlens, scale)
    dev = q.device
    if _fi_override is not None:
        st = _fi_override
    else:
        st = _FI_STATE.get(dev)
        if st is None:
            st = make_fi_state(dev)
            _FI_STATE[dev] = st
    wrapper = st["wrapper"]
    H, D = q.shape[1], q.shape[2]
    Dp = 64 if D <= 64 else (128 if D <= 128 else 256)   # FlashInfer-supported head_dim
    qp, kp, vp = (_fi_pad_hd(t, Dp).contiguous() for t in (q, k, v))
    if st["cu_obj"] is not cu_seqlens:          # plan once per forward
        cu = cu_seqlens.to(torch.int32)
        wrapper.plan(cu, cu, H, H, Dp, causal=False, sm_scale=float(scale),
                     q_data_type=q.dtype)
        st["cu_obj"] = cu_seqlens
    out = wrapper.run(qp, kp, vp)
    return out[..., :D].contiguous()


# Backend selectable via env for A/B benchmarking. Default = flashinfer: it is the
# only capture-legal varlen backend, so the encoder CUDA graph (default on, see
# _cuda_graph_enabled) needs it. Falls back gracefully if flashinfer is missing.
# MSTAR_VARLEN_BACKEND in {adaptive, per_segment, dense, padded, flashinfer}.
_VARLEN_BACKEND = os.environ.get("MSTAR_VARLEN_BACKEND", "flashinfer")
_VARLEN_FALLBACKS = {"adaptive": _sdpa_varlen_adaptive,
                     "per_segment": _sdpa_varlen_per_segment, "dense": _sdpa_varlen_dense,
                     "padded": _sdpa_varlen_padded, "flashinfer": _flashinfer_varlen}


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


@dataclass
class _AudioGraph:
    """One captured layer-loop graph for a fixed audio-length layout. ``cu_seqlens``
    is retained so the storage the captured kernels read outlives the graph;
    ``fi_state`` keeps the dedicated FlashInfer wrapper alive (never re-planned)."""
    graph: "torch.cuda.CUDAGraph"
    static_x: torch.Tensor
    out: torch.Tensor
    cu_seqlens: torch.Tensor
    fi_state: dict


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

        # CUDA-graph capture state for the encoder layer-loop tail. One captured
        # graph per distinct audio-length layout (keyed on seq_len + cu_seqlens).
        # Opt-in via MSTAR_ENCODER_CUDA_GRAPH=1; only legal with the FlashInfer
        # varlen backend (SDPA's data-dependent mask build is not capture-safe).
        self._cg_cache: dict = {}
        self._cg_max_keys = int(os.environ.get("MSTAR_ENCODER_CG_MAX_KEYS", "16"))
        self._cg_warmed = False

    def _maybe_cg_warmup(self, input_features, feature_lens):
        """Pre-capture graphs for all configured batch sizes BEFORE serving, so
        lazy capture never lands inside a measured request. Triggered once, on
        the first forward, when MSTAR_ENCODER_CG_WARMUP is a comma list of batch
        sizes. Replicates the first clip's per-segment length to each batch size
        (the graph key is the audio-length layout, identical across clips of the
        same length)."""
        if self._cg_warmed:
            return
        self._cg_warmed = True
        spec = os.environ.get("MSTAR_ENCODER_CG_WARMUP", "1,2,4,8")
        if not spec or not self._cuda_graph_enabled() or feature_lens is None:
            return
        try:
            batch_sizes = sorted({int(b) for b in spec.split(",") if b.strip()})
        except ValueError:
            return
        fl = feature_lens.reshape(-1)
        L0 = int(fl[0])                                   # frames in clip 0
        mel = input_features.shape[0]
        dev, dt = input_features.device, input_features.dtype
        logger.info("audio encoder CUDA-graph warmup: pre-capturing batch sizes %s", batch_sizes)
        for k in batch_sizes:
            try:
                lens = torch.full((k,), L0, dtype=torch.long, device=dev)
                feats = input_features[:, :L0].repeat(1, k) if input_features.shape[1] >= L0 \
                    else torch.zeros(mel, L0 * k, device=dev, dtype=dt)
                self.forward(feats, feature_lens=lens)       # triggers capture for key_k
            except Exception:
                logger.warning("audio CG warmup failed for bs=%d", k, exc_info=True)

    def _cuda_graph_enabled(self) -> bool:
        if os.environ.get("MSTAR_ENCODER_CUDA_GRAPH", "1") not in ("1", "true", "True"):
            return False
        return _FLASHINFER_AVAILABLE and _VARLEN_BACKEND == "flashinfer"

    def _layer_loop_tail(self, hidden_states, cu_seqlens, max_seqlen):
        """The expensive, capture-legal region: encoder layer loop + post-norm /
        projection head. ``cu_seqlens``/``max_seqlen`` depend only on the audio
        length layout, so for a fixed layout this whole region replays as one graph."""
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens, max_seqlen)
        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states

    def _capture_graph(self, key, hidden_states, cu_seqlens, max_seqlen):
        """Capture ``_layer_loop_tail`` for one audio-length ``key`` with a
        dedicated FlashInfer wrapper (planned once, never re-planned). Returns an
        ``_AudioGraph`` or None if capture failed (caller then runs eager)."""
        dev = hidden_states.device
        fi_state = make_fi_state(dev)
        if fi_state is None:
            return None
        static_x = hidden_states.clone()
        set_fi_override(fi_state)
        graph = torch.cuda.CUDAGraph()
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._layer_loop_tail(static_x, cu_seqlens, max_seqlen)
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(graph):
                out = self._layer_loop_tail(static_x, cu_seqlens, max_seqlen)
        except Exception:
            logger.warning("audio encoder CUDA-graph capture failed for key=%s; "
                           "falling back to eager", key, exc_info=True)
            try:
                torch.cuda.synchronize(dev)
            except Exception:
                pass
            return None
        finally:
            set_fi_override(None)
        return _AudioGraph(graph=graph, static_x=static_x, out=out,
                           cu_seqlens=cu_seqlens, fi_state=fi_state)

    @torch.no_grad()
    def forward(self, input_features, feature_lens=None, return_dict=True, **kwargs):
        # ``return_dict``/``**kwargs`` accepted for signature-compatibility with
        # HF's Qwen3OmniMoeAudioEncoder (some callers pass them); the native path
        # always returns the same AudioEncoderOutput regardless.
        assert feature_lens is not None, "native AuT requires feature_lens"
        self._maybe_cg_warmup(input_features, feature_lens)
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

        if self._cuda_graph_enabled():
            # Key on the exact varlen layout: same key => identical cu_seqlens, so
            # the captured graph is valid. Audio content only enters via
            # ``hidden_states``, which we copy into the static buffer.
            key = (int(hidden_states.shape[0]), tuple(cu_seqlens.tolist()))
            ag = self._cg_cache.get(key)
            if ag is None and len(self._cg_cache) < self._cg_max_keys:
                ag = self._capture_graph(key, hidden_states, cu_seqlens, max_seqlen)
                if ag is not None:
                    self._cg_cache[key] = ag
            if ag is not None:
                ag.static_x.copy_(hidden_states)
                ag.graph.replay()
                return AudioEncoderOutput(last_hidden_state=ag.out.clone())

        hidden_states = self._layer_loop_tail(hidden_states, cu_seqlens, max_seqlen)
        return AudioEncoderOutput(last_hidden_state=hidden_states)
