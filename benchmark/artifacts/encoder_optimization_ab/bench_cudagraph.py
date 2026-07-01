#!/usr/bin/env python
"""MEASURED CUDA-graph A/B for the native vision encoder block-loop (the part a
piecewise CUDA-graph runner would capture). Compares EAGER vs CUDA-graph REPLAY
at FIXED shape, per batch. Uses the dense varlen backend (no CPU sync -> actually
capturable). This is the BEST CASE for graphs (fixed shape, no padding waste);
real variable shapes would force padding on top, only making graphs worse.
"""
import sys, os, time, statistics as st
sys.modules.setdefault("flash_attn", None)
os.environ["MSTAR_VARLEN_BACKEND"] = "dense"   # dense = capturable (no .tolist())
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeVisionEncoderConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    get_vision_bilinear_indices_and_weights, get_vision_position_ids, get_vision_cu_seqlens)
from mstar.model.qwen3_omni.components.vision_encoder import NativeQwen3OmniVisionEncoder
from torch.nn.attention import sdpa_kernel, SDPBackend
DEV, DT = "cuda:0", torch.bfloat16


def frontend(enc, pixel_values, grid_thw):
    """Replicate encoder.forward up to the block loop (runs once, eager)."""
    dtype = enc.patch_embed.proj.weight.dtype
    pixel_values = pixel_values.to(dtype)
    bidx, bw = get_vision_bilinear_indices_and_weights(
        grid_thw, num_grid_per_side=enc.num_grid_per_side, spatial_merge_size=enc.config.spatial_merge_size)
    pos_ids = get_vision_position_ids(grid_thw, enc.spatial_merge_size)
    cu = get_vision_cu_seqlens(grid_thw)
    h = enc.patch_embed(pixel_values)
    pe = (enc.pos_embed(bidx) * bw[:, :, None]).sum(0)
    h = h + pe.to(h.dtype)
    rot = enc.rotary_pos_emb(pos_ids).reshape(h.shape[0], -1)
    emb = torch.cat((rot, rot), dim=-1)
    posemb = (emb.cos(), emb.sin())
    maxs = int((cu[1:] - cu[:-1]).max())
    return h, cu, maxs, posemb


def vis_input(cfg, n):
    rows = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    grid = torch.tensor([[1, 26, 28]] * n, dtype=torch.long, device=DEV)
    npatch = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
    return torch.randn(npatch, rows, device=DEV, dtype=DT), grid


def time_ms(fn, iters=20, warm=5):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


def main():
    cfg = Qwen3OmniMoeVisionEncoderConfig()
    enc = NativeQwen3OmniVisionEncoder(cfg).to(DEV, DT).eval()
    print(f"device={torch.cuda.get_device_name()} torch={torch.__version__}")
    print(f"\n{'batch':>6s} {'eager(blockloop)':>17s} {'cudagraph':>11s} {'speedup':>8s}")
    import torch.nn.functional as F
    import mstar.model.qwen3_omni.components.vision_encoder as VE
    rows = []
    for n in (1, 4, 8):
        pv, g = vis_input(cfg, n)
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):   # MATH = capture-safe
            h, cu, maxs, posemb = frontend(enc, pv, g)
            # Precompute the block-diagonal mask ONCE, OUTSIDE capture (the data-
            # dependent index-scatter that builds it is NOT capture-legal). This is
            # exactly what a piecewise runner would put in a static buffer.
            total = h.shape[0]
            seg = torch.zeros(total, dtype=torch.int32, device=DEV)
            seg[cu[1:-1].long()] = 1
            seg = torch.cumsum(seg, 0)
            fixed_mask = (seg[:, None] == seg[None, :])
            def fixed_attn(q, k, v, cu_seqlens, max_seqlen, scale):
                qb, kb, vb = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
                o = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=fixed_mask, scale=scale)
                return o.squeeze(0).transpose(0, 1)
            VE.varlen_attention = fixed_attn   # patch encoder to use the static mask
            def loop(x):
                for blk in enc.blocks:
                    x = blk(x, cu, maxs, posemb)
                return x
            # eager
            static_x = h.clone()
            e = time_ms(lambda: loop(static_x))
            # cuda graph capture/replay (best case, fixed shape)
            try:
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3): loop(static_x)
                torch.cuda.current_stream().wait_stream(s)
                gph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gph):
                    static_y = loop(static_x)
                c = time_ms(lambda: gph.replay())
                sp = f"{e/c:.2f}x"
            except Exception as ex:
                c, sp = float("nan"), f"ERR:{str(ex)[:14]}"
        rows.append((n, e, c, sp))
        print(f"{n:>6d} {e:17.2f} {c:11.2f} {sp:>8s}")
        torch.cuda.empty_cache()
    print("\n(block-loop only = the piecewise-capturable part; frontend/attention-prep runs once outside)")
    import json
    json.dump([{"batch": n, "eager_ms": e, "graph_ms": c} for n, e, c, _ in rows],
              open("/workspace/autoresearch/bench_artifacts/optimization/cudagraph_ab.json", "w"))


if __name__ == "__main__":
    main()
