#!/usr/bin/env python
"""Prove the warmup pre-capture moves CUDA-graph capture cost OUT of the hot path.

Without warmup, the FIRST forward at each new batch layout pays capture (3 warmup
forwards + capture) -- this is what lands in the measured serving window and made
mid-batch TTFT regress. With MSTAR_ENCODER_CG_WARMUP, all layouts are captured on
the first forward (startup), so every subsequent first-at-bs forward is a pure
replay. We measure the FIRST-forward latency at each bs in both modes."""
import os, sys, time
sys.modules.setdefault("flash_attn", None)
os.environ["MSTAR_VARLEN_BACKEND"] = "flashinfer"
os.environ["MSTAR_ENCODER_CUDA_GRAPH"] = "1"
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeVisionEncoderConfig
from mstar.model.qwen3_omni.components.vision_encoder import NativeQwen3OmniVisionEncoder
import mstar.model.qwen3_omni.components.audio_encoder as AE
DEV, DT = "cuda:0", torch.bfloat16
BS = [1, 4, 8, 16, 32]


def vis_input(cfg, n):
    rows = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    g = torch.tensor([[1, 26, 28]] * n, dtype=torch.long, device=DEV)
    npp = int((g[:, 0] * g[:, 1] * g[:, 2]).sum())
    return torch.randn(npp, rows, device=DEV, dtype=DT), g


def first_forward_ms(enc, pv, g):
    """Latency of ONE forward from a cold cache for this bs (capture if not yet
    captured, else replay)."""
    torch.cuda.synchronize(); t = time.perf_counter()
    enc(pv, grid_thw=g)
    torch.cuda.synchronize(); return (time.perf_counter() - t) * 1000


def main():
    cfg = Qwen3OmniMoeVisionEncoderConfig()
    print(f"device={torch.cuda.get_device_name()}  first-forward latency (ms) at each bs")
    print(f"{'bs':>4} {'no-warmup (capture in path)':>28} {'warmup (pre-captured)':>22}  {'speedup':>8}")

    # --- no warmup: fresh encoder, first forward at each bs pays capture ---
    os.environ.pop("MSTAR_ENCODER_CG_WARMUP", None)
    enc_nw = NativeQwen3OmniVisionEncoder(cfg).to(DEV, DT).eval()
    nw = {}
    with torch.no_grad():
        for n in BS:
            pv, g = vis_input(cfg, n)
            nw[n] = first_forward_ms(enc_nw, pv, g)   # cold -> capture happens here

    # --- warmup: pre-capture all bs on the very first forward, then time ---
    os.environ["MSTAR_ENCODER_CG_WARMUP"] = ",".join(map(str, BS))
    enc_w = NativeQwen3OmniVisionEncoder(cfg).to(DEV, DT).eval()
    with torch.no_grad():
        pv1, g1 = vis_input(cfg, 1)
        t = time.perf_counter(); enc_w(pv1, grid_thw=g1); torch.cuda.synchronize()
        warmup_cost = (time.perf_counter() - t) * 1000   # one-time startup cost
        wm = {}
        for n in BS:
            pv, g = vis_input(cfg, n)
            wm[n] = first_forward_ms(enc_w, pv, g)        # all pre-captured -> replay

    for n in BS:
        print(f"{n:>4} {nw[n]:>28.2f} {wm[n]:>22.2f}  {nw[n]/wm[n]:>7.2f}x")
    print(f"\none-time warmup cost (startup, off the hot path): {warmup_cost:.1f} ms")
    print(f"captured keys after warmup: {len(enc_w._cg_cache)}")


if __name__ == "__main__":
    main()
