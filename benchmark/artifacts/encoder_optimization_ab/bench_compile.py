#!/usr/bin/env python
"""A/B the torch.compile lever on the native encoders (dense SDPA baseline,
no flash-attn). Compares eager vs torch.compile(dynamic=True) on the encoder
forward over a SPREAD of shapes, reporting steady-state ms AND recompile count
(the issue #131 pitfall). Keep-only-if-it-wins.
"""
import sys; sys.modules.setdefault("flash_attn", None)
import torch, time, statistics as st
import torch._dynamo as dyn
DEV, DT = "cuda:0", torch.bfloat16
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoderConfig, Qwen3OmniMoeAudioEncoderConfig)
from mstar.model.qwen3_omni.components.vision_encoder import NativeQwen3OmniVisionEncoder
from mstar.model.qwen3_omni.components.audio_encoder import NativeQwen3OmniAudioEncoder


def measure(fn, r=8, ni=5, nw=3):
    for _ in range(nw): fn()
    torch.cuda.synchronize(); out = []
    for _ in range(r):
        t = time.perf_counter()
        for _ in range(ni): fn()
        torch.cuda.synchronize(); out.append((time.perf_counter() - t) / ni * 1000)
    return out


def vis_input(cfg, n):
    rows = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    grid = torch.tensor([[1, 26, 28]] * n, dtype=torch.long, device=DEV)
    npatch = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
    return torch.randn(npatch, rows, device=DEV, dtype=DT), grid


def run(kind):
    if kind == "VISION":
        cfg = Qwen3OmniMoeVisionEncoderConfig()
        enc = NativeQwen3OmniVisionEncoder(cfg).to(DEV, DT).eval()
        comp = torch.compile(enc, dynamic=True)
        def mk(n):
            pv, g = vis_input(cfg, n)
            return lambda: enc(pv, grid_thw=g)
        def mkc(n):
            pv, g = vis_input(cfg, n)
            return lambda: comp(pv, grid_thw=g)
        unit = "ms/img"
    else:
        cfg = Qwen3OmniMoeAudioEncoderConfig(); cfg.n_window = 50; cfg.n_window_infer = 800
        enc = NativeQwen3OmniAudioEncoder(cfg).to(DEV, DT).eval()
        def mk(n):
            lens = torch.full((n,), 3000, dtype=torch.long, device=DEV)
            feats = torch.randn(cfg.num_mel_bins, int(lens.sum()), device=DEV, dtype=DT)
            return lambda: enc(feats, feature_lens=lens)
        comp = torch.compile(enc, dynamic=True)
        def mkc(n):
            lens = torch.full((n,), 3000, dtype=torch.long, device=DEV)
            feats = torch.randn(cfg.num_mel_bins, int(lens.sum()), device=DEV, dtype=DT)
            return lambda: comp(feats, feature_lens=lens)
        unit = "ms/req"
    print(f"\n===== {kind} ({unit}); eager vs compile(dynamic=True) =====")
    print(f"{'batch':>6s} {'eager':>12s} {'compiled':>12s} {'speedup':>8s}")
    dyn.reset(); base_recompiles = dyn.utils.counters["stats"].get("unique_graphs", 0)
    for n in (1, 4, 8, 16, 32):
        with torch.no_grad():
            e = st.mean([v / n for v in measure(mk(n))])
            try:
                c = st.mean([v / n for v in measure(mkc(n))])
                sp = f"{e/c:.2f}x"
            except Exception as ex:
                c, sp = float("nan"), f"ERR:{str(ex)[:8]}"
        print(f"{n:>6d} {e:12.2f} {c:12.2f} {sp:>8s}")
        torch.cuda.empty_cache()
    rec = dyn.utils.counters["stats"].get("unique_graphs", 0) - base_recompiles
    print(f"  -> compiled graphs (recompiles) across the shape spread: {rec}  (1 = good; many = the #131 pitfall)")


if __name__ == "__main__":
    print(f"device={torch.cuda.get_device_name()} torch={torch.__version__}")
    run("VISION")
    run("AUDIO")
