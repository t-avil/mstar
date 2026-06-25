"""Performance benchmark for the native Qwen3-Omni encoders vs the HF wrappers.

Issue #131 acceptance: the native encoder path must be "at least as fast,
ideally faster with concurrent requests." This script measures that with
variance, across batch sizes.

Methodology (documented so the numbers are reproducible and trustworthy)
------------------------------------------------------------------------
* **dtype = bfloat16 = the production setup.** The Qwen3-Omni encoders run on
  the ``enc_dec`` stateless engine, whose ``autocast_dtype`` defaults to
  ``torch.bfloat16`` (``mstar/engine/stateless_engine.py``); only the audio-codec
  flavor forces fp32. So bf16 is what actually runs in production — and it is
  exactly where the HF vision Conv3d patch-embed hits a cuDNN low-precision cliff
  (~3.2 s for a 728-patch image; the *same* Conv3d in fp32 is ~0.2 ms). We report
  bf16 to avoid reporting false (fp32-only) numbers.
* **Attention backend = SDPA**, flash-attn excluded (blocked in ``sys.modules``);
  the HF baseline is built with ``attn_implementation="sdpa"``. Both paths use the
  same attention kernel, so the comparison isolates the structural changes the
  port introduces. For the vision encoder this barely matters: ~97% of the HF
  time is the Conv3d patch-embed, not attention.
* **Batch sizes** ``(1, 2, 4, 8, 16)`` and **repeats** ``10`` reuse the existing
  ``perf_testing/offline_homogenous.sh`` convention (``BATCH_SIZES`` / ``NUM_BATCHES``).
  Timing uses CUDA events (the ``test/modular/test_fused_rmsnorm.py`` pattern).
* Each datapoint is ``repeats`` independent measurements (each the mean of
  ``n_iter`` timed calls). We report mean / std / p50 / p95 and persist **every
  raw sample** so variance is auditable.
* HF vision is measured only up to batch 4: one call is ~3.2 s/image and its
  per-image cost is batch-independent (no cross-request batching), so larger
  batches add minutes of wall-clock without new information. This is logged, not
  hidden.
* Weights are random (perf is value-independent). Numerical parity is a separate
  artifact (``qwen3_omni_encoder_parity.py`` + ``test_qwen3_omni_native_encoders.py``).

Run (CUDA + transformers; NO flash-attn):
    python -m benchmark.qwen3_omni_encoders --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics as st
import sys
from functools import partial

# --- hard-block flash-attn: force the SDPA varlen fallback everywhere -------- #
sys.modules.setdefault("flash_attn", None)

import torch  # noqa: E402

DTYPE = torch.bfloat16
BATCH_SIZES = [1, 2, 4, 8, 16]          # reuse perf_testing/offline_homogenous.sh
AUDIO_N_WINDOW = 50                      # published Qwen3-Omni-30B audio frontend
AUDIO_N_WINDOW_INFER = 800
HF_VISION_MAX_BATCH = 4                  # HF vision is ~3.2 s/img; cap it (logged)


def measure(fn, repeats: int, n_iter: int, n_warmup: int) -> list[float]:
    """Return ``repeats`` independent timings (ms), each the mean of ``n_iter``
    CUDA-event-timed calls. Variance across the returned list is the signal."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(n_iter):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / n_iter)
    return samples


def summarize(samples: list[float]) -> dict:
    s = sorted(samples)
    return {
        "mean": round(st.mean(s), 4),
        "std": round(st.pstdev(s), 4) if len(s) > 1 else 0.0,
        "p50": round(s[len(s) // 2], 4),
        "p95": round(s[min(len(s) - 1, int(0.95 * len(s)))], 4),
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
        "raw": [round(x, 4) for x in samples],
    }


# --------------------------------------------------------------------------- #
def _audio_cfg():
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
    )
    cfg = Qwen3OmniMoeAudioEncoderConfig()
    cfg.n_window = AUDIO_N_WINDOW
    cfg.n_window_infer = AUDIO_N_WINDOW_INFER
    return cfg


def bench_audio(device, repeats):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder,
    )

    from mstar.model.qwen3_omni.components.audio_encoder import NativeQwen3OmniAudioEncoder

    cfg = _audio_cfg()
    hf = Qwen3OmniMoeAudioEncoder._from_config(cfg, attn_implementation="sdpa").to(device, DTYPE).eval()
    nat = NativeQwen3OmniAudioEncoder(cfg).to(device, DTYPE).eval()
    frames = 3000
    out = []
    for n in BATCH_SIZES:
        lens = torch.full((n,), frames, dtype=torch.long, device=device)
        feats = torch.randn(cfg.num_mel_bins, int(lens.sum()), device=device, dtype=DTYPE)
        with torch.no_grad():
            hf_s = [v / n for v in measure(partial(hf, feats, feature_lens=lens), repeats, 10, 5)]
            nat_s = [v / n for v in measure(partial(nat, feats, lens), repeats, 10, 5)]
        rec = {"batch": n, "hf": summarize(hf_s), "native": summarize(nat_s),
               "speedup_mean": round(st.mean(hf_s) / st.mean(nat_s), 2)}
        out.append(rec)
        print(f"  audio  n={n:>2}  HF={rec['hf']['mean']:7.2f}±{rec['hf']['std']:.2f}  "
              f"native={rec['native']['mean']:6.2f}±{rec['native']['std']:.2f} ms/req  "
              f"speedup={rec['speedup_mean']}x")
    return out


def _vision_cfg():
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeVisionEncoderConfig,
    )
    return Qwen3OmniMoeVisionEncoderConfig()


def _vision_input(cfg, n, device):
    rows = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    grid = torch.tensor([[1, 26, 28]] * n, dtype=torch.long, device=device)
    npatch = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
    return torch.randn(npatch, rows, device=device, dtype=DTYPE), grid


def bench_vision(device, repeats):
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeVisionEncoder,
    )

    from mstar.model.qwen3_omni.components.vision_encoder import NativeQwen3OmniVisionEncoder

    cfg = _vision_cfg()
    hf = Qwen3OmniMoeVisionEncoder._from_config(cfg, attn_implementation="sdpa").to(device, DTYPE).eval()
    nat = NativeQwen3OmniVisionEncoder(cfg).to(device, DTYPE).eval()
    out = []
    for n in BATCH_SIZES:
        pv, g = _vision_input(cfg, n, device)
        with torch.no_grad():
            nat_s = [v / n for v in measure(partial(nat, pv, grid_thw=g), repeats, 5, 3)]
            if n <= HF_VISION_MAX_BATCH:
                hf_s = [v / n for v in measure(partial(hf, pv, grid_thw=g), repeats, 1, 1)]
                hf_sum, sp = summarize(hf_s), round(st.mean(hf_s) / st.mean(nat_s), 1)
            else:
                hf_sum, sp = None, None  # skipped: HF per-image cost is batch-independent
        rec = {"batch": n, "patches_per_image": 728, "hf": hf_sum,
               "native": summarize(nat_s), "speedup_mean": sp}
        out.append(rec)
        hf_txt = f"HF={hf_sum['mean']:8.1f}±{hf_sum['std']:5.1f}" if hf_sum else "HF=  (skipped)   "
        print(f"  vision n={n:>2}  {hf_txt}  native={rec['native']['mean']:6.2f}±{rec['native']['std']:.2f} ms/img"
              + (f"  speedup={sp}x" if sp else "  (HF batch-independent)"))
    return out


def bench_patch_embed(device, repeats):
    """Headline: HF bf16 Conv3d vs native matmul, plus the fp32 Conv3d control
    that proves the slowness is a bf16/fp16 cuDNN cliff, not the convolution
    itself."""
    from mstar.model.qwen3_omni.components.vision_encoder import VisionPatchEmbed

    cfg = _vision_cfg()
    rows = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    out = []
    for patches in (728, 1600, 3136):
        rec = {"patches": patches}
        for dt_name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
            pe = VisionPatchEmbed(cfg).to(device, dt).eval()
            x = torch.randn(patches, rows, device=device, dtype=dt)
            xc = x.view(patches, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
            n_iter = 1 if dt is torch.bfloat16 else 20
            with torch.no_grad():
                rec[f"conv3d_{dt_name}"] = summarize(measure(partial(pe.proj, xc), repeats, n_iter, 1))
                rec[f"matmul_{dt_name}"] = summarize(measure(partial(pe, x), repeats, n_iter, 2))
        out.append(rec)
        print(f"  patch_embed p={patches:>4}  Conv3d/bf16={rec['conv3d_bf16']['mean']:9.2f}ms  "
              f"Conv3d/fp32={rec['conv3d_fp32']['mean']:6.3f}ms  matmul/bf16={rec['matmul_bf16']['mean']*1000:6.1f}us")
    return out


# --------------------------------------------------------------------------- #
def make_charts(artifact, out_dir, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping charts (JSON still written).")
        return []
    paths = []

    # 1) audio + vision: ms/item vs batch size, with std error bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, ylab in [(axes[0], "audio", "ms / request"), (axes[1], "vision", "ms / image")]:
        rows = artifact[key]
        xs = [r["batch"] for r in rows]
        nat_m = [r["native"]["mean"] for r in rows]
        nat_e = [r["native"]["std"] for r in rows]
        ax.errorbar(xs, nat_m, yerr=nat_e, marker="o", capsize=4, label="native")
        hx = [r["batch"] for r in rows if r["hf"]]
        hm = [r["hf"]["mean"] for r in rows if r["hf"]]
        he = [r["hf"]["std"] for r in rows if r["hf"]]
        if hx:
            ax.errorbar(hx, hm, yerr=he, marker="s", capsize=4, label="HF wrapper")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(xs)
        ax.set_xlabel("batch size (concurrent requests)")
        ax.set_ylabel(ylab + " (log)")
        ax.set_title(f"{key} encoder — bf16, {artifact['env']['repeats']} repeats (±std)")
        ax.legend()
        ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.suptitle(f"Qwen3-Omni encoders: per-item latency vs batch — {artifact['env']['gpu']}")
    fig.tight_layout()
    p = os.path.join(out_dir, f"qwen3_omni_latency_vs_batch_{tag}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    # 2) patch-embed: Conv3d bf16 vs Conv3d fp32 vs matmul (log), error bars
    rows = artifact["patch_embed"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [r["patches"] for r in rows]
    idx = range(len(xs))
    series = [("conv3d_bf16", "HF Conv3d (bf16)", -0.25),
              ("conv3d_fp32", "Conv3d (fp32 control)", 0.0),
              ("matmul_bf16", "native matmul (bf16)", 0.25)]
    for kkey, lbl, off in series:
        ax.bar([i + off for i in idx], [r[kkey]["mean"] for r in rows], width=0.25,
               yerr=[r[kkey]["std"] for r in rows], capsize=3, label=lbl)
    ax.set_yscale("log")
    ax.set_xticks(list(idx))
    ax.set_xticklabels(xs)
    ax.set_xlabel("patches / image")
    ax.set_ylabel("ms (log)")
    ax.set_title(f"Vision patch-embed — bf16 Conv3d cliff (H100, {artifact['env']['repeats']} repeats)")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, f"qwen3_omni_patch_embed_{tag}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--repeats", type=int, default=10, help=">=10 for variance")
    ap.add_argument("--out", default="benchmark/artifacts")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "benchmark requires CUDA"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = args.device

    import transformers
    env = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dtype": "bfloat16 (= production enc_dec autocast)",
        "attention_backend": "sdpa (flash-attn excluded)",
        "batch_sizes": BATCH_SIZES,
        "repeats": args.repeats,
        "audio_n_window": AUDIO_N_WINDOW,
        "audio_n_window_infer": AUDIO_N_WINDOW_INFER,
        "hf_vision_max_batch": HF_VISION_MAX_BATCH,
        "python": platform.python_version(),
    }
    print(f"Qwen3-Omni encoder benchmark — {env['gpu']}, bf16, repeats={args.repeats}")
    print("audio:")
    audio = bench_audio(device, args.repeats)
    print("vision:")
    vision = bench_vision(device, args.repeats)
    print("patch_embed:")
    patch = bench_patch_embed(device, args.repeats)

    artifact = {"env": env, "audio": audio, "vision": vision, "patch_embed": patch}
    os.makedirs(args.out, exist_ok=True)
    tag = env["gpu"].replace(" ", "_").replace("/", "_")
    jpath = os.path.join(args.out, f"qwen3_omni_encoders_{tag}.json")
    with open(jpath, "w") as f:
        json.dump(artifact, f, indent=2)
    charts = make_charts(artifact, args.out, tag)
    print(f"\nwrote {jpath}")
    for c in charts:
        print(f"wrote {c}")


if __name__ == "__main__":
    main()
