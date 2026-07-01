"""Headline before/after for the Qwen3-Omni encoder port (issue #131).

A focused two-number-per-encoder summary (HF wrapper -> native) at bs=1 plus one
batched (concurrent-request) point, with a reproducible bar chart. The full
batch sweep + patch-embed cliff breakdown lives in ``qwen3_omni_encoders.py``;
this is the at-a-glance artifact.

Honest framing (so the numbers aren't oversold):
  * The HF baseline is forced onto SDPA (flash-attn blocked) so both paths use
    the same attention kernel and the comparison isolates the *structural* port,
    not a kernel swap. Production HF would use flash-attn-2; the vision number is
    barely affected (it is ~97% patch-embed Conv3d, not attention) but the audio
    number is therefore an SDPA-vs-SDPA comparison, not vs HF's fastest path.
  * Nearly all of the vision gain is the patch-embed Conv3d->F.linear swap, which
    dodges a bf16 cuDNN cliff (see qwen3_omni_encoders.py:bench_patch_embed); the
    same swap could in principle be applied to the HF module.
  * Weights are random (timing is value-independent); parity is a separate
    artifact (qwen3_omni_encoder_parity.py + the parity tests).

Run (CUDA + transformers; no flash-attn needed):
    python -m benchmark.encoder_before_after --device cuda:0 --out benchmark/artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial

sys.modules.setdefault("flash_attn", None)  # force SDPA on both sides

import torch  # noqa: E402

from benchmark.qwen3_omni_encoders import measure, summarize  # noqa: E402

DT = torch.bfloat16


def _vision(device, repeats, batch):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeVisionEncoderConfig,
    )
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeVisionEncoder,
    )

    from mstar.model.qwen3_omni.components.vision_encoder import NativeQwen3OmniVisionEncoder

    cfg = Qwen3OmniMoeVisionEncoderConfig()
    hf = Qwen3OmniMoeVisionEncoder._from_config(cfg, attn_implementation="sdpa").to(device, DT).eval()
    nat = NativeQwen3OmniVisionEncoder(cfg).to(device, DT).eval()
    rows = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    res = {}
    for n in (1, batch):
        g = torch.tensor([[1, 26, 28]] * n, device=device)
        pv = torch.randn(26 * 28 * n, rows, device=device, dtype=DT)
        with torch.no_grad():
            # HF has no cross-request batching: per-image cost is ~3.2 s and
            # batch-independent, so 1 timed iter at the batched point too.
            hf_ms = [v / n for v in measure(partial(hf, pv, grid_thw=g), repeats, 1, 1)]
            nat_ms = [v / n for v in measure(partial(nat, pv, grid_thw=g), repeats, 5, 3)]
        res[f"bs{n}"] = {"hf_per_img": summarize(hf_ms), "native_per_img": summarize(nat_ms),
                         "speedup": round(summarize(hf_ms)["mean"] / summarize(nat_ms)["mean"], 1)}
    return res


def _audio(device, repeats, batch):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
    )
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder,
    )

    from mstar.model.qwen3_omni.components.audio_encoder import NativeQwen3OmniAudioEncoder

    cfg = Qwen3OmniMoeAudioEncoderConfig()
    cfg.n_window, cfg.n_window_infer = 50, 800
    hf = Qwen3OmniMoeAudioEncoder._from_config(cfg, attn_implementation="sdpa").to(device, DT).eval()
    nat = NativeQwen3OmniAudioEncoder(cfg).to(device, DT).eval()
    res = {}
    for n in (1, batch):
        lens = torch.full((n,), 3000, dtype=torch.long, device=device)
        feats = torch.randn(cfg.num_mel_bins, int(lens.sum()), device=device, dtype=DT)
        with torch.no_grad():
            hf_ms = [v / n for v in measure(partial(hf, feats, feature_lens=lens), repeats, 10, 5)]
            nat_ms = [v / n for v in measure(partial(nat, feats, lens), repeats, 10, 5)]
        res[f"bs{n}"] = {"hf_per_req": summarize(hf_ms), "native_per_req": summarize(nat_ms),
                         "speedup": round(summarize(hf_ms)["mean"] / summarize(nat_ms)["mean"], 2)}
    return res


def _chart(artifact, out_dir, tag, batch):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; skipping chart.")
        return []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    specs = [("vision", "hf_per_img", "native_per_img", "ms / image"),
             ("audio", "hf_per_req", "native_per_req", "ms / request")]
    for ax, (key, hfk, natk, ylab) in zip(axes, specs):
        labels, hf_v, nat_v = [], [], []
        for n in (1, batch):
            r = artifact[key][f"bs{n}"]
            labels.append(f"bs={n}")
            hf_v.append(r[hfk]["mean"]); nat_v.append(r[natk]["mean"])
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], hf_v, 0.4, label="HF wrapper (SDPA)", color="#9467bd")
        ax.bar([i + 0.2 for i in x], nat_v, 0.4, label="native", color="#d1791f")
        ax.set_yscale("log"); ax.set_xticks(list(x)); ax.set_xticklabels(labels)
        ax.set_ylabel(ylab + " (log)")
        ax.set_title(f"{key} encoder — bf16, {artifact['env']['repeats']} repeats")
        for i in x:
            ax.text(i + 0.2, nat_v[i], f"{artifact[key][f'bs{batch if i else 1}']['speedup']}x",
                    ha="center", va="bottom", fontsize=8)
        ax.legend()
        ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.suptitle(f"Qwen3-Omni encoder port: before/after (HF wrapper -> native) — {artifact['env']['gpu']}")
    fig.tight_layout()
    p = os.path.join(out_dir, "encoder_before_after_chart.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    return [p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4, help="the batched concurrency point")
    ap.add_argument("--out", default="benchmark/artifacts")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "benchmark requires CUDA"
    import transformers
    env = {"gpu": torch.cuda.get_device_name(args.device), "torch": torch.__version__,
           "transformers": transformers.__version__, "dtype": "bfloat16",
           "attention": "sdpa (flash-attn blocked; see module docstring)",
           "repeats": args.repeats, "batch_point": args.batch,
           "weights": "random (timing is value-independent)"}
    artifact = {"env": env,
                "vision": _vision(args.device, args.repeats, args.batch),
                "audio": _audio(args.device, args.repeats, args.batch)}
    os.makedirs(args.out, exist_ok=True)
    tag = env["gpu"].replace(" ", "_").replace("/", "_")
    jpath = os.path.join(args.out, "encoder_before_after.json")
    with open(jpath, "w") as fh:
        json.dump(artifact, fh, indent=2)
    print(json.dumps(artifact, indent=2))
    charts = _chart(artifact, args.out, tag, args.batch)
    print(f"wrote {jpath}")
    for c in charts:
        print(f"wrote {c}")


if __name__ == "__main__":
    main()
