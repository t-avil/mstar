"""Per-layer numerical parity: native Qwen3-Omni encoders vs HF, *through depth*.

Issue #131 asks to "validate parity against HF encoder outputs." The shipped
test (``test/modular/test_qwen3_omni_native_encoders.py``) checks the final
output and every DeepStack level. This script complements it by measuring
divergence at **every** layer, to answer the question "is it similar in the
middle, not just at the end?" — and to separate genuine implementation error
from bf16 rounding.

Method: build the HF encoder and the native encoder from the same config, copy
HF's (random) weights into the native module (names match 1:1, so
``load_state_dict`` reports 0 missing / 0 unexpected — itself a structural parity
check), feed identical inputs, and capture the residual-stream hidden state
after each block via forward hooks. No checkpoint needed: this tests
implementation *equivalence*, which is weight-value-independent.

Run in fp32 (isolates algorithm) and bf16 (the production dtype). Persists JSON
+ a divergence-vs-depth chart.

    python -m benchmark.qwen3_omni_encoder_parity --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.modules.setdefault("flash_attn", None)

import torch  # noqa: E402

COS_MIN = 0.999
RELL2_MAX = 0.05


def _cmp(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    rel = (a - b).norm().item() / max(b.norm().item(), 1e-9)
    mx = (a - b).abs().max().item()
    return {"cos": cos, "relL2": rel, "max_abs": mx}


def _hook_all(modules, store):
    for i, m in enumerate(modules):
        m.register_forward_hook(
            lambda _m, _in, out, i=i: store.__setitem__(
                i, out[0] if isinstance(out, tuple) else out))


def run_one(device, dt):
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
        Qwen3OmniMoeVisionEncoderConfig,
    )
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder,
        Qwen3OmniMoeVisionEncoder,
    )

    from mstar.model.qwen3_omni.components.audio_encoder import NativeQwen3OmniAudioEncoder
    from mstar.model.qwen3_omni.components.vision_encoder import NativeQwen3OmniVisionEncoder

    res = {}

    # ---- vision ----
    vc = Qwen3OmniMoeVisionEncoderConfig()
    hf = Qwen3OmniMoeVisionEncoder._from_config(vc, attn_implementation="sdpa").to(device, dt).eval()
    nat = NativeQwen3OmniVisionEncoder(vc).to(device, dt).eval()
    miss, unexp = nat.load_state_dict(hf.state_dict(), strict=False)
    rows = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
    g = torch.tensor([[1, 26, 28]], device=device)
    pv = torch.randn(26 * 28, rows, device=device, dtype=dt)
    hcap, ncap = {}, {}
    _hook_all(hf.blocks, hcap)
    _hook_all(nat.blocks, ncap)
    with torch.no_grad():
        o = hf(pv, grid_thw=g)
        on = nat(pv, grid_thw=g)
    res["vision"] = {
        "load_missing": len(miss), "load_unexpected": len(unexp),
        "per_layer": [_cmp(ncap[i], hcap[i]) for i in range(len(hf.blocks))],
        "final": _cmp(on[0], o.pooler_output),
        "deepstack_indexes": list(vc.deepstack_visual_indexes),
    }

    # ---- audio ----
    ac = Qwen3OmniMoeAudioEncoderConfig()
    ac.n_window, ac.n_window_infer = 50, 800
    ha = Qwen3OmniMoeAudioEncoder._from_config(ac, attn_implementation="sdpa").to(device, dt).eval()
    na = NativeQwen3OmniAudioEncoder(ac).to(device, dt).eval()
    miss, unexp = na.load_state_dict(ha.state_dict(), strict=False)
    lens = torch.tensor([3000], device=device)
    f = torch.randn(ac.num_mel_bins, 3000, device=device, dtype=dt)
    hcap, ncap = {}, {}
    _hook_all(ha.layers, hcap)
    _hook_all(na.layers, ncap)
    with torch.no_grad():
        o = ha(f, feature_lens=lens)
        on = na(f, lens)
    res["audio"] = {
        "load_missing": len(miss), "load_unexpected": len(unexp),
        "per_layer": [_cmp(ncap[i], hcap[i]) for i in range(len(ha.layers))],
        "final": _cmp(on.last_hidden_state, o.last_hidden_state),
    }
    return res


def make_chart(artifact, out_dir, tag):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; skipping chart.")
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key in [(axes[0], "vision"), (axes[1], "audio")]:
        for dt_name in ("float32", "bfloat16"):
            pl = artifact[dt_name][key]["per_layer"]
            ys = [max(p["relL2"], 1e-9) for p in pl]
            ax.plot(range(len(ys)), ys, marker=".", label=f"{dt_name} relL2")
        if key == "vision":
            for di in artifact["bfloat16"][key]["deepstack_indexes"]:
                ax.axvline(di, color="gray", ls=":", alpha=0.6)
        ax.axhline(RELL2_MAX, color="red", ls="--", alpha=0.5, label=f"test bar {RELL2_MAX}")
        ax.set_yscale("log")
        ax.set_xlabel("layer index" + (" (dotted = DeepStack captures)" if key == "vision" else ""))
        ax.set_ylabel("relative L2 vs HF (log)")
        ax.set_title(f"{key} encoder — native vs HF divergence through depth")
        ax.legend()
        ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.suptitle(f"Per-layer parity (random weights, 0 missing/unexpected) — {artifact['env']['gpu']}")
    fig.tight_layout()
    p = os.path.join(out_dir, f"qwen3_omni_parity_depth_{tag}.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return [p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="benchmark/artifacts")
    args = ap.parse_args()
    assert torch.cuda.is_available()
    import transformers
    artifact = {"env": {"gpu": torch.cuda.get_device_name(args.device),
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "cos_min": COS_MIN, "rell2_max": RELL2_MAX,
                        "note": "random weights copied HF->native; tests implementation equivalence"}}
    for dt_name, dt in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        print(f"=== {dt_name} ===")
        r = run_one(args.device, dt)
        artifact[dt_name] = r
        for key in ("vision", "audio"):
            f = r[key]["final"]
            worst = max(p["relL2"] for p in r[key]["per_layer"])
            print(f"  {key}: load {r[key]['load_missing']}miss/{r[key]['load_unexpected']}unexp | "
                  f"final cos={f['cos']:.6f} relL2={f['relL2']:.2e} | worst-layer relL2={worst:.2e}")
    os.makedirs(args.out, exist_ok=True)
    tag = artifact["env"]["gpu"].replace(" ", "_").replace("/", "_")
    jpath = os.path.join(args.out, f"qwen3_omni_parity_depth_{tag}.json")
    with open(jpath, "w") as fh:
        json.dump(artifact, fh, indent=2)
    charts = make_chart(artifact, args.out, tag)
    print(f"wrote {jpath}")
    for c in charts:
        print(f"wrote {c}")


if __name__ == "__main__":
    main()
