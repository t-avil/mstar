# Qwen3-Omni serving benchmark — figs 5/6 (I2S) + I2T/S2T (TTFT/ITL)

Reproduces the deliverable: the paper's fig-5/6 RTF+throughput batch sweep, but
for **I2S** instead of TTS, with **both M\* variants** (native vs HF encoders)
plus the two competitors; and the **I2T/S2T TTFT & ITL** charts across all four
runtimes. Everything here is path-parametrized (`MSTAR_REPO`, `BENCH_ROOT`,
`PY`, `GPUS`, `OUT_ROOT`, `BATCHES`, …) — defaults match the original bench node.

## The four runtimes
| dir / series | what |
|---|---|
| `ours_native` | M\*-new — native encoders (`serve_mstar.sh VARIANT=native`) |
| `ours_hf`     | M\*-old — HF-wrapper encoders (`serve_mstar.sh VARIANT=hf`) |
| `vllm_omni`   | vLLM-Omni |
| `sglang_omni` | SGLang-Omni |

The two M\* variants are the **same server** flipped via the env toggle
(`MSTAR_QWEN3_NATIVE_{AUDIO,VISION}_ENCODER`), wired in
`mstar/model/qwen3_omni/config.py`.

## Procedure (per topology)

Fig 5 = 2-GPU disaggregated (`CONFIG=configs/qwen3omni_2gpu.yaml`).
Fig 6 = Thinker TP=2, 3-GPU (`CONFIG=configs/qwen3omni_thinker_tp2.yaml`,
competitors: SGLang only — vLLM TP not evaluable, matching the paper).

For each runtime, launch its server, then drive the sweep:

```bash
# --- M*, both variants (restart the server between variants) ---
CONFIG=configs/qwen3omni_2gpu.yaml VARIANT=native bash serve_mstar.sh &   # then:
VARIANT=native bash run_mstar_paths.sh                                    # text+speech sweep
CONFIG=configs/qwen3omni_2gpu.yaml VARIANT=hf bash serve_mstar.sh &       # then:
VARIANT=hf     bash run_mstar_paths.sh

# --- competitors (speech + text) ---
SYSTEM=vllm_omni   URL=http://localhost:8091 bash bench_speech.sh
SYSTEM=vllm_omni   URL=http://localhost:8091 bash bench_system.sh
SYSTEM=sglang_omni URL=http://localhost:8092 bash bench_speech.sh
SYSTEM=sglang_omni URL=http://localhost:8092 bash bench_system.sh
```

`run_mstar_paths.sh` writes M\* into `ours_<variant>/`; the competitor scripts
write into `<system>/`. All land under `$OUT_ROOT`
(default `$BENCH_ROOT/bench_artifacts`).

## Plot

```bash
python -m benchmark.serving_scripts.plot_serving \
    --data-root "$OUT_ROOT" --out-dir benchmark/artifacts/serving \
    --topology "2-GPU disaggregated"      # or "Thinker TP2, 3-GPU" for fig 6
```

Emits:
- `qwen3_omni_i2t_s2t_ttft.png` — TTFT vs batch, I2T & S2T, 4 runtimes
- `qwen3_omni_i2t_s2t_itl.png` — ITL vs batch, I2T & S2T, 4 runtimes
- `qwen3_omni_i2s_t2s_rtf_throughput.png` — RTF + throughput vs batch for
  **both** speech paths: I2S (image→speech, the figs-5/6 analog) and T2S
  (text→speech, the paper's original TTS path) — so the optimization level on
  each is visible side by side.

Pure plotting over saved `results.json` — no model re-run, fully auditable.
Sanity-check the plumbing with `--selftest` (synthetic data, no GPU).

## Known blocker (handoff #8)
vLLM-Omni and SGLang-Omni currently **crash on the speech path**
(`_thinker_to_talker_prefill → torch.cat()` empty list — the thinker yields no
text under greedy decode). Until that's fixed, the I2S figure has only the two
M\* series with real data; competitor cells render as absent. Try temperature>0,
the real seed-tts prompt format, and/or a pinned vendor release.
