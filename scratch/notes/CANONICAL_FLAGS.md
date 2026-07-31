# Canonical "full optimization flag set" — Qwen3-Omni serving (mstar-encoders)

Reconstructed and validated 2026-06-29. This is the env-flag set the **original good**
S2T runs used and that the later **"fair" re-runs dropped**, causing the ~3.5x slowdown
(S2T B=1: TTFT 0.339s / 32.6 tok/s vs original 0.097-0.101s / ~84 tok/s).

Build used for validation: editable `mstar` install -> `/home/tim/exp/combined-wt/mstar`
(branch `exp/combined-coalesce-piggyback`). Interpreter `/home/tim/mstar-encoders/.venv/bin/python`.

---

## The canonical flag set

```bash
# --- optimization flags that MUST be exported at SERVER launch ---
export MSTAR_GPU_MEL=1              # GPU log-mel extraction (THE S2T/TTFT lever)
export MSTAR_GPU_IMAGE_PREPROCESS=1 # on-device image resize/patchify (I2T/I2S; no-op for S2T)
export MSTAR_VLLM_PROMPT_LAYOUT=1   # vLLM prompt layout: model *answers* (longer output) + audio M-RoPE h/w parity
```

These three default **OFF** in the code and must be set explicitly. The other "wins"
are already baked in as **defaults** on this build and need no flag:
- `native_audio_encoder = True`, `native_vision_encoder = True`
  (`mstar/model/qwen3_omni/config.py:380-400`; override only with
  `MSTAR_QWEN3_NATIVE_AUDIO_ENCODER=0` / `..._VISION_ENCODER=0`).
- `codec_chunk_frames = 15`, `codec_left_context_frames = 15`
  (`config.py:291-292`; Talker-only, irrelevant to S2T).

For speech-output paths only (S2S/I2S — NOT needed for S2T), the harness also sets
`BENCH_SPEECH_THINKER_TEMPERATURE=0.7` on the **client**; it is harmless on S2T.

### Exact launch command (validated, GPUs 2,3)

```bash
WT=/home/tim/exp/combined-wt
SOCK=/home/tim/tmp/canonflags; rm -rf "$SOCK"; mkdir -p "$SOCK"
setsid bash -c "cd $WT && source /home/tim/mstar-encoders/.venv/bin/activate && \
  export PYTHONPATH=$WT/.edit_override CUDA_VISIBLE_DEVICES=2,3 \
         HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/home/tim/hf_datasets \
         TMPDIR=/home/tim/tmp/launch-tmp; \
  export MSTAR_GPU_MEL=1 MSTAR_GPU_IMAGE_PREPROCESS=1 MSTAR_VLLM_PROMPT_LAYOUT=1; \
  timeout 1500 numactl --cpunodebind=0 --membind=0 python -m mstar.cli.main serve qwen3_omni \
    --config configs/qwen3omni_2gpu.yaml --host 0.0.0.0 --port 8160 \
    --tensor-comm-protocol SHM --socket-path-prefix $SOCK --log-level INFO \
  > server.log 2>&1" </dev/null >/dev/null 2>&1 &
```

S2T client (B=1):

```bash
cd /home/tim/exp/combined-wt && source /home/tim/mstar-encoders/.venv/bin/activate
python -m benchmark.runner --url http://127.0.0.1:8160 --model qwen3omni \
  --request-type audio_to_text --dataset libri --profiling-type closed_loop \
  --max-concurrency 1 --num-requests 24 --num-warmup 5 --inference-system ours \
  --local-cache /home/tim/tmp/libri_wavs --output-dir <out>
```

---

## Which flags were MISSING in the "fair" runs

The fair re-runs (`fair_sweep_s2t.log`, `mixedwalk_fair_s2t.log`) launched the server with
**none** of the three flags above. Missing:
- `MSTAR_GPU_MEL=1`  <-- dominant cause of the 3.5x S2T slowdown
- `MSTAR_GPU_IMAGE_PREPROCESS=1`  (only matters for image paths)
- `MSTAR_VLLM_PROMPT_LAYOUT=1`

Origin of the canonical set: the original "integrated" good run
(`/home/tim/exp_rebench/out_mstar_new_integrated/...`, server `integ_server.log`, and
`/home/tim/exp_throughput/command.txt`) launched the integrated build with
`MSTAR_GPU_MEL=1 MSTAR_GPU_IMAGE_PREPROCESS=1` plus prompt-layout, native encoders, and
codec_chunk=15. On `combined-wt`, native+codec are defaults, so only the three flags above
must be re-supplied.

---

## Measured S2T B=1 (this validation, isolated on GPUs 2,3, ×24 + 5 warmup, dataset=libri)

| Server config (only difference = flags) | TTFT mean | text tok/s | tok/req | E2E mean |
|---|---|---|---|---|
| **Full canonical set** (GPU_MEL + IMG + PROMPT_LAYOUT) | **0.101 s** | **83.9** | 19.4 | 0.231 s |
| GPU_MEL only (prompt-layout OFF) | 0.104 s | 71.6 | 14.4 | 0.201 s |
| **No flags** (= the "fair" run config) | 0.365 s | 31.5 | 14.6 | 0.463 s |
| --- reference --- | | | | |
| Original good (`out_mstar_new_integrated`) | 0.101 s | 83.7 | 19.4 | 0.232 s |
| "Fair" bad run (`fair_sweep_s2t.log`) | 0.339 s | 32.6 | 14.5 | 0.445 s |

**Verdict:** the full canonical set reproduces the original good numbers to within noise
(0.101 s / 83.9 tok/s vs original 0.101 s / 83.7 tok/s). The no-flags isolated run
reproduces the "fair" slowdown (0.365 s / 31.5 tok/s vs fair 0.339 s / 32.6 tok/s),
**proving the slowdown was the missing flags, not GPU co-location.**

### Attribution of each flag (S2T B=1)
- **`MSTAR_GPU_MEL=1` is the TTFT lever:** TTFT 0.365 s -> 0.104 s (~3.5x) on its own.
  It moves log-mel spectrogram extraction off the CPU critical path onto the GPU.
- **`MSTAR_VLLM_PROMPT_LAYOUT=1`** adds the rest of the tok/s: output goes 14.4 -> 19.4
  tokens/req (the model *answers* in vLLM layout instead of emitting a short transcript),
  lifting 71.6 -> 83.9 tok/s. It also gives audio M-RoPE h/w parity for same-output
  comparisons vs vLLM.
- **`MSTAR_GPU_IMAGE_PREPROCESS=1`** is a no-op for S2T (no image); kept for I2T/I2S.

---

## All optimization flags that exist in the build (for reference)

From `grep -rhoE 'MSTAR_[A-Z0-9_]+' /home/tim/exp/combined-wt/mstar`:
`MSTAR_GPU_MEL`, `MSTAR_GPU_IMAGE_PREPROCESS`, `MSTAR_VLLM_PROMPT_LAYOUT`,
`MSTAR_QWEN3_NATIVE_AUDIO_ENCODER`, `MSTAR_QWEN3_NATIVE_VISION_ENCODER`,
`MSTAR_VARLEN_BACKEND`, `MSTAR_VIT_BATCHING`, `MSTAR_VLLM_AUDIO_SENTINELS`,
`MSTAR_ENCODER_COALESCE(_MAX_BATCH/_WAIT_MS)`, `MSTAR_ENCODER_CUDA_GRAPH`,
`MSTAR_ENCODER_CG_MAX_KEYS`, `MSTAR_ENCODER_CG_WARMUP`, `MSTAR_MIXED_WALK`,
`MSTAR_MIXED_MAX_PREFILL_REQS`, `MSTAR_MIXED_PREFILL_CHUNK`, `MSTAR_MIXED_TOKEN_BUDGET`,
`MSTAR_PRE_PLAN_SPEC`, `MSTAR_MAX_CONSECUTIVE_SPEC_STEPS`, `MSTAR_SPEC_PEEK_FOR_FAIRNESS`,
`MSTAR_NUM_SLOTS`, `MSTAR_WORKSPACE_BUFFER_MB`, `MSTAR_PHASE_TIMING`, `MSTAR_DUMP(_DIR)`,
`MSTAR_PY_SWITCH_INTERVAL_SEC`, `MSTAR_ZMQ_*`.

The encoder-coalesce / mixed-walk flags are **batch-throughput** levers (B>=4) and do
**not** affect B=1 TTFT; they are not part of the B=1 canonical S2T set.
