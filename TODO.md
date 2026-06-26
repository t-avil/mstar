# TODO — Qwen3-Omni native encoders (#131) + figs 5/6 serving deliverable

Consolidated from: the code review of this branch, the original #131 acceptance
criteria, the figures-5/6 (I2S) + I2T/S2T (TTFT/ITL) deliverable, and the
handoff TODO list. Deduplicated and tagged.

Legend: **[now]** = doable without a GPU/checkpoint · **[gpu]** = needs the
30B checkpoint + ≥2 H100/H200 + competitor stacks · **[both]** = code now,
verify on GPU.

---

## 0. Branch split (handoff #13) — [x] SCRIPTED & VERIFIED

`split_qwen3_omni_pr.sh` creates two clean stacked branches and was test-run in
a throwaway checkout: Branch A = 17 encoder files, Branch B = 19 serving files,
**full coverage, zero overlap**. Non-destructive (new branches, no force-push) —
review the diffs, then push. NB: the stale hand-made serving PNGs
(`artifacts/serving/{3way_compare,serving_compare,speech_rtf_throughput}.png`)
land on Branch B but are superseded by `plot_serving.py`; regenerate or drop
them once real data exists.

The branch currently bundles two distinct deliverables. Split them so the
encoder PR stays minimal and reviewable:

- **Branch A — `native-qwen3-omni-encoders` (issue #131 only):** the native
  audio/vision encoders, config toggle, weight loading, parity test,
  encoder-only before/after microbench, and the encoder docstrings.
  - [now] Keep: `mstar/model/qwen3_omni/components/{audio,vision}_encoder.py`,
    `submodules.py` native parts, `config.py` toggle, `qwen3_omni_model.py`
    wiring, `test/modular/test_qwen3_omni_native_encoders.py`,
    `benchmark/qwen3_omni_encoder_parity.py`, `benchmark/qwen3_omni_encoders.py`,
    `benchmark/encoder_before_after.py`.
  - [now] Move OUT: `ENVIRONMENT.md`, `benchmark/HANDOFF_*.md`,
    `benchmark/serving_scripts/*`, edits to `benchmark/dataset.py` +
    `benchmark/runner.py`, `benchmark/artifacts/serving/*`.
- **Branch B — serving benchmark (figs 5/6 + I2T/S2T):** everything moved out
  above, plus the new `plot_serving.py` and the M*-old serving config.

---

## 1. Honesty / claim accuracy [now] — handoff #2, #3, #5, #6 + review

- [x] **Audio "5–11×" overclaim.** `qwen3_omni_model.py` audio docstring now
      states the measured ~1.2–1.7× (cross-request batching, not a faster
      kernel); no "overclaim" comment, just the real figure.
- [x] **Attribute the vision speedup honestly.** Vision docstring now says the
      gain is the patch-embed Conv3d→`F.linear` swap (bf16 cuDNN cliff), not a
      new attention kernel, and that the same swap could apply to the HF path.
- [x] **Soften the "optimized attention path" claim.** Vision docstring now
      states attention is the same `flash_attn_varlen_func` HF uses, not a
      shape-specialized kernel.
- [x] **Before/after script hardened.** Rewrote `encoder_before_after.py`:
      proper warmup/repeats (reuses the rigorous `measure`/`summarize`), a
      batched concurrency point (handoff #4), env-parametrized output (no
      `/home/timchick`), honest forced-SDPA labeling, and it now emits the
      `encoder_before_after_chart.png` reproducibly (previously no generator).
      The *numbers* still need a real-weight GPU run (→ §6) but the harness is
      defensible.
- [x] **Fixed setup-script crash.** `setup_and_run_qwen3_omni_encoders.sh`
      passed `--iters/--warmup`, which the bench's argparse rejects (only
      `--repeats`) — would have crashed on launch. Now passes `--repeats`
      (back-compat `ITERS` alias kept).

## 2. PyTorch / CUDA version + encoder settings [now/both] — handoff #5, #6 (+ user flag)

**User flag "incorrect pytorch / check cuda version" = the version/build (RESOLVED,
see below). The decorator/graph items remain as separate handoff items.**

The `enc_dec` engine already provides the inference context and optimization
(see `mstar/engine/stateless_engine.py`):
`_inference_context` = `autocast(bf16) + no_grad`; `make_enc_dec_config` sets
`apply_torch_compile=True` and `enable_piecewise_runner=True`.

- [ ] **(handoff item, not the user flag) Redundant/conflicting decorators.**
      `@torch.no_grad()` on `forward` is redundant with the engine context;
      `@torch.compiler.disable` on `varlen_attention` forces a graph break.
      Decide the correct combination so the engine's `torch.compile` applies.
- [ ] **Manual dtype casts vs autocast.** `forward` does
      `x.to(self.weight.dtype)`; under the engine's autocast this can fight the
      cast. Confirm the patch-embed bf16 path is preserved (the headline result
      depends on bf16) while not double-casting.
- [x] **Torch/CUDA build (RESOLVED).** Box is CUDA 12.8 (`nvcc` 12.8,
      `torch.version.cuda==12.8`) → correct wheel is `torch==2.9.1+cu128`, NOT
      the `+cu130` the encoder README documented (cu130 is CUDA-13.x-only and
      breaks flash-attn compile here). README corrected. The setup script
      already auto-selects cu128 via `UV_TORCH_BACKEND=auto`. NB: this sandbox
      has torch 2.8.0 which cannot load `sgl-kernel`; real runs need 2.9.1.
- [ ] **CUDA-graph / piecewise capture gap (handoff #5, biggest ticket
      deviation).** The encoders implement neither `get_cuda_graph_configs()`
      nor `get_piecewise_runner_config()`, so the enc_dec engine's
      graph+piecewise machinery is inert on them. Either (a) implement
      `get_piecewise_runner_config()` mirroring `vjepa2`
      (`mstar/model/vjepa2/{submodules.py,components/predictor.py}` — captures a
      variable-length block loop as preamble→captured-loop→postamble) and record
      the win, or (b) keep the opt-out but back it with a measurement showing it
      regresses. Make it an *evidenced* decision. **[both]**

## 3. Code correctness edge cases [now] — handoff #12 + review

- [x] **Audio `_Out` ad-hoc return.** Replaced with a module-level
      `AudioEncoderOutput` namedtuple; `forward` now accepts
      `feature_lens=None, return_dict=True, **kwargs` for HF-signature
      compatibility. CI test asserts the typed return.
- [x] **Audio `can_batch` is narrower than documented.** Docstring on
      `NativeAudioEncoderSubmodule` now states the one-segment-per-request
      condition and the multi-clip sequential fallback explicitly.
- [~] **Vision dead branch.** Could not locate a dead empty-DeepStack branch in
      the current `vision_encoder.py forward()` (the loop always appends for the
      configured deepstack indexes). Likely already removed in a later commit;
      the only `[torch.tensor([])]` fallbacks are defensive, in `submodules.py`.
      Left as-is — flag if you still see it somewhere.
- [x] **SDPA fallback O(N²) mask.** Added an explicit comment in `_sdpa_varlen`
      that it materializes an O(total_tokens²) mask and is a CI/parity-only
      fallback, not the production flash-varlen path.

## 4. Parity test runnable in CI [now] — handoff #15

- [x] Added `test/modular/test_qwen3_omni_native_encoders_ci.py`: seeded,
      CPU, fp32, flash-attn-forced-off (SDPA) structural parity on SMALL HF +
      native encoders — no checkpoint/GPU/flash-attn. Asserts 0 missing/0
      unexpected on `load_state_dict`, parity at EVERY block/layer (covers the
      intermediate-capture nit), the merged pooler, and every DeepStack level.
      Skips only if `transformers` is absent. **(Code-verified via py_compile;
      needs transformers to actually run — present in CI.)**

## 5. Serving benchmark deliverable — figs 5/6 (I2S) + I2T/S2T [both]

Acceptance criteria for the deliverable (this is the figs-5/6 ask):

- [x] **Reproducible plotting.** Added `benchmark/serving_scripts/plot_serving.py`
      — reads persisted `results.json`, emits the I2T/S2T TTFT + ITL charts and
      the I2S RTF/throughput batch-sweep, series keyed by directory so M*-old vs
      M*-new are distinct. **Verified here via `--selftest` (synthetic data).**
- [x] **M*-old vs M*-new split (mechanism).** `model_kwargs` is request-time in
      this codebase, so added an ENV toggle in `config.py`
      (`MSTAR_QWEN3_NATIVE_{AUDIO,VISION}_ENCODER`) + `serve_mstar.sh
      VARIANT=native|hf`. Both variants run as separate series → the required
      **4 runtimes**. **[now: done; gpu: run]**
- [~] **Fig 5 analog (I2S, 2-GPU disaggregated):** harness + plotter ready
      (`CONFIG=configs/qwen3omni_2gpu.yaml`, `--topology "2-GPU disaggregated"`,
      `BATCHES="1 4 8 16 32"`). **[gpu: run]**
- [~] **Fig 6 analog (I2S, Thinker TP=2, 3-GPU):** same harness with
      `CONFIG=configs/qwen3omni_thinker_tp2.yaml`, `--topology "Thinker TP2,
      3-GPU"`, SGLang-only competitor. Produced as its own chart. **[gpu: run]**
- [~] **I2T/S2T TTFT + ITL across 4 runtimes:** `bench_system.sh` + plotter
      ready; sweeps concurrency, not bs=1 smoke. **[gpu: run]**
- [x] **Batch sweep in harness.** `run_mstar_paths.sh` / `bench_speech.sh` now
      default to `BATCHES="1 4 8 16 32"` (override-able).
- [x] **Parametrize hardcoded paths.** All serving scripts now use
      `${MSTAR_REPO}`, `${BENCH_ROOT}`, `${PY}`, `${OUT_ROOT}`, `${GPUS}`, …
      with the old node's values as defaults. See `serving_scripts/README.md`.

## 6. GPU-blocked verification (ready-to-run, needs hardware) [gpu]

- [ ] **Real-weight parity** on the actual 30B checkpoint: vision pooler, every
      DeepStack level, audio hidden states — single and batched (handoff #1).
- [ ] **Real before/after** with proper warmup/repeats on real weights;
      replace synthetic numbers (handoff #2). Include the concurrency benchmark
      (multi-image / multi-clip native-batched vs HF-sequential) (handoff #4).
- [ ] **Fix competitor TTS/I2S baselines** (handoff #8). vLLM-Omni + sglang-omni
      both crash on TTS (`_thinker_to_talker_prefill → torch.cat()` empty list —
      thinker yields no text). Try temperature>0, the real seed-tts prompt
      format, and/or a pinned vendor release. Without this there is **no**
      competitor data for figs 5/6.
- [ ] **TTS figures end-to-end** on ≥2 H200: reproduce fig 5 (2-GPU disagg) and
      fig 6 (Thinker TP2, 3-GPU) on Seed-TTS across batch; confirm the 2.7× vs
      vLLM / 4.0× vs sglang throughput claims (handoff #7).
- [ ] **I2T/S2T at proper scale** with real batching/concurrency before any
      win/lose conclusion (handoff #9). Current bs=1/8-req has M* at parity or
      slightly behind on ITL.
- [ ] **Single-GPU H200 smoke** for S2T/I2T via `configs/qwen3omni_colocated.yaml`
      (full model fits in 141 GB; config never exercised) (handoff #10).
- [ ] **T2S no-regression** check after the encoder change (handoff #11).

---

## Resolved

- **PyTorch/CUDA version (user flag):** the documented `torch 2.9.1+cu130` was
  wrong for the CUDA 12.8 hardware — corrected to `+cu128` in
  `README_qwen3_omni_encoders.md` and `ENVIRONMENT.md`. Authoritative pin
  (`pyproject.toml torch==2.9.1`), `docs/installation.rst`, and the setup
  script's `UV_TORCH_BACKEND=auto` were already correct.
