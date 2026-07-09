# Reviewer — start here

**One sentence:** Start at `RESULTS.md` (honest, length-fair headline + the audio-length caveat up front),
verify every figure against `NUMBERS.md` (the single source of truth that all docs are filled from) and the
*why* in `EXPLANATION_GRID.md` (file:line-rooted across M*-new/M*-old/vLLM, mechanisms in `research/`), then
review the shippable code as one small PR on branch **`qwen3-omni-unified`** (the single `OptimizationConfig`
switchboard + `RUNBOOK_QWEN3_OMNI_UNIFIED.md` + parity tests in `test/modular/`).

## What is already done (CPU / from committed data) — reviewable now
- Data integrity: `datapoints ≡ aggregates` (93/93 cells), B2 mstar_old filled, self-describing units
  (`sample_rate`, `audio_seconds_method`, `token_count_source`) in every `raw_<path>.json`.
- Plain ratios (M*-new/vLLM, M*-new/M*-old) — the ≥10%-over-both / PASS framing is removed.
- Charts (200 dpi) with error bars, missing-cell open markers, and the S2S mstar_old B32 ITL outlier dropped+annotated.
- Docs reconciled to NUMBERS.md: length ratios (S2S new/vLLM ~0.82, I2S ~0.78, **M*-old ~0.57 → flagged as
  better-than-fair**), "fp32-exact; bf16 cos≥0.9999" (not "identical"), sourced GPU-img 7–100×, piggyback =
  considered-and-deferred (order-invariant reorder ruled out; out of #131 scope).
- Unified config-gated entrypoint + ablation routing + parallel n=10 orchestrator + runbook + parity tests (pushed).

## What we are WAITING FOR GPUs to do (NOT yet run — validation pending)
These do not change the mechanism conclusions, but the **headline numbers should be treated as provisional**
until item 1, and parity is **asserted-but-not-executed** until items 2–4:
1. **Uniform n=10 re-run** of M*-new (+ gpumel/gpuimg/prompt-layout variants) via
   `qwen3-omni-unified` `orchestrate_rerun.sh` across isolated H200 pairs (5 warmup / n=10), **spliced with the
   reused M*-old + vLLM datapoints** from the prior isolated runs (documented reuse). The currently committed
   numbers come from earlier runs with non-uniform n and some host contention — the re-run makes n uniform and
   clean; figures may shift within run-to-run variance (~1–5%).
2. **bf16 encoder parity confirmation** — run `test_qwen3_omni_native_encoders.py` on the real 30B checkpoint and
   capture the **true MIN cos across audio + every vision DeepStack level**; the `COS_MIN=0.9999` bar is
   knife-edge and must be set just below the measured min (it has not been executed on GPU).
3. **Run the GPU-mel + GPU-image parity tests** (`test_qwen3_omni_gpu_mel_parity.py` with non-hop-multiple
   durations + frame-count asserts; `test_qwen3_omni_gpu_image_parity.py`) — written, never executed.
4. **Audio-output parity test** (`test_qwen3_omni_audio_output_parity.py`, currently skipped) — greedy
   (temperature=0) + fixed seed, M*-new generated tokens vs HF/vLLM; needs M* + reference servers.
5. **Greedy-speech test** — run the speech path with `thinker_temperature=0` to confirm M* does not emit
   empty/crash; if clean, revert the harness speech-path thinker override to match `main`. (Note: this alone
   will NOT make audio 1-to-1 — the unseeded talker@0.9 is the dominant divergence; a shared sampling seed +
   greedy would be required for true output parity.)
6. (minor) Reconcile the GPU-image speedup range against `exp_imageenc/micro_raw.json` (docs use the sourced
   7–100×; an alternate microbench cited 12–440× at different image sizes — confirm on the run).

## Honesty notes a reviewer should know
- Outputs are **not** identical to vLLM (stochastic talker@0.9 + thinker@0.7, unseeded — M*'s native defaults,
  same as `main`); per issue #131 parity = **encoder + performance**, not audio-output identity (verified
  against the issue). Speech is compared on **length-normalized RTF**; M*-old's ~0.57 length makes its speech
  numbers flatter-than-fair (called out in RESULTS).
