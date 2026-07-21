# Single-config submission — what to check & how to replicate

**Branch:** `submission/single-config` (this branch). It is a strict superset of
`winning/dual-goal` + the single-config commit, i.e. **all the code AND all the
benchmark artifacts in one place**. 143 commits ahead of `origin/main` (`9ee13699`).

## The result (B32, natural-EOS, closed-loop, continuous batching, no DP/PD)
ONE served config beats vLLM-Omni 0.24 on every path:

| path | single config | vLLM-Omni 0.24 | ratio |
|------|---------------|----------------|-------|
| i2t (tok/s) | 1817.5 | 1744.6 | **1.04×** |
| s2t (tok/s) | 846.1  | 664.9  | **1.27×** |
| i2s (req/s) | 2.48   | 1.14   | **2.17×** |
| s2s (req/s) | 14.19  | 4.88   | **2.91×** |
| t2s (req/s) | 1.66   | 0.83   | **2.00×** |

## What to check (in this order)
1. **The mechanism** — `mstar/conductor/conductor.py`, `_assign_worker_graphs_to_workers`
   (routes the multi-rank DP-replica encoder group to the idle rank by output
   modality; call site in `_do_ingest_request` passes `body.initial_output_modalities`)
   and `configs/qwen3omni_2gpu_dpenc.yaml` (encoders `ranks:[0,1]`, no `tp_size` ⇒
   two BF16 replicas; Thinker r1, Talker+Code2Wav r0). ~30 lines, byte-identical.
2. **The flag stack** — `FEATURES.txt` (repo root): every winning flag, what it does,
   and that it is default-OFF/byte-identical when off.
3. **The data** — `submission/raw/<label>/<rt>_b<B>/results.json` for 4 series
   (`dpenc`=single config, `winning`=per-modality, `upmain`=m-star main, `vllm024`=competitor),
   5 paths × 6 batches. Charts: `submission/charts/final_{i2t,s2t,i2s,s2s,t2s}.png`.
4. **Regenerate the charts from the raw** (self-contained, no GPU):
   `python submission/scripts/gen_charts.py` → rewrites `submission/charts/*.png`.
5. **Parity** — `test/modular/test_merged_prefill_audio_autogate.py` (+ the other
   `test/modular/test_*_identity/parity` suites). CPU-only: `pytest test/modular/`.

## How to replicate the numbers on GPU (2× H200)
Prereqs (adjust paths at the top of the scripts for your box):
- **Server venv** with M* installed — `SVENV` (default `…/mstar-new/.venv`).
- **Client venv + bench harness** — `CVENV` (default `…/mstar-encoders/.venv`) and
  `BENCH` (dir containing the in-repo `benchmark` package). `LIBRI` = cached
  LibriSpeech wavs for the speech/s2t datasets.
- Two **idle, whole** GPUs (scripts abort if a GPU has >1 GiB used — no co-location).

```bash
# 1) boot THE one config (serves all 5 paths; no topology swap)
submission/scripts/boot_single_config.sh 6,7 8296     # <gpus> <port>; ~7–10 min to WARM+READY

# 2) sweep all 5 paths against it → writes submission/raw/dpenc/<rt>_b<B>/results.json
submission/scripts/run_sweep.sh 8296

# 3) regenerate charts
python submission/scripts/gen_charts.py
```
Notes: s2t B32 is wave-lottery sensitive — validate at n≥256 (the committed i2t/s2t
B32 cells use the n=256 read). The competitor (`vllm024`) and `upmain` series were
measured separately and are committed here as reference; per the owner rule, re-run
vLLM-0.24 through your own pipeline before defending those numbers.

## Everything-in-one-config, in one sentence (vs origin/main `9ee13699`)
See `PR_BODY.md` for the single-paragraph feature summary and `WRITEUP.md` for the
full method, parity evidence, and honest caveats (speech "2–3×" is req/s not
audio-sps; t2s ~2.0× is the weakest speech path; vLLM measured locally).
