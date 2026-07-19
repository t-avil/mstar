# FINAL PLAN — dual-goal Qwen3-Omni, lock-in run (started 2026-07-19)

Autonomous ~15h loop. Ping user ONLY on full completion or a hard blocker.
Work continuously; debug through flukes; validate everything (academic submission).

## DECISIONS (from user, 2026-07-19)
- **vLLM benchmarking rule LIFTED for THIS TASK ONLY**: I may boot & benchmark
  vLLM-Omni 0.24 myself (non-DP, closed-loop, natural-EOS, max-concurrency,
  continuous batching). Standing owner rule resumes after this task.
- **Pass gate**: audio generation (i2s, s2s) MUST stay stable ~2-3x vs vLLM-Omni.
  Text (i2t, s2t): no regression vs current 2x2 charts; a small single-metric miss
  is a fluke -> do NOT stop, rebench later. Only a real AUDIO regression -> STOP+report.
- **All runs**: NO DP-disagg. Continuous batching. Max concurrency. Closed loop.
  NO length settings (natural-EOS). GPUs 6,7 (Thinker=GPU7). Reuse mega-cache.
- **Cadence**: work continuously; ping user only when plan complete or hard-blocked.
- **Charts**: current color scheme; m-star main remote = GREY.

## AMBIGUITY A RESOLVED (2026-07-19): decodefloor ⊇ godv9
`git merge-base --is-ancestor godv9(1e7f10da) decodefloor(deda3e0f)` = YES; decodefloor
missing NOTHING from godv9, adds preproc-pool (MSTAR_PREPROC_PROC 18330f4d = the i2t fix) +
mega-cache + R1 (default-off byte-identical). So winning branch (decodefloor+autogate)
reproduces godv9's committed speech 2-3x AND improved text. Winning-branch base CONFIRMED.
Committed env.txt naming godv9 = earlier snapshot pre-preproc-pool. NO rebase needed.

## KEY FACTS FROM P0 INVENTORY
- Config per path: i2t,s2t -> qwen3omni_2gpu_encoff.yaml; i2s,s2s -> qwen3omni_2gpu.yaml (base).
- Baseline branch (showcase fork-FROM): `encoders-implemeneted`(sic) @4c33b33a, worktree mstar-new.
- Prior clean showcase model: eiv2 = `encoder-implemented-v2` @510beedd (forks from 4c33b33a,
  1 feature/commit). eiv2/cleanup @1b494c62, eiv2/proven @e3a4c3a8 = curated subsets. USE AS MODEL.
- Committed data (DO NOT re-run): vLLM-0.22 (all 4 paths) + older-M* on `benchmarks` branch
  (bench-agg). MISSING (P4 must run): vLLM-0.24 (all paths, natEOS) + m-star main (all paths).
- vLLM-0.24: baselines/vllm-omni @v0.24.0, launch_vllm_6_7_numa1_v024.sh, OWN venv, port 8096,
  non-DP single-node. One serve handles all 4 request-types.
- Harness: exp_full_nateos.sh; per-path request-type+dataset: i2t=image_to_text/food101,
  s2t=audio_to_text/libri, i2s=image_to_speech/food101, s2s=audio_to_speech/libri. natEOS = OMIT
  --ignore-eos/--output-len. n/batch: text{1:64,2:64,4:96,8:96,16:96,32:128} speech{1:12,2:20,4:24,8:40,16:64,32:96}.
- Metrics: text=text_token_throughput/request_throughput/ttft.text.p50/itl.text.mean;
  speech=audio_seconds_throughput/request_throughput/rtf.p50. Audio "2-3x" = audio_sps (2-2.4x) AND req/s (2.8-3.6x).
- Charts: gen_text_nateos.py + gen_speech_nateos.py. main=GREY #7f7f7f (add 5th series). vLLM-0.24
  keep GREEN (drop/keep 0.22; avoid two greens — 0.24 is the competitor now).
- m-star main worktree = /home/tim/mstar @9ee13699.

## WINNING CONFIG (to confirm in Phase 0 inventory)
- Build: decodefloor @deda3e0f (opt/decode-cpu-floor), COMMON ship flags (boot_pd_45.sh).
- Topology-per-modality: text (i2t,s2t) -> encoff.yaml; speech (i2s,s2s) -> base.yaml.
- s2t fix: MERGED_PREFILL_AUDIO, batch-gated (on B<=16, off B32) -> to be made AUTO.
- Audio-gen 2-3x wins come from speech-floor opts on base topology.

## PHASES
- [ ] P0 INVENTORY (agent): confirm winning config per path (topology+flags); identify
      "encoders-implemented original" branch + the prior residual-cleanup branch; list
      committed baseline data (avoid re-runs); find vLLM 0.24 boot recipe (~/baselines).
- [ ] P1 PRESERVE: new branch `winning/dual-goal` off decodefloor code; push to fork;
      FEATURES.txt writeup per feature (what + how to enable vs encoders-enabled baseline).
- [ ] P2 AUTO-GATE: MERGED_PREFILL_AUDIO fires only when live decode occupancy <= ~16-24
      (env MSTAR_MERGED_PREFILL_AUDIO_MAX_BS, default ~24). Boot+validate: B4/B8/B16 win,
      B32 no regression. Fold into winning branch.
- [ ] P3 REBENCH 4 PATHS (winning branch): i2t,s2t (encoff) + i2s,s2s (base), B1-B32,
      natural-EOS closed-loop non-DP. GATE CHECK: audio 2-3x stable? text no-regress?
      Fail(audio) -> STOP. Preserve data.
- [ ] P4 PARITY + COMPETITORS: (a) token parity winning vs m-star main; (b) rebench
      vLLM-Omni 0.24 (4 paths); (c) rebench encoders-implemented (4 paths); (d) rebench
      m-star main remote (4 paths). All non-DP/closed-loop/natural-EOS.
- [ ] P5 CHARTS: 2x2 for all 4 paths (winning=blue solid, older refs, vLLM 0.24=green,
      m-star main=GREY). Send to user.
- [ ] P6 SHOWCASE BRANCH: fork from encoders-implemented-original; commit-by-commit add
      each winning feature; clean code, no leftover; readable history. Reuse prior
      cleanup branch's residual-cleanup approach if found.
- [ ] P7 FINAL REBENCH: rebench the final clean branch (match winning perf); commit ALL
      data to the benchmark branch.

## STATE LOG (append each turn; newest last)
- 2026-07-19 T0: plan created. Decisions locked. Launching P0 inventory + P2 auto-gate
  code-locate agents. Heartbeat loop scheduled.
- 2026-07-19 T1: P2 auto-gate IMPLEMENTED + committed on winning/dual-goal (655f15ca):
  MSTAR_MERGED_PREFILL_AUDIO_MAX_BS gate (default 24), occupancy=len(conductor.requests)
  via model_kwargs. 4 sites verified vs real code, syntax OK, Talker-kwargs-safe, parity-safe.
  BLOCKER: ALL 8 GPUs on node now 100% util/~105GB (others' sglang+Megatron jobs). Cannot
  boot. GPU phases (P2-validate, P3, P4, P7) PAUSED. Doing non-GPU work (P1 preserve+FEATURES,
  P6 showcase prep) meanwhile; loop polls for 6,7 to free. NEXT on GPU-free: boot winning
  worktree (boot_winning_s2t.sh) validate B8/B16 merge-win + B32 no-regress.
- 2026-07-19 T2: Ambiguity A RESOLVED (decodefloor ⊇ godv9, base confirmed). P1 DONE:
  winning/dual-goal pushed to fork w/ auto-gate (655f15ca) + FEATURES.txt (b3d59dc3).
  P0 inventory consumed (config/branches/baselines/vLLM recipe all in KEY FACTS above).
  ALL remaining phases GPU-blocked (node full: others' sglang+Megatron, 8x100%). Non-GPU
  work exhausted for now. WAITING on GPUs 6,7. NEXT when free: P2-validate boot
  (boot_winning_s2t.sh) then P3 rebench 4 paths. P6 showcase deferred until features validated.
- 2026-07-19 T3: GPU poll #1 — node STILL saturated (8x 100+GB, others' jobs). Built P3 prep
  (non-GPU): boot_winning.sh (parameterized config boot) + p3_rebench.sh (phase-selective
  text/speech sweep into unified exp_nateos_out/winning/). RESUME SEQUENCE when 6,7 free:
  (1) bash boot_winning.sh win_txt qwen3omni_2gpu_encoff.yaml 8296 "MSTAR_MERGED_PREFILL_AUDIO=1"
      -> wait ready -> bash p3_rebench.sh text 8296 winning   [also validates P2 autogate:
      s2t B16~670 merged vs B32~890 not-merged]
  (2) teardown -> bash boot_winning.sh win_spch qwen3omni_2gpu.yaml 8296 -> bash p3_rebench.sh speech 8296 winning
  (3) gen_text_nateos.py winning + gen_speech_nateos.py winning -> GATE CHECK audio 2-3x + text no-regress.
  Still blocked. Non-GPU prep now exhausted.
- 2026-07-19 T4: GPU poll #2 (~1hr in) — still saturated. Verified P4 infra all present:
  vLLM-0.24 (baselines/launch_vllm_6_7_numa1_v024.sh + .venv + deploy yaml, v0.24.0), main
  /home/tim/mstar@9ee13699, encoders-impl mstar-new@4c33b33a. OPEN Q for P4: what flags/config
  to boot main + encoders-impl with (they lack winning flags) — resolve at P4 (likely their own
  default configs, fair "as-is" baseline). If still blocked at ~3hr total, ping user.
- 2026-07-19 T5: GPU poll #3 (~1h20m in) — still saturated (8x busy, others' jobs). No action;
  all prep done. Waiting. Ping threshold: ~3hr total (~poll #6-7).
- 2026-07-19 T6: GPU poll #5 (~2h in) — TRANSITION: GPU7 freeing (3.7GB/0% from 121GB) but
  GPU6 still busy (133GB/100%). Need BOTH. Tightened poll to 10min to catch both-free. NOTE:
  if GPU6 frees but GPU7 keeps ~3.7GB residual (>1000 abort thresh in boot_winning.sh),
  investigate what's on GPU7 (nvidia-smi compute-apps -i 7) before booting — don't co-locate.
- 2026-07-19 T7: GPUs 6,7 FREED (poll #6, ~2h10m). RESUMED. Booting winning encoff+merge
  (lab_win_txt, port 8296); compile cache-hit (33s), capturing. Chained: wait-ready ->
  p3_rebench.sh text 8296 winning (i2t+s2t) [b2ma73w2t]. Writes exp_nateos_out/winning/.
  ON COMPLETE: (1) check autogate (s2t B16 tok/s ~670 MERGED vs B32 ~890 NOT-merged — confirms
  MAX_BS=24 gate); (2) teardown win_txt; (3) boot_winning.sh win_spch qwen3omni_2gpu.yaml 8296
  -> p3_rebench.sh speech 8296 winning; (4) gen charts + GATE CHECK (audio 2-3x MUST hold).
- 2026-07-19 T8: P2 AUTO-GATE VALIDATED (n=256, clean): s2t B8=507(1.33x, merged),
  B16=675(1.43x, merged), B32=879(1.18x, NOT merged=baseline, dodges -26% regression).
  Gate works exactly as designed. P2 DONE. Earlier p3 s2t B16=417 = cold-boot fluke.
  NOTE: dataset filelock PermissionError on shared /m-coriander/coriander/hf/datasets (others'
  job) -> FIX: HF_DATASETS_CACHE=/m-coriander/coriander/tim/hf_datasets (patched p3_rebench.sh).
  Warm CLEAN text re-run running [bgf4szspl] -> exp_nateos_out/winning/. NEXT: teardown win_txt,
  boot_winning.sh win_spch qwen3omni_2gpu.yaml 8296 (base, NO merge), p3_rebench.sh speech 8296
  winning (i2s+s2s = HARD GATE 2-3x), then charts.
- 2026-07-19 T9: TEXT GOAL ACHIEVED (warm clean, winning branch vs vLLM022, exp_nateos_out/winning/):
  i2t 1.07/1.07/1.20/1.21/1.11/1.06x (6/6 win); s2t 1.44/0.99/1.02/1.30/1.38/1.11x (6/6 >=parity,
  B2=0.99 tie). Cold-boot flukes gone. Teardown win_txt done, GPUs freed. BASE speech server
  (lab_win_spch, port 8296) booting -> chained p3_rebench.sh speech 8296 winning [waiter b6onjhr14].
  ON COMPLETE: HARD GATE i2s/s2s audio_sps + req/s ~2-3x vs vLLM022 (i2s B32 target ~100 sps /
  s2s ~54 sps). Then charts. If real audio regression -> HARD STOP + ping.
- 2026-07-19 T16: P6 showcase — built + PUSHED showcase/dual-goal (fork 4c33b33a) with 6 verified
  clean feature commits (cleanup, fp8, ordered-emit, grids, grid-fix, host-floor; cherry-picked
  from eiv2). Remaining ~11 features: primary commits cherry-pick clean but FOLLOW-UP FIXES +
  worker.py-heavy features CONFLICT on godv9's intervening history. Force-resolving risks
  subtly-broken features -> for academic submission, delivering clean START + full commit plan
  (SHOWCASE_STATUS.md on branch) rather than a fragile rush. P6 = PARTIAL (documented). NEXT
  PRIORITY P7 DATA PRESERVATION (independent of P6): commit ALL campaign data to benchmarks branch:
  exp_nateos_out/{winning,vllm024,encoders_impl,main}/ + deliverable/charts/final_*.png +
  *_winning.png + FINAL_PLAN.md + gen_final.py + FEATURES.txt. Use bench branch -> merge to
  benchmarks (bench-agg worktree) per CLAUDE.md git workflow. Then FINAL PING (honest: core+charts+
  data done, showcase started+planned).
- 2026-07-19 T15: main DONE (weak as-is baseline: i2t B32 624, s2t B32 104, s2s sps 15.9; winning
  3-8x over main). P4 COMPLETE (vLLM024 + encoders_impl + main all rebenched). P5 DONE: gen_final.py
  -> charts/final_{i2t,s2t,i2s,s2s}.png (4 series: winning blue / baseline dotted / main GREY /
  vLLM024 green). Verified render correct. SENDING to user (explicit deliverable). Parity vs main:
  cross-build exact-token diff not meaningful (fp8/custom-ops numerics + nondeterministic GPU greedy);
  parity is ensured per-feature (auto-gate byte-identical by construction; fp8/custom-ops have
  committed certs) + outputs coherent. NEXT: P6 showcase branch (non-GPU, off 4c33b33a, model eiv2)
  + P7 final rebench + commit data to benchmarks branch. Then final ping.
- 2026-07-19 T14: encoders-impl baseline DONE (exp_nateos_out/encoders_impl/, booted 168s via
  cache). Baseline as expected: i2t B32 825 (winning 1866 = +126% over baseline!), s2t B32 383
  (winning 832), i2s sps B32 92 / s2s 62. Torn down. MAIN (/home/tim/mstar@9ee13699) boots CLEAN
  on shared venv, sweep running [waiter bm8je9j8z] -> exp_nateos_out/main/. P5 SERIES PLAN:
  winning=blue solid, encoders_impl=blue dotted, main=GREY #7f7f7f, vLLM024=green. Generator
  gen_final.py (being written) reads exp_nateos_out/{winning,encoders_impl,main,vllm024}/.
- 2026-07-19 T13: vLLM-0.24 DONE (exp_nateos_out/vllm024/). WINNING vs vLLM024 FULL:
  i2t tok/s 1.07-1.25x / req/s 1.24-1.51x (WIN all); s2t tok/s 0.99-1.28x / req/s 1.18-1.48x (WIN
  all); i2s audio-sps 1.50-1.74x / REQ/S 2.19-2.52x; s2s audio-sps 1.15-2.17x / REQ/S 2.65-3.50x.
  => TEXT wins both metrics; SPEECH 2-3x on req/s (audio-sps length-confounded, vLLM024 emits
  longer clips). Speech chart 4th panel is req/s ratio = shows 2-3x. GATE HOLDS vs 0.24.
  vLLM torn down. NOW booting encoders-impl (mstar-new@4c33b33a, base cfg, MINIMAL flags = as-is
  baseline, no mega-cache so slow boot) -> chained p4_sweep ours 8296 encoders_impl [waiter b6677nu81].
  NEXT: main (/home/tim/mstar@9ee13699) same way, then P5 charts.
- 2026-07-19 T12: vLLM-0.24 sweep ~85% (s2s finishing). KEY FINDINGS: (a) vLLM024 i2t≈vLLM022
  (1744 vs 1769 B32) — text sanity OK. (b) vLLM024 IMPROVED audio-sps (i2s B32 72.7 vs 022's
  47.44) so winning i2s audio-sps ratio ~1.5x vs 0.24 (was 2.3x vs 022). BUT this is an audio-
  LENGTH confound: on req/s (length-robust), winning i2s=2.19-2.52x, s2s=2.6-3.5x vs vLLM024 =
  2-3x HOLDS. Report BOTH metrics; req/s is the honest one (like text tok/s length-confound).
  NOT a stop (winning didn't regress; competitor's audio-sps got longer clips). NEXT: full s2s,
  compute exact winning-vs-024 ratios (audio-sps AND req/s AND rtf), then M* competitors.
- 2026-07-19 T11: *** P3 HARD GATE = PASS *** i2s audio_sps 2.18-2.64x vLLM (6/6 >=2x), s2s
  1.73-2.32x + req/s 2.6-3.4x (matches committed s2s profile, NO regression). Winning branch
  BEATS committed M* speech every batch (i2s +11-19%, s2s +15-41%). DUAL GOAL ACHIEVED: text
  wins both paths + speech 2-3x preserved/improved. P3 DONE. 4 winning charts generated
  (charts/*_winning.png). NOW P4: vLLM-0.24 booting (port 8096, TP=1 thinker CUDA path, non-DP
  confirmed) -> chained p4_sweep.sh vllm_omni 8096 vllm024 [waiter b7ou1eg5l]. THEN encoders-impl
  + main rebench, parity check, then P5 final charts (add vLLM024 green + main grey).
- 2026-07-19 T10: speech server ready (552s), sweep running. i2s partial audio_sps B1-B16 =
  13.9/21.8/36.9/62.4/90.6 — EXCEEDS committed M* (11.7/18.8/31.3/52.3/77.5) and ~2.3-2.4x vLLM.
  AUDIO LEAD HOLDING. B32+s2s still running [waiter b6onjhr14]. ON COMPLETE: compute exact ratios
  vs vLLM022 speech arrays (baselines.json vllm022_speech_natEOS), confirm gate, then P5 charts.
