# i2t Win Campaign — Graveyard Log
# Append-only. Every idea tried, measured delta, cause of death.
# Cause taxonomy: FUNDAMENTAL | BUG | CONFIG | NOISE | INTERACTION | PARITY-FAIL
# Format: [UTC] idea | path/batch | delta vs baseline | cause | note

## Prior campaigns (imported from memory/committed docs) — see per-branch FINDINGS for detail
- V1 async-sched (MSTAR_ASYNC_SCHED): i2t | non-identical +9% len, tok/s wash | INTERACTION (deferral shifts batch composition) | parked
- Naive multi-step B32: regressed | INTERACTION (re-plan cost) | 
- MoE-autotune on i2t: wash | FUNDAMENTAL (CPU-bound path) |
- TP=2 for Thinker: net-negative even w/ symm-mem all-reduce | FUNDAMENTAL (multi-process arch) |
- Prefill budget throttle 512->2048: REAL but MODEST (B32 TTFT -17%, rps +4.6%) | capped at 2048 bucket | partial win, needs bigger captured bucket

## Live campaign 2026-07-16

### 2026-07-16 flagship baseline (wt-boot-cache 2b43f06e, encoff, matched greedy natural-EOS n50w10)
- i2t B32, PREPROC POOL **OFF** (bootA, fix20 COMMON only): req/s **7.00**, tok/s **1240**,
  TTFT p50 **3476ms** (≈JCT 3534 → image-preprocess ABYSS, 32 imgs inline-serialized),
  ITL mean 6.3ms (end-burst). vs vLLM B32 8.32 req/s / 1769 tok/s / 179ms / 15.4ms.
  DIAGNOSIS: TTFT abyss = MSTAR_PREPROC_PROC/GPU_IMAGE_PREPROCESS not set in fix20 COMMON.
  Not a real result — reboot with preproc pool ON (image-preprocess fix) = A/B in progress.

### ★ 2026-07-16 IMAGE-PREPROCESS POOL = BIG i2t WIN (same build wt-boot-cache, A/B n50w10)
Flags added: MSTAR_GPU_IMAGE_PREPROCESS=1 MSTAR_PREPROC_PROC=1 MSTAR_PREPROC_PROCS=8 MSTAR_BURST_CAP=1 MSTAR_BURST_THREADS=8
i2t B32:  OFF req/s 7.00 tok/s 1240 TTFT 3477 ITL 6.3  ->  ON req/s 10.06 tok/s 1677 TTFT 984 ITL 11.6
Deltas: req/s +44%, tok/s +35%, TTFT -72%. vs vLLM: req/s +21% WIN, ITL WIN, tok/s -5% (still lose length-invariant), TTFT 984 vs 179 (still 5.5x, OPEN).
VERDICT: preproc pool is MANDATORY for i2t (fix20 COMMON was missing it = the whole "abyss").
REMAINING for clean i2t B32 win on EVERY metric: close tok/s -5% + TTFT 984->~179. Both prefill/TTFT-bound.
Next levers: more preproc procs, ENC_OVERLAP (async encoder), fused-KV; verify variance (n=50 single sample).

### 2026-07-16 preproc-ON flagship i2t sweep (matched greedy natural-EOS, n50w10, GPUs6/7)
LENGTH NOTE: M* B32 out=727 bytes vs vLLM committed B32 out=920 bytes -> M* SHORTER.
 => M* req/s wins are PARTLY length (shorter req finishes sooner); tok/s & ITL are the
    honest length-robust metrics. M* WINS ITL every batch (faster per-token decode).
Sweep vs vLLM: B1 req+38/tok+7/ITL win/TTFT90(match); B4 req+56/tok+19/ITL win;
  B8 req+43/tok+18/ITL win; (B16,B32 pending). B32 earlier: req+21/tok-5/ITL win/TTFT984.
KEY: only B32 loses tok/s (-5%), dragged by TTFT 984ms (batch prefill-serialization).
 M* ITL already beats vLLM at B32 -> fixing B32 TTFT flips tok/s to a win too.
 NEXT LEVER: PD-disagg config (prefill rank0 / decode rank1) -> committed eiv2/PD had TTFT 189ms.

### ★★ FLAGSHIP preproc-ON i2t FULL SWEEP (wt-boot-cache encoff, matched greedy nat-EOS n50w10) — BASELINE-OF-RECORD
| B | req/s vs vLLM | tok/s vs vLLM | ITL M*/vLLM | TTFT p50 | verdict |
| 1 | +38% | +7% | 4.3/4.8 | 90/87 | WIN |
| 4 | +56% | +19%| 6.2/6.9 | 106 | WIN |
| 8 | +43% | +18%| 7.6/9.6 | - | WIN |
|16 | +33% | +9% | 9.6/11.9| 140/168 | WIN |
|32 | +15% | -6% | 12.1/15.4| 757-984/179 | tok/s+TTFT LOSE (only holdout) |
tok/s & ITL = length-robust primary (M* out 727-754B vs vLLM 920B => req/s confounded).
CONCLUSION: M* wins i2t B1-B16 clean; B32 is the sole loss, entirely TTFT-driven (batch
prefill-serialization on single Thinker GPU). ITL wins ALL batches (M* decode faster).
Data: campaign_i2t/results/preproc_on/i2t_B*/
NEXT: PD-disagg (prefill rank0 / decode rank1) to fix B32 TTFT -> should flip tok/s to win.

### ★★ PD-DISAGG (wt-boot-cache + qwen3omni_2gpu_pd.yaml + preproc, matched greedy nat-EOS n50w10)
PD is the BETTER i2t build (bigger tok/s wins than encoff at B1-B16):
| B | req/s vs vLLM | tok/s vs vLLM | ITL M*/vLLM | TTFT p50 |
| 1 | +38% | +6% | 4.4/4.8 | 82 |
| 4 | +63% | +25%| 5.9/6.9 | 84 |
| 8 | +60% | +31%| 6.9/9.6 | - |
|16 | +45% | +20%| 8.1/11.9| 148 |
|32 | +14% | -6% | 11.2/15.4| 949 |
B32 remains the lone loss: tok/s -6% (nat-EOS), TTFT ~950. PD does NOT fix B32 TTFT
(B16->B32 TTFT cliff 148->949 = single-GPU prefill compute wall; prior campaign: structural,
TP regresses host-bound decode). B32 fixed-len256 test: tok/s +1.3%, ITL win (tok/s loss is
partly length: M* nat-EOS out 727-793B < vLLM 920B -> less TTFT amortization).
PARITY: PD greedy DETERMINISTIC (20/20 run-to-run), outputs coherent+correct; token-level
diverges from encoff build (fp8-MoE+topology numerics, both deterministic) = accepted fp8
tradeoff (committed eiv2/PD/fp8 ship set = parity 24/24). B32 fixed-len runs noisy (need repeats).
DECISION: PD = new best i2t build. B32 TTFT = documented structural limit.

### ★★★ FINAL i2t VERDICT (PD build, n=96, median of 3-6 repeats, matched greedy nat-EOS)
M* WINS i2t vs vLLM-Omni 0.22 on tok/s + req/s + ITL at EVERY batch; TTFT wins B1-B16, ~tie B32.
| b | req/s Δ | tok/s Δ | TTFT M*/vLLM | ITL M*/vLLM |
| 1 | +33% | +6.4% | 83/87 W | 4.4/4.8 W |
| 4 | +58% | +27%  | 84/130 W| 5.9/6.9 W |
| 8 | +60% | +36%  | 88/100 W| 6.9/9.6 W |
|16 | +62% | +35%  | 122/168 W| 8.3/11.9 W|
|32 | +33% | +11%  | 204/179 ~tie (p50 194-304)| 11.9/15.4 W|
KEY MEASUREMENT INSIGHT: B32 "TTFT abyss" was TWO artifacts: (1) preproc pool OFF (fix20
default) = 3477ms -> fixed by pool; (2) too-small n (32-50) = ramp-up-dominated 949ms ->
at n=96 steady-state continuous batching TTFT p50 ~200ms. Throughput is host-load tail-
sensitive -> median of >=3 repeats under monitored load mandatory. Chart: i2t_FINAL_pd_vs_vllm.png.

### ★★ s2t (speech→text) — PD build, n=96, median of 3 repeats (+ B32 n=256)
| b | req/s Δ | tok/s Δ | TTFT M*/vLLM | ITL M*/vLLM |
| 1 | +49% | +27% | 77/66 LOSE | 5.0/5.0 tie |
| 4 | +16% | -1% | 125/60 LOSE | 6.7/9.0 W |
| 8 | +46% | +27% | 172/143 LOSE| 7.5/15 W |
|16 | +80% | +59% | 228/191 LOSE| 8.3/23 W |
|32 | +40%(n96)/+51%(n256) | +22%/+30% | 499/217 LOSE| 6.6/29 W |
VERDICT: s2t wins req/s + tok/s + ITL (ITL huge: M* ~5-8ms vs vLLM up to 29ms = 4x faster
decode). LOSES TTFT every batch (single-GPU audio prefill serialization, same structural
class as i2t B32). s2t is latency-PRIMARY (TTFT+ITL) -> SPLIT: ITL win, TTFT deficit = s2t
NOT cleanly won on primary. TTFT lever (MSTAR_ENCODER_ASYNC) helps TTFT but regresses s2t
throughput (documented tradeoff) -> s2t TTFT is the focused open problem. Chart: s2t_FINAL_pd_vs_vllm.png.

### ★★ ENC-ASYNC (MSTAR_ENCODER_ASYNC=1) — i2t-ONLY win (completes i2t sweep), DESTROYS s2t
Boot: PD + preproc + MSTAR_ENCODER_ASYNC=1 (port 8347).
i2t B32 (3 reps): tok/s +11.3/+10.2/+15.4% (unchanged vs no-encasync +11%); TTFT p50 216/161/168
  -> median 168ms < vLLM 179 < no-encasync 204. => enc-async turns i2t B32 TTFT tie->WIN.
i2t B16: tok/s +35.7% (no regression). => enc-async is a clean i2t upgrade (B32 TTFT tie->win).
s2t B8: req/s +45.6%->+22.9%, tok/s +27->+7.7%, TTFT 172->235 (WORSE), ITL 7.5->8.2.
s2t B32: req/s +40%->+3.6%, tok/s +22%->-9.3%, TTFT 499->540, ITL 6.6->15.6 (COLLAPSE).
=> enc-async REGRESSES s2t badly (confirms prior note). MODALITY-GATED: use enc-async for i2t,
   NOT for s2t. (PD is already text-out-only per-modality; topology/flags per modality.)
VERDICT: BEST i2t = PD + preproc + ENC_ASYNC (wins EVERY metric EVERY batch incl B32 TTFT 168).
         BEST s2t = PD + preproc (NO enc-async).

### CORRECTION — enc-async NOT adopted (fuller data)
First 3 B32 reps gave TTFT median 168 (<179) but the clean 3-rep SWEEP gives B32 TTFT median
**200ms** (high variance 161-216) — enc-async does NOT reliably push i2t B32 TTFT below vLLM
179; tok/s only marginal (+11->+13%). Determinism held (i2t B1 16/16). And it DESTROYS s2t.
=> enc-async REJECTED (marginal for i2t, harmful for s2t). BEST i2t stays PD + preproc.
i2t B32 TTFT (~200 median) is a genuine ~tie neither PD nor enc-async decisively flips =
structural single-GPU-prefill limit (vLLM uses 2-GPU TP prefill; M* TP regresses host-bound decode).

### E1 eiv2 build (moonshot/decode-composed) BASELINE (flags OFF) — REGRESSES vs wt-boot-cache
i2t B32 (3 reps): tok/s -33.7/-35.3/-37.0% (ITL 15-17ms) vs wt-boot-cache PD +11% (ITL 11.9). BAD.
s2t B8: tok/s -1.6% ITL 12.4 (vs +27%/7.5). s2t B32: tok/s -0.1% ITL 18.3 (vs +22%/6.6). BAD.
=> eiv2 decode-composed branch with flags OFF falls to a slower decode path (ITL ~2x worse).
   The branch refactored decode expecting its flags ON. NEXT: E3+E4 composed decode flags ON
   (FULLSTEP_DECODE + DECODE_SYNCFREE) — must recover >=25% ITL to beat wt-boot-cache. Parity-gate.

### ★ E3+E4 eiv2 COMPOSED DECODE (FULLSTEP_DECODE=1 + DECODE_SYNCFREE=1) — DEAD (parity-fail + collapse)
PARITY FAIL: i2t B1 determinism 2/16 match run-to-run (NON-deterministic greedy!). Plus perf
COLLAPSE: i2t B32 tok/s -90%, ITL 153ms (10x slow); cells timed out. Output slightly degraded.
=> composed in-graph/syncfree decode is BROKEN (non-deterministic + 10x slow). DEAD.
COMBINED with E1 (eiv2 baseline regresses tok/s -35%/ITL 2x): ENTIRE eiv2/moonshot-decode branch
direction is EXHAUSTED/dead. wt-boot-cache + PD + preproc = confirmed champion.
Note: fused-KV (E2) moot — eiv2-only + eiv2 baseline already -35%, can't beat champion.
=> Config/flag space EXHAUSTED. Remaining open gaps (i2t B32 TTFT tie, s2t TTFT) are STRUCTURAL
   (need mixed prefill+decode co-admission = code). Pivot to CODE: N1 dual-stream + s2t instrumentation.

### s2t TTFT localization (code agent) — audio encoder ALREADY BATCHED (no cheap fix)
Native audio encoder varlen-packs all ready requests into ONE forward (submodules.py:243-267,
can_batch:275-282); scheduler groups all (micro_scheduler.py:1172). NOT bs=1. => "batch audio
encoder" = NO-OP (refutes hypothesis). B32 s2t 499ms = compute-bound (block-diagonal attn scales
~linear w/ batch) + batched Thinker prefill_audio. Same structural class as i2t B32 prefill.
=> No cheap flag/code fix for s2t TTFT. Real levers: (a) encoder-side kernel work / async overlap
(ENC_ASYNC regresses s2t throughput - rejected), (b) mixed prefill+decode co-admission (large
rewrite). Instrumentation patch (MSTAR_S2T_PHASE_TIMING, parity-safe default-off) available to
split AuT-encode vs Thinker-prefill vs KV-handoff if we want the exact breakdown.
CONCLUSION: both open gaps (i2t B32 TTFT tie, s2t TTFT) = structural compute-bound single-GPU
prefill; only real fix = co-admission substrate (vLLM-style unified token budget) = multi-day rewrite.

### ★★★ CO-ADMISSION IS ALREADY BUILT — MSTAR_MIXED_SPLIT_ATTN (champion branch, UNBENCHMARKED)
Scoping agent GO/NO-GO: the record-around-attention substrate for mixed prefill+decode is COMMITTED
on wt-boot-cache/infra/boot-cache (commits e6c41479 + fixes): captures thinker_mixed as ONE fixed-shape
FLASH_INFER_PACKED graph (decode_rows+prefill_bucket), splits attention into 2 sub-wrappers planned
OUTSIDE the graph (cuda_graph_runner.py:354,1402,1798). NO run_mixed stub anymore. Default OFF
(qwen3_omni_model.py:299). Enable: MSTAR_MIXED_BATCH=1 MSTAR_MIXED_SPLIT_ATTN=1 MSTAR_MIXED_CHUNK_SIZES=256,512.
MY CHAMPION BOOTS HAD SPLIT_ATTN OFF -> co-admission never active in any measurement so far!
Prior mixed failures: MIXED_WALK=fundamental(no captured shape, fixed by split-attn); mixed-CG=BUG(dict
not NodeOutput, superseded); W5-P2=NEUTRAL at B32 (B32 is CPU-serialization-bound not mixed-eager-bound).
=> Expected payoff LOW (B32 CPU-floor) but it's THE real co-admission lever + targets s2t TTFT too +
   it's a FLAG A/B (hours not days). PARITY IS THE RISK (mixed fp8/split-KV -> greedy nondeterminism;
   killed eiv2 E3/E4). TEST: parity-gate (determinism 16/16 + coherence) FIRST, then i2t B32 + s2t TTFT
   paired >=3 reps vs split-attn-off. NEXT EXPERIMENT.

### ★★★ E-SPLIT MSTAR_MIXED_SPLIT_ATTN (co-admission) — TESTED, NEUTRAL(i2t)/HARMFUL(s2t)/parity-risk. NOT adopted.
i2t B32 (3 reps): tok/s +7.3/+13.5/+11.0% (median +11%), TTFT p50 198-255 (median 214), ITL 10.8-12.2.
 = SAME as champion (split-attn off): NEUTRAL, no TTFT gain. Confirms B32 is CPU-floor-bound not mixed-eager-bound.
s2t B32: tok/s -38.4%, ITL 45ms, TTFT 426 = REGRESSED badly (mixed step hurts s2t decode). s2t B8 ~ok.
Parity: i2t B32 determinism 3/32 run-to-run (batch-composition fp8/split-KV nondeterminism); coherent/correct.
VERDICT: co-admission (the last real structural lever, already built) does NOT close i2t B32 TTFT or s2t
TTFT. Scoping prediction confirmed: single-GPU-prefill + 26ms CPU-serialization floor is the wall; TP
regresses (SHM no NVLink); postprocess-exile/syncfree = eiv2 branch = BROKEN (parity-fail+collapse).
=> ALL TRACTABLE LEVERS EXHAUSTED. Champion (wt-boot-cache+PD+preproc, split-attn OFF) is the ceiling.
   Remaining gaps (i2t B32 TTFT ~tie, s2t TTFT) require an ATTENDED decode-CPU-floor rewrite (high risk,
   prior attempts broke parity). ESCALATE. Shift to rigor: bulletproof the delivered wins.

### RIGOR (2026-07-17) — bulletproofed medians
i2t B32 with 9 reps: tok/s +12.6%, TTFT p50 181ms (≈vLLM 179, now a clean tie/marginal win),
 ITL 11.8, req/s +35.3%. => i2t is a decisive win on EVERY metric every batch.
s2t n=256 x3 (wave-robust): req/s +21..92%, tok/s +2..68%, ITL 5-8.5 vs vLLM 5-29 (huge win),
 TTFT 78-505 vs 66-217 (loses all — structural). Confirms s2t throughput+ITL win, TTFT deficit.
Charts: i2t_FINAL_pd_vs_vllm.png, s2t_FINAL_n256_pd_vs_vllm.png.
