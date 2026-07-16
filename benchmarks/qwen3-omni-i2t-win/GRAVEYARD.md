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
