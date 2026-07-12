# LEARNINGS: the i2t batch-TTFT investigation (2026-07-12 loop, 4h in)

## The verdict (what actually happened — backtrack complete)

**There is NO code regression.** The suspicion "we introduced a bug that
exploded TTFT" is falsified three independent ways:
1. Code archaeology: the flags-off i2t admission→encode→prefill→token path is
   byte-identical from `4c33b33` (encoders) to HEAD (`RESEARCH_MSTAR_TTFT_PATH.md`);
   every schedule/mixed/chunked/merged change is default-off. Only two
   default-ON flags exist; both exonerated by measurement:
   conductor-poll (POLL=0 vs 1: 671 vs 600ms — no effect) and FUSED_TOPK (µs).
2. Live bisect: the OLD encoders build measures 447/840ms (B16/B32) gated
   tonight — and the MODERN build with default flags measures **428/600-671ms
   at load 55-58**. The modern default build is as fast AND as load-robust as
   the old one.
3. Flag-group ladder ×2 rounds on the ship stack: results incoherent across
   rounds (same flags 0.34s↔6.2s) — no flag group is the culprit.

**The real phenomenon:** M* at i2t B32 bursts ~21 CPU cores (measured 2126%
CPU) during prefill/admission waves. Under *same-priority* neighbor CPU load
(root/kanzhu jobs; NOT nice-19 background — spinners at nice 19 don't
preempt and leave TTFT at 0.43-0.93s even at loadavg 122), the wave stalls
stochastically → multi-second, bimodal TTFT. Every historical "regression"
datapoint (576-vs-2532 lean table, chart 3426ms cells) is a cross-load
comparison artifact. The committed encoders sweeps (532-717ms) and the modern
quiet cells (629ms bs16v4) are the SAME number within the boot lottery.

**Why vLLM stays flat (~170ms) on the SAME single GPU** (`RESEARCH_VLLM_OMNI_TTFT.md`,
file:line refs inside): Thinker is TP=1 single-GPU (the old "they TP across 2
GPUs" diagnosis is WRONG); i2t is stage-0-only. Their recipe:
- ONE varlen-packed eager ViT forward for every scheduled image
  (per-step encoder budget = 32768 embed tokens);
- unified per-step token budget co-schedules the FULL prefill + 31 decodes in
  ONE model step → new request's first token ≈ one ITL later; the stall is
  bounded to ~1 step by construction;
- async scheduling (no CPU bubble between steps), greedy batched argmax,
  detokenize in a separate OS process.
Net: their host-CPU demand per step is small and constant; ours is a
21-core burst orchestrated across processes — that is the entire difference.

## Scoreboard tonight (ship stack, natural load 25-80, port 8344)
- i2t B32: **8.68 req/s (ABOVE the vLLM committed band 8.03-8.59)**,
  1536 tok/s raw = **1843 length-normalized (> band top 1814)**; TTFT p50
  1.44s (r1) / 3.43s (r2) — variance-dominated; quiet windows: 0.6-0.9s.
- s2t B32 n=256: 40.3 req/s (+44%), **843 tok/s raw (+27% over 598-664)**,
  TTFT 284ms (band 215-282).
- i2t B1: 269ms/0.92rps at load 25 (vLLM 83-118/0.85-0.89 — rps wins, TTFT
  behind under load; 108-135ms in quiet windows).
- Length asymmetry reminder: M* generates ~176 tok/req vs vLLM ~212 on
  food101 (both engines' natural sampling; identical in committed h2h) —
  raw tok/s comparisons carry this ~1.20× factor.

## THE 20 FIXES (ranked; ★ = attacks the TTFT variance mechanism directly)

### A. Scheduling/admission (port vLLM's recipe — biggest lever)
1. ★ One-step co-admission: fold a NEW request's full prefill chunk into the
   current decode step under a large unified token budget (our MIXED_BATCH
   machinery + budgets is 80% of this; missing piece = admit at ARRIVAL, not
   at spec-yield/readiness boundaries; target budget ~32k like vLLM).
2. ★ Encoder step budget: batch ALL pending image encodes into one varlen
   ViT forward per step (BATCH_VISION_PREFILL batches per-wave; make it
   per-step with an embed-token budget instead of grid-capped waves).
3. ★ Async scheduling: build step N+1's inputs while N runs (DEFER-SAMPLE
   narrow variant from HANDOFF_V7 — still the right shape; vLLM proves the
   async-sched pattern is the load-robustness key, not just throughput).
4. ★ Admission fast-path: bypass the spec-chain yield gate for BRAND-NEW
   requests (first prefill) — arrival-triggered, like vLLM's per-step
   admission (today ~8% of steps admit via must_yield_away).
5. PD topology (qwen3omni_2gpu_pd.yaml, validated): decode isolation removes
   prefill-freezes-decode; adopt as ship topology (+63% s2t B32, ITL p99
   29ms; B32 8.08 at load 79).

### B. Host-CPU demand cuts (the 21-core burst is the vulnerability)
6. ★ Producer-side emit sequence numbers (replaces the consumer FIFO;
   review's top deferred item) — removes ordered-emit's arrival-order
   dependence AND the api-server hold logic.
7. Prefill-wave host work batching: one plan_attention + one rope plan for
   the whole vision wave instead of per-walk (plan reuse, board #22).
8. Per-rid cleanup index for ordered-emit maps (O(1) vs O(all-pending)).
9. Ordered-emit inline fast-path (skip FIFO churn when queue empty).
10. Shared sample+unpack helper for eager/captured paths (drift + one less
    per-wave Python body).
11. Batched WGD/control-plane pack (msgspec, one send per step — board #20);
    conductor hops are pure host latency on every walk transition.
12. Exile per-token Python further: extend the sidecar pattern to i2t rids
    (SIDECAR_WALKS excludes prefill_vision — that's why i2t rides legacy).
13. Move api_server detok/postprocess fully off the serve process (vLLM
    separates EngineCore from API proc; our uvicorn+preprocess share).

### C. Ops/environment (immediate, no code)
14. ★ CPU-set discipline: boot workers with a reserved core set (numactl
    --physcpubind) sized to the burst (24 cores node1), so neighbor jobs
    land elsewhere; document in launch scripts.
15. Benchmark protocol: load-gate <25 + n>=96 (B32) / n>=256 (s2t B32) +
    paired A/B only — single loaded cells produced every false alarm this
    week (budget "regression", ordered-emit "30% cost", encoders "576-vs-2532").

### D. Throughput polish (tok/s specifically)
16. Length-matched tok/s certification cells (--ignore-eos --output-len
    212/512) so tok/s comparisons stop carrying the 1.20× length artifact.
17. Greedy-parity default for benchmark configs (temp=0 both engines —
    vLLM 0.22 default IS greedy; our sampling produces shorter outputs and
    muddies every tok/s and rps comparison).
18. Batched argmax shortcut when temperature==0 across the batch (skip
    top-p/k machinery per-step; vLLM does this — sampler.py:239).
19. DP/long-output mode: at len>=512 the DP-replica topology already showed
    +16% tok/s over vLLM — certify it for long-output workloads.
20. Keep mixed/chunked machinery ON (this week's data: it is load-armor,
    not overhead — L1-off cells were the worst under contention; the old
    "lean" folklore inverted under load).

## Protocol lessons (write into every future session)
- Never compare cells across load regimes; the box swings 9→213.
- nice-19 synthetic load does NOT reproduce neighbor contention.
- Never edit a script a live bash is executing (incremental read corrupts).
- Walk registration is boot-time: dynflagging a merged/new walk ON routes
  admissions into a black hole. Merged walks are colocated-topology-only.
- s2t B32 at n<=96 is a wave lottery (39.7 vs 24.6 same flags): n>=256.
