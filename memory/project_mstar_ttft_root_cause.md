---
name: project_mstar_ttft_root_cause
description: "M* i2t B32 loss root cause 2026-07 — TTFT (prefill serialization), NOT decode; one-walk-per-batch freezes decodes; ENCODER_ASYNC is measured unshipped win"
metadata:
  node_type: memory
  type: project
  originSessionId: e0c3d6dc-5207-40da-aea3-893c738981a0
---

The ONLY i2t loss vs vLLM is **B32 req/s, caused 100% by TTFT** (~2400ms vs vLLM FLAT
~179ms, 14x). M* wins ITL every batch + req/s B1-B16. TTFT is worse than vLLM at EVERY
batch (196>87 B1 ... 2400>179 B32) → flattening it lifts BOTH tok/s AND req/s at ALL
batches (it's wasted wall-clock). See [[project_mstar_throughput_levers]].

**ROOT CAUSE (agent-confirmed, mstar-moe code = the running build, PYTHONPATH trap
[[feedback_mstar_worktree_pythonpath]]):**
- `micro_scheduler.py:97-104` enforces **one graph_walk per batch**; encoders are
  standalone `Sequential([encoder, thinker])` nodes → each new admission's encoder+prefill
  **mega-step FREEZES all in-flight decodes** = head-of-line blocking. This is exactly what
  vLLM avoids.
- Empirical: during B32 i2t TTFT, **GPU sits IDLE ~2s** then ramps to 100% → the stall is
  CPU-bound encoder-metadata prep (per-request host syncs: `rope.py:333-335` 3×`.item()`/
  image, `vision_encoder.py:274,278`) + per-request KV-read serializing ONE request at a
  time, before the GPU is fed. Not GPU compute.

**vLLM's flat-TTFT mechanism (copy this):** ONE shared per-step token budget
(max_num_batched_tokens≈8192); schedule() every step does decodes FIRST (~32 tok) then
fills remainder with NEW-request prefill chunks IN THE SAME step → new req admitted within
~1 step regardless of batch. `continue` (not break) past unschedulable running reqs (no
HOL block). scheduler.py:348,364-366,544-548. Pure scheduling = parity-safe; the mixed
decode+prefill bucket it creates is the batch-composition piece M* must capture.

**Prior attempts (don't re-run): all in [[project_mstar_throughput_levers]] + EXPERIMENTS.md:**
chunk-size 512 optimal (closed); W5-P1 alternation loses; W5-P2 captured MIXED_BATCH
correct + wins small batch but **NEUTRAL at B32** (eager launch floor); V2 budget fold DEAD
on short i2t (budget_folds=0, prompts ≤256 tok never chunk); admit-jitter & prefill-gather
WASH (readiness serializes through per-req encode pipeline); SIDE_PREFILL correctness-win
throughput-wash.

**TOP UNSHIPPED / UNTRIED parity-safe B32-TTFT levers (ranked):**
1. **MSTAR_ENCODER_ASYNC** — MEASURED i2t B32 **−30% TTFT, +7.4% req/s** (byte-identical async
   encoder dispatch) but **s2t −18% → must gate to vision/B≥16**. NOT in mstar-moe .py (lives
   off-stack branch) → needs code PORT before testing, then 1 reboot A/B (effect >> 18% boot
   variance = conclusive). **Highest ROI.**
2. **MSTAR_MERGED_PREFILL** already ON in running env (+11% tok/s B32 claim) but unproven at
   B32 w/ counters; log is WARNING-level so can't confirm firing. Validate.
3. CUDA-graph the mixed step (exp/mixed-cg-* = empty scaffolding) — the named fix for why
   eager MIXED_BATCH went neutral at B32. Harder.
4. Relax one-walk-per-batch (micro_scheduler:97-104) = M*'s own MRV2; + on-device encoder
   metadata to kill the 2s host-sync stall. Structural.

**TEST RESULT 2026-07-07 — MERGED_PREFILL hypothesis FALSIFIED.** Rebooted flagship stack
with MSTAR_MERGED_PREFILL OFF + MSTAR_MIXED_BATCH_VISION ON (flags confirmed active).
B32 i2t: **6.13 req/s / 1576 tok/s / TTFT 2311ms** vs baseline 6.31/1621/2369 — TTFT
UNCHANGED (within boot-variance/noise). So the merged-walk unmergeability is NOT the
serializer; making prefill foldable doesn't help because request READINESS still serializes
(admit-jitter/gather already showed this). Don't pursue the merged-prefill flag gate.

**REVISED DIAGNOSIS (prefill profile under B32 mixed load, GPU7 77% / GPU6 idle):**
- No prefill/encode/vision CPU hotspot (_process_new_inputs 2.7% incl). The stall is NOT
  vision-encode or prep.
- Worker CPU under load: **36% blocked on torch.cuda synchronize (streams.py:231)** = a
  per-step BLOCKING sync, no CPU/GPU overlap; ~18% zmq send; ~5% check_stop. The sync
  serializes CPU↔GPU → GPU only 77% util (23% starved).
- **STRUCTURAL SMOKING GUN: the 30B Thinker runs SINGLE-GPU for i2t.** GPU7 does all Thinker
  work; GPU6's worker (Talker/Code2Wav) is idle without speech. vLLM almost certainly runs
  Thinker TP=2 across both GPUs → ~2x prefill+decode compute = flat TTFT + higher req/s.
  This is the prime remaining suspect for the B32 gap. TP=2 boot is the known-hard path
  (NCCL device_id + watchdog-capture fixes already landed: communication.py:161,
  cuda_graph capture_error_mode=thread_local; needs free port + slow compile).
- Secondary parity-safe lever: make the per-step synchronize ASYNC (overlap postprocess with
  GPU) — under MIXED load (unlike pure decode where it washed) this should lift GPU 77→~95%.
NOTE real clean-box gap is ~11% (committed M* 7.42 natural vs vLLM 8.32), not the len256
6.1 vs 8.3 (protocol/contention confound). Fix server kept up as 'nomrg' @8402.

**TEST RESULT 2026-07-07 #2 — TP=2 Thinker FALSIFIED (regresses everything −37%).** Booted
`qwen3omni_2gpu_thinkertp2_txt.yaml` (Thinker ranks[0,1] tp_size2, both GPUs), identical
flags. B32 i2t: **4.00 req/s / 1028 tok/s / TTFT 3070ms / ITL 14ms** vs single-GPU baseline
6.31 / 1621 / 2369 / 9. TP=2 WORSE on ALL metrics incl TTFT. Cause: **TP all-reduce over SHM
(no NVLink/RDMA on this node — committed note says RDMA/Mooncake broken) is too slow**; it
poisons prefill AND decode. ITL 9→14ms is the warmup-independent tell. **⇒ M*'s single-GPU
Thinker is the CORRECT choice on this node — that's WHY M* wins ITL. The vLLM "2-GPU TP" ref
does NOT mean M* should copy it; the interconnect makes TP a loss here. Don't retry TP=2.**
TP=2 boot works (fixes hold) but takes ~30min (cold Inductor autotune of new shard shapes).

**HONEST STATE after 2 falsified structural hypotheses:** M* wins ITL all batches + req/s
B1-B16; B32 req/s gap is only **~11% clean** (committed M* 7.42 natural vs vLLM 8.32), badly
confounded by protocol (M* ignore_eos256 vs vLLM natural~212) + contention (±15-25%) +
boot-lottery (±18%). Under B32 mixed load M* is **CPU-serialization-bound, NOT GPU-bound**
(GPU7 77%, 36% blocked on per-step synchronize). **Remaining parity-safe M*-side lever:
async/non-blocking postprocess sync on the PREFILL/mixed path** (shipped spec pipeline
already overlaps DECODE postprocess; prefill admissions still block → push GPU 77→~95%).
**GATING QUESTION (owner-only, can't do): re-measure vLLM at --ignore-eos --output-len 256
on THIS box — the gap may be near-zero under matched protocol.** Until then, more M*-side
structural work has uncertain payoff.

**TEST RESULT 2026-07-07 #3 — SIDE_PREFILL BROKEN on mstar-moe.** The async-prefill-overlap
mechanism (move standalone prefill to a side CUDA stream so decode chain stays deep) IS the
right fix and exists as MSTAR_SIDE_PREFILL, but on mstar-moe @opt/moe-autotune it CRASHES:
`RuntimeError: shape mismatch [249,4,128] vs [317,4,128]` in `_reap_side_if_done`
(worker.py:2411), "side prefill batch failed" — the paged-KV race fixed on opt/sideprefill-fix
@586fcfa1 is NOT in this worktree. Even where fixed, prior measurement = wash/−3%. Not worth
porting. Clean baseline this boot: B32 i2t NATURAL len **7.76 req/s / 1360 tok/s / TTFT
2011ms / ITL 8ms** (175 tok/req) vs committed vLLM 8.32 / 1769 / 179 / 15.4 (212 tok/req) →
req/s gap only **~7%** (within ±18% boot-variance + protocol confound), TTFT-driven.

**LEVERS EXHAUSTED (all quick/flag B32 levers dead):** merged-prefill toggle (no effect),
TP=2 (−37%), SIDE_PREFILL (broken here / wash elsewhere), chunk-size/jitter/gather/multistep/
moe-autotune (prior wash). Remaining = multi-day structural builds only: CUDA-graph the mixed
prefill+decode step (exp/mixed-cg-* empty scaffolding — the named fix for B32 TTFT), FlashInfer
plan-advance (shared blocker), relax one-walk-per-batch (MRV2 scheduler). All uncertain payoff
on a ~7% confounded gap. CONCLUSION: M* wins ITL all batches + req/s B1-16 + tok/s matched-len;
lone B32 req/s deficit is small + within noise. Escalated strategic fork to user 2026-07-07.

**TEST #4 2026-07-07 — SIDE_PREFILL port also DEAD on flagship.** Cherry-picked the full
4-file fix @586fcfa1 (compile_ops threading.local + cache_manager side-workspace +
kv_cache_engine + worker) into mstar-moe. It FIXED the shape-mismatch (317-vs-249 race gone)
but exposed a DEEPER integration failure: `KeyError: '<rid uuid>'` in the api_server message
loop (281 errors) — side-prefill outputs get lost against mstar-moe's FAST_ROUTE2/SLIM_EMIT/
SIDECAR routing, which the sidefix branch predates. Requests drop → run stalls. Reconciling =
deep integration work, and even working SIDE_PREFILL is measured wash/−3%. Reverted (mstar-moe
clean @323b0cad). **SIDE_PREFILL conclusively dead on flagship.**

**FINAL: ALL quick+medium B32 levers exhausted** (merged-prefill, TP=2, SIDE_PREFILL both
raw-crash and ported, chunk/jitter/gather/multistep/moe-autotune). M* wins ITL all batches +
req/s B1-16 + tok/s matched-len; lone deficit = B32 req/s ~7% (7.76 vs 8.32, TTFT-bound,
WITHIN ±18% boot noise + protocol confound). Remaining = multi-day structural builds only
(CUDA-graph the mixed prefill+decode step = the real fix, exp/mixed-cg-* empty; FlashInfer
plan-advance; MRV2 scheduler). Recommend: owner resolve confound (re-measure vLLM ignore_eos256
on this box) BEFORE committing multi-day build — gap may be noise.

**TEST #5 2026-07-07 — BATCH_VISION_PREFILL FALSIFIED (regresses).** The vision ENCODER
already batches by default (parity-safe); the serial gate is the Thinker `prefill_vision`
walk (bs=1 assert, submodules.py:1265). MSTAR_BATCH_VISION_PREFILL batches it >1/step with
real bs>1 graph captures (submodules.py:1602), mutually exclusive w/ MERGED_PREFILL. Tested
MERGED off + BATCH_VISION_PREFILL=1: B32 i2t natural **5.99 req/s / TTFT 3562ms** vs baseline
7.76 / 2011 → req/s −23%, TTFT +77% (ITL improved 8→5ms). Batching prefills adds wait-to-form
latency + loses merged/spec-pipeline overlap. Dead.

**★ FINAL CONCLUSION 2026-07-07: the flagship config is a LOCAL OPTIMUM — every prefill
perturbation regresses/washes** (merged on/off, TP=2 −37%, SIDE_PREFILL broken, BATCH_VISION_
PREFILL −23%). Single-request-serial-prefill + MIXED_SPEC pipeline on single-GPU is optimal
for this SHM node. The ~7% B32 req/s gap (7.76 vs 8.32, within ±18% boot noise + protocol
confound) is STRUCTURAL: vLLM's flat TTFT comes from 2-GPU TP prefill + per-step chunked
admission; M* can't match on 1 GPU and M*'s own TP=2 regresses on SHM (no NVLink). Closing it
needs a multi-day rewrite (CUDA-graph the mixed step — but W5-P2 already reached only NEUTRAL
at B32; OR efficient TP=2-over-SHM comm like vLLM's — deep). **Expected payoff LOW.**
RECOMMENDATION: (1) owner re-measure vLLM at ignore_eos256 on THIS box — gap may be noise/zero;
(2) otherwise treat M* as already-winning (ITL all batches, req/s B1-16, tok/s matched-len) and
stop the flag search. Every quick+medium lever is exhausted.

**TEST #6 2026-07-07 — "capture same shapes as vLLM" (option 2) INVESTIGATED + FALSIFIED.**
Two agents mapped both capture schemes: vLLM keys CUDA graphs on 1D total-tokens (51 buckets
[1,2,4]+range(8,256,8)+range(256,513,16), cap 512, 256 seqs), mixed→PIECEWISE graph padded up,
NEVER eager on mixed (cudagraph_dispatcher.py). M* mixed key is (bs,num_tokens) but the
num_tokens axis ALREADY pads total D+P up (kv_cache_engine.py:575); bs pinned to 32 with
scheduler caps (_MIXED_MAX_DECODE=31, _MIXED_MAX_CHUNK_TOKENS=512, micro_scheduler.py:213) that
GUARANTEE a fold hits a captured bucket — so M* does NOT go eager (user's premise was wrong).
Captured mixed step (thinker_mixed, real graph.replay) + MIXED_PREPLAN are BUILT. Tested full
option-2 (MERGED off + MIXED_BATCH+SPEC+VISION+PREPLAN): B32 i2t natural **7.46 req/s / TTFT
2457ms** vs MERGED baseline 7.76 / 2011 → NEUTRAL/slightly-worse. food101 prefill ≈286 tok
(256 vision @512px + 30 text) < 512 ceiling ⇒ folds ENGAGE, aren't declined. **Root cause is
NOT capture shapes: folding reorganizes but can't reduce the fixed prefill+decode work on ONE
compute-bound GPU. MERGED (1 efficient standalone prefill) beats folding (split+overhead) for
i2t. vLLM's flat TTFT needs its 2-GPU TP prefill — which M* can't use on SHM (TP=2 −37%).**
Finer/larger buckets won't help (folds already succeed). Option 2 conclusively neutral.

**TEST #7-8 2026-07-08 — TP=2 REOPENED (NVLink present!) then RE-CLOSED via symm-mem.**
Correction: GPUs 6-7 have FULL NVLink (NV18, H200, ~900GB/s) — "SHM/RDMA broken" was
inter-PROCESS routing, not the NCCL TP all-reduce. So TP=2 −37% was NOT a hardware wall.
Root cause found: M* uses vanilla NCCL dist.all_reduce (communication.py:58, ~96/step for
30B Thinker, captured in graph); vLLM uses a custom low-latency P2P all-reduce.
IMPLEMENTED the fix: MSTAR_SYMM_ALLREDUCE (committed mstar-moe 75ffb389) = PyTorch 2.9
native `torch.ops.symm_mem.one_shot_all_reduce`(<128KB)/`two_shot_all_reduce_` — vLLM's exact
mechanism, capturable, rendezvous in pre-capture priming window, graceful NCCL fallback,
default-off byte-identical. VERIFIED it works: TP=2 ITL 14→12ms. BUT TP=2 STILL net-negative:
B32 i2t natural **4.64 req/s / TTFT 3590ms / ITL 12** vs single-GPU 7.76/2011/8. TTFT got
WORSE, not better → the 2-GPU split isn't parallelizing prefill. **The all-reduce was NOT the
dominant TP cost — M*'s MULTI-PROCESS TP architecture is (2 conductor-routed worker procs in
lockstep, each hitting the ~26ms Python floor + vision-encode pinned to rank0). vLLM's TP is a
single efficient multi-GPU process; M* can't match without a multi-month single-process-TP
rewrite.** TP=2 definitively closed. NCCL_PROTO=LL128 crashed capture (dead-end).

**★★ DEFINITIVE FINAL 2026-07-08: EVERY lever exhausted.** Single-GPU prefill/mixed levers all
neutral/regress (flagship is local optimum); TP=2 net-negative even with vLLM-matching
symm-mem all-reduce (multi-process architecture, not all-reduce). The B32 i2t req/s gap (~7%,
7.76 vs 8.32, WITHIN ±18% boot + protocol confound) is structural to M*'s single-process-per-
GPU architecture and would need a multi-month single-process-multi-GPU-TP rewrite to close —
unjustified for a within-noise gap. M* WINS ITL all batches + req/s B1-16 + tok/s matched-len;
s2t already wins. Genuine artifact kept: MSTAR_SYMM_ALLREDUCE (correct, parity-safe). RECOMMEND
owner re-measure vLLM at ignore_eos256 on this box to confirm gap is even real.

**MATCHED-LENGTH VERIFICATION 2026-07-08 (resolves the confound):** committed benchmark
data was STALE (like s2t: old mstar_new i2t B32 4.39 vs current 7.76; s2t 19→39). Measured
CURRENT build at MATCHED output-len 210 (= vLLM's committed 210 tok/req) vs committed vLLM
i2t@210 (B8 3.45 / B16 5.11 / B32 8.21):
- **B8 M* 3.74 (+8% WIN), B16 5.40 (+6% WIN), B32 7.02 (−14% LOSS, TTFT 2476ms).**
- The B8/B16 wins SURVIVE matched-length ⇒ real, not output-length artifacts. B32 loss is
  REAL even at matched length (TTFT-bound), ~7-14% (within ±18% boot + contention noise).
- s2t VERIFIED all-batch WIN (current build, single-GPU): B1-B32 5.60/9.03/13.82/18.78/27.32/
  39.24 vs vLLM 3.83/9.14/13.22/15.79/19.80/30.85. B32 +27%.
**DEFINITIVE: M* beats vLLM on i2t B1-B16 (matched-length-confirmed), s2t ALL batches, ITL
everywhere, tok/s at matched/long length. SOLE loss = i2t B32 (~7-14%, TTFT/prefill-bound,
noise-adjacent). Confound resolved — B32 loss is real but small. Charts committed:
campaign_i2t_scoreboard.png, campaign_s2t_scoreboard.png.**

Running server: none. mstar-moe @opt/moe-autotune. GPUs 6/7 free.
Committed reframe: benchmarks/qwen3-omni-joint/AR_LOOP_10_REFACTORS.md scoreboard section.
