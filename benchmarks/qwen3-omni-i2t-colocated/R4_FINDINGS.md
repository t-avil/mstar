# R4 Deep Investigation — findings (5 Opus agents, challenge prior conclusions)

## Agent 2 (M*-floor-challenge) — prior work was PARTLY LAZY. Reframe:
- plan() is NOT the floor: already overlapped (pre_plan on plan_executor, gated advance_event, concurrent
  with replay(N); plan is token-independent). 3-Opus R3 verdict proved plan can't go IN-graph then wrongly
  equated that with plan=floor. LAZY LEAP.
- NOBODY wall-clock-measured plan vs replay vs postprocess. MSTAR_PHASE_TIMING only records
  await_gpu/submit_spec/iter_total (worker.py:4494,5052,5188). "floor" inferred from py-spy self-% (not
  critical-path wall-clock). => DECISIVE MISSING EXPERIMENT: measure per-step plan()/replay/postprocess wall-clock.
- ★ -7% at B32 is LARGELY A LENGTH/TTFT ARTIFACT, not decode loss: ITL TIES 15.4/15.4 yet tok/s -7% because
  M* out ~770B vs vLLM 920B -> M*'s higher B32 TTFT amortizes over ~20% fewer tokens. Tied ITL + lost tok/s =>
  loss is TTFT-per-token, NOT decode speed. => target TTFT (prefill) or length-normalize, NOT the decode rewrite.
- HOST-SIDE MULTISTEP = the untested lever: K GPU-resident DIRECT_FEED replays (worker.py:3372-3404) -> K-deep
  pre-plan (extend plan_executor depth 1->K) -> ONE batched D2H of K*bs tokens -> K stop-decisions -> TRIM before
  emit (async_sched +9% was emitting overrun; trim fixes). Removes same-step round-trip for K-1 of K steps.
  GATES: NUM_SLOTS=2->K workspaces (mem); the risk = if plan() Python ~ replay GPU time, single-thread
  plan_executor becomes new bottleneck (MEASURE first); B1 det N/N + trim correctness.
- R1/DIRECT_FEED neutrality = correct test of WRONG lever (both kept the round-trip). Doesn't license "irreducible".
NEXT: (1) MEASURE decomposition (cheap, decisive). (2) If host>replay: host-side multistep. (3) Re-frame B32 as
TTFT-amortization -> also consider TTFT levers. Awaiting agents 1,3,4,5 for cross-validation.

## Agent 4 (zero-sync-arch) — RANKED REFACTOR (build order #1->#2->#3)
#1 HOST-SIDE MULTISTEP (highest ROI, NO backend change, BUILD FIRST): inner loop K replays on gpu_executor
   thread (worker.py:2444, _run_basic_batched), GPU-resident DIRECT_FEED feed (worker.py:3372, cuda_graph_runner:2253),
   K-deep overlapped pre-plan (existing plan_executor issued K times), accumulate [bs,K] GPU buffer, ONE D2H after K
   (_d2h_new_tokens 3920), K stop-decisions (_compute_new_stops 3975 over columns) + TRIM before emit. Amortizes
   emit-tail + _try_speculate_next rebuild + barriers K-fold. Side branch opt/host-multistep, MSTAR_HOST_MULTISTEP=K
   default 0. Slice0 shadow K=2 (assert [bs,2]==2x single), Slice1 K=2 live A/B, Slice2 K=4/8 (sweet spot ~K=4 @178tok).
   Gates: NUM_SLOTS 2->K workspaces; SIDECAR_CHECKSTOP on; SAMPLE_RENDEZVOUS off; B1 det + B32 length-dist.
#2 PERSISTENT O(1) BATCH: kill per-step _try_speculate_next rebuild/register_request/uuid4 (worker.py:2946,883);
   persistent decode batch mutated only on membership change. Compounds #1. Second.
#3 FULL ZERO-SYNC (month, backend-gated by Agent3 Path A): in-graph argmax + per-slot-private buffers
   (_intern_static_buffer key +slot_idx cuda_graph_runner.py:456) + on-device stop; reuses #1 window semantics.

## Agent 3 (backend) — ★★★ THE PRIOR R3-INFEASIBLE VERDICT WAS WRONG (lazy miss)
FlashInfer 0.6.13 ALREADY HAS flashinfer.decode.trtllm_batch_decode_with_kv_cache = PLAN-FREE single kernel,
reads seq_lens from a DEVICE uint32[batch] buffer (in-graph capturable), block_tables device page-table. H200 sm_90
auto-selects xqa backend. NO version bump, NO torch-2.9.1/sgl-kernel compat risk. The R3 agents only examined
BatchDecodeWithPagedKVCacheWrapper (host plan()) and MISSED this plan-free kernel in the SAME lib. = the lazy miss.
PATH A (BEST, 2-4wk): swap FlashInferDecodeWrapper.run -> trtllm_batch_decode_with_kv_cache; static device seq_lens;
advance_seq_lens -> in-graph seq_lens+=1; ragged indptr/indices -> dense [batch,max_pages] block_table; xqa-vs-FA2
parity gate. = vLLM MRV2's exact mechanism (device seq_lens + full-decode cudagraph). UNBLOCKS full in-graph multistep.
Path B (FA3 flash_attn_with_kvcache, sgl-kernel) = hot spare. Paths C/D/E not recommended. Refs: arxiv 2501.01005,
flashinfer trtllm_batch_decode docs, vLLM issue #29134 (FlashInfer plan D2H is library-wide, not M* bug).

## SYNTHESIS (3/5 in): prior "infeasible/irreducible" was PARTLY LAZY. TWO viable wins:
(near) HOST-SIDE MULTISTEP (#1, no backend, days) + (real) PATH A trtllm_batch_decode zero-sync (weeks, pinned-dep-safe).
DECISIVE FIRST EXPERIMENT: MEASURE per-step plan()/replay/postprocess wall-clock at B32 (never done) to confirm
host-bound + size the K-amortization. Also: -7% is partly TTFT-amortization (ITL tied) -> quantify. Awaiting agents 1,5.

## Agent 1 (vLLM FORENSICS, read vLLM 0.24 source) — ★★★ THE REAL GAPS (vLLM re-plans on host too!)
"vLLM avoids per-step host plan via in-graph seq_len" = FALSE. vLLM ALSO re-plans host every decode step.
The gap is HOW: vLLM does 3 things M* doesn't (all doable in pinned flashinfer 0.6.13):
1. ★ CHEAP REPLAY-PLAN: vLLM runs full .plan() ONCE (vllm_first_call) then fast_decode_plan (flashinfer/decode.py:2898)
   = stripped _cached_module.plan over PERSISTENT PINNED buffers + tiny non_blocking H2D each step. M* calls FULL stock
   BatchDecodeWithPagedKVCacheWrapper.plan() EVERY step (~750us on critical path, M*'s own TODO cache_manager.py:392-430,
   flashinfer_utils.py:392). = SINGLE BIGGEST GAP.
2. ★ NO per-step per-rid Python REBUILD: vLLM persistent InputBatch (gpu_input_batch.py) numpy-inplace + 1 pinned H2D.
   M* rebuilds index tensors every step in `for rid` loop + alloc_manager.alloc() per rid + torch.tensor(list) over
   PAGEABLE mem (cache_manager.py:313-352). = M*'s CPU floor at B32.
3. ★ KILL .item() D2H: M* prefill-path plan does int(lens.sum().item()) (flashinfer_utils.py:215) per step; vLLM none.
   Also vLLM one batched host sync (_to_list: pinned copy+event+1 .tolist(), gpu_model_runner.py:3672/7513) NOT per-rid.
vLLM sample=GPU argmax (sampler.py:240), stop=host same-step+trim, CUDA graph = forward+attn replay only (sample/plan/
stop outside). Optional TRTLLM decode (flashinfer.py:1273) skips plan entirely (= Agent3 Path A).
FIX MENU to match B32 (targeted, NOT a rewrite): (1) adopt fast_decode_plan; (2) persistent O(1) batch (kill per-rid
rebuild/alloc/pageable-tensor); (3) excise .item() sync. All verified from vLLM's own code. Then Agent4 #1 host-multistep
compounds, Agent3 Path A trtllm = the endpoint.

## ★★★★ SYNTHESIS (4/5): prior "irreducible/infeasible" was LAZY. Concrete achievable path to match/beat vLLM B32:
DECISIVE MEASURE FIRST: per-step plan()/replay/rebuild/postprocess wall-clock at B32 (never done).
Then implement in order (side branch, parity-gated): fast_decode_plan (biggest gap) -> persistent O(1) batch ->
kill .item() -> host-side multistep (amortize K-fold) -> [weeks] trtllm_batch_decode zero-sync. Also -7% partly
TTFT-amortization (ITL tied) - quantify. Awaiting Agent5 (academic/roofline) for the fundamental-vs-fixable verdict.

## ★★★★★ Agent 5 (academic/roofline) — VERIFIED, overturns the campaign. THE ANSWER.
CROSS-CHECKED committed vLLM steadyflag (h2h_out_steadyflag/vllm_i2t_B32_*, n=384) — VERIFIED by me:
  vLLM B32: tok/s 1724 (1706-1731), ITL mean 17.4/p50 12.0ms, TTFT mean 169/p50 160ms (FLAT).
  M* B32:   tok/s 1640 (1594-1695), ITL mean 15.4/p50 11.2ms, TTFT mean 470/p50 ~190ms (HEAVY TAIL).
=> M* WINS B32 DECODE (ITL 15.4 vs 17.4 = -11%!). tok/s -4.9% (inside M*'s variance). At MATCHED LENGTH M* +1.3%.
Roofline: B32 decode GPU floor ~3.3ms (HBM-bound, 110/128 experts, 15.8GB/4.8TB/s); measured ITL ~11ms => 70-80%
host-bound (TRUE) BUT vLLM equally host-bound (~8.7ms/step) and M* FASTER host (~7.9ms). Decode-occupancy: M* 79%
vs vLLM 94% -> the 15pt = wall NOT at full decode concurrency = PREFILL/TTFT STALLS. THE LOSS IS TTFT-TAIL + LENGTH.
PRIOR WORK LAZY (4 ways): (1) claimed ITL tie 15.4/15.4 - actual vLLM 17.4, M* WINS; (2) cherry vLLM tok/s 1769 vs
committed 1724 (-7% -> -4.9% within variance); (3) R1/R3/3-Opus all attacked DECODE ITL = a metric M* ALREADY WINS;
(4) never decomposed the real gap (TTFT mean 470). Architectural fork = PREFILL ADMISSION: vLLM unified token budget
(chunked prefill folded into every decode step -> flat TTFT, fat ITL p95 70.8) vs M* separate prefill/decode NodeBatch
(32-img wave stalls decode -> TTFT tail 470, clean ITL p95 39.8). Co-admission EXISTS default-off (mixed_preplan
kv_cache_engine.py:1095-1106; qwen3_omni_model.py:299). SPLIT_ATTN test scored p50 not mean-TTFT tail = misjudged.
VERDICT: B32 loss = ~2-5% (near-noise) TTFT-tail + shorter-output artifact; DECODE IS A WIN. FIXABLE via prefill
co-admission tuning on EXISTING substrate — NOT a rewrite. R3/decode path answered the WRONG question.

## ═══ DEFINITIVE R4 SYNTHESIS (all 5 Opus) ═══
M* is NOT decode-slow at B32 — it WINS decode (ITL 15.4<17.4). The tok/s -4.9% (near noise, +1.3% matched-length)
is the TTFT-mean tail (470 vs 169) from prefill-wave stalling decode. REAL LEVER = prefill co-admission (fold
prefill into decode = vLLM unified token budget), an existing default-off switch mis-tested before (wrong metric).
Decode micro-opts (fast_decode_plan, persistent batch, Agent1/3/4) are SECONDARY (free host headroom, but decode
already wins). NEXT: (1) decompose TTFT-mean-470 tail (queue-wait vs prefill-compute vs decode-stall) on colocated;
(2) tune co-admission (mixed_preplan / CHUNKED_PREFILL / SPLIT_ATTN) scored on MEAN-TTFT + tok/s, parity-gated;
(3) report matched-length tok/s. CORRECT SCOREBOARD: M* wins i2t B1-16 + B32 DECODE + matched-length; only blemish =
B32 aggregate tok/s via TTFT tail (fixable).

## ★★★★★ ROOT CAUSE of i2t B32 TTFT tail PINNED (code-grounded) — the co-admission is off FOR VISION
qwen3_omni_model.py:188-299: co-admission = MSTAR_MIXED_BATCH (fold prefill chunk into a captured thinker_mixed
decode step). My boots have MIXED_BATCH=1 + CHUNKED_PREFILL_V2=1 + MIXED_SPEC=1 — BUT those co-admit TEXT prefill.
i2t prefill is VISION (32 images). Vision co-admission needs MSTAR_MIXED_BATCH_VISION=1 (:217-237) +
MSTAR_CHUNKED_PREFILL_V2_VISION (produces vision chunk rows) — BOTH DEFAULT-OFF, NOT in my boots. So the 32-image
VISION prefill wave CANNOT ride decode -> serializes -> TTFT tail 470ms. Also MSTAR_MIXED_PREPLAN default-OFF
(:192-214): the code SAYS its absence = "the ~unity i2t B32 residual (measured 0.969-1.032 band)" — the packed
prefill-wrapper plan (~0.75-1.5ms/32-row) runs INLINE = serial bubble, un-hidden. MSTAR_MIXED_SPLIT_ATTN (:288)
= the mixed-shape plan speedup (6.11->1.84ms attn) I tested before but on PD/wrong-metric.
=> DECISIVE EXPERIMENT: boot colocated with MSTAR_MIXED_BATCH_VISION=1 MSTAR_CHUNKED_PREFILL_V2_VISION=1
MSTAR_MIXED_PREPLAN=1 (+MIXED_SPLIT_ATTN=1) so the VISION prefill co-admits + pre-plans + splits attn -> flatten
TTFT-mean 470->~170. Boot-time (capture must provision deepstack statics; IMA-latch mark_mixed_vision_provisioned
:260). Parity-gate B1 det + B32 length-dist; measure TTFT-MEAN + tok/s + ITL (co-admit trades ITL-tail for TTFT-tail;
M* has ITL headroom 15.4<17.4). If TTFT-mean drops + tok/s rises w/o losing ITL lead -> B32 WIN. = the real fix.
