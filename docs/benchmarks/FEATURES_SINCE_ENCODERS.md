# FEATURES_SINCE_ENCODERS — complete feature set of the M* code delta

Baseline: `encoders-implemeneted` @ `4c33b33` (the "encoders implemented" state).
Winning build: `opt/prep-h2d` @ `620de91`. Cumulative diff `4c33b33..620de91`:
**47 files, +11,865/−301** — 92 commits on a linear first-parent chain.
All branches live on fork `git@github.com:t-avil/mstar.git`; every SHA below was
read from the diffs, not from commit messages. Measured effects cite
EXPERIMENTS.md entries; grades per ab_verdict.py protocol.

Topology notes: the lineage begins directly at `4c33b33` (no separate
merge-config branch). `opt/sidecar-checkstop` exists twice: standalone
`5414a5a` (off `12ce776`) and re-landed as stack head `7ef5150` (=`opt/stack-n2`)
with a byte-identical diff. `opt/decode-v2` @ `2239005` doubles as tag
`encoders-implemeneted-v2` — the base worker-opt checkpoint most exp/* branches
fork from.

Winning lineage milestones, in order:
`4c33b33` → opt/decode-v2 `2239005` → opt/sched-pack `74a985c` → opt/speech-floor
`01856e8` → opt/v2-policy `56e65f8` → opt/prefill-merge `41200ec` → opt/custom-ops
`12ce776` → opt/cfgcache-v2 `ded928d` → opt/stack-n2 `7ef5150` → opt/prep-h2d
`620de91`. On-chain experiments: exp/chunked-prefill-v2 `23d07ee`,
exp/mixed-batch-p2 `4d9d6e3`, exp/bucket24 `845faff`,
exp/batched-postprocess-v2 `1e171e1`.

## A. Features IN the winning stack (shipped, in lineage order)

### A1. Base worker-opt block (opt/decode-v2 `2239005`; 17 commits, +1636/−80)
- **Block-fp8 MoE** (`MSTAR_MOE_FP8`/`_TALKER`): new `fused_moe/fp8.py` (+367) —
  per-(128,128)-block w8a8 grouped GEMM (`per_block_cast_to_fp8_weight`,
  Triton `per_token_group_quant_fp8`, `_fused_moe_fp8_kernel`, BLOCK_SIZE_K=128).
  Wired at `moe.py::_dispatch_fp8` with lazy `_ensure_fp8_experts` that frees
  bf16. `2239005` retunes only the prefill-M tile (BLOCK_M=64, GROUP_M=8;
  1.63× over BLOCK_M=32). Measured: **1.13× e2e** (decode-bottleneck entry);
  bake-off #24: shipped Triton grouped GEMM beats DeepGEMM masked at all
  decode M (keep BLOCK_M=16 there).
- **Fused topk router** (`MSTAR_FUSED_TOPK`, default ON): `sgl_kernel.topk_softmax`
  on the CUDA router path, guarded fallback.
- **check_stop D2H batching**: `worker.py::_prematerialize_for_check_stop` —
  one cat + pinned D2H per step reused by the send path.
- **Side-stream prefill** (`MSTAR_SIDE_PREFILL`): second executor + CUDA stream
  runs prefill/encoder concurrent with decode (`_maybe_dispatch_side` /
  `_reap_side_if_done` / `_drain_side`).
- **Inline/batched emit** (`MSTAR_INLINE_EMIT`, `MSTAR_BATCH_EMIT`): token ints
  inline; one message per step.
- **Memoized postprocess** (`MSTAR_FAST_POSTPROC`, via `4ce0125`):
  `tensors.py::store_and_populate_graph_edges_fast` — `_FastPopulatePlan` per
  (rid,node,walk); replay patches only fresh `uuid4()` + `data_ptr()`.
- **Decode buckets 24/28** (via `845faff`): cuts B32 padding waste while the
  live batch churns 17–31.

### A2. Chunked prefill W5-P1 (exp/chunked-prefill-v2 `23d07ee`; +1234/−104)
`MSTAR_CHUNKED_PREFILL_V2`(`_VISION`,`_ASSERT`), `MSTAR_PREFILL_CHUNK_TOKENS`.
Splits long Thinker prefill into ≤C-token chunks as separate steps so the
round-robin interleaves them with decode (vs one ~27.5 ms mega-step).
`qwen3_omni_model.py::plan_prefill_chunk` picks the largest bucket ≤
min(remaining,cap), floor 128; conductor threads `prefill_chunk_offset`; vision
variant splits `prefill_vision` into standalone `encode_vision` + chunked
Thinker walk. Text-output only (Talker thinker_states forbid splitting).
NOTE: 100% of food101/libri prefills are ≤256-tok unchunkable spans — the fold
family was falsified for THIS workload; the mechanism ships for long-prompt
workloads.

### A3. Captured mixed batch W5-P2/P3 (exp/mixed-batch-p2 `4d9d6e3`; +1823/−30)
`MSTAR_MIXED_BATCH`(`_ASSERT`,`_VISION`), `MSTAR_MIXED_SPEC`,
`MSTAR_MIXED_PREPLAN`, `MSTAR_WALK_STATS`. Requires CHUNKED_PREFILL_V2.
Assembles a real `thinker_mixed` capture batch (31 decode + 1 prefill chunk →
padded 32) — `micro_scheduler.py::_try_assemble_mixed`, chunk gates
(metadata present, C ≤ largest bucket, repetition_penalty==1.0). SPEC folds
the mixed step into the running spec chain; PREPLAN adds the
FLASH_INFER_PACKED pre-plan surface in cuda_graph_runner.py. Ships the
WALK_STATS counters used for all later mechanism-liveness checks.

### A4. Scheduler pack + fairness backoff (opt/sched-pack `74a985c`; +96/−11)
`MSTAR_SCHED_PACK`, `MSTAR_SCHED_PACK_PEEK_CAP`(8), `MSTAR_DIRECT_FEED`
refreshable. Precomputes emit routing flattens once per step; exponential
backoff on the fairness peek (reset on positive peek/fresh chain).

### A5. Speech floor (opt/speech-floor `01856e8`; +843/−8)
- `MSTAR_FAST_CHECKSTOP_TALKER`: batches per-rid `layer0_codes` into one
  cat+pinned D2H, then one `flat.tolist()` + int compares vs cached
  `_talker_codec_eos_id`. Walk-gated to `talker_decode`.
- `MSTAR_CODEC_CHUNK_EMIT`: coalesces the colocated Talker→Code2Wav codec edge
  (`chunk_policy.py::coalesce_size`, `stream_buffer.py` stage/flush,
  `worker.py::_route_streaming_tensor_coalesced`).
- 2-GPU taco yaml. (Speech cells all ≥2.1× refs — this plus A1 is why.)

### A6. V2 budget policy (opt/v2-policy `56e65f8`; +446/−288)
`MSTAR_MIXED_BUDGET_TOKENS`, `_MIN_DECODE`, `MSTAR_MIXED_MIN_DECODE`,
`MSTAR_MIXED_SINGLE_CHUNK`. Every-step budgeted chunk fold
(`micro_scheduler.py::_chunk_over_budget` = n_decode + C > budget, identical
in peek and pop), decode-occupancy floor default 24 (single-chunk decode-starve
anti-lesson), dynflags-refreshable, nothing baked into capture. Also merges
opt/integration-v4 (A9) and reverts two compile-tracing experiments that
wedged dynamo.

### A7. Merged multimodal prefill (opt/prefill-merge `41200ec`; +789/−105)
`MSTAR_MERGED_PREFILL`. Collapses `[prefill_text, prefill_vision]` into one
`prefill_multimodal` walk, killing a conductor round-trip.
`_maybe_merge_prefill_schedule` gates (text-only output, CHUNKED_V2_VISION
off, exactly 1+1); `_build_merged_multimodal_inputs` concatenates
embeds/pos_ids/deepstack in modality order, threads MRoPE start_pos, rides the
existing `prefill_vision` capture (bs=1). Byte-identical OFF. KEY: it's a
vision-strategy SWAP — only fires with CHUNKED_PREFILL_V2_VISION off.

### A8. torch.library custom ops (opt/custom-ops `12ce776`; +310/−10)
`MSTAR_CUSTOM_OPS`. New `compile_ops.py`: `mstar::run_attention`,
`mstar::fused_experts_fp8`, `mstar::apply_rope`, each with register_fake;
active-manager registry (`set_active_manager`). Replaces
`@torch.compiler.disable` fusion boundaries so compile traces
proj/norm/residual chains whole; fp8 experts pre-quantized BEFORE compile
(kills lazy-quant trace hazard). Measured: compiled-thinker graph breaks
**816 → ~165**. Boot time needs FX_GRAPH_CACHE + CACHE_DIR +
DYNAMO_CACHE_SIZE_LIMIT=128.

### A9. GC/GIL tuning (opt/integration-v4 `52d9116`, merged via A6; +27)
`MSTAR_GC_TUNE`, `MSTAR_PY_SWITCH_INTERVAL_SEC`. gen-0 threshold 700→50,000;
post-capture `gc.collect(); gc.freeze()`; tunable interpreter switch interval.

### A10. Sampler config cache V2 (opt/cfgcache-v2 `ded928d`; +227/−25 sampling.py)
`MSTAR_SAMPLER_CFG_CACHE_V2`. Root cause: per-step `torch.tensor(cfg, device=)`
pageable H2D + cudaStreamSynchronize ×6 = **22% of B32 wall**
(sampling.py:419). V1 keyed on exact request-id tuple and missed nearly every
B32 step under churn. V2: persistent per-slot device tensors written once at
admission (pinned, non_blocking), batch assembled by `index_select`;
`rand_offset` an on-device per-slot counter. No pageable H2D on any path;
byte-identical output. Measured: **+5.2% raw, ~+2%±4% after harness debias** —
mechanism real, partially GIL-shade-overlapped.

### A11. Sidecar Stage-2 check_stop (opt/stack-n2 `7ef5150` ≡ `5414a5a`; +452/−77)
`MSTAR_SIDECAR_CHECKSTOP`(`_SHADOW`). Removes the `side.synchronize()`
graph-tail wait (~1.1–2.1 ms) without deferring the stop decision:
`_checkstop_barrier` records a reusable event; `_await_checkstop` polls
`event.query()`; decision still made this step from this step's tokens (no
route move — one-way data-flow preserved). Shadow mode asserts equality.
Measured: **+2.3%** at B32 when the wall re-flipped host-side after A10.

### A12. Device-side pos_ids (opt/prep-h2d `620de91`; +65/−9)
`MSTAR_PREP_DEVICE_POS`. Builds per-step `thinker_decode` MRoPE pos_ids ON
DEVICE (kills the per-step pageable pos_ids H2D that was **24% of B1 wall**,
submodules.py:620) and hoists the cfgv2 gather. Input-build only,
capture-safe, dynflags-refreshable. Measured: **+3.1% B2**, B1 to 1.17–1.21×.

### A13. Env-only: Inductor CDT (no code)
`TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1` +
`TORCHINDUCTOR_CACHE_DIR=inductor_cache_cdt` + `TORCHINDUCTOR_FX_GRAPH_CACHE=1`
+ `TORCHDYNAMO_CACHE_SIZE_LIMIT=128`. Measured: **+3.4% B32**. Cold CDT
compile ≈ 40 min; the committed cache makes boots normal. See
TORCH_COMPILE_FINAL.md for the full compile audit.

## B. Off-stack features (built, measured, NOT in the winning env)

- **opt/prefill-merge-audio `a2d2f47`** (+683/−123, `MSTAR_MERGED_PREFILL_AUDIO`):
  s2t twin of A7 — one `prefill_multimodal_audio` walk; no deepstack, MRoPE
  advance == seq_len so it reuses the `prefill_text` capture; handles 2-entry
  legacy AND 3-entry vLLM-layout schedules. Byte-identical OFF. GPU A/B pending.
- **opt/prefill-gather `092d18f`** (+183/−1, `MSTAR_PREFILL_GATHER_MS`,
  `_TARGET` 4, `MSTAR_ENCODER_GATHER_TARGET` 8): defers a lone bs=1 prefill so
  packed captures fill; occupancy-guarded (≥24 floor). FALSIFIED for i2t B32 —
  readiness serializes through the KV/encode pipeline.
- **opt/admit-jitter `0c68268`** (+79, `MSTAR_ADMIT_JITTER_MS`): deterministic
  hash-fraction admission stagger, surplus-only. FALSIFIED — guard
  self-suppresses at the trough.
- **opt/sidecar-batch `8b775a3`** (+113/−7, `MSTAR_SIDECAR_BATCH`=K): coalesces
  K StepRecords into one pickle send (21.7µs→4.75µs/record). WASHED-NEGATIVE
  live (B/A ~0.94, 2×) — freed host ms absorb into the worker.py:3193
  await-GPU wait. Parked default-off.
- **opt/async-sched `18b1244`** (+499/−17, `MSTAR_ASYNC_SCHED`): defers a
  decode step's postprocess one iteration (GPU-side loop-back via DIRECT_FEED,
  `_async_trim` for overrun steps). PARKED — non-identical +9% length,
  tok/s wash; deferral shifts batch composition.
- **opt/w2-retest `a2788a9` / exp/step-txn `b98de66`** (+668,
  `MSTAR_STEP_TXN`(`_VERIFY`)): full per-rid postprocess memoization
  (`step_txn.py::StepTransaction`, failure-atomic fast_execute). CLOSED —
  4/4 boot failures across 2 environments; cost exceeds 2–5% EV.
- **opt/compile-fix `1733fab`** (+14 norm.py, no flag): compile-aware RMSNorm —
  pure-torch fp32 path under `torch.compiler.is_compiling()` (FlashInfer
  rmsnorm graph-broke ~311×/boot). Eager unaffected. Mechanism folded into
  the custom-ops era; branch kept as the minimal standalone form.
- **exp/direct-feed `6644913`** (+107/−1, `MSTAR_DIRECT_FEED`): feeds next spec
  step's text_inputs from the batched sampled tensor (2nd handle, no copy),
  per-rid fallback for non-token edges.
- **exp/two-step-decode `6bf6136`** (+638/−139, `MSTAR_MULTISTEP_DECODE`=N):
  N decode steps in one GPU submission (one swap/restore bracket, in-graph
  direct feed, `extra_step_outputs`). Gated on uniform decode + ≥2N remaining
  iters; seen-token penalty forces single-step.
- **exp/prefill-buckets `08e7123`** (+4/−1, unconditional): prefill buckets
  +384/768/1536 (i2t pads ~45% into 512 otherwise).
- **opt/mixed-walk `4334810`** (29 commits, +5964/−40) and
  **exp/combined-coalesce-piggyback `0671191`** (26 commits, +8252/−128, off
  old upstream `2e6465a`): the eager varlen mixed-walk line
  (`engine/mixed_walk.py::build_mixed_varlen_layout`, vLLM-v1-style
  decode-first + piggybacked prefill chunks, token-budget admission) plus the
  native-encoder superset (varlen audio/vision encoder modules with layered
  SDPA fallbacks, `MSTAR_ENCODER_COALESCE` bounded cross-request window,
  GPU log-mel, GPU image preprocess, vLLM prompt-layout parity). Superseded
  on-stack by A2/A3 (captured) and the encoder work already in `4c33b33`;
  kept as the reference implementation of the eager path.
- **exp/combo-w1w7 `c8cebc7` / exp/batched-postprocess-v2 `1e171e1`**: the two
  landing sites of the shared `MSTAR_FAST_POSTPROC` commit `4ce0125` (on
  bucket24 and on the W1 base respectively); the mechanism shipped via A1.

## C. Complete MSTAR_* flag inventory (opt/prep-h2d, defaults from code)

Booleans default OFF unless marked ON.

Perf (winning env sets: CUSTOM_OPS, MOE_FP8, SAMPLER_CFG_CACHE_V2,
SIDECAR_CHECKSTOP, PREP_DEVICE_POS, MERGED_PREFILL):
`MSTAR_MOE_FP8`, `MSTAR_MOE_FP8_TALKER`, `MSTAR_FUSED_TOPK`(**ON**),
`MSTAR_SIDE_PREFILL`, `MSTAR_INLINE_EMIT`, `MSTAR_BATCH_EMIT`,
`MSTAR_SLIM_EMIT`, `MSTAR_SLIM_EMIT2`, `MSTAR_FAST_ROUTE`, `MSTAR_FAST_ROUTE2`,
`MSTAR_FAST_SEND`, `MSTAR_FAST_POSTPROC`, `MSTAR_FAST_CHECKSTOP`,
`MSTAR_FAST_CHECKSTOP_TALKER`, `MSTAR_CODEC_CHUNK_EMIT`, `MSTAR_DIRECT_FEED`,
`MSTAR_SCHED_PACK`, `MSTAR_SCHED_PACK_PEEK_CAP`(8),
`MSTAR_CONDUCTOR_POLL`(**ON**), `MSTAR_EMIT_SIDECAR`,
`MSTAR_SIDECAR_CHECKSTOP`, `MSTAR_SIDECAR_CHECKSTOP_SHADOW`,
`MSTAR_SIDECAR_BATCH`(1), `MSTAR_SAMPLER_CFG_CACHE`,
`MSTAR_SAMPLER_CFG_CACHE_V2` (OFF — the "default ON" code comment is stale),
`MSTAR_MERGED_PREFILL`, `MSTAR_MERGED_PREFILL_AUDIO` (off-stack),
`MSTAR_CUSTOM_OPS`, `MSTAR_PREP_DEVICE_POS`, `MSTAR_MIXED_BATCH`(`_ASSERT`,
`_VISION`), `MSTAR_MIXED_SPEC`, `MSTAR_MIXED_PREPLAN`,
`MSTAR_MIXED_SPLIT_ATTN`, `MSTAR_MIXED_SINGLE_CHUNK`,
`MSTAR_MIXED_BUDGET_TOKENS`, `MSTAR_MIXED_BUDGET_MIN_DECODE`,
`MSTAR_MIXED_MIN_DECODE`(24 floor), `MSTAR_CHUNKED_PREFILL_V2`(`_VISION`,
`_ASSERT`), `MSTAR_PREFILL_CHUNK_TOKENS`,
`MSTAR_SPEC_PEEK_FOR_FAIRNESS`(**ON**),
`MSTAR_MAX_CONSECUTIVE_SPEC_STEPS`(1024), `MSTAR_PRE_PLAN_SPEC`(**ON**),
`MSTAR_DECODE_BUCKETS`([1,2,4,8,16,24,28,32]), `MSTAR_PREFILL_BUCKETS`,
`MSTAR_BATCH_VISION_PREFILL`, `MSTAR_GC_TUNE`,
`MSTAR_PY_SWITCH_INTERVAL_SEC`, `MSTAR_STEP_TXN`(`_VERIFY`) (off-stack),
`MSTAR_ASYNC_SCHED` (off-stack), `MSTAR_ADMIT_JITTER_MS` (off-stack),
`MSTAR_PREFILL_GATHER_MS`/`_TARGET`/`MSTAR_ENCODER_GATHER_TARGET` (off-stack),
`MSTAR_MULTISTEP_DECODE` (off-stack), `MSTAR_MIXED_WALK` family (off-stack).

Model/env (mostly ON): `MSTAR_ENCODER_CUDA_GRAPH`(**ON**),
`MSTAR_ENCODER_CG_MAX_KEYS`(16), `MSTAR_ENCODER_CG_WARMUP`("1,2,4,8"),
`MSTAR_QWEN3_NATIVE_AUDIO_ENCODER`/`_VISION_ENCODER`(config default),
`MSTAR_GPU_MEL`(**ON**), `MSTAR_GPU_IMAGE_PREPROCESS`(**ON**),
`MSTAR_VLLM_PROMPT_LAYOUT`(**ON**), `MSTAR_VLLM_AUDIO_SENTINELS`,
`MSTAR_VIT_BATCHING`, `MSTAR_VISION_GRAPH_ALIGN`,
`MSTAR_VARLEN_BACKEND`("flashinfer"), `MSTAR_WORKSPACE_BUFFER_MB`(512),
`MSTAR_NUM_SLOTS`(2), `MSTAR_WALK_STATS`, `MSTAR_PHASE_TIMING`,
`MSTAR_DYNFLAGS`, `MSTAR_DUMP`/`_DIR`, `MSTAR_ZMQ_TRANSPORT`/`_TCP_HOST`/
`_TCP_BASE_PORT`.

## D. Measured bottom line (vs committed vLLM refs)

Speech 12/12 GREEN 2.1–2.9×; s2t 6/6 GREEN 1.08–3.06×; i2t B1–B16 GREEN
1.04–1.22×; i2t B32 band-parity (mean 7.98–8.37, peak 8.95 vs band 8.03–8.50)
with the residual proven structural (worker.py:3193 await-GPU gate — see
HANDOFF_V8 §3).
