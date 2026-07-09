# M* AR decode loop — empirical CPU floor + 10 refactors to make it Python-free / torch.compile-able

Date: 2026-07-07. Goal: kill the ~26 ms per-step Python/GIL floor that pins i2t B32
decode (GPU ~50% idle, one worker core saturated). Two inputs: (1) a **live py-spy
profile** of the actual decode worker under sustained B32 i2t decode, and (2) an
8-agent line-by-line comparison of how vLLM-Omni 0.22 does the same work.

Paths: **M/** = `/m-coriander/coriander/tim/mstar-new/mstar/` (running build; on-disk
worker.py was updated post-boot so live line numbers differ — function anchors below
are current-checkout). **V/** = vLLM `…/vllm-omni/.venv/…/site-packages/vllm/`.

---

## ⚠️ SCOREBOARD REFRAME (2026-07-07) — the decode floor is NOT the i2t loss

Fresh measurement on the current best build (`mstar-new` @ 8402, GPUs 6/7, all opt
flags ON: SIDECAR_CHECKSTOP, FAST_POSTPROC/ROUTE/SEND, SLIM_EMIT, PREP_DEVICE_POS,
MIXED_BATCH, MERGED_PREFILL, CHUNKED_PREFILL_V2, MOE_FP8/AUTOTUNE, CUSTOM_OPS) vs the
committed vLLM 0.22 h2h. i2t, warmup 10, `--ignore-eos`:

| B | M* req/s | vLLM req/s | M* ITL | vLLM ITL | **M* TTFT** | **vLLM TTFT** |
|---|---|---|---|---|---|---|
| 1 | **0.98** ✅ | 0.88 | **4.3** ✅ | 4.8 | 196 | 87 |
| 8 | **4.20** ✅ | 3.67 | **6.1** ✅ | 9.6 | 486 | 100 |
| 16 | **5.89** ✅ | 5.62 | **7.1** ✅ | 11.9 | 912 | 168 |
| 32 | 7.42 ❌ | **8.32** | **9.3** ✅ | 15.4 | **2400** | **179** |

**M* already beats vLLM on ITL at every batch, and on req/s at B1–B16.** At B32 len512
(decode-dominated) M* does **1771 tok/s vs vLLM 1769** with **ITL 10 vs 15.4** — decode
is at parity-or-better. **The one and only i2t loss is B32 req/s, caused 100% by TTFT:**
~2400 ms vs vLLM's flat 179 ms (**14×**). vLLM holds TTFT flat across batch (chunked +
fairly-scheduled prefill); M* prefill serializes at high concurrency. The already-on
mixed-prefill flags only nudged it (2532→2369).

**Consequence for this doc:** R1–R10 optimize the *decode* floor (ITL, tok/s) — exactly
the axes M* already wins. **They cannot close the B32 loss.** The real best-ROI target is
**B32 TTFT = prefill admission/chunking scheduling** (make a waiting request's first
prefill chunk run promptly instead of behind the full queue — vLLM's flat-TTFT behavior).
Parity note: pure chunking of a single prefill is parity-safe (same attention math, same
tokens); it's the *co-batching* of prefill+decode (already shipped as MIXED_BATCH) that
carries the batch-composition parity risk. Pursue the chunk/admission-order lever, not more
decode work. Keep R1/R4 only as ITL-tail (p99) polish — decode p99 spikes to 133 ms from
GIL contention, a latency (not throughput) nicety.

## A. Empirical floor — where the GIL thread actually spends decode CPU

Profiled PID 2521088 (the Thinker decode worker, GPU 7) under `--max-concurrency 32
--ignore-eos --output-len 768` i2t, py-spy `record --rate 300` over 40 s + 8 instant
dumps. The worker MainThread runs `worker.run()` → `_postprocess_batch` per step. GPU 6
sat at **0%** while GPU 7 decoded — the 30B Thinker decodes single-GPU, so the whole
per-step cost is one Python thread. Self-time (leaf) ranking:

| Rank | Self % | Frame | What it is |
|---|---|---|---|
| 1 | **18.0%** | `_prematerialize_for_check_stop` (worker.py:1871) | D→H materialize of new tokens every step to run stop checks on CPU |
| 2 | **~13%** | `_send_outputs`→`send_pyobj` (worker.py:946) | pickle + zmq-send each step's outputs to the conductor |
| 3 | **7.9%** | `torch.cuda…synchronize` (worker.py:1660/1664) | hard per-step stream barrier before check_stop |
| 4 | **8.1%** | `run` body (worker.py:1949 loop) | input/plan/route bookkeeping between replays |
| 5 | ~5% | `_replay_populate_plan`/`_plan_matches`/`store_and_populate_graph_edges`/`_replay_route_plan` (tensors.py) | per-step FlashInfer plan replay + tensor-edge routing |
| 6 | ~1.5% | `[edge.clone() for edge in …]` (base.py:75) | per-request output edge clone |
| 7 | ~1.2% | `uuid4` (uuid.py:725) | a fresh UUID per output tensor per request per step |

Inclusive: **`_postprocess_batch` = 23.4%** of worker wall, and it is 100% serial Python
(a per-request `for` loop, worker.py:1673-1761, doing stop-loop handling, zmq peer
sends, `store_and_populate_graph_edges`, uuid set-comp, edge clones, `process_node_outputs`).
The forward itself is captured in a CUDA graph and cheap; **postprocess + plan-replay +
sync are the floor.** vLLM does *none* of this on its critical thread.

**One-line thesis:** M* returns to Python (and to a host sync) *between every decode
step* for three reasons — re-plan FlashInfer, rebuild positions, and postprocess/
check-stop/route on the engine thread. vLLM does all three either in-graph (tensor ops)
or off-thread. Removing those three is what unlocks a captured multi-token decode
(multi-step is already proven **byte-identical-correct** — it only regressed because it
re-planned inline per micro-step).

---

## PARITY FILTER (2026-07-07) — only ship refactors that keep the output byte-identical

Constraint: **greedy output must be unchanged, token-for-token.** No speculative decoding,
no sampler/temperature change, nothing that shifts batch composition (which flips logits
via FP non-associativity — the `project_mstar_v1_async_sched` trap: async-sched shifted
batch composition → +9% length, NON-identical). Verdict per refactor:

| Refactor | Parity | Why |
|---|---|---|
| **R1** exile postprocess | ✅ identical | moves *where* work runs, not *what* — same tokens |
| **R2** capturable plan-advance | ✅ identical | same plan values; the multi-step it unblocks is already byte-identical; NOT speculation (replays the same model, no guess/verify) |
| **R3** on-GPU positions | ✅ identical | same position ids; existing `MSTAR_PREP_DEVICE_POS` is identical-by-design |
| **R4** in-graph stop-check | ✅ identical | same eq/isin decision, on GPU not CPU |
| **R5** persistent slot routing | ✅ identical | uuid/clone/dict bookkeeping only |
| **R6** batched zmq | ✅ identical | transport only |
| **R9** in-graph seq-len advance | ✅ identical | same counter values |
| **R10** in-graph sampler | ⚠️ **argmax-only** | greedy argmax-on-GPU is identical; **the Gumbel-Max stochastic path is EXCLUDED** (touches RNG/temperature) |
| ~~**R7**~~ ready-set + budget/**chunked/mixed**~~ | ⛔ **EXCLUDED** | changes batch composition + prefill chunking → non-identical (async-sched trap) |
| ~~**R8**~~ split-at-attention **torch.compile**~~ | ⛔ **EXCLUDED** | Inductor fusion changes FP reduction order → logits flip at ties; not bit-identical unless a byte-identical gate passes |

**Parity-safe implementation set: R1, R2, R3, R4, R5, R6, R9, R10(argmax-only).** These
remove the *entire* measured CPU floor (postprocess 23% + plan 5% + positions + sync)
without touching a single emitted token. Every one must still pass a greedy
byte-identical A/B (n=1 vs baseline, `--ignore-eos` fixed len) before it lands — same
gate that certified `MSTAR_DECODE_MULTISTEP`.

**Excluded because they can change the output:** R7 (batch-composition/chunked-prefill),
R8 (compile fusion), R10's stochastic sampler. R7's *throughput* intent (mixed
prefill+decode for TTFT) is real but must be re-derived in a parity-preserving form —
identical batch composition, only the *packing* changed — before it's admissible; treat
it as blocked, not scheduled.

## B. The 10 refactors (ranked by ROI on the measured floor)

### R1 — Exile `_postprocess_batch` off the engine thread ⭐ biggest single win (~30–40% of floor)
- **Measured target:** the whole `_postprocess_batch` (M/worker/worker.py:1603-1762), 23%
  inclusive + its `send_pyobj` 13% — all serial on the GIL thread that must also launch
  the next forward.
- **vLLM:** the model runner returns raw GPU token tensors; detokenize / stop-check /
  output serialization live in a *separate* `OutputProcessor` on the engine-core output
  thread (`V/v1/engine/core.py`, `V/v1/engine/output_processor.py`) — the GPU thread
  never blocks on them. `execute_model` → async output queue.
- **M* change:** hand `output` (GPU tensors + metadata) to a single-consumer output
  thread the instant the forward's completion event is recorded; the run loop launches
  step N+1 immediately. check_stop, `store_and_populate_graph_edges`, uuid, clone,
  zmq-send all move off the critical thread. This is the design already sketched as
  `MSTAR_SIDECAR_CHECKSTOP` (memory `project_mstar_sidecar_checkstop`) but generalized
  from check_stop-only to the *whole* postprocess.
- **Risk:** the run loop must not re-enter a batch whose stop hasn't been decided — keep a
  1-step-behind ready-set (see R7). GIL contention between two Python threads is real but
  both are mostly in C (torch, zmq) so they interleave.

### R2 — Capturable FlashInfer plan-advance: kill the per-step `plan()` ⭐ keystone (unblocks multi-step)
- **Measured target:** `_replay_populate_plan`/`_plan_matches` (tensors.py) ≈ 5% here, but
  the real cost is the ~750 µs `plan()` + 2 internal D→H CUB reads that force a return to
  Python each step (M/engine/cache_manager.py plan path; in-code TODO admits it).
- **vLLM:** persistent paged-KV buffers allocated once (`V/v1/attention/backends/flashinfer.py:688-693`),
  refreshed in place via `np.cumsum(out=…)` + a Triton page-index copy (829-884); build
  calls **`fast_decode_plan`** (1858-1948) which after the first call does "only H2D copy
  of indptr and last_page_len", skipping workspace re-partition. TRTLLM path skips `.plan()`
  entirely.
- **M* change:** allocate `paged_kv_indptr/indices/last_page_len` once as persistent CPU+GPU
  buffers per decode-bs bucket (replace the fresh `torch.tensor(...)` per step). Steady-state
  update is `last_page_len += 1` (append one page index at a page boundary) — a capturable
  tensor op, plus a `fast_plan`-style tiny H2D. Never the full `plan()`.
- **Why it matters most:** this is the single blocker that made `MSTAR_DECODE_MULTISTEP`
  (byte-identical n=1/2/4) regress 3.5–4.8× — it re-planned inline per micro-step. Crack
  this and multi-step wins immediately.

### R3 — On-GPU TM-RoPE position advance: no `.item()`, no `torch.tensor(...,device=cuda)` per step
- **Measured target:** per-step position build is a pageable H2D every step
  (M/model/qwen3_omni/submodules.py:640 batched / :701-705 default); in-code comment flags
  it at **24% of wall at bs=1**. Prefill build loops with 3× `.item()` per image
  (components/rope.py:332-335).
- **vLLM:** compute `mrope_position_delta` once at prefill (one `.item()`, `qwen2_5_vl.py:1278`);
  decode advance = `np.arange(delta+ctx …)` into a pinned buffer + one async H2D
  (`mrope.py:401-414`); spec path drifts as a pure GPU add.
- **M* change:** promote the already-present-but-default-OFF `MSTAR_PREP_DEVICE_POS_BATCHED`
  (pinned host buffer + async H2D, submodules.py:625-638) to the always-on path, and swap
  the prefill per-image `.item()` loop for vectorized `arange`/`meshgrid`/`repeat_interleave`
  driven by `grid_thw` kept as a tensor. Removes the last per-step host sync besides plan.

### R4 — In-graph stop-token check (drop the D→H prematerialize) — attacks the #1 leaf (18%)
- **Measured target:** `_prematerialize_for_check_stop` = **18% self**, the single hottest
  frame. It copies new tokens D→H (worker.py:1856-1948, `_get_pinned_d2h_buffer` + `.cpu()`)
  so `check_stop_for_batch` can compare on CPU.
- **vLLM:** stop detection is done against GPU tensors where possible; EOS/stop-id compare
  is a GPU `eq`/`isin`, and only a *small* per-step "did any finish" bit is synced — not the
  full token materialization. Detokenizer stop-strings run off-thread on already-synced ids.
- **M* change:** compute the stop mask on-GPU (`new_tokens.eq(eos_id) | isin(stop_ids)`),
  D→H only the boolean mask (or a popcount) once per step, and defer string-stop /
  max-len bookkeeping to the R1 output thread. With ignore_eos benchmarks this is nearly
  free; in production it removes an 18% hot copy from the critical thread.

### R5 — Batched, persistent output routing: kill per-request uuid + clone + dict churn
- **Measured target:** the per-request `for` loop worker.py:1733-1761 —
  `store_and_populate_graph_edges` (tensors.py:628), a `{info.uuid …}` set-comp
  (`uuid4` = 1.2%), `real_outputs = [edge.clone() …]` (clone 1.5%), `process_node_outputs`
  (node_manager_utils.py:478, 1.2%), all rebuilt each step.
- **vLLM:** outputs are index-addressed in a persistent `InputBatch` (fixed slots, O(1)
  add/remove/condense, `V/v1/worker/gpu_input_batch.py`) — no per-request dict keyed by a
  fresh UUID, no per-step clone; the sampled-token tensor is sliced by slot.
- **M* change:** replace UUID-keyed edge maps with a persistent slot table (rid→slot fixed
  for a request's life); reuse pre-allocated edge tensors (write in place instead of
  `clone()`); generate uuids only at request admission, not per step. Pure-Python-overhead
  removal, no numerics change.

### R6 — Batch the per-step zmq output send (one message, not one send_pyobj per hop)
- **Measured target:** `send_pyobj` 9.6% + `send` 3.3% ≈ **13%** — pickling Python objects
  and sending per step (worker.py:946-1006, `_send_outputs`), plus extra small STOP_LOOPS
  sends inside the postprocess loop (worker.py:1714-1725).
- **vLLM:** engine↔worker uses msgpack over a single shared-memory ring per step, one
  serialized blob, not per-object pickle.
- **M* change:** coalesce all of a step's outputs + loop-stops into one `send` with a
  compact codec (msgpack/raw buffers, not `send_pyobj` pickle). M* already has SHM tensor
  protocol; extend it to the control message so the 13% pickle cost drops toward memcpy.

### R7 — ⛔ EXCLUDED (parity): incremental ready-set + budgeted (Sarathi-style) scheduling
> Changes batch composition + prefill chunking → non-identical output (async-sched trap).
> The incremental ready-set *bookkeeping* alone is parity-safe and can be salvaged for R1;
> the budgeted/chunked/mixed scheduling is what's excluded.
- **Target:** `run()` body (8.1%) rescans request state to form each batch; combined with
  the R1 exile it needs a lock-free "which rids are ready for step N+1" structure so the
  loop never waits on postprocess.
- **vLLM:** `Scheduler` keeps `running`/`waiting` deques and a token budget; each step pops
  a budgeted set in O(active), and chunked-prefill folds prefill tokens into the same step
  (`V/v1/core/sched/scheduler.py`). No full rescan.
- **M* change:** maintain an incrementally-updated ready-set (add on admission, remove on
  stop) and a per-step token budget so mixed prefill+decode packs into one forward — this
  is also the TTFT lever (see §C) and what makes captured mixed-batch feasible.

### R8 — ⛔ EXCLUDED (parity): split-at-attention torch.compile (`fullgraph` around the block)
> Inductor fusion reorders FP reductions → logits can flip at ties → not byte-identical.
> Only admissible if a greedy byte-identical A/B passes on the compiled graph.
- **Target:** M* runs the Thinker forward eager-with-custom-ops (fullgraph=False); the
  per-op Python dispatch between kernels adds to the floor and blocks whole-block fusion.
- **vLLM:** `@support_torch_compile` compiles the model with `fullgraph=True` and an FX
  split *only* at the attention op (`splitting_ops`, `V/compilation/{decorators,backends,
  partition_rules}.py`), so everything between attentions is one fused Inductor graph
  captured into the CUDA graph; attention stays a custom op with `register_fake`.
- **M* change:** wrap the decoder block in `torch.compile(fullgraph=True)` with attention
  as a `splitting_op` custom op (M* already has `register_fake` custom ops from
  `opt/custom-ops`). Reduces Python dispatch to one graph entry per block and lets Inductor
  fuse RMSNorm/residual/RoPE-apply. (Boot-time cost needs the FX-cache flags from the
  custom-ops memo.)

### R9 — In-graph seq-len / last-page advance (fold `advance_seq_lens` into tensor ops)
- **Target:** `advance_seq_lens` is a Python loop bumping python-int `position_id_start`/
  `seq_len` per request *after every replay* (cache_manager.py advance path) — the piece
  that forces a Python return between micro-steps even after R2/R3.
- **vLLM:** decode counters live in GPU int tensors advanced by an in-graph `+= num_new`
  (`V/v1/worker/gpu_model_runner.py:2119-2126`); Python ints are only a lazily-synced mirror.
- **M* change:** back seq_len / last_page_len / position with small persistent GPU int
  tensors advanced by a captured `+= step`; keep the Python mirror synced once per burst,
  not per step. With R2+R3+R9 no per-step host work remains → K steps replay as one graph.

### R10 — In-graph **argmax** sampler (remove the sampler host round-trip) — ⚠️ argmax-only
> Parity-safe scope: greedy **argmax** on GPU (identical). The **Gumbel-Max stochastic
> path is EXCLUDED** — it touches RNG/temperature and changes the output.
- **Target:** sampling currently returns to Python between forward and the next input build;
  for greedy it's an argmax but still a host-visible tensor read that gates the step.
- **vLLM:** the sampler (`V/v1/sample/`) runs entirely on GPU; sampled ids feed straight
  back into the persistent input buffer with no D→H except the final output copy.
- **M* change (argmax-only):** keep **greedy argmax** on-GPU and write the sampled id
  directly into the persistent decode-input slot (R5) so the next replay consumes it
  without a host hop. Together with R2/R3/R9/R4 this closes the last per-step sync, making
  a fully in-graph K-token decode loop possible — M*'s equivalent of the MRV2 zero-sync
  decode. **Do NOT** port the stochastic Gumbel-Max path (RNG/temperature → output
  changes) under the parity constraint.

---

## C. Dependency order (what unlocks what)

```
R2 (capturable plan-advance) ── keystone ───┐
R3 (on-GPU positions)                       ├─→ captured K-token multi-step decode
R9 (in-graph seq-len advance)               │   (MSTAR_DECODE_MULTISTEP already byte-identical)
R10 (in-graph sampler)                      ┘
R1 (exile postprocess) ─ independent, biggest immediate win ─→ enables R4,R5,R6 off-thread
R4/R5/R6 (check-stop / routing / zmq)  ── ride on R1's output thread
R7 (incremental ready-set + budget) ─→ also the TTFT lever (mixed prefill+decode)
R8 (split-at-attention compile) ─ independent, reduces dispatch floor
```
Do **R1 first** (largest measured floor, no numerics risk, independent), then **R2** (the
keystone that makes multi-step actually win). R3/R9/R10 finish the in-graph decode. R7 is
the bridge to the TTFT fix (B32 req/s is TTFT-gated; see BEAT_VLLM_MASTERPLAN.md).

---

## D. Did earlier advances help ITL/TTFT on *other* paths? (side-question)

- **`MSTAR_SIDE_PREFILL` — real ITL win, overlooked.** Once the thread-safety bug was fixed
  (`_ACTIVE_MANAGER` module-global → `threading.local()`, opt/sideprefill-fix @586fcfa1),
  overlapping the encoder/prefill on a side stream **cut i2t B32 ITL ~27%** (latency-shaping)
  even though it washed on throughput (no decode to overlap during the initial burst). This
  is a genuine **ITL** improvement worth shipping for latency-sensitive serving, independent
  of the throughput story. Re-confirm the ITL delta under warmup=10/n=10 and bank it.
- **`MSTAR_MOE_AUTOTUNE` — inconclusive on ITL/TTFT, worth a targeted re-measure.** It gave
  +4–12% on the isolated MoE GEMM but e2e throughput washed (decode CPU-bound). TTFT is
  GPU-bound (prefill), so autotune *should* help TTFT even where it doesn't help decode
  throughput — but the loop runs only captured decode-side ITL/throughput, not a clean TTFT
  A/B. **Action:** measure TTFT with/without `MSTAR_MOE_AUTOTUNE` at B8/B16/B32 (prefill is
  where the tuned tiles bite); if TTFT drops it's a free win on the #2-priority metric.
- **All other levers** (multistep-as-is, prefill-chunk 256, config symmetry) washed or
  regressed on every metric — see LOOP_THROUGHPUT_RESULTS.md.

## E. Provenance
Empirical: py-spy of PID 2521088 under live B32 i2t decode, 2026-07-07 (dumps +
speedscope in /home/tim/tmp/decode_prof3, not committed — regenerate via a sustained
`--ignore-eos --output-len 768 --max-concurrency 32` load). Comparative: 8 vLLM-vs-M*
source agents + 1 design agent (this session). No vLLM was booted (owner rule); vLLM
mechanism read from committed source only.
