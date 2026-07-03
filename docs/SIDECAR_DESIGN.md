# SIDECAR_DESIGN — exiling per-token emit/postprocess work from the worker GIL

Design document only. No implementation. Branch `exp/overlap-sched` @ 333b7dd.
Produced 2026-07-03 from a full code read of the postprocess/emit/route paths
plus the measured record in the knowledge base (EXPERIMENTS.md at
`bench-merge/benchmarks/qwen3-omni-joint/`, referenced below by section name)
and PLAN_BEAT_VLLM_V4.md (vLLM-Omni source dive).

---

## 0. TL;DR and recommendation

**GO on Stage 1** (emit + WGD sidecar), ranked **first** of the three
day-scale options:

| Rank | Option | Predicted e2e (i2t B32) | Size | Why this rank |
|---|---|---|---|---|
| 1 | **Stage-1 sidecar** (this doc) | **+10–20%** (net −1.3–2.0 ms off an 8.7 ms Python wall) | 2–4 d | Attacks the measured wall directly: main-thread Python ≈ 100% of step time |
| 2 | V1 full async-sched (vLLM pattern) | +0–5% *now*; real value post-sidecar | 3–5 d | Deepens the pipeline but cannot push the wall below the per-step Python sum under one GIL; prof_final already ruled "V1 re-enters only after main-thread < GPU time" |
| 3 | GPU-side kernel work (e.g. #24 MoE bake-off) | ~0% *now* | 2–4 d | GPU is 51% busy; every GPU ms saved converts ≈0 while the host floor binds (the fp8 lesson in reverse) |

Stage-1 expected main-thread removal, honestly discounted: **gross movable
≈ 1.7–2.5 ms/step, minus 0.3–0.5 ms hand-off cost ⇒ net ≈ 1.3–2.0 ms** of the
8.74 ms median postprocess wall. If it converts ms-for-ms (the SLIM_EMIT
precedent), step 8.8 → ~7.0–7.5 ms ⇒ +15–25% e2e; discounted for the
FAST_SEND counter-precedent (small trims on this path inverted), predict
**+10–20%**, kill below +2%.

**Biggest risk found:** *split-brain WGD accumulation state.* The three
per-request accumulators that ride WORKER_GRAPHS_DONE
(`pending_new_tokens`, `current_output_chunks`, `output_loop_indices` —
`node_manager_utils.py:283-287`, flushed at `worker.py:1497-1504`) are written
on the per-token path (moves to the sidecar) **and** on boundary/prefill/
non-inline paths (stay on the worker). If ownership is split, WGD is assembled
from two half-truths and the conductor's completion accounting silently
corrupts — the same failure class as the SLIM_EMIT template-mutation bug that
cost a 5× regression before it was found (EXPERIMENTS §"MSTAR_SLIM_EMIT
REGRESSED 5x"). The design therefore moves accumulator ownership *wholesale*:
the worker never writes them for sidecar-scoped walks; every write becomes a
record field. This is the invariant to shadow-verify hardest.

NO-GO criteria are in §9. The line-item accounting (§4.4) clears the <1 ms
NO-GO bar with margin, so the recommendation is GO — but it is a *discounted*
GO: roughly half the profiler-attributed send cost is load-bearing state
mutation or GIL-shade filler that does not leave.

---

## 1. The measured wall (why incremental cuts are over)

All numbers from the committed record (i2t B32, pair 0,1, prof_final2 +
adjacent A/Bs):

- Decode step **median 8.8 ms**; GPU busy **51.4%**. Main thread
  `postprocess_batch` **median 8.74 ms ≈ the step** — the main thread *is*
  the pipeline wall. (EXPERIMENTS §"Re-profile with ROUTE2+SLIM2")
- Component medians after six landed optimizations (FAST_POSTPROC, SLIM_EMIT/2,
  FAST_ROUTE/2, sampler cache, FAST_CHECKSTOP, N2):
  **route 1.82 ms** (mark_node_complete + process_new_inputs + clones remain),
  **send_outputs 3.14 ms**, **check_stop ~1.1–2.1 ms** (deliberate side-stream
  D2H wait), **register ~1 ms**, **shell ~2 ms**.
- **The GIL-valve law** (EXPERIMENTS §"Sampler config-tensor cache"): on a
  multi-GIL-thread worker, removing gpu-thread *waits* pays only after
  main-thread Python shrinks or moves off-GIL. Corollary proven twice since:
  removing main-thread *work* converts (SLIM_EMIT +15–25%, FAST_ROUTE +7–10%),
  removing interleaved *micro-Python* does not — **MSTAR_FAST_SEND cut a
  CPU-verified −0.4 ms and regressed −3% e2e, 4/4 adjacent pairs**
  (EXPERIMENTS §"MSTAR_FAST_SEND — REGRESSION"). Two independent audits agree
  the remaining items are load-bearing or wait-valves.
- E9/E10 (direct GPU feed, two-step decode): both flat — the loop-cycle
  overhead hypothesis is falsified; the floor is the per-token
  route/store/emit Python, and E9's implementation proved the route/store
  block load-bearing (Loop semantics). (EXPERIMENTS §Queue E9/E10)

Conclusion inherited from the FAST_SEND verdict: the incremental-Python-cut
region is exhausted. The next win must **remove work from the worker process
entirely** or restructure who waits where. This is vLLM V1's answer: the
engine process does model-forward + scheduling only; detokenization,
serialization, and client-bound output processing live in a different process
(EngineCore pattern; EXPERIMENTS option board V4 "detok/serialization out of
the conductor process; ZMQ IO threads release GIL; one batched
EngineCoreOutputs per step"; PLAN_BEAT_VLLM_V4 Part 2: their async D2H at
`vllm_omni/worker/gpu_model_runner.py:249-266` is the V1 half; the
output-processor exile is the V4 half M* is missing).

---

## 2. Anatomy of the per-step main-thread work

`Worker._postprocess_batch` (`worker.py:2598-2944`), per decode step at B32:

| # | Block | file:line | ms (median) | Character |
|---|---|---|---|---|
| 1 | cleanup + overstay dedup + pending-stop clear | `worker.py:2604-2645` | shell | scheduler-state mutation |
| 2 | `per_req_nested_idxs` fetch (per rid) | `worker.py:2647-2651` | shell | feeds emit loop_indices |
| 3 | completion_event sync | `worker.py:2666-2676` | ~0.9 (older prof) | GIL-releasing wait |
| 4 | `_prematerialize_for_check_stop` — side-stream cat + pinned D2H + `side.synchronize()` | `worker.py:2693, 3001-3120` (sync at 3066) | 1.1–2.1 | GIL-releasing **deliberate wait** (graph-tail) |
| 5 | check_stop decision (FAST_CHECKSTOP int compares) | `worker.py:2697-2726` | in #4's figure | pure ints, cheap |
| 6 | stop_loops + peer STOP_LOOPS msgs | `worker.py:2733-2788` | boundary-only | scheduler-state mutation |
| 7 | **route/store loop**: `store_and_populate*` (2820/2829), `mark_node_complete` (2842), edge clones (2845), `process_node_outputs` (2847), `set_output_ref_counts` (2862) | `worker.py:2794-2864` | **1.82** | **load-bearing** (see §3.2) |
| 8 | prem dict build | `worker.py:2878-2884` | small | already-materialized ints |
| 9 | `_register_outputs` (SHM-skip decision, per-rid register ctx) | `worker.py:2889, 1138-1198` | ~1.0 | worker-local tensor lifecycle |
| 10 | **per-rid `_send_outputs` loop** + one batch pickle/send | `worker.py:2922-2939, 1200-1510` | **3.14** | mixed — the Stage-1 target, dissected in §4.4 |

The pipeline shape (`worker.py:1513-1526`): speculate/build N+1 → await
GPU(N) → thread outputs → submit GPU(N+1) → postprocess(N) overlapping
GPU(N+1). Since postprocess(N) ≈ 8.7 ms > GPU(N+1) ≈ 4.5 ms, the next
`await_gpu` returns immediately and iteration time is postprocess-bound.

---

## 3. Q1 — what can leave the worker process, and what cannot

### 3.1 Clearly movable: emit/serialization + client-bound bookkeeping

The api_server **already consumes asynchronously** — it is a separate process
draining a ZMQ PULL socket (`api_server/entrypoint.py:419-480`), inflating
slim items from cached templates (`entrypoint.py:348-394`) and routing per rid
under its own lock (`entrypoint.py:396-417`). Nothing the worker computes on
the emit path is read back by the scheduler:

- **Emit message construction**: per-rid `SlimResultTokens`/`ResultTensors`
  building, slim-template bookkeeping (`_slim_emit_sent`,
  `_slim_emit_loop_layout`), loop_key derivation, metadata dicts
  (`worker.py:1297-1440`).
- **The batch pickle + send**: `communicator.send` is `send_pyobj` — the
  pickle runs on the worker main thread holding the GIL
  (`communication/communicator.py:116-131`); only the socket write is on
  ZMQ's IO thread.
- **WGD assembly on completions**: flushes + message build at
  `worker.py:1475-1510` (boundary-rate, ~1 per rid per partition).
- **Profiling accounting** (`worker.py:2907-2913`) — off in benchmarks, moves
  for free.

### 3.2 Cannot leave: route/store/loop bookkeeping (the scheduler contract)

The route/store block (`worker.py:2794-2864`, 1.82 ms) drives **next-step
readiness**. Exact load-bearing structures:

- **`Loop.complete_iter`** (`graph/base.py:505-537`, called via
  `mark_node_complete` → `graph_io.py:76` → `base.py:688`): advances
  `curr_iter`, resets the inner registry, and returns either the loop-back
  re-injection edges (steady state) or the terminal declared outputs + 
  filtered loop-back signals (stop). The *decision* of which shape the next
  step takes lives here. E9's implementation note is the controlling
  precedent: "mark_node_complete drives Loop.complete_iter; the stored
  loop-back edge feeds next-step readiness for the non-spec fallback, so it
  cannot be skipped" (EXPERIMENTS §Queue E9).
- **Queue readiness**: `process_node_outputs` ingests loop-back edges into
  `WorkerGraphQueues.process_new_inputs` → `WorkerGraphIO.ingest_input` →
  `ready_node_names` (`node_manager_utils.py:451-627`, local ingest at
  506-527; `graph_io.py:48,59`). The scheduler's `get_next_batch` reads this
  ready set; the speculation chain's fairness/fold peeks
  (`worker.py:3404-3474`) read it too. If this lags a step, the non-spec
  fallback schedules stale batches and the mixed-batch fold peeks lie.
- **`tensor_manager` stored refs**: `store_and_populate_graph_edges*` assigns
  the `TensorPointerInfo`/uuids that next step's `prepare_inputs` resolves,
  and `set_output_ref_counts` (`worker.py:2862`) + `dereference`
  (`tensors.py:1045`) guard tensor_store reuse. GPU-adjacent, worker-local
  by construction — a CUDA-less sidecar cannot own it.
- **The is_done sweep + `queue.reset`** (`node_manager_utils.py:534-541`)
  produces `completed_worker_graph_ids` — the trigger for WGD and for the
  request's walk transition.

Moving this block out means the sidecar owns scheduler ground truth and the
worker asks it what to run next — that is not a sidecar, that is the vLLM V1
EngineCore *rewrite* (scheduler owns state, workers stateless). W2 already
proved the cheaper half of that bet doesn't pay here: full memoized replay of
the walk bookkeeping (3-4× fewer ops) was correctness-perfect and e2e-flat
(EXPERIMENTS §W2). Stage 3 evaluates this honestly (§8) but the prior is
strongly against.

**Boundary cases sitting between the two piles:**

- `buffer_new_tokens` / `buffer_output_signals` /
  `register_output_loop_indices` (`node_manager_utils.py:982-995,441-446`):
  client-bound accumulation, read *only* by the WGD flush — movable, **but
  only with whole-owner transfer** (§4.3, risk §0).
- `check_stop`: gates scheduling one step late already (the speculative N+1
  is submitted at `worker.py:3736` before check_stop(N) runs at 2693; the
  overstay dedup at 2607-2645 exists precisely to absorb that). Movable in
  principle with a stop-feedback message — Stage 2, with the lag/waste math
  in §6.2.
- The local-release `dereference` loop (`worker.py:1446-1447`): stays —
  tensor_store is worker state; it is ~0.1–0.2 ms of dict ops.

---

## 4. Q2 — the minimal viable sidecar

### 4.1 Topology

One **sidecar process per worker** (2 workers ⇒ 2 sidecars on the 2-GPU
config). The worker keeps route/store/check_stop; the sidecar owns emit + WGD:

```
                 per-step compact record (ZMQ PUSH, pickled small tuple)
  worker main ────────────────────────────────► sidecar (no CUDA context)
  thread      ── boundary record on completions ─►   │
                                                     ├─► api_server: ResultTensorsBatch
                                                     │   (SlimResultTokens items, templates
                                                     │    OWNED by the sidecar)
                                                     └─► conductor: WORKER_GRAPHS_DONE
```

- Transport: **ZMQ PUSH/PULL, one pair, `send_pyobj`** of a compact record.
  Rationale: the payload is a few KB of ints/strs (pickle ~50–150 µs); the
  existing communicator/poller/EventWakeup machinery
  (`communicator.py:74-79`) is battle-tested; a single PUSH/PULL pair
  preserves the FIFO the slim-template protocol requires
  (`entrypoint.py:443-447`: "same FIFO stream guarantees the template
  precedes any slim item"). An SHM ring is the escalation if the record
  pickle ever shows up in a profile — do not build it first (the audio SHM
  file transport at `tensors.py:1272-1342` is the wrong shape for
  per-step control records).
- The sidecar sets `CUDA_VISIBLE_DEVICES=""` and never initializes CUDA; it
  is a pure-CPU Python process with its own GIL. Spawned (not forked — the
  worker holds a CUDA context) at worker init, hidden behind model load.
- Scope: **walk-gated** to the text paths (`thinker_decode`, `prefill_text`,
  `thinker_mixed`), per the E4b lesson that ungated fast paths taxed Talker
  steps 17%. Talker/Code2Wav emit rides streaming edges, not this path, and
  stays untouched.

### 4.2 The per-step record (steady decode, B32)

```python
StepRecord(
    step_id: int,                  # monotonic, per worker
    walk: str, partition: str,
    items: list[tuple[
        rid_idx: int,              # index into a session-interned rid table
        token: int,                # from the prem dict (already CPU ints)
        signal_idx: int,           # interned edge-name index
        loop_key: tuple[int, ...], # wg_fwd_pass_idx + loop iters (SLIM_EMIT2 shape)
        flags: int,                # inline-qualifying | stop | final
    ]],
)
```

plus a rid-registration record on admission (rid string, edge-name template
material — the first full `ResultTensors` the sidecar must send so the
api_server can cache its template, `entrypoint.py:455-478`) and a
**boundary record** on wg completion carrying the worker-only WGD fields:
`completed_worker_graph_ids`, `is_first_tp_rank`, flushed persist signals,
`per_label_seq_info`, `partition_done`, `stream_tokens_consumed`,
`rx_info`/`tx_info` (`worker.py:1491-1509`). Boundary records are
~1/(177 steps)/rid — their construction cost is amortized noise.

### 4.3 Ownership moves (all-or-nothing per structure)

| State | Today | Stage 1 |
|---|---|---|
| `_slim_emit_sent`, `_slim_emit_loop_layout` (`worker.py` init; used 1337-1431) | worker | **sidecar** (it decides full-template vs slim per (rid,name)) |
| `pending_new_tokens`, `current_output_chunks`, `output_loop_indices` (`node_manager_utils.py:283-287`) | worker (`buffer_*`/`register_*`, flushed into WGD) | **sidecar** — the worker *never* writes them for sidecar-scoped walks; non-inline/boundary emit paths append record fields instead |
| WGD assembly + send (`worker.py:1475-1510`) | worker | **sidecar** (conductor tolerates late WGD: `conductor.py:849`) |
| inline-emit SHM-skip decision `_inline_emit_uuids` (`worker.py:1092-1136`) + `register_for_send` (`tensors.py:1310`) | worker | **worker** (tensor lifecycle; the record's `inline` flag is derived from it) |
| producer-side ref release (`worker.py:1446-1447`) | worker | **worker** |
| route/store/check_stop/stop_loops | worker | **worker** |

### 4.4 Honest ms accounting — what actually leaves

Skeptical dissection of the measured 3.14 ms send + adjacent shell, against
the FAST_SEND audit's cost ranking (EXPERIMENTS §"MSTAR_FAST_SEND built"):

| Item | ms est. | Leaves? |
|---|---|---|
| Per-rid emit construction: slim/loop-key logic, item dataclasses, `_inline_emit_uuids` send-side recompute, metadata dicts, per-rid manager-call overhead (`worker.py:1297-1440`) | 1.2–1.6 | **yes** |
| Batch message pickle + `send_pyobj` (1 fat msg/step → 1 compact msg/step) | 0.2–0.4 | **yes**, net of replacement |
| `buffer_new_tokens`/`buffer_output_signals`/`register_output_loop_indices` ownership (`worker.py:1282-1332`) | 0.2–0.3 | **yes** (whole-owner move) |
| WGD flushes + assembly, amortized (`worker.py:1475-1510`) | 0.1–0.2 | **yes** |
| `per_req_nested_idxs` fetch (`worker.py:2647-2651`) — replaced by loop-iter ints the worker already tracks | 0.1–0.2 | partly |
| to_workers / streaming / persist sends (steady decode: empty) | ~0 | n/a |
| Local-release `dereference` loop | 0.1–0.2 | **no** |
| `_register_outputs` ~1.0 (SHM decision, per-rid register ctx) | 1.0 | **no** (Stage-1 rider: re-land FAST_SEND's empty-register skip *inside* this change — its −3% GIL-economics objection is voided once the surrounding Python leaves) |
| route 1.82 / check_stop 1.1–2.1 / stop shell | 4–5 | **no** (Stages 2–3) |
| **Gross movable** | **1.7–2.5** | |
| Record construction + compact pickle (add-back) | 0.3–0.5 | |
| **Net main-thread removal** | **1.3–2.0** | |

Two calibration points, both measured on this exact path: SLIM_EMIT removed
~1.5–2 ms of pickle/construction and converted +15–25%; FAST_SEND removed
0.4 ms of interleaved lookups and converted **−3%**. The sidecar move is
SLIM-shaped (one contiguous block of construction + serialization leaves the
process, transport included), not FAST_SEND-shaped (micro-trims between
load-bearing waits). Predicted: **+10–20% e2e at i2t B32**, with the
downside case a wash, not a collapse — the fallback path is the byte-identical
legacy code behind the flag.

Secondary benefit, unpriced above: with ~2 ms less main-thread Python, the
gpu-thread valves change economics again — re-run the sampler-cache-style
re-tests after landing (the "trend toward crossover" was measured at
−7% → −1% → ~0% as the main thread lightened; EXPERIMENTS §"Stack test").

### 4.5 Rejected alternative shape: fold into the api_server

vLLM V1 exiles output processing into the *consumer* process, and M*'s
consumer already exists — extending `SlimResultTokens` into a per-step batch
record consumed directly by the api_server would avoid a new process. Rejected
for Stage 1 because (a) WGD targets the **conductor**, not the api_server —
the accumulator ownership has to leave the worker anyway, and parking half of
it in the api_server and half in the conductor doubles the split-brain
surface; (b) the api_server's message loop is already implicated in a
measured collapse under emit-path load (the SLIM_EMIT 5× incident was
diagnosed as downstream throttling in exactly this loop,
EXPERIMENTS §"MSTAR_SLIM_EMIT REGRESSED 5x"); adding per-token inflate work
there risks moving the wall rather than removing it. A dedicated sidecar sits
upstream of both consumers and keeps both hot loops untouched.

---

## 5. Q3 — the in-process alternative: a third postprocess thread

Move `_postprocess_batch` onto a third thread; main thread does only
scheduling/speculation/submit. **Rejected on GIL arithmetic:**

- The wall is Python *volume*, not dependency structure: main-thread GIL-held
  Python ≈ 8.7 ms/step while GPU needs ≈ 4.5 ms. A third thread does not
  shrink the 8.7 ms; CPython serializes it against the gpu-thread's
  ~1.4 ms (sample submit, `prepare/plan`) and the plan-executor's Python
  (`worker.py:3211,3234` — there are already 2–3 GIL threads). Total
  GIL demand per step is unchanged ⇒ wall unchanged, minus new
  context-switch losses. Measured priors all point the same way: E7's side
  thread collapsed B32 to 0.098× (eager + GIL contention); the GIL
  switch-interval sweep regressed −22% (W6); the sampler-cache saga proved
  the existing overlap *depends* on wait-shade, which a third compute
  thread consumes.
- The only way a third thread deepens the pipeline is if the main thread
  stops *waiting* for postprocess(N) before speculating N+2 — but then
  stops/readiness lag a step, which is exactly the Stage-2/V1 async-sched
  ordering problem **with none of the off-GIL benefit**. The pipeline is
  already 1-deep by speculation (`worker.py:1513-1526`); the second
  await→postprocess gap it would open is the same gap V1 opens, minus the
  GIL relief a real process boundary gives.
- Compare the measured valve structure: `await_gpu` (`worker.py:3611`) and
  the check_stop D2H wait already release the GIL and give the gpu/plan
  threads their shade. A postprocess thread would compete for precisely
  those windows.

Verdict: strictly dominated by the sidecar. Same correctness surface,
zero GIL-volume reduction. Do not build.

---

## 6. Q4 — ordering and correctness

### 6.1 Loop-back readiness does NOT wait on the sidecar (Stage-1 proof)

Everything step N+1 needs is produced worker-side before any record is sent:

1. **Spec path** (steady decode): `_thread_outputs_to_speculative`
   (`worker.py:2473`) splices batch_N's `NodeOutput` directly into the
   speculative batch after `await_gpu` — before `_postprocess_batch` runs at
   all (`worker.py:3611→3660→3736→3781`). The sidecar is not in this path.
2. **Non-spec fallback**: readiness comes from the route/store block —
   `mark_node_complete`/`Loop.complete_iter` re-injection ingested into
   `ready_node_names` (`worker.py:2842-2850`,
   `node_manager_utils.py:506-527`) — which **stays on the worker** (§3.2).
3. **Stops**: check_stop + `stop_loops` + `_pending_loop_stops` stay
   worker-side in Stage 1 (`worker.py:2693-2788`).
4. The record is a *copy* of ints and interned names. The sidecar produces
   nothing the scheduler, tensor_manager, or engine ever reads in Stage 1.
   The only feedback edge in the whole design is Stage 2's stop message,
   which is advisory-with-lag by construction (§6.2).

One-way data flow is the design's central safety property. Preserve it: any
future "small" read-back from the sidecar re-couples the wall.

### 6.2 Stage 2 — moving check_stop's decision to the sidecar

Mechanics: the sidecar already receives every sampled token int; EOS detection
(`token == im_end and not ignore_eos`, `worker.py:2715-2723`) is pure int
compares it can do for free. It returns `StopFeedback{rid, loop_names,
observed_step}` on the worker's existing message path
(`_process_messages`, drained at `worker.py:3368`), which converts it into
the same `stop_loops` + `_pending_loop_stops` calls.

What the worker keeps, non-negotiable:

- **The D2H itself stays.** The sidecar has no CUDA context; the pinned-buffer
  copy (`worker.py:3057-3066`) is the only way tokens become ints. What
  changes is the *wait*: instead of `side.synchronize()` on the critical
  path, the worker polls `event.query()` and consumes step N's buffer during
  iteration N+1 (double-buffer the pinned slab, `worker.py:2986-2999` already
  indexes buffers). Removes the 1.1–2.1 ms graph-tail wait from the wall.
- **max_tokens enforcement stays worker-side** — it is a pure counter
  (`dynamic_loop_iter_counts`, no D2H needed), so a stalled/dead sidecar can
  never cause unbounded generation. Only EOS detection lags.

Cost of the lag — extra tokens per request: stop decisions currently apply
one step late (speculation); Stage 2 adds the deferred-consume step plus
message latency ⇒ stops land **2–3 steps late** total. Per stopping request
that is 1–2 extra speculated tokens ≈ **0.6–1.1% wasted compute** at 177
tok/req — and *zero client-visible overrun*, because the sidecar knows the
stop the instant it computes it and simply does not emit post-stop tokens.
The waste is KV/compute only.

Correctness work item: the overstay machinery assumes stops are known exactly
one step late — `_pending_loop_stops` is cleared after a single iteration
(`worker.py:2645`) and matched against the immediately-following batch
(`worker.py:2607-2642`). Stage 2 must generalize it to a persistent
per-(rid, loop) stop set applied on arrival: discard the rid from the spec
chain, suppress routing of its overstayed outputs for the K in-flight steps,
invalidate its populate/route plans (same sites as today,
`worker.py:2637-2641`). This is the riskiest single edit in the plan; it
ships behind its own flag with an assert-mode that cross-checks against the
legacy worker-side check_stop running in shadow.

Honest expected value: the wait being removed is a **GIL valve** ("the new
GIL valve, mostly harmless" — EXPERIMENTS §"Final-stack decomposition"), so
by the valve law its removal converts only if the main thread is still the
wall after Stage 1. Predict +3–10%, wide error bars, genuine regression risk.
That is why it is Stage 2 with a hard gate, not part of the MVS.

### 6.3 Message-ordering invariants

- **Worker→sidecar**: single PUSH/PULL, FIFO per connection. Template-before-
  slim and token-order-per-rid follow from FIFO + the sidecar processing
  records in order. Never add a second worker→sidecar socket.
- **Sidecar→api_server**: the api_server routes each item independently under
  its lock and tolerates late items for completed rids
  (`entrypoint.py:412-417`), same as today.
- **Sidecar→conductor WGD vs worker-side conductor traffic**: the conductor
  ignores late WGD for completed requests (`conductor.py:849`) and processes
  WGD by rid (`conductor.py:1114-1128`); REMOVE_REQUEST races are already
  absorbed worker-side by deferred removes
  (`worker.py:3122-3149`). One new ordering rule: the worker must not send
  anything that implies partition completion before the sidecar's WGD for
  that partition can exist — audit the (rare) non-WGD conductor messages
  during implementation.
- **Flag flips (dynflags)**: template ownership means ON→OFF→ON flips must
  reset `_slim_emit_sent`-equivalent state on both sides. Cheapest safe rule:
  the sidecar flag is refreshable but a flip forces full-template re-emission
  for every live (rid, name) (the consumer handles duplicate templates fine —
  it just re-caches).

---

## 7. Q5 — crash, backpressure, lifecycle

**Sidecar death.** Worker checks `Process.is_alive()` on the dynflags cadence
(every 50 iters, `worker.py:3355-3358` — ~0.4 s at 8.8 ms/step) and treats a
ZMQ send failure/HWM as equivalent. On detection: (a) log CRITICAL and flip
to the legacy inline path (which is the retained flag-off code — byte-
identical behavior, zero new logic); (b) for rids with emit state stranded in
the dead sidecar (unsent tokens, unflushed WGD accumulators), the worker
cannot reconstruct — those in-flight requests are **failed fast** via the
existing remove/abort path rather than left to ride the api_server's 15 s
TTL. New requests proceed on the fallback path. No restart-and-resume in v1:
resuming means replaying accumulator state, which is the split-brain trap
again. A benchmark engine wants loud fail-fast here.

**Backpressure.** Sidecar per-step work is ~1–2 ms against a 7–9 ms producer
period — 4–7× headroom; the queue only grows if the sidecar degrades
(usually: api_server slow). Policy: bounded SNDHWM (~512 steps ≈ 4.5 s of
buffer), send with NOBLOCK. **The worker never blocks on the sidecar** —
blocking recreates the coupling this design exists to remove. On HWM breach:
treat as sidecar failure (drain-and-disable → permanent fallback until
restart), *not* per-step fallback — mixing paths step-by-step breaks the
per-(rid,name) FIFO the template protocol needs (§6.3). Counter
(`sidecar_hwm_trips`) rides WALK_STATS.

**Startup.** Spawn at worker init (spawn, not fork; no CUDA in the child;
keep `torch.cuda` uninitialized — records are plain ints + interned strings).
Import cost ~2–5 s, fully hidden behind model load/capture (~minutes).
RSS ~100–150 MB per sidecar, off-GPU. Pin to the worker's NUMA node on a
distinct core (composes with PLAN #14b). Shutdown: sidecar drains its queue
on SIGTERM with a 5 s deadline, then exits; worker waits at most that long.

---

## 8. Q6 — staged plan

**Stage 1 — emit + WGD sidecar** (`MSTAR_EMIT_SIDECAR`, default off)

- Worker keeps route/store/check_stop/register; sidecar owns emit
  construction, api_server transport, slim templates, WGD assembly +
  conductor transport, profiling accounting. Accumulator ownership moves
  wholesale (§4.3).
- Expected: net −1.3–2.0 ms main-thread ⇒ **+10–20% i2t B32** (discounted);
  riders: re-land the empty-register skip; re-test sampler-cache stacking.
- Risk: **medium** — split-brain WGD state (§0), FIFO/template protocol,
  fallback-path divergence.
- Size: 2–4 days.
- Validation recipe:
  1. CPU byte-identity: capture a legacy run's api_server + conductor message
     streams on a fixed trace; assert the sidecar build emits an identical
     stream (the FAST_SEND 5-test pattern, extended to WGD bodies).
  2. Token-identity smoke: tok/req **176.9–177.0 exact**, zero
     template-miss warnings (`entrypoint.py:360`), zero tracebacks.
  3. Mechanism-alive counters (no counter, no verdict): records/step,
     sidecar queue depth p50/p95, `sidecar_hwm_trips == 0`, WGD-from-sidecar
     count == legacy WGD count on the same workload.
  4. Adjacent A/B: dyn_ab where flippable (with template-reset-on-flip),
     else two-server ab_* alternation; ≥3 adjacent pairs, full n=96 cells,
     i2t:32 sentinel; promote at geomean ≥ +2%.
  5. Coverage guard: s2t B8 + i2s B8 cells to prove the walk-gating (E4b
     lesson) — audio paths must be exactly flat.

**Stage 2 — check_stop offload** (`MSTAR_SIDECAR_CHECKSTOP`, default off,
requires Stage 1 converted)

- Deferred-consume D2H (poll, no sync), EOS decision in sidecar,
  StopFeedback message, persistent overstay set; max_tokens stays worker-side.
- Expected: +3–10% (valve-law caveat: this removes a *wait*, so it converts
  only if the main thread is still the wall post-Stage-1 — re-profile first;
  if prof shows main thread < GPU time, skip to V1 instead).
- Risk: **high** — overstay generalization touches the spec chain's
  correctness core. Shadow-mode (legacy check_stop asserting against sidecar
  decisions, W2 pattern, zero mismatches over thousands of steps) is
  mandatory before any perf cell.
- Size: 2–3 days. Extra-token waste ≤1.1% by construction (§6.2); assert
  tok/req unchanged (client-visible stream must not grow).

**Stage 3 — route/store restructure: evaluate, expect NO**

- The only honest versions are (a) batched loop bookkeeping across rids
  (vectorize `mark_node_complete`/ingest over the batch — stays in-process,
  ~1 ms ceiling on a 1.82 ms block that W2 already showed overlaps GPU), or
  (b) the full EngineCore split (scheduler owns readiness, workers
  stateless) — a week+-scale rewrite of the scheduler contract
  (`get_next_batch`, spec chain, mixed-batch folds all read worker-local
  ready state). Decision gate: only if post-Stage-1/2 profiling shows route
  as the top remaining item AND the projected step time puts vLLM parity
  within reach of (a). Otherwise close the campaign here and move to the
  V1/GPU column.

---

## 9. Ranked verdict and NO-GO criteria

Ranking rationale (measured, not vibes):

1. **Sidecar Stage 1** — the wall is main-thread Python (8.74 ms ≈ step);
   this is the only option of the three that removes GIL-held work from the
   process. Precedent says contiguous emit-path removals convert
   (SLIM_EMIT), and the accounting clears >1 ms net.
2. **V1 full async-sched** — right idea, wrong order. With one GIL, pipeline
   depth cannot push throughput past (Python per step)⁻¹; prof_final's
   redirect stands ("V1 re-enters only after main-thread < GPU time").
   After Stages 1–2, if the wall flips to the submit/gpu side, V1 is the
   next build — and the sidecar makes V1 *easier* (less postprocess Python
   per deferred step to reason about).
3. **GPU-side kernel work** — at 51% GPU busy, kernel ms convert ≈0 e2e until
   the host floor drops. Keep #24 as an in-graph microbench curiosity only.

**NO-GO honesty check** (mandated): would Stage 1 remove <1 ms? The line-item
audit (§4.4) says no — the movable block is ≥1.3 ms net even with every
individual estimate at its pessimistic end, because it includes the whole
construction+serialization chain SLIM_EMIT already proved is real, converting
main-thread cost. So: **GO**. But the following outcomes flip it to NO-GO
retroactively, and the campaign then proceeds directly to V1:

- Stage-1 adjacent A/B geomean < +2% across 3 pairs (the effect-size gate),
  or any tok/req drift, or `sidecar_hwm_trips > 0` on healthy cells;
- shadow/byte-identity failures that require weakening the one-way data-flow
  invariant (§6.1) to fix — re-coupling is a design failure, not a bug to
  patch;
- post-Stage-1 profile shows the removed ms did not shorten the step (pure
  GIL-shade transfer — the FAST_SEND signature at block scale). In that
  world the two-thread GIL economics are stranger than both audits believe,
  and the honest next step is V1 async-sched plus a CPU-pinning pass (#14b),
  not more surgery on postprocess.

---

## 10. Open questions for implementation time

1. Exact inventory of worker→conductor messages besides WGD on the text
   walks (audit for the §6.3 ordering rule).
2. Whether `per_label_seq_info` / `rx_info` / `tx_info` are cheap enough to
   ship per boundary record or need interning (they are per-completion, so
   almost certainly fine).
3. Record schema versioning for the dynflags flip path (template re-emission
   handshake).
4. Whether the Stage-1 sidecar should also absorb the *non-inline* emit
   edges (prefill-boundary full ResultTensors). Default no — they carry
   GraphEdges with SHM tensor_info and are boundary-rate; shipping them to
   the sidecar just moves the same pickle one hop earlier.
5. NUMA/core placement interaction with #14b once both land.
