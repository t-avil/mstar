# M* i2t B32 TTFT regression: prefill/encode path trace (encoders 4c33b33 → HEAD 1e7f10da)

Repo: `/m-coriander/coriander/tim/mstar-godv9`, branch `opt/prep-pos-batched-v9`.
Range: `4c33b33` (encoders-era, 2026-06-30) → `1e7f10da` (HEAD, 2026-07-12), 110 commits.
Observation: i2t B32 batch TTFT ~576ms → ~2300ms+, **with all `MSTAR_*` flags off** (lean A/B).

## Headline result (surprising)

Four independent traces (schedule construction, scheduler/spec serialization,
host-syncs, git archaeology) all converge on the same conclusion: **the default
(all-`MSTAR_*`-off) admission → encode → prefill → first-token path is
byte-identical between the two commits**, with exactly two exceptions, and both
exceptions are flags that **default ON** — i.e. "turning the flags off" (unsetting
`MSTAR_*`) does NOT disable them. So the lean A/B's premise ("regression persists
with flags off") does not exclude them.

- **`MSTAR_CONDUCTOR_POLL`** — default `"1"` (ON). `conductor.py:57-59`.
- **`MSTAR_FUSED_TOPK`** — default `"1"` (ON). `moe.py:61-64`.

Every other change in the window (merged prefill, chunked-prefill-v2 (+vision),
mixed-batch/mixed-spec, sched-pack, side-prefill, encoder-async, custom-ops,
MoE-fp8, batch-vision-prefill) is behind an `MSTAR_*` flag that defaults **OFF**,
each documented and code-verified as byte-identical when off. These are the
branch's *fix attempts*, consistent with the regression surviving flags-off.

Concretely nothing new was added to the default path in these areas:
- **Schedule** (`qwen3_omni_model.py`): `_build_thinker_prefill_schedule` still emits
  `[prefill_text, prefill_vision] → thinker_decode`, same order/splicing.
  `_maybe_merge_prefill_schedule` (L1454) and `_append_vision_schedule` (L1534)
  are no-ops with `MSTAR_MERGED_PREFILL` / `MSTAR_CHUNKED_PREFILL_V2_VISION` off.
- **Scheduler** (`micro_scheduler.py:741 get_next_batch`): same round-robin, a
  new prefill walk still wins RR the instant it appears. Spec system
  (`MAX_CONSECUTIVE_SPEC_STEPS`=1024, `SPEC_PEEK_FOR_FAIRNESS`=1) is unchanged and
  **predates** 4c33b33 — the "spec is new" premise is false for this window. A new
  prefill waits ~1–2 decode steps (~9–18ms), identical in both versions.
- **Host-syncs** (`worker.py`, `cuda_graph_runner.py`, `cache_manager.py`,
  `submodules.py`): no new `.item()/.cpu()/synchronize()/wait_event()` on the
  flags-off prefill path. The per-walk `completion_event.synchronize()`
  (worker.py:3333, old:1660) and check_stop `side.synchronize()` (worker.py:3819,
  old:1928) run unconditionally in BOTH versions, byte-identical mechanism.

So **code archaeology does not, by itself, explain a 4× TTFT blowup on the default
path.** That negative result is itself the most important finding: the next step
is empirical (A/B + profile), not more reading — and the two default-ON flags plus
config/measurement provenance are where to look.

## Ranked 5 most likely regression mechanisms

### 1. `MSTAR_CONDUCTOR_POLL` (default ON) × the pre-existing 32 serialized bs=1 vision-prefill walks
The **only structural default-path change** to the conductor's per-hop wait.
- `conductor.py:1162-1165`: replaces the old unconditional `time.sleep(0.001)` per
  loop with `self.communicator.wait_for_work(timeout_ms=50)` (blocking ZMQ poll).
- Commit trail (`git log -S wait_for_work 4c33b33..HEAD`):
  - `314b5a25` (07-03) introduce, default ON.
  - `4a19bbb2` (07-03) **revert default to OFF** — first smoke *wedged, zero
    requests completed*; message: *"some conductor work source doesn't wake the
    poller ... loop work triggered by internal state rather than an inbound message."*
  - `b1c1ff18` (07-03) fixes only the crash (unguarded `self.event.fd` in
    `communicator.wait_for_work`; conductor registers no EventWakeup) and **flips
    default back ON "pending re-smoke."** No re-smoke lands in the remaining commits.
- **Magnitude fit**: i2t B32 issues **32 sequential single-request `prefill_vision`
  walks** (`assert len(inputs)==1` submodules.py:1387, pre-existing). If any
  conductor hop misses its wakeup and eats the 50ms timeout, the tax stacks across
  32 walks: 32×~50ms ≈ 1600ms — which matches the 576→2300 delta almost exactly.
- **Caveat (why it's not proven)**: my own read of the loop shows the state machine
  is message-driven — `WORKER_GRAPHS_DONE` (conductor.py:1114) is processed and
  `_process_done_forward`→`_send_partition_inputs` (L974) sends the next walk *in
  the same iteration*, and that message wakes the poller. So on paper the 50ms poll
  should be harmless. I could not confirm a residual wakeup-source gap after the
  crash fix. **This is the #1 suspect on priors (only per-hop default change +
  magnitude potential + documented history of wedging), but it is unverified.**
- **Test first**: A/B i2t B32 `MSTAR_CONDUCTOR_POLL=0` vs default. If TTFT drops to
  ~576ms, confirmed.

### 2. The "flags off" A/B never actually disabled the default-ON flags
Reframes the hunt. Unsetting `MSTAR_*` leaves `MSTAR_CONDUCTOR_POLL` and
`MSTAR_FUSED_TOPK` ON. So the regression can still be a default-ON flag introduced
this window — the A/B did not rule that out. Re-run the "encoders vs modern" A/B
with `MSTAR_CONDUCTOR_POLL=0 MSTAR_FUSED_TOPK=0` explicitly set on the modern side.

### 3. Not code — config / measurement provenance / boot variance
- `6c70f758` adds `configs/qwen3omni_2gpu_pd.yaml` (prefill/decode disaggregation)
  as a **new opt-in file**; no existing `.py`/yaml changed (`git diff 4c33b33..HEAD
  -- configs/qwen3omni*.yaml` is empty), selected only via `--config`. If the bench
  pointed at it, prefill/decode split across 2 GPUs could serialize TTFT. Confirm
  which config the 576 vs 2300 numbers used.
- Memory notes an ~18% boot lottery / cross-run variance on this stack. The two
  numbers may not be apples-to-apples (different boot, build, or config). Verify
  both came from the same config on the same GPUs.

### 4. `MSTAR_FUSED_TOPK` (default ON, `856d103b`)
Fused `topk_softmax` in every MoE router forward (`moe.py:105`). ~microseconds/layer
— magnitude far too small to explain seconds; also silently no-ops if `sgl_kernel`
isn't importable. Low prior; cheap to rule out (`MSTAR_FUSED_TOPK=0`).

### 5. Pre-existing amplifier: 32 serialized bs=1 vision-prefill walks (not itself new)
`prefill_vision` has always been bs=1 (assert submodules.py:1387, unchanged since
before 4c33b33), so B32 runs 32 sequential vision-encoder+prefill walks, each
paying its own `completion_event.synchronize()` + check_stop `side.synchronize()`.
This is the amplifier that turns any small per-walk overhead increase (e.g. #1)
into seconds. Intended fix `MSTAR_BATCH_VISION_PREFILL` exists but defaults OFF.

## Recommended next actions
1. A/B `MSTAR_CONDUCTOR_POLL=0` at i2t B32 (settles #1/#2, the cheapest decisive test).
2. Confirm which config each of 576ms / 2300ms was measured on (settles #3).
3. If both come back clean, the regression is not on the default *code* path —
   profile a live i2t B32 run (py-spy / NVTX around the conductor loop,
   `wait_for_work`, and the 32 `prefill_vision` walk handoffs) rather than more
   reading.

## Note
One sub-agent reported seeing `<system-reminder>`-style text appear inside command
stdout (a date-change notice, task-tool nudges). These match the harness's own
top-level reminders and are almost certainly harness-injected context being
misattributed to tool output, not repo-based prompt injection — but worth a glance
if reproduced.
