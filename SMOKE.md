# SMOKE — MSTAR_ASYNC_SCHED (V1 async scheduling / GPU-resident sampled ids)

CODE landed on `opt/async-sched` (worktree `/m-coriander/coriander/tim/mstar-v1async`).
This file is the GPU-side validation the code session could NOT run (no GPU).
Do this on the project GPU set only when the box is idle (`nvidia-smi`); GPUs
6,7 are the canonical pair.

## What the flag does (one-line)

Defers each decode step's postprocess (check_stop / route+store / emit / WGD) by
one loop iteration so it consumes tokens whose D→H already finished — the
blocking `completion_event`/prematerialize waits leave the critical path. Stop
detection lags, so a completing request runs up to TWO extra (wasted) decode
steps; those overrun tokens are TRIMMED before emit, so the client stream must
be byte-identical to baseline.

## Preconditions / gotchas

- **Requires `MSTAR_DIRECT_FEED=1`.** With it off the worker logs CRITICAL and
  disables async (stays synchronous). Always set both.
- **NOT dynflags-refreshable.** `lab_ab.sh` flips flags via `dynflags.json`,
  which async ignores (read once at init). **A/B MUST be two servers** (flag in
  the STATIC flag string), never a dyn_ab flip. Putting `MSTAR_ASYNC_SCHED` in a
  lab_ab FA/FB JSON measures nothing.
- Default OFF and byte-identical when off (every added path is gated).

## Canonical winning-stack flag string (base for both legs)

```
WINSTACK="MSTAR_MOE_FP8=1 MSTAR_BATCH_EMIT=1 MSTAR_FAST_POSTPROC=1 MSTAR_CHUNKED_PREFILL_V2=1 MSTAR_CHUNKED_PREFILL_V2_VISION=1 MSTAR_MIXED_BATCH=1 MSTAR_MIXED_BATCH_VISION=1 MSTAR_MIXED_SPEC=1 MSTAR_PREFILL_CHUNK_TOKENS=256 MSTAR_SLIM_EMIT=1 MSTAR_SLIM_EMIT2=1 MSTAR_FAST_ROUTE=1 MSTAR_FAST_ROUTE2=1 MSTAR_SAMPLER_CFG_CACHE=1 MSTAR_FAST_CHECKSTOP=1 MSTAR_FAST_SEND=1"
```

(To smoke the intended "post-sidecar" regime, append `MSTAR_EMIT_SIDECAR=1` —
it self-requires SLIM_EMIT2, already present. async+sidecar both-on is untested
on GPU but should be order-preserving: the sidecar step record just emits one
iteration later. Isolate V1 first WITHOUT the sidecar, then stack it.)

## 1. Correctness smoke (single server, one i2t:8 cell)

Boot ON and run one small closed-loop cell:

```bash
cd /m-coriander/coriander/tim
./lab_server.sh v1async_on /m-coriander/coriander/tim/mstar-v1async \
  "$WINSTACK MSTAR_DIRECT_FEED=1 MSTAR_ASYNC_SCHED=1 MSTAR_WALK_STATS=1" \
  configs/qwen3omni_2gpu_encoff.yaml 6,7 8299

# after "WARM+READY on port 8299":
CVENV=/home/tim/mstar-encoders/.venv ; BENCH=/m-coriander/coriander/tim/bench-v2
OUT=/m-coriander/coriander/tim/lab_v1async_on/smoke_i2t8 ; mkdir -p "$OUT"
PYTHONPATH=$BENCH HF_HOME=/m-coriander/coriander/hf timeout 1200 "$CVENV/bin/python" -m benchmark.runner \
  --url http://127.0.0.1:8299 --model qwen3omni --request-type image_to_text \
  --dataset food101 --profiling-type closed_loop --max-concurrency 8 \
  --num-requests 96 --num-warmup 2 --inference-system ours \
  --local-cache /m-coriander/coriander/hf --output-dir "$OUT" | tail -20
```

PASS criteria — all must hold:

- **tok/req ≈ 177** (the i2t decode length; a shift means the stream is not
  byte-identical — trimming is wrong). Check:
  ```bash
  python -c "import json;d=json.load(open('$OUT/results.json'));print('tok/req', (d.get('total_output_tokens') or d.get('output_tokens_total'))/d['num_requests'])"
  ```
  If those keys differ, eyeball `completion`/token counts in results.json; the
  number must match an OFF run (section 3) to the token.
- **Zero errors / tracebacks:**
  ```bash
  grep -iE 'error|traceback|exception|dropped rid|CRITICAL' /m-coriander/coriander/tim/lab_v1async_on/server.log | grep -v 'Deferring cleanup' | head
  ```
  (expect nothing; "Deferring cleanup of tensor" is a normal SHM ACK log)
- **New WALK_STATS counters present and sane** (worker logs them at WARNING
  every 200 steps):
  ```bash
  grep -oE "WALK_STATS step=[0-9]+ .*" /m-coriander/coriander/tim/lab_v1async_on/server.log | tail -1
  ```
  Expect in the printed list:
  - `async_sched_steps` > 0 (the deferral fired — this is the mechanism proof;
    ~= number of speculative decode steps),
  - `late_stop_trims` present (>= 0; nonzero once requests complete — the extra
    overrun tokens being dropped),
  - `async_d2h_wait_us` present and SMALL relative to step time (the residual
    completion-event wait; near-zero confirms the block moved off the critical
    path).
- Confirm async actually engaged (not silently disabled): the CRITICAL
  "requires MSTAR_DIRECT_FEED" line must be ABSENT from server.log.

Kill: `./lab_kill.sh v1async_on` (or by the boot pgid).

## 2. Byte-identity of the emitted stream (the load-bearing correctness gate)

Async must not change a single emitted token vs OFF. Run the SAME cell against
an OFF server and an ON server (same seed/dataset/order) and diff the per-request
completions:

```bash
# OFF server on 4,5:8298, ON server on 6,7:8299 (both from this worktree)
./lab_server.sh v1async_off /m-coriander/coriander/tim/mstar-v1async \
  "$WINSTACK MSTAR_DIRECT_FEED=1 MSTAR_WALK_STATS=1" \
  configs/qwen3omni_2gpu_encoff.yaml 4,5 8298
# ... run the identical benchmark.runner cell against :8298 into smoke_off/
# then compare the ordered completion texts / token-id lists request-by-request:
python -c "
import json
a=json.load(open('/m-coriander/coriander/tim/lab_v1async_off/smoke_i2t8/results.json'))
b=json.load(open('/m-coriander/coriander/tim/lab_v1async_on/smoke_i2t8/results.json'))
ka=[r.get('generated_text') for r in a['per_request']]; kb=[r.get('generated_text') for r in b['per_request']]
print('identical' if ka==kb else 'MISMATCH', 'n=',len(ka))
"
```
(Adjust the results.json field names to whatever the runner emits — the point is
a request-by-request token/text equality check. Any mismatch = STOP, the trim
logic is dropping/keeping a wrong token.)

## 3. A/B throughput (two servers, i2t B32 — the target metric)

Not dynflags-flippable → two servers, identical flags except the two async vars.
For comparability prefer the SAME GPU pair back-to-back (kill/boot) or two pairs
concurrently accepting the ~+15% cross-NUMA caveat on 0,1 (6,7 vs 4,5 are both
valid; DYN/LAB_NUMA_NODE is derived from the PCI bus). Run a discarded warm cell
first (server maturity spans ~100 reqs).

```bash
# A = async OFF (baseline):  $WINSTACK MSTAR_DIRECT_FEED=1 MSTAR_WALK_STATS=1
# B = async ON:              $WINSTACK MSTAR_DIRECT_FEED=1 MSTAR_ASYNC_SCHED=1 MSTAR_WALK_STATS=1
# For each: boot, discard one i2t:32 warm cell, then 3 rounds of i2t:32 (n=96).
```

Report per-round req/s A vs B (i2t B32), plus B1 (`i2t:1`). Expected per the
design record: **+0–5%** now (SIDECAR_DESIGN §0 ranked V1 "real value
post-sidecar"; the win only converts once main-thread Python is below GPU time —
so also read `async_d2h_wait_us` and the step cadence to see whether the wait
actually collapsed). tok/req must stay 177 on BOTH sides.

## 4. What "broken" looks like (stop here and report)

- tok/req drifts from 177, or the section-2 diff MISMATCHes → trim is wrong.
- `dropped rid` warnings, or garbage/repeated tokens → the 2-step-old GPU token
  tensor was overwritten before its deferred postprocess read it (the held-clone
  lifetime assumption — see the risk list in the handoff message).
- req/s craters (not just flat) → the chain-break flush is firing every step
  (deferral never engaging); check `async_sched_steps` is a large fraction of
  total decode steps.
- CRITICAL "requires MSTAR_DIRECT_FEED" in server.log → you forgot DIRECT_FEED.
