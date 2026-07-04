# vLLM-Omni reliability ledger

Evidence for the acceptance-protocol reliability rider (GOAL Tier S4: "zero M*
self-inflicted crashes during acceptance"). This is a factual log of what the
committed artifacts and server logs show on **one box, one window
(2026-07-04)**. It is NOT a marketing claim: the vLLM deaths are not
root-caused by us, and vLLM-Omni 0.23 may fix them (see Method & caveats).
Assembled by the code agent from read-only log forensics; timestamps are the
log-local clock (server logs stamp `MM-DD HH:MM:SS`, treated here as the box's
wall clock for the 2026-07-04 session).

## 1. vLLM-Omni 0.22 crash events (2026-07-04)

Events A-C are all from `/home/tim/exp_vllm_i2s_s2s/server.log` (the i2s/s2s
experiment server), signature `[StageEngineCoreClient] stage-N [rep-0]
subprocess died unexpectedly (exit code None)` emitted by
`stage_engine_core_client.py:265` — **four** such subprocess-death lines across
**three distinct collapse events**. Event D is a **fourth, distinct failure
mode** (silent engine death behind a live API — see its row), evidenced from a
different artifact (the p2verify small-batch race run logs):

| # | Time (2026-07-04) | Failure | Load state | Evidence |
|---|---|---|---|---|
| A | 17:40:50 | stage-0 **and** stage-1 subprocess died | pre-race (before race.log cell #1) | server.log:433-434 (pid 1755838) |
| B | 18:24:34 + 18:25:38 | stage-1, then stage-2 subprocess died | **mid-race, under load** | server.log:2523, :2748 (pid 2113390) |
| C | 20:16:14 | stage-0 subprocess died → full teardown | **IDLE, no benchmark load** | server.log:3388 (pid 2875178) |
| D | ~20:46-21:12 | **ZOMBIE**: EngineCore silently died, API still 200 | **idle between races** (relaunch @~20:30 served the 20:41-20:46 race fine) | h2h_out_smallbatch/vllm_*/run.log (HTTP 500 × 12/12 per cell, 5 cells) |

Detail per event:

- **Event A (17:40:50, pre-race).** Two stages (0 and 1) died together before
  the head-to-head race began (race.log cell #1 ran ~18:00 and the race
  completed 18:11:57). Cause is **not established from the log** — it predates
  the race and could be a genuine early collapse or a setup-phase restart.
  Recorded here for completeness; not claimed as a load-induced crash.
  Plausibly the "1st collapse today" implied by EXPERIMENTS.md's "2nd collapse"
  wording (§2).

- **Event B (18:24:34 → 18:25:38, mid-race, under load).** stage-1 died at
  18:24:34, stage-2 at 18:25:38, during the extended head-to-head race. This is
  corroborated by **committed** race output:
  `bench-merge/benchmarks/qwen3-omni-joint/h2h_v2/race_ext.log` reads
  `H2H vllm i2t B32 #6 req/s 0.000 tok/s 0.0` and
  `#7 req/s 0.000 tok/s 0.0` — while the adjacent M* cells the same rounds read
  `#6 7.599` and `#7 7.395` (M* healthy). Round #5 immediately prior had vLLM
  healthy at `8.243`, so the collapse is bracketed to between rounds #5 and #6.
  race.log cells #1-4 (18:00-18:11:57) show vLLM healthy (8.463 / 8.406 …), so
  the death is specifically mid-race, not a cold server.

- **Event C (20:16:14, IDLE).** stage-0 died with no benchmark running; the
  death immediately cascaded into a full engine teardown:
  server.log:3393-3399 shows `[AsyncOmni] Shutting down` (20:16:15) →
  `[Orchestrator] Received shutdown signal` → all 3 stage replicas shut down by
  20:16:20. This is the cleanest self-inflicted signature: an idle server lost a
  subprocess and tore itself down.

- **Event D (~20:46-21:12, ZOMBIE — distinct failure mode).** After event C the
  vLLM server was relaunched (~20:30) and served the p2verify race cleanly from
  20:41-20:46. Its EngineCore then died **silently** sometime between 20:46 and
  21:12 while the front-end API server kept answering `/v1/models` with HTTP 200
  — a zombie state. It was discovered only when the 21:12 small-batch race hit
  the engine: **every request of every cell returned HTTP 500** `{"error":
  {"message":"EngineCore encountered an issue. See stack trace (above) for the
  root cause.","type":"InternalServerError","code":500}}` — 12/12 requests
  across all 5 cells (`h2h_out_smallbatch/vllm_{i2t_B1,i2t_B2,i2t_B4,s2t_B2,
  s2t_B4}/run.log`). The structured 500 body (not connection-refused) confirms
  the API layer was alive while the engine was dead. This is materially worse
  than A-C for an acceptance harness: **a liveness check on `/v1/models`
  (or any front-end endpoint) passes while the server cannot actually serve.**
  - *Ops consequence (operator, noted here for the record):* the race reboot
    procedure now requires an **engine-level liveness probe — a real
    completion** — before racing, since front-end 200s are not sufficient to
    prove the engine is up.

**Honest count note:** events {A, B, C} are three subprocess-death collapse
*events* (four raw `subprocess died` lines; event B killed two stages a minute
apart). Event D is a **fourth, different** failure — a silent EngineCore death
behind a live API, evidenced by 500s in the race logs rather than a
`subprocess died` line. So the session shows **four distinct vLLM failures**,
two of them (C, D) while not under race load, and D specifically defeats a
naive front-end liveness check.

## 2. Earlier documented vLLM deaths (EXPERIMENTS.md)

`bench-merge/benchmarks/qwen3-omni-joint/EXPERIMENTS.md`:
- Line 1357: "**RELIABILITY: vLLM DIED MID-RACE (2nd collapse today; pairs 6-7
  lost)**; M* served every cell all day, zero self-inflicted deaths." — the
  contemporaneous note for event B, and its "2nd collapse today" wording
  independently implies an earlier collapse the same day (consistent with event
  A).
- Line 1442: the acceptance-gate validation record explicitly rejects race
  rounds r6/r7 as "vLLM crashes" and notes the gate "catches vLLM's two
  mid-race deaths," pooling the clean cells to the honest 0.883 B32 read. So the
  mid-race deaths are already baked into the committed methodology as
  crash-rejections, not silently dropped.

No vLLM crash events earlier than 2026-07-04 were found in EXPERIMENTS.md.
(Other "collapse" hits in that file are M* *performance* collapses — e.g. the
E7 side-thread 0.098× at line 102 — or M* servers killed **externally**
mid-cell at line 973 "the sweep (8299) and speech (8311) servers died mid-cells
(external kill)", i.e. operator-initiated, not self-inflicted. They are not
vLLM deaths and are excluded.)

## 3. M* side of the ledger (same box, same window)

- **Six M* server boots tonight (as of 21:35)** — `lab_crusade`, `lab_pmerge`,
  `lab_jit`, `lab_parm2`, `lab_gather`, `lab_arm3` (dirs under
  `/m-coriander/coriander/tim/`). Each `server.log` was grepped for crash
  signatures (`traceback|fatal|segfault|core dumped|CUDA error|died
  unexpectedly`): **zero matches in all six** (directly checked).
- **Every M* process death tonight was intentional**, initiated by the operator
  via `lab_kill` (e.g. diag/pmerge). This is the load-bearing distinction: an
  operator teardown is not a self-inflicted crash. (Corroborated pattern:
  EXPERIMENTS.md:973 records earlier M* servers ending by *external kill*, never
  an unexpected subprocess death.)
- **"2 days continuous benching, zero self-inflicted deaths"** — this is a
  **handoff claim** (HANDOFF_V5 §1: "M* zero self-inflicted deaths in 2 days of
  continuous benching"). It is cited as-is; it was NOT independently
  re-verified end-to-end by this ledger, which only confirms the six boots
  tonight are crash-signature-clean. Treat the 2-day span as an operator
  assertion, tonight's six-boot cleanliness as directly checked.

## 4. Method & caveats (read before quoting any of this)

- **Same box, same window.** All events above are 2026-07-04 on the shared
  H200 box. M* and vLLM ran in the same period, so environmental factors
  (thermals, neighbors, driver) are shared, not a confound favoring either.
- **Asymmetric churn — favors the vLLM side, not ours.** M* was under *heavier*
  operational churn tonight (6 boots, continuous A/B cell fire, dynflags flips)
  than vLLM, yet showed zero unexpected deaths. The comparison is not "idle M*
  vs stressed vLLM"; if anything the stress asymmetry runs against M*.
- **Not root-caused.** We did not diagnose *why* vLLM's EngineCore subprocesses
  die (exit code None = no clean exit status captured). We are not asserting a
  code defect, only logging observed behavior. Event A's cause is explicitly
  undetermined.
- **0.23 may fix it.** vLLM-Omni's own logs warn of deprecated paths
  (`VLLM_USE_FLASHINFER_MOE_FP16 ... removed in v0.23`); their known
  per-chunk/transport issues are one release from a fix. A reliability claim
  built on 0.22 must be re-tested against 0.23 before it survives.
- **Absence-of-signature ≠ proof-of-no-crash.** The M* "zero crash signatures"
  check is log-grep evidence; a hard crash that never logged would be missed.
  It is corroborated by M* "served every cell" but is not a formal proof.

## 5. One-line summary (for the acceptance rider, if it holds up)

On 2026-07-04, on one shared H200 box, vLLM-Omni 0.22 exhibited **four distinct
failures** — three EngineCore subprocess collapses (one mid-race with committed
lost cells, one while idle) plus a zombie mode (EngineCore silently dead behind
a live API, 500-ing an entire race) — while M* completed every cell across six
boots under heavier churn with zero self-inflicted deaths. Two of the four vLLM
failures occurred off race load, and the zombie mode defeats a naive front-end
liveness check. Claimable only while the logs keep showing it, and pending
re-test against vLLM-Omni 0.23.

---
*Sources: `/home/tim/exp_vllm_i2s_s2s/server.log` (lines cited inline);
`bench-merge/benchmarks/qwen3-omni-joint/h2h_v2/{race.log,race_ext.log}`;
`h2h_out_smallbatch/vllm_*/run.log` (event D 500s);
`bench-merge/benchmarks/qwen3-omni-joint/EXPERIMENTS.md:1357,1442,973`;
`lab_{crusade,pmerge,jit,parm2,gather,arm3}/server.log`; HANDOFF_V5 §1.*
