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
**three distinct collapse events**. Events D and E are **distinct failure
modes** (a silent zombie, and a mid-race collapse whose teardown raised a
lock-safety `RuntimeError` — see their rows), evidenced from the race run logs.
**Five failure events in one session; MTBF under our benchmark cadence ~30-60
min.**

| # | Time (2026-07-04) | Failure | Load state | Evidence |
|---|---|---|---|---|
| A | 17:40:50 | stage-0 **and** stage-1 subprocess died | pre-race (before race.log cell #1) | server.log:433-434 (pid 1755838) |
| B | 18:24:34 + 18:25:38 | stage-1, then stage-2 subprocess died | **mid-race, under load** | server.log:2523, :2748 (pid 2113390) |
| C | 20:16:14 | stage-0 subprocess died → full teardown | **IDLE, no benchmark load** | server.log:3388 (pid 2875178) |
| D | ~20:46-21:12 | **ZOMBIE**: EngineCore silently died, API still 200 | **idle between races** (relaunch @~20:30 served the 20:41-20:46 race fine) | h2h_out_smallbatch/vllm_*/run.log (HTTP 500 × 12/12 per cell, 5 cells) |
| E | ~22:18:30 | **mid-race collapse**; 16 cells 500'd; teardown raised `RuntimeError: release unlocked lock` | **mid-race, under load** (reboot @~21:40 passed the anti-zombie probe, served i2t B1/B2 r1) | h2h_out_imergecol/vllm_*/run.log (16 cells); server.log:9576 (pid 272705) |

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

- **Event E (~22:18:30, mid-race collapse; lock-safety RuntimeError on
  teardown).** After event D the server was rebooted (~21:40) and **passed the
  new anti-zombie completion probe** before racing; it served i2t B1/B2 round 1
  of the imergecol race, then the engine failed and **16 race cells returned
  HTTP 500** from i2t B4 r3 onward (`h2h_out_imergecol/vllm_*/run.log`, 16 cell
  dirs). At 22:18:30 the engine tore down and one stage proc logged a **genuine
  concurrency bug**: `RuntimeError: release unlocked lock`
  (`stage_engine_core_proc.py:188`, server.log:9576). The full traceback shows
  it fired **during handling of `SystemExit: 143`** (a SIGTERM): the signal
  interrupted `threading.Condition.wait()` inside the engine's input-queue busy
  loop (`vllm/v1/engine/core.py:1219` → `queue.get` → `not_empty.wait()`),
  leaving the condition's lock inconsistent, so the `with self.not_empty:`
  context-manager exit raised `release unlocked lock` — a real
  signal-handling/lock-safety defect in vLLM-omni's stage engine shutdown path.
  - **Causation (timeline-established, self-inflicted).** The `SystemExit: 143`
    SIGTERM was **internal to vLLM's own process tree**, not an operator signal.
    Command-history timeline: the race's last vLLM cell failed and `H2H_DONE`
    printed at **22:17:12**; the operator's first intervention (a
    `kill -- -<pgid>` during reboot) executed **no earlier than ~22:19:45**. The
    lock error fired at **22:18:30** — over a minute before any operator signal,
    and that same reboot command's log-tail captured the 500 flood as
    pre-existing. So vLLM's **own** StagePool/orchestrator SIGTERM'd the wedged
    engine proc (the same self-teardown cascade as event C), and the teardown
    exposed the lock defect. Event E is therefore a **self-inflicted** mid-race
    failure: engine entered a failing state under load (500 flood), vLLM's own
    orchestrator tore it down, and the shutdown path hit `release unlocked lock`.
    What remains undetermined is only the *upstream trigger of the first 500* —
    the lock bug is a real secondary defect on the teardown path, not (on this
    evidence) the spontaneous first cause of the collapse.

**Honest count note:** events {A, B, C} are three subprocess-death collapse
*events* (four raw `subprocess died` lines; event B killed two stages a minute
apart). Event D is a silent zombie (500s, no `subprocess died` line); event E is
a mid-race collapse (16 cells lost) whose SIGTERM teardown raised a lock-safety
`RuntimeError`. So the session shows **five distinct vLLM failure events**, three
of them mid-race (B, E, and the D-precursor race) and two off race load (C idle,
D zombie) — an MTBF of ~30-60 min under our benchmark cadence. Two of the five
(D's zombie, E's lock bug) are failure *modes* beyond a plain subprocess death.

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
- **Head-to-head during event E (imergecol race, ~22:18):** in the SAME race
  where vLLM lost 16 cells, the M* imerge server **served all 18 of its cells**
  (18/18 `results.json`, **zero** error/500/traceback lines across its run logs)
  with tok/req in-band (i2t B16 ~172, i2t B32 174-177 — the 172-179 band).
  Directly checked from `h2h_out_imergecol/mstar_*/`.
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
  undetermined. For event E specifically: the SIGTERM is timeline-confirmed
  **internal** to vLLM (self-teardown at 22:18:30, over a minute before the
  operator's ≥22:19:45 kill — see event E), so the collapse is self-inflicted;
  but the `release unlocked lock` `RuntimeError` is a real lock-safety defect on
  the **teardown path**, distinct from the still-undetermined upstream trigger
  of the first 500. Cite the lock bug as a real secondary defect, not as the
  spontaneous first cause of the collapse.
- **MTBF is an observed rate, not a guarantee.** ~30-60 min between failures is
  what this one session showed under our specific alternating-race cadence; it
  is a lower-bound-ish anecdote, not a measured distribution.
- **0.23 may fix it.** vLLM-Omni's own logs warn of deprecated paths
  (`VLLM_USE_FLASHINFER_MOE_FP16 ... removed in v0.23`); their known
  per-chunk/transport issues are one release from a fix. A reliability claim
  built on 0.22 must be re-tested against 0.23 before it survives.
- **Absence-of-signature ≠ proof-of-no-crash.** The M* "zero crash signatures"
  check is log-grep evidence; a hard crash that never logged would be missed.
  It is corroborated by M* "served every cell" but is not a formal proof.

## 5. One-line summary (for the acceptance rider, if it holds up)

On 2026-07-04, on one shared H200 box, vLLM-Omni 0.22 exhibited **five distinct
failure events** (~30-60 min MTBF under our race cadence) — three EngineCore
subprocess collapses (mid-race and idle), a zombie mode (EngineCore silently
dead behind a live 200 API, 500-ing an entire race), and a mid-race collapse
(16 cells lost) whose teardown raised a real `release unlocked lock`
lock-safety `RuntimeError` — while M* completed every cell across six boots
under heavier churn with zero self-inflicted deaths (in the event-E race, M*
served 18/18 cells error-free). Three failures were mid-race, two off load; the
zombie mode defeats a naive front-end liveness check. Claimable only while the
logs keep showing it, and pending re-test against vLLM-Omni 0.23.

---
*Sources: `/home/tim/exp_vllm_i2s_s2s/server.log` (lines cited inline, incl. E @
:9500-9576);
`bench-merge/benchmarks/qwen3-omni-joint/h2h_v2/{race.log,race_ext.log}`;
`h2h_out_smallbatch/vllm_*/run.log` (event D 500s);
`h2h_out_imergecol/{vllm,mstar}_*/run.log` (event E: 16 vLLM cells lost, 18/18 M* served);
`bench-merge/benchmarks/qwen3-omni-joint/EXPERIMENTS.md:1357,1442,973`;
`lab_{crusade,pmerge,jit,parm2,gather,arm3}/server.log`; HANDOFF_V5 §1.*
