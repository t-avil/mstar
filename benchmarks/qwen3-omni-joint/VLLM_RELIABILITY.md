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

All from `/home/tim/exp_vllm_i2s_s2s/server.log` (the i2s/s2s experiment
server), signature `[StageEngineCoreClient] stage-N [rep-0] subprocess died
unexpectedly (exit code None)` emitted by `stage_engine_core_client.py:265`.
The raw log contains **four** such subprocess-death lines across **three
distinct collapse events**:

| # | Time (2026-07-04) | Stage(s) died | Load state | APIServer pid | Evidence |
|---|---|---|---|---|---|
| A | 17:40:50 | stage-0 **and** stage-1 | pre-race (before race.log cell #1) | 1755838 | server.log:433-434 |
| B | 18:24:34 + 18:25:38 | stage-1, then stage-2 | **mid-race, under load** | 2113390 | server.log:2523, :2748 |
| C | 20:16:14 | stage-0 | **IDLE, no benchmark load** | 2875178 | server.log:3388 |

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

**Honest count note:** the "three deaths, third tonight" framing maps to
{A, B, C} as three collapse *events*; the raw log has *four* `subprocess died`
lines because event B killed two stages a minute apart. Either way the log
evidence is unambiguous: ≥3 unexpected vLLM subprocess collapses in one
2026-07-04 session, one of them while idle.

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

- **Four M* server boots tonight** — `lab_crusade`, `lab_pmerge`, `lab_jit`,
  `lab_parm2` (dirs under `/m-coriander/coriander/tim/`). Each `server.log` was
  grepped for crash signatures (`traceback|fatal|segfault|core dumped|CUDA
  error|died unexpectedly`): **zero matches in all four**.
- **The one M* process death tonight was intentional**, initiated by the
  operator via `lab_kill` (diag/pmerge). This is the load-bearing distinction:
  an operator teardown is not a self-inflicted crash. (Corroborated pattern:
  EXPERIMENTS.md:973 records earlier M* servers ending by *external kill*, never
  an unexpected subprocess death.)
- **"2 days continuous benching, zero self-inflicted deaths"** — this is a
  **handoff claim** (HANDOFF_V5 §1: "M* zero self-inflicted deaths in 2 days of
  continuous benching"). It is cited as-is; it was NOT independently
  re-verified end-to-end by this ledger, which only confirms the four boots
  tonight are crash-signature-clean. Treat the 2-day span as an operator
  assertion, tonight's four-boot cleanliness as directly checked.

## 4. Method & caveats (read before quoting any of this)

- **Same box, same window.** All events above are 2026-07-04 on the shared
  H200 box. M* and vLLM ran in the same period, so environmental factors
  (thermals, neighbors, driver) are shared, not a confound favoring either.
- **Asymmetric churn — favors the vLLM side, not ours.** M* was under *heavier*
  operational churn tonight (4 boots, continuous A/B cell fire, dynflags flips)
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

On 2026-07-04, on one shared H200 box, vLLM-Omni 0.22 logged ≥3 unexpected
EngineCore subprocess collapses (one mid-race with committed lost cells, one
while idle) while M* completed every cell across four boots under heavier churn
with zero self-inflicted deaths — claimable only while the logs keep showing it,
and pending re-test against vLLM-Omni 0.23.

---
*Sources: `/home/tim/exp_vllm_i2s_s2s/server.log` (lines cited inline);
`bench-merge/benchmarks/qwen3-omni-joint/h2h_v2/{race.log,race_ext.log}`;
`bench-merge/benchmarks/qwen3-omni-joint/EXPERIMENTS.md:1357,1442,973`;
`lab_{crusade,pmerge,jit,parm2}/server.log`; HANDOFF_V5 §1.*
