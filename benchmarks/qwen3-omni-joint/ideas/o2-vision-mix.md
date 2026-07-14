# o2 — vision-mix at boot (G5 unlock): IMA-safe capture-provisioning latch

Worktree/branch: `/m-coriander/coriander/tim/wt-o2-vision-mix` @ `idea/o2-vision-mix`,
commit `c3ea8856`. Base: fix20 tip `1c4c25c3`. No GPU touched, no server started.
Flag: **`MSTAR_MIXED_BATCH_VISION`** (boot-time). Default off = byte-identical.

## TL;DR

The W5-P3-lite vision-mix machinery (flag helper, capture provisioning, scheduler
gates, deepstack preprocess) was **already implemented in the fix20 base** — G5 is
mechanically built. What was NOT safe is the exact thing the task flags: routing a
`prefill_vision` chunk into the captured `thinker_mixed` step was gated on a **live**
`mixed_batch_vision_enabled()` env read on both the capture side and the scheduler
side. Because `MSTAR_DYNFLAGS` mutates `os.environ` mid-run for one-server A/Bs, a
**boot-OFF → runtime-ON flip** would make the scheduler route a vision chunk into a
text-signature mixed graph that lacks the per-layer deepstack static buffers = the
UNCAP-IMA failure. This change closes that hole with a capture-provisioning latch
(the established `_split_attn_env_snapshot` pattern), so the boot capture's ground
truth — not the live flag — decides routing. ~157 LoC across 4 files.

## (1) G5 mechanics: what "deepstack statics baked at boot" means, in code

The `thinker_mixed` step is a **captured CUDA graph**. Its packed static input
buffers and their keys (`static_input_keys`) are fixed at capture time and cannot
grow afterwards. A `prefill_vision` chunk row carries, on top of the text signature
(`input_embeds`/`cos_3d`/`sin_3d`), a **per-layer `deepstack_<i>`** tensor and a
custom-MRoPE `mrope_pos_advance` side-channel; the Thinker forward splices deepstack
additively at `vision.deepstack_visual_indexes` layers (`[8,16,24]`).

- `_build_prefill_vision_packed` (submodules.py:1663–1712): builds the packed dict
  with one `deepstack_<i>` buffer per index — the extra keys vs the text builder.
- `_build_prefill_text_packed` (submodules.py:1625–1661): text signature only.
- `get_cuda_graph_configs` (submodules.py:1901–1960): the `thinker_mixed` capture
  chooses its packed builder by `mixed_batch_vision_enabled()` —
  `_build_prefill_vision_packed` (adds `deepstack_<i>`) when the vision flag is on,
  else `_build_prefill_text_packed`. `static_input_keys` are frozen from these keys
  at capture (cuda_graph_runner.py:730,805). A text-signature mixed capture has **no
  `deepstack_<i>` static buffer**, so replaying a vision chunk there gives
  `preprocess` nowhere to copy deepstack into.
- Per-request dispatch inside the mixed step: `prepare_inputs` sees the batch walk
  `thinker_mixed` and re-dispatches on each row's real walk (submodules.py:696–697);
  a vision chunk row routes to `_vision_chunk_inputs` (submodules.py:806) which emits
  `deepstack_<i>` tensor_inputs + `mrope_pos_advance` (submodules.py:910–917).
- Why a dynflag is impossible: `static_input_keys` is baked once at capture; you
  cannot add deepstack buffers to an already-captured graph. Enabling vision folding
  after a text-only boot would route to a graph with a missing static key. Same class
  of hazard the runner already guards for split-attn bucket selection
  (`_config_uses_split_attn` / `_split_attn_env_snapshot`, cuda_graph_runner.py:397–419).

Scheduler side (single chokepoint): `_mixed_chunk_walks()` (micro_scheduler.py:374)
is the only place `prefill_vision` is added to the mixable-chunk set; it is shared by
the peek (`has_mixed_opportunity`), the assembler (`_try_assemble_mixed`), and the
mid-chain pop (`pop_mixed_chunk_for_spec`), so all three agree.

## (2)+(3) What this change adds

`MSTAR_MIXED_BATCH_VISION=1` (boot) already provisions the mixed graphs with the
deepstack static buffers (the `_build_prefill_vision_packed` branch above). The new
work is the **IMA-safe runtime check** so routing follows what was *actually*
captured:

- `qwen3_omni_model.py`: process-global `_MIXED_VISION_PROVISIONED` (None until
  capture runs) with `mark_mixed_vision_provisioned(bool)` and
  `mixed_vision_capture_provisioned() -> bool`. After boot the recorded truth is
  authoritative (runtime flag flips ignored); before any capture (None) it falls
  back to the live flag so CPU tests / boot ordering behave as written — production
  routing only happens post-capture, where the record wins.
- `submodules.get_cuda_graph_configs`: records the ONE capture's truth at boot
  (`mark_mixed_vision_provisioned(mixed_batch_enabled() and mixed_batch_vision_enabled())`).
- `micro_scheduler._mixed_chunk_walks`: admits `prefill_vision` iff
  `mixed_vision_capture_provisioned()` **and** the live flag. The provisioning check
  is the hard IMA gate (a runtime ON-flip after a vision-off boot can never route to
  the unprovisioned graph); the live flag is ANDed only to honor a **safe-direction**
  runtime OFF-flip (stop routing even though the graph could still replay it), which
  dyn_ab A/Bs use.

Fails closed in every ambiguous case: no capture record + no flag → text-only;
booted off → vision never routed regardless of later env.

## (4) Extra capture memory

Incremental cost of vision-mix vs the text-only mixed capture = the added
`deepstack_<i>` static input buffers on the 3 mixed buckets (`BS+C`, C∈{256,288,512}
→ 288/320/544 tokens), 3 deepstack layers, hidden=2048, bf16:

| bucket (tok) | 3 × (tok×2048) × 2B |
|---|---|
| 288 | 3.375 MiB |
| 320 | 3.750 MiB |
| 544 | 6.375 MiB |
| **total** | **~13.5 MiB** |

Negligible: the existing standalone `prefill_vision` captures already hold the same
deepstack buffers at up to 16384 tokens (192 MiB for that one bucket alone), and the
Thinker is a 30B-A3B model. The captured graph gains 3 elementwise deepstack-add
kernels per bucket (no workspace). No decode/text-mixed memory change.

## Design / risks

- **Byte-identical when off**: `MSTAR_MIXED_BATCH_VISION` off → `mark(False)` →
  `_mixed_chunk_walks` returns `{prefill_text}` (P2), and the mixed capture uses the
  text builder — identical captures and gates to base. Verified reading the off-path.
- **Process co-location**: capture (`get_cuda_graph_configs`) and the mixed-assembly
  scheduler both live in the GPU worker process (the scheduler queries the runner's
  captured graphs), so the module global is shared. If ever split, the record reads
  None in a non-capturing process → fail-closed (never routes vision).
- **Capture OOM edge**: recording at declaration is IMA-safe — if the whole
  `thinker_mixed` capture fails, no graph exists and replay lookup returns None (a
  routing miss handled by the existing path), never a missing-key IMA. This edge is
  identical for the text mixed batch and not vision-specific.
- **Target scope**: a small image (258 vision tokens → C=288 bucket) can now
  co-admit into a decode step like vLLM. Large images/video (>512) overflow the
  captured chunk cap (G1) — that's o1's eager-fold territory, out of scope here.

## A/B recipe (i2t B32 TTFT, COADMIT + vision-mix)

Boot the server with vision-mix capable (boot-time flag, cannot be enabled by
dynflag):

```
MSTAR_MIXED_BATCH=1 MSTAR_MIXED_SPEC=1 MSTAR_CHUNKED_PREFILL_V2=1 \
MSTAR_CHUNKED_PREFILL_V2_VISION=1 MSTAR_MIXED_BATCH_VISION=1 \
MSTAR_COADMIT=1  <boot server>
```

Paired, load-gated (host loadavg <25), n≥96 at i2t B32 on small-image food101
(258-tok spans → C=288 bucket). Cells (same boot; MSTAR_MIXED_BATCH_VISION honored
in the safe-off direction via dynflags for the A):

1. **A (vision-mix off)**: `MSTAR_MIXED_BATCH_VISION=0` — vision prefill runs
   standalone (the encode+prefill step that freezes decodes); COADMIT can fold only
   the ≤512 text portion.
2. **B (vision-mix on)**: `MSTAR_MIXED_BATCH_VISION=1` — the 258-tok vision chunk
   co-admits into the decode step in ONE captured mixed step.

Metrics: i2t B32 TTFT p50/p99 and req/s. Watch WALK_STATS `_coadmit_fold` (arrivals
that co-admitted) and the mixed-vs-standalone split. Expected win: removes the
per-request standalone vision-prefill step for small images at B32, folding it into a
live decode step — the G5 lever COADMIT documented but could not pull without a
vision-capable boot. Bounded (small images only; large/video still overflow G1), and
does not touch the host-CPU burst (#2/#3/#14 territory), so it is one component of
the i2t TTFT gap, not full parity by itself.

## Files touched (my scope)

- `mstar/model/qwen3_omni/qwen3_omni_model.py` (+48): provisioning latch
  (`_MIXED_VISION_PROVISIONED`, `mark_mixed_vision_provisioned`,
  `mixed_vision_capture_provisioned`).
- `mstar/model/qwen3_omni/submodules.py` (+14): record capture truth in
  `get_cuda_graph_configs`.
- `mstar/worker/micro_scheduler.py` (+19): `_mixed_chunk_walks` gates on
  provisioning + safe-direction live flag.
- `test/modular/test_qwen3_omni_mixed_batch_vision.py` (+81): reset the latch in the
  fixture; 4 new tests (record authority, pre-capture fallback, boot-off/runtime-on
  IMA scenario, boot-on with safe OFF-flip). 21/21 pass.

## Validation (no GPU, no server)

- `compileall` of the 3 source files → OK.
- `PYTHONPATH=<worktree> <mstar-new venv>/python -c "import mstar; ..."` → latch and
  routing chokepoint behave: record False + flag on → `prefill_vision` NOT in
  `_mixed_chunk_walks()`; record True → it is.
- `pytest test/modular/test_qwen3_omni_mixed_batch_vision.py` → **21 passed**.
</content>
