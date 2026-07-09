# REVIEW V9 — Why M* loses to vLLM-Omni 0.22 on text paths, and how to fix it

**Date:** 2026-07-06. **Method:** four independent audit agents (inventory / vLLM-research /
code / propose), each instructed to **distrust every comment, markdown doc, and commit message**
in this repo and verify against raw JSON, source code, and external release notes only. Every
claim below is tagged with its evidence. Where a prior doc (HANDOFF_V8, EXPERIMENTS.md, MEMORY)
disagrees with the raw data, the raw data wins and the doc is flagged as wrong.

Booted/audited commit: **`opt/prep-h2d` @ 620de91** (the tree `launch_mstar_best.sh` boots from
`mstar-b1fix/`), with the exact flag stack the launcher sets.

---

## 0. TL;DR — the three-sentence version

1. We didn't get slower — **vLLM got faster**: core **v0.22 shipped Model Runner V2** (GPU-native
   input prep, *zero CPU↔GPU sync in the decode loop*, Triton sampler), on by default for Qwen3.
   That erased exactly the CPU/GIL floor M* has been fighting.
2. Our own "20/24 win" scoreboard was measured against **broken vLLM 0.21** (`raw_vllm_021.json`,
   i2t B32 = **2.55** req/s); against healthy **0.22** (band **8.03–8.59**) the honest raw data says
   we **lose i2t B2 (0.97×) and B32 (0.93×), tie B1/B16, win B4/B8** — a 3–7% loss, *not* "lose by a
   lot," and s2t is not actually losing in any committed file.
3. The "winning build won't boot" blocker was **not** a loader race — it was **289 GB of our own
   orphaned `/dev/shm` segments** (477k files from dead servers) starving CUDA context creation.
   Cleaned; host free 46 GB → 343 GB; boots again. **No node reboot needed.**

---

## 1. Root cause analysis

### 1a. External driver — vLLM Model Runner V2 (the thing that beat us)
`RESEARCH`, HIGH confidence. vLLM-Omni's minor version is pinned to vLLM core, so 0.21→0.22 rode
**core** 0.21→0.22. The text-decode wins are in core, not the omni plugin (the two "Qwen3-Omni perf"
omni commits touch ~16/~24 lines of encoder/TTS, not the decode loop).

- **Model Runner V2 (MRV2)** — ground-up execution-core rewrite: GPU-native input prep via Triton
  (`input_ids`/`positions`/`query_start_loc`/`seq_lens` built on-device), persistent-batch state
  table gathered per step, **zero CPU↔GPU sync across the decode loop**, Triton Gumbel-Max sampler.
  **Default for Qwen3 dense in core v0.22** (PR #39337). Blog: +56% throughput, −6.3% TPOT.
  Source: https://vllm.ai/blog/2026-03-24-mrv2 , https://docs.vllm.ai/en/v0.22.1/design/model_runner_v2/
- Reinforcements 0.22→0.24: GPU↔CPU **sync-elimination series** (PR #41429/#42347, v0.22),
  **FlashInfer/Triton fused sampler** (v0.23), **FULL_AND_PIECEWISE CUDA-graph dispatch** for
  uniform-decode vs mixed batches (#42304 v0.22, #44050 v0.23), **FP8/MoE kernel** upgrades
  (SM90 CUTLASS `swap_ab` +180–290%; Qwen3-Next `fused_moe` +25%; v0.24).
- **Do NOT re-attempt naive async scheduling.** vLLM-Omni issue #4442 shows async-sched alone does
  *not* overlap decode launches — the same failure mode our `MSTAR_ASYNC_SCHED` hit. MRV2's win is
  the *zero-sync state design*, not deferral. Track vLLM-Omni **RFC #1770** for when the omni
  *Thinker text runner* itself flips to MRV2.

### 1b. Internal — three code-identifiable structural gaps (why our text path is CPU-bound)
`CODE` + `PROPOSE`, verified in source at the booted commit:
1. **Per-step Python/GIL floor (~26 ms at B32).** The micro-scheduler rescans all queues per
   request per node every step; prepare/postprocess run per-rid Python loops that scale with batch.
   GPU sits ~50% idle. (This is exactly what MRV2 removed on the vLLM side.)
2. **Prefill is serialized with decode.** The scheduler enforces **one graph_walk per batch**
   (`micro_scheduler.py:97-104`); encoders are standalone `Sequential([encoder, thinker])` nodes.
   A new i2t/s2t request injects an encoder+prefill mega-step that **freezes all in-flight decodes**
   → ITL spikes + TTFT tax. vLLM 0.22 mixes prefill+decode in one step.
3. **MoE is bf16.** The fp8/int8 grouped-GEMM + DeepGEMM/TMA paths are **stripped**
   (`utils/fused_moe/kernels.py:4-7`); the 30B-A3B decode is MoE-GEMM bound.

---

## 2. TEN ISSUES with the current implementation (distrust-verified)

Mix of **code defects** (hurt latency/throughput) and **methodology defects** (made us believe we
were winning when we weren't). Both are "mistakes we can fix."

### Code defects (verified in source at 620de91, with the shipped flags ON)
1. **Batched pos-ids do a pageable, synchronous H2D every decode step at B>1.**
   `submodules.py:609` `pos = torch.tensor(starts, dtype=torch.float).to(device, non_blocking=True)`
   — source is unpinned so `non_blocking` is silently ignored → blocking H2D on the GPU thread every
   step, in fp32. `MSTAR_PREP_DEVICE_POS` (the win this branch is *named* for) only patched the
   **per-request B1** path (`submodules.py:654-668`, gated `len>1` at `kv_cache_engine.py:844`). **The
   entire throughput regime — including the losing cells B2 and B32 — never got the fix.** *(This is
   the single most on-target, lowest-risk lever for the cells we lose. It is the centerpiece of the
   god branch below.)*
2. **`forward_batched` packs `thinker_states` even for text-only (i2t/s2t) requests.**
   `submodules.py:1882-1897` unconditionally `torch.cat([layer_0_embed, layer_n_hidden])` every
   batched decode step, filtering per-rid audio only afterward. The sequential path *does* gate this
   (`:1359`); the batched path (the throughput regime) does not → wasted `(bs, 2·hidden)` cat/step.
3. **Two unconditional `hidden_states.clone()` in the Thinker forward** feed #2:
   `thinker.py:222` (`layer_0_embed`) and `:249` (`layer_n_hidden`), baked into every captured
   text-decode/prefill graph, not gated on talker use → wasted clone kernels + memory for text-only.
4. **Encoder is an exclusive GPU step that stalls decode** (structural, §1b.2):
   `qwen3_omni_model.py:745-752,805-872`. `CHUNKED_PREFILL_V2`+`MIXED_BATCH` fold only *Thinker*
   prefill chunks, never the encoder node; `MSTAR_SIDE_PREFILL` is **OFF** in the booted config.
5. **Encoder/vision-prefill host syncs on the TTFT critical path.**
   `vision_encoder.py:274` `int(max())`, `:278` `cu_seqlens.tolist()`; audio `.max().item()`/
   `.nonzero()`/`.tolist()` (`audio_encoder.py:233,241,235,57…`); Thinker vision pos-id build
   `.item()`+python grid loop (`submodules.py:920,933-945`); `rope.py:333-335` 3×`.item()`/image.
   Each serializes encode→prefill before the first token.
6. **Per-step full scheduler rescan on the GIL** (`micro_scheduler.py:645-689`) — O(reqs×nodes)
   Python between GPU steps; the backoff at `worker.py:4262-4307` exists *because* these are
   "thousands of guaranteed-negative scans." Half the B32 CPU floor.
7. **Main-thread hard sync before postprocess.** `worker.py:3193`
   `output.completion_event.synchronize()` blocks the scheduler on step N even though N+1 is in
   flight; plus `future.result()` (`:4448`) and a `wait(0.005)` (`:4583`). Caps postproc↔GPU overlap.
8. **CUDA-graph batch-size padding waste** (suspected, needs measurement): decode buckets
   `[1,2,4,8,16,24,28,32]` pad up by cloning real rows (`cuda_graph_runner.py:1621-1625`); bs 17→24,
   29→32 ⇒ up to ~30% padded compute for in-between sizes.

### Methodology / data-integrity defects (why the scoreboard lied to us) — `INVENTORY`
9. **We averaged in vLLM crash runs and a mislabeled `system` field.** Every results.json has
   `system:"ours"` even for vLLM runs — only `inference_system` distinguishes engines; any
   `system`-keyed aggregation silently merges vLLM into M*. And vLLM crash runs (`thr=0.0,
   completed=0/96`, e.g. `h2h_flagship_stack/vllm_i2t_B32_{3,4,5}`) are committed next to good runs;
   averaging them drops vLLM's mean 8.3→5.6 and fabricates a fake "1.43× win." **Cause of the
   optimism.**
10. **We quoted 2-decimal ratios inside 20–29% run-to-run noise, and config-shopped the winners.**
    Genuine warmed variance: i2t B32 repeats span **7.21–8.67 (20%)**, i2t B2 span **1.22–1.58
    (29%)**. `sweep_mstar_v5_verified/i2t/B4` commits **2.565** while its own five repeats max at
    **2.341** (the committed number beats every real run). The i2t B2 "win" only appears when quoting
    the low-request-count `ab_b2check` variant (1.85) instead of the honest median (~1.46 vs vLLM
    ~1.50). **HANDOFF_V8 §2 calls B1/B2/B16 green wins; raw ratios are 0.98/0.97/1.00.** Sibling
    `VLLM_022_GRID.md` disagrees with HANDOFF on the same cells; neither matches raw. **`V2 budget
    policy` is documented as a "WIN" in EXPERIMENTS.md but the committed A/B (`w5_mixed_datapoints/
    ab_p2/`) regresses every cell −4…−9%.**

**Bonus operational mistake (the boot blocker):** 289 GB of orphaned `/dev/shm` from dead M*
servers (SHM tensor-comm segments never reaped) starved host RAM → `torch.cuda.set_device` OOM at
`worker.py:448`, misread as a loader race. See §5.

---

## 3. TEN EASY boosts (localized, low-risk)

| # | Lever | Targets | Evidence / status |
|---|-------|---------|-------------------|
| E1 | **Pinned+reusable batched pos-ids buffer** (fix issue #1) — extend PREP_DEVICE_POS to `prepare_inputs_batched`; write `starts` into a pinned CPU buffer, `non_blocking` copy into a reusable GPU buffer; drop fp32→int. | itl, tput @ B2/B32 | **New, on-target for losing cells.** god-branch centerpiece. |
| E2 | **Gate the text-only `thinker_states` cat** (issue #2) + the two clones (#3). | itl, tput | New, localized. |
| E3 | **fp8 grouped-GEMM MoE** — re-add the stripped w8a8 path. | itl, tput | `exp/fp8-quant`,`opt/decode-v2 MSTAR_MOE_FP8`; MEMORY: **1.13× e2e validated**. Land it. |
| E4 | **Tune decode buckets** (add 24/28 already present; verify no pad at B17-32). | tput | `opt/decode-v2`; unverified — measure. |
| E5 | **Async encoder dispatch, vision-only** `MSTAR_ENCODER_ASYNC`. | ttft (i2t) | **Measured:** i2t B32 +7.4% req/s, −30% TTFT; but s2t −18% → **gate vision-heavy B≥16 only**. |
| E6 | **Denser prefill token buckets** (384/768/1536). | ttft | `exp/prefill-buckets`; unverified. |
| E7 | **Move encoder rope/grid metadata on-device** (kill issue #5 syncs); cache `cu_seqlens` per resolution. | ttft | New, localized. |
| E8 | **Drop the completion_event sync from the critical path** (issue #7) — rely on the deferred-copy event SIDECAR_CHECKSTOP already lays down. | itl | Localized but needs care. |
| E9 | **Prune unbounded scheduler dicts** (`node_and_walk_to_last_batch_num` never pruned, `held_until` rebuilt) — issue #10-code. | tput (slow leak) | Trivial cleanup. |
| E10 | **CDT / inductor cache env** (already in launcher) — keep, but honest number is **+1.0%** not +3.4% (`ab_cdt_b32`). | tput | Already on; don't overclaim. |

## 4. FIVE HARD things (structural — where the fight is actually won vs 0.22)

| # | Lever | Targets | Status |
|---|-------|---------|--------|
| H1 | **Mixed prefill+decode step** — one varlen forward batching a chunked prefill with in-flight decodes; needs a `thinker_mixed` walk + relaxing the one-walk-per-batch rule. **#1 structural gap vs 0.22.** | ttft, itl, tput | `opt/mixed-walk`, `exp/mixed-batch-p2` — **code-only, UNMEASURED**; prior naive fold −5%. |
| H2 | **CUDA-graph the mixed step** (FULL_AND_PIECEWISE-style dispatch). H1 eager reintroduces the launch floor; capture is what makes it pay. Do H1→H2 as a pair. | itl, tput | `exp/mixed-cg-*` branches are **empty scaffolding** — genuinely new. |
| H3 | **Speculative decode / MTP** — Qwen3 ships an MTP head; amortizes the 26 ms Python step over k tokens, biggest win at low/mid batch where we're CPU-floor bound. | itl, tput | `exp/spec-decode-mtp` — design only, **no impl**. Largest *new* algorithmic lever. |
| H4 | **Persistent/incremental scheduler ready-set** (kill issue #6) — vLLM keeps a persistent running batch; we rebuild it every step. This is our version of MRV2's zero-sync state table. | itl, tput | New; the principled fix for the CPU floor (not naive async). |
| H5 | **PD-disaggregation at batch** (`qwen3omni_pd_disaggregated.yaml`) — prefill on its own worker stops contending decode SMs. | ttft, tput | Regresses at B1 (findings) — retest **B≥16 under load**, not B1. |

---

## 5. Boot blocker — real root cause and fix (no reboot)
**Symptom:** best build printed its serving banner then `worker_1 failed to initialize: CUDA error:
out of memory` at `torch.cuda.set_device(self.device)` (`worker.py:448`) on an *empty* 143 GB GPU;
3/3 (now 4/4) failures; handoff blamed a "rank-placement race" and floated a node reboot.

**Actual cause:** host-RAM exhaustion. `torch.cuda.set_device` creates the CUDA primary context
(needs pinned host memory). The launcher pins `numactl --membind=1` — and because GPUs 6,7's sysfs
`numa_node` reads empty, the launcher's fallback forces node **1**, which was down to **14.6 GB
free**. Meanwhile tim owned **477,095 orphaned `/dev/shm` segments = 289 GB** (torch_*/cuda.shm.*/
chatcmpl lockfiles from hundreds of dead SHM-protocol servers), host free 46 GB, **swap 100 % full**.
Context alloc on the starved node → OOM.

**Fix applied:** reaped tim-owned `/dev/shm` (other users untouched) + stale `sk_lab_*` sockets.
Result: `/dev/shm` 295 GB → **3.1 GB**, host free 46 GB → **343 GB**, NUMA node 1 free 14.6 GB →
**243 GB**. Build boots again.

**Prevent recurrence:** (a) the server cleanup trap must reap its own `/dev/shm` segments on exit;
(b) fix the launcher NUMA fallback — when a GPU's `numa_node` is unknown, bind to the node with the
most free memory (or don't `membind` at all), never hardcode 1; (c) periodic `/dev/shm` sweep of
tim-owned dead segments.

---

## 6. Recommended sequence (fast path to beating 0.22)
1. **E1** (batched pinned pos-ids) — on-target for the cells we actually lose (B2/B32); measure A/B
   on one warm server, interleaved, to beat the 20–29% noise. *(god branch, done first.)*
2. **E3** (fp8 MoE, 1.13× already validated) + **E2** (text-only cat/clone gating).
3. **H1→H2** (mixed step, then graph it) — the structural fix for prefill serialization, the real
   gap vs 0.22.
4. **H4** (persistent ready-set) + **H3** (MTP spec decode) — attack the CPU floor the principled
   way MRV2 did; measure everything *in-CUDA-graph* (out-of-graph A/Bs lie).
5. **E5** (vision-gated async encoder) + **H5** (PD-disagg at batch).

**Measurement discipline going forward:** filter `inference_system=="ours"` (never `system`); drop
`thr<=0`/partial runs; quote medians of ≥5 warmed repeats with the spread, never a single
config-shopped 2-decimal ratio; A/B on one warm server (dynflags), not across boots.
