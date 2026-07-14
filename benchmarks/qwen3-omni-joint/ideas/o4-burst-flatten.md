# o4 — Burst flattening: MSTAR_BURST_CAP

Branch: `idea/o4-burst-flatten` (worktree `wt-o4-burst-flatten`), commit `a6f55ba1`.

## TL;DR

The i2t B32 TTFT-under-load problem is a host-CPU **oversubscription** burst,
not a per-op cost. **Nothing in the codebase ever calls
`torch.set_num_threads`**, so every spawned M* process defaults its torch
intra-op (OpenMP/MKL) pool to the machine's full physical-core count (128 on
the bench box). During a prefill/admission wave several processes each try to
fan a CPU op across dozens of cores at the same instant. On a quiet box the OS
soaks it; under same-priority neighbor load the threads schedule stochastically
and the wave stalls → bimodal multi-second TTFT. `MSTAR_BURST_CAP` (default
off, byte-identical off) caps each process to a small, fixed thread budget so
our per-step host demand becomes small and constant — the vLLM property.

## Burst inventory (core deliverable: fan-out sites, file:line, est. cost)

Process topology at i2t (single-GPU): **api_server** (uvicorn + the serial
preprocess worker thread), **conductor** (spawned), **1 Worker/rank** (spawned),
plus optional **detok** / **sidecar** children. All are `mp.get_context("spawn")`
children — each a fresh interpreter that re-imports torch.

| # | Fan-out site | file:line | Mechanism | Est. core-seconds per B32 wave |
|---|---|---|---|---|
| 1 | **torch default intra-op pool, EVERY process** | (absence of any `set_num_threads` — grep is empty) | torch sizes the OMP/MKL pool to physical cores (128 here) per process; a CPU op in any process fans across all of them | **Dominant.** Each CPU op in preprocess/worker momentarily demands up to 128 threads; ≥3 processes → up to ~3×128 thread-demand, thrashes under contention. This is the 21-core burst’s amplifier. |
| 2 | Image decode + resize (preprocess) | `model/base.py:379` `load_image` (torchvision `decode_image`, `.float()/255`); `model/bagel/components/modeling_utils.py:243,280,304` `F.resize(BICUBIC, antialias)` | torchvision/torch CPU ops → intra-op pool (site #1) | High: BICUBIC antialias resize of a wave of images; each resize alone can light every core. |
| 3 | HF `AutoProcessor` feature-extract | `model/qwen3_omni/qwen3_omni_model.py:2316` `process_prompt` (image_processor / feature_extractor); numpy round-trip `:2376` | numpy/torch CPU → OMP/MKL pool | High during the wave; the CPU heavy part of preprocess. |
| 4 | Tokenization / chat-template | `process_prompt` (HF tokenizer) | mostly single-thread; small | Low. |
| 5 | SHM tensor store/copies | `data_worker.py:429` `store_and_return_tensor_info`; transport pools `communication/tensors.py:228` (`max_workers=3`), `engine/kv_store.py:319` | bounded (3) transport threads + torch copies | Bounded already; **not** a fan-out source. |
| 6 | Worker host-side prep | `worker/worker.py` plan/reshape/sample around the GPU forward | torch CPU ops → intra-op pool (site #1) | Medium; GPU forward itself is unaffected by CPU thread count. |
| 7 | Worker GPU/plan/side executors | `worker/worker.py:4199,4222,4249` | **`max_workers=1`** each | None — not a fan-out. |

**Key finding — the preprocess wave is already staggered.** The preprocess
worker (`data_worker.py:1022` `run`) pulls **one request per loop iteration**
(serial), so a 32-request wave is naturally amortized across iterations, not
spiked all-at-once. So the spike is **not** request concurrency — it is the
per-op thread fan-out (site #1). That means the right lever is a thread cap, and
adding request-level chunking/staggering would be redundant complexity. I
deliberately did **not** add it.

**Key finding — the real vulnerability is oversubscription, not total work.**
The measured burst is ~21 cores (2126% CPU), well under 128. The problem is
that this demand is spread as *unbounded, all-core* fan-out across multiple
processes; under neighbor load at equal priority those threads thrash. Capping
each process to a fixed small pool keeps the same real work but makes the
demand bounded and predictable.

## What is capped, where

`mstar/utils/burst_cap.py` — `apply_process_thread_cap(role)`: when
`MSTAR_BURST_CAP=1`, sets `torch.set_num_threads(n)` (intra-op — the big one),
attempts `torch.set_num_interop_threads(n)` (guarded; can only be set before
inter-op work), and exports `OMP_NUM_THREADS` / `MKL_NUM_THREADS` /
`OPENBLAS_NUM_THREADS` / `NUMEXPR_NUM_THREADS` so native pools and any
subprocess inherit the cap. Returns `None` and touches nothing when off.

Wired at all five process entry points (boot-time, each re-reads env so per-role
overrides apply):
- api_server: `api_server/entrypoint.py` `main()` (role `api_server`) — before the
  conductor is spawned, so env propagates to children too.
- conductor: `api_server/entrypoint.py` `_conductor_process_target` (role `conductor`).
- worker: `conductor/conductor.py` `_worker_process_target` (role `worker`) — before
  `Worker` init; caps CPU threads only, **GPU forward untouched**.
- detok child: `api_server/detok_proc.py` (role `detok`).
- sidecar child: `worker/emit_sidecar.py` (role `sidecar`).

Transport pools (`tensors.py`, `kv_store.py`, `max_workers=3`) left **uncapped
on purpose** — clamping 3→budget-8 is a no-op and would only risk narrowing
transport concurrency. `capped_workers(default, role)` helper exists for any
future explicit pool (only ever narrows, never widens; no-op when off).

## Flags

- `MSTAR_BURST_CAP` — `0`/unset (default) = off, **byte-identical** to today
  (no `set_num_threads`, no env writes). `1` = on.
- `MSTAR_BURST_THREADS` — per-process thread budget (default `8`).
- `MSTAR_BURST_THREADS_<ROLE>` — per-role override, ROLE ∈
  `{API_SERVER, CONDUCTOR, WORKER, DETOK, SIDECAR}` (e.g.
  `MSTAR_BURST_THREADS_WORKER=16`). Intended discipline: the per-process caps
  **sum** to the host-CPU budget the whole engine should occupy in a wave (e.g.
  api_server 8 + conductor 4 + worker 8 ≈ 20, replacing unbounded 3×128).
- All **boot-time** (thread-pool size is a process property; does NOT follow
  `MSTAR_DYNFLAGS`). A/B via two boots.

## Design tension / risks

- **Quiet-box slowdown (expected, by design):** fewer threads for the resize /
  feature-extract → a single request’s preprocess is slower on an idle box. The
  win is variance (p95/p99) under load, not p50 on a quiet box. Tune
  `MSTAR_BURST_THREADS` to trade the two. Report BOTH.
- Too-low a cap could bottleneck the preprocess enough to hurt even loaded
  throughput; sweep 4/8/16.
- `set_num_interop_threads` can throw if the inter-op pool is already fixed —
  guarded (degrades to a debug log), so it can’t crash a process.
- Worker cap is CPU-only; if any host-side worker step is genuinely
  parallel-CPU-bound it slows in isolation — but that is exactly the burst we
  want bounded.

## A/B recipe (paired, at HIGH load; p95 TTFT is the metric)

Two boots, same commit, i2t B32 food101 closed-loop, n≥96, **paired under
neighbor load** (the contention lottery makes single loaded cells lie —
LEARNINGS_FIX20). Report p50 AND p95/p99 TTFT plus rps/tok-s:
- Boot A (baseline): `MSTAR_BURST_CAP=0`.
- Boot B (capped): `MSTAR_BURST_CAP=1 MSTAR_BURST_THREADS=8`.
- Sweep B over `MSTAR_BURST_THREADS ∈ {4,8,16}` to find the knee.
- Also run one quiet-box (load <25) pair to quantify the isolation cost so the
  robustness win is stated against a known quiet-box price.
- Expected: p95/p99 TTFT tightens materially under load (fewer multi-second
  stalls); p50 flat-to-slightly-worse on a quiet box.

## Validation done (no GPU / no server)

`test/test_burst_cap.py` (all pass, each case in a fresh subprocess):
off = untouched (returns None, threads unchanged); on = intra-op threads +
OMP/MKL env == budget; per-role override wins; `capped_workers` only narrows;
**preprocess BICUBIC-antialias resize is byte-identical with cap off vs on**
(the load-bearing correctness guarantee — capping cores changes speed, never
result). `python -c "import mstar"` OK; `compileall` OK on all touched files.
