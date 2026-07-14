# o5 — MSTAR_PREPROC_PROC: off-process request-side multimodal preprocessing

**Flag:** `MSTAR_PREPROC_PROC` (boot-time; default off = byte-identical inline path).
Tuning: `MSTAR_PREPROC_PROCS` (pool size, default 4), `MSTAR_PREPROC_THREADS`
(intra-op threads per child, default 0 = auto = `cpu_count // procs`).
**Branch/worktree:** `idea/o5-api-split` @ `/m-coriander/coriander/tim/wt-o5-api-split`.
**Composes with** `MSTAR_DETOK_PROC` (both children coexist; independent directions).

## The bigger split #13 pointed at

Fix #13 (`MSTAR_DETOK_PROC`) moved only the *output*-side `tokenizer.decode` off
the serve process. Everything else the uvicorn process does still shares one GIL.
The largest remaining cost is the *input*-side multimodal preprocessing.

## uvicorn-process inventory (what shares the serve GIL today)

Everything below runs inside the single uvicorn/serve OS process:

- **uvicorn event loop** (main thread, async): HTTP parse + SSE/NDJSON streaming
  — `entrypoint.py` `generate` (`:742`), `iter_result_chunks` (`:560`),
  `async_stream_results` (`:617`), `_chunk_to_ndjson` (`:622`).
- **Upload persistence**: `await f.read()` + `run_in_threadpool(write_bytes)`
  (`entrypoint.py:787-788`) — already off the loop, on the threadpool.
- **`APIServer._process_messages`** thread (`entrypoint.py:432`): drains ZMQ
  results from the conductor and appends chunks to each pending request. This is
  the thread whose GIL starvation delays token emission for in-flight decodes.
- **`PreprocessWorkerThread.run`** thread (`data_worker.py`): the data worker.
  - **Request preprocessing — `_process_input` (`data_worker.py:371`)**, the heavy
    part:
    - `model.load_image(filepath, "cpu")` (`:389`) — image decode.
    - `model.process_prompt(...)` (`:412`) — **image resize + normalize +
      patchify + tokenize**. For Qwen3-Omni this is `_gpu_image_preprocess`
      (`qwen3_omni_model.py:590`, default-on `MSTAR_GPU_IMAGE_PREPROCESS=1`) which,
      because the API-server model's device is `cpu`, runs the torchvision
      bicubic-resize/patchify port **on CPU** — our note put this at the ~175 ms
      I2T TTFT cost, and it fans out across ~21 cores via torch's intra-op pool.
    - `load_audio` / `load_video` for those modalities.
  - Output-side: detok/emit ordering, tensor reads (unchanged here).

**Root cause this shaves:** a burst of 32 image requests runs all 32 preprocesses
**serially on the ONE data-worker thread**, holding the GIL the uvicorn loop and
the ZMQ-drain thread also need. So while a new batch is admitted, in-flight
decodes' tokens can't be streamed out, and the last request in the burst waits for
31 preprocesses before it is even admitted. This is the 21-core burst.

## Chosen split + why

**Move `load_* + process_prompt` into a pool of CUDA-free child processes.** The
CPU burst leaves the serve process entirely; the serve worker thread keeps only a
per-tensor memcpy (unpickle + `torch.frombuffer`) plus the transport/conductor
handoff it already owned. Chosen over the alternatives for the LoC budget because
it directly removes the GIL-holding burst *and* parallelizes it across cores (a
single child would relieve the GIL but keep the burst serial). The transport tail
(`tensor_manager.store_and_return_tensor_info`, register/persist, the
`NewRequestConductor` send) **stays on the worker thread** — it owns the ZMQ
socket + SHM tensor manager and must not move.

## Architecture

- **`mstar/api_server/preproc_proc.py`** (new):
  - `preprocess_tensors(model, input, device)` — the extracted `load_* +
    process_prompt` block. **Single implementation called by BOTH the inline path
    and the child**, so they cannot drift (byte-identity by construction).
  - `PreprocClient` — pool of N spawn children, **each with its own bounded
    in-queue** (round-robin dispatch), **one shared out-queue**. The serve worker
    thread polls `get_ready()` in its existing run loop — **no receiver thread**
    (unlike detok, which needed one for async emission).
  - **In-order admission**: completions are emitted in submit order via a reorder
    buffer keyed by a monotonic submit index, so a concurrent burst reaches the
    conductor in the exact order the inline path used. The flag's only happy-path
    effect is *where* preprocessing ran.
  - `run_preproc_proc` — child entry: hides CUDA, caps `torch.set_num_threads`,
    builds the SAME tokenizer/processor model as the serve process (same as
    detok), loops `preprocess_tensors` → returns wire tensors.
- **`data_worker.py`**: `_process_input` refactored into `preprocess_tensors(...)`
  + `_admit_request(...)`; `_on_new_input` dispatches to the pool or inline;
  `_complete_preproc` admits pool/recovered results; `run()` drains `get_ready()`
  each iteration; abort → `cancel_rid`; shutdown → pool teardown.

## Byte-identity

- Child builds the identical model + calls the identical `preprocess_tensors`; env
  is inherited by spawn so the same `MSTAR_GPU_IMAGE_PREPROCESS`/processor path is
  taken. i2t image/text preprocessing is **device-agnostic on CPU** (data worker
  device = `cpu`), so hiding CUDA in the child changes nothing — identical
  torchvision/HF ops on identical bytes → identical tensors.
- Tensors cross as **raw bytes + dtype + shape**, rebuilt with
  `torch.frombuffer(...).reshape(...)` — the exact bytes reinterpreted with the
  exact dtype/shape (bitwise identical; validated across float32/64/16, **bf16**,
  int64, uint8, bool, and zero-element). **Deliberately not torch tensors over the
  queue** → nothing rides torch's mp shared-memory reducer → **no `/dev/shm`
  leak** (the shm-leak lesson). No double-preprocess: each job is preprocessed
  once (pool) or once (inline fallback), never both.
- **AUDIO CAVEAT:** with `MSTAR_GPU_MEL=1` (default) + CUDA present, the serve
  process computes log-mel on the GPU, which is not bit-identical to a CUDA-hidden
  child's CPU mel. So audio-input requests are **not offloaded** in that mode
  (`_should_offload` keeps them inline, byte-identical). i2t is always offloaded.
  With `MSTAR_GPU_MEL=0` audio is CPU on both sides and could be offloaded too.

## Failure semantics (mirror the detok sidecar)

Child death, an errored job, or a saturated in-queue → **permanent inline
fallback**, never a hang, never a dropped/duplicated request:

- A returned job is popped from the outstanding table before it is buffered, so a
  late result for an already-recovered index is dropped (exactly-once).
- On failure every still-outstanding job is recovered by running the identical
  inline `preprocess_tensors + _admit_request` on the serve worker thread, in
  submit order; future submits return `False` → the worker preprocesses inline.
- **Abort-before-admit race** (an abort arriving while the job is still in the
  child): `cancel_rid` marks the rid (only if it has a live pool job) and
  `get_ready` drops that completion — so an early ABORT can never be followed by a
  late NEW_REQUEST that leaves the request admitted-but-un-aborted. The marker
  self-clears when the job drains (bounded set, no `forget_rid` needed).

## Risks

- **Core oversubscription**: a pool of N children each running torch's intra-op
  burst can exceed core count. Mitigated by the per-child thread cap (default
  `cpu_count // procs`); tune `MSTAR_PREPROC_PROCS`/`MSTAR_PREPROC_THREADS`.
- **IPC cost**: pixel_values cross as bytes (one memcpy each way). Far cheaper
  than the resize/patchify it replaces, but not free; if it dominates, revisit
  (shared memory would reintroduce the /dev/shm leak risk — avoided on purpose).
- **Boot**: N children each load the HF processor (~seconds) at `PreprocessWorker`
  init — parallel, and finished well before `finalize_setup` lets the server
  serve, so no first-request penalty in practice.
- **In-order reorder buffer** can head-of-line-block if one child is much slower;
  round-robin + similar image sizes keep this bounded to ~one image's time.

## A/B recipe (owner-run; do not boot vLLM)

Two M*-only boots, identical config, i2t B32:

- Baseline: `MSTAR_PREPROC_PROC=0` (+ current best flags, incl. `MSTAR_DETOK_PROC`
  as shipped).
- Treatment: `MSTAR_PREPROC_PROC=1 MSTAR_PREPROC_PROCS=4` (same other flags).

Metrics: **i2t B32 TTFT p50/p95** (primary — the burst-serialization target) and
**rps under load**; ITL/JCT for regression check. Expect the biggest movement on
**TTFT p95** at high concurrency (the last-admitted request in a burst) and on rps
under host-load, where GIL relief for the ZMQ-drain thread lets in-flight decodes
keep streaming during a new batch's admission. Confirm output bytes are identical
to baseline (byte-identity), and check host-core saturation drops on the serve
process.

## Validation (CPU-only, no GPU)

- `import mstar` + `compileall` of `preproc_proc.py` / `data_worker.py`: OK.
- `test/modular/test_preproc_proc_identity.py` (real Qwen tokenizer/processor,
  real small image, CPU-only): wire round-trip bitwise-identical across 8 dtypes;
  pool preprocess byte-identical to inline; in-submit-order admission;
  cancel-before-admit drops the completion; child-death inline fallback recovers
  exactly-once with no hang.  [status filled below]
