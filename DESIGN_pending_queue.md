# Device-backed Talker text FIFO (`MSTAR_TALKER_PENDING_QUEUE`)

Port of SGLang-Omni's `PendingTextTensorQueue` optimization into M*'s
Qwen3-Omni Talker text/hidden feed.

## Finding: M*'s talker feed IS a per-row Python-list buffer (optimizable)

The Thinker streams one `thinker_states` row (a `[1, thinker_hidden]` projected
hidden state) per decode step to the Talker. On the consuming worker this is
buffered by `mstar/streaming/stream_buffer.py::StreamBuffer`, gated by a
`ChunkPolicy`. The Thinker→Talker edge uses
`FixedChunkPolicy(chunk_size=1, continue_after_done=True)`
(`mstar/model/qwen3_omni/qwen3_omni_model.py:705-706`), i.e. window == stride ==
1 — a one-row-per-step FIFO.

The per-step hot loop (`talker_decode`) currently churns Python objects per row:

- `StreamBuffer._buffer` is a `list`; every `pop_chunk` re-slices it
  (`self._buffer = self._buffer[stride:]`, `stream_buffer.py:153`), reallocating
  a list object each decode step.
- Arrival bookkeeping uses a `deque` (`_tensor_ids_in_order`) plus a `dict`
  (`_id_to_tensor`) keyed by tensor uuid, with a per-row `dict` insert/delete.
- `_route_streaming_tensor` (`worker.py:644`) does a per-row `tensor.clone()`.

The consumed value is reached via `prepare_inputs` `talker_decode`
(`mstar/model/qwen3_omni/submodules.py:1598-1604`): `inputs["thinker_states"][0]`
→ `text_projection`. So yes — this is the per-row deque/list buffering with
per-step churn that SGLang's `PendingTextTensorQueue` eliminates. The
optimization applies (a modest, real win given B=1 S2S RTF is ~98.6% Talker AR +
Code2Wav).

## What changed

| File | Change |
|------|--------|
| `mstar/streaming/pending_text_queue.py` (new) | Faithful port of SGLang's `PendingTextTensorQueue`: one device tensor + integer cursor. `popleft`/`pop_slice` advance the cursor (no copy); `append` adopts by reference when empty, else concatenates. Adds a 2D-preserving slice API and `pending_queue_mode()` env reader. |
| `mstar/streaming/chunk_policy.py` | `ChunkPolicy.is_single_row_fifo()` (default `False`); `FixedChunkPolicy` returns `True` when `chunk_size == 1`. Gates eligibility — only window==stride==1 policies use the FIFO. |
| `mstar/streaming/stream_buffer.py` | `__post_init__` reads the env var; `_use_pending_queue` true only for single-row-FIFO policies. `_update_buffer`/`has_chunk_ready`/`pop_chunk` route through `_pop_chunk_pending()` (device FIFO) when enabled. `parity` mode runs both paths and asserts equality. |
| `test/modular/test_qwen3_omni_pending_text_queue.py` (new) | FIFO unit tests vs reference `deque`, cursor/wrap/zero-copy, StreamBuffer off/on/parity equivalence, eligibility, CUDA-skipif device test. |

## Env gate (`MSTAR_TALKER_PENDING_QUEUE`)

- `off` / `0` / unset (**default**): list path, **byte-identical** to before.
  The original `pop_chunk`/`_update_buffer` code runs verbatim — the new code is
  reached only when the flag is set.
- `on` / `1` / `true`: device-backed single-tensor FIFO for eligible (single-row
  FIFO) edges; all other edges (codec_tokens `FixedChunkPolicy(25)`, Code2Wav
  `LeftContextChunkPolicy`, any sliding window) keep the list path.
- `parity` / `2` / `shadow`: maintain both the list and the FIFO in lockstep;
  every pop asserts `torch.equal(fifo_row, list_row)` and returns the list
  result. Runtime parity gate for GPU validation.

## Parity argument

For a single-row FIFO (`chunk_size == 1`, the only eligible case):

- **Normal pop** — list path returns `_collate([_buffer[0]])` → `{"data":
  items[0]}` where `items[0]` is the `[1, hidden]` row. FIFO path returns
  `pop_slice(1)` → `rows[cursor:cursor+1]`, a `[1, hidden]` view of the same
  values. Same shape, same values.
- **Empty flush after producer-done** — `is_ready(0)` is `False` for
  `chunk_size == 1`, so both paths emit `{"data": None}`, `stride == 0`.
- **`continue_after_producer_done`** — handled identically: drained buffer with
  the flag set still reports `has_chunk_ready()` and emits empty,
  never-final chunks.
- **Ordering** — `_update_buffer` keeps the existing head-of-line gating
  (`_tensor_ids_in_order` + `_id_to_tensor`); the FIFO only replaces the ordered
  *storage*, so out-of-order RDMA arrival is still serialized before append.
- `start_offset` (`_consumed`), `chunk_index`, `is_final`, and
  `policy.register_chunk(stride)` are advanced with the same values on both
  paths (verified by the StreamBuffer test).

Multi-item / overlapping policies are never eligible (`is_single_row_fifo()`
returns `False`), so their stacking semantics (`torch.stack`) are untouched.

## GPU validation command

No GPU was used for this branch (code + `py_compile` + CPU parity script only).
On a GPU box, validate parity first, then measure RTF + throughput with the flag
OFF vs ON at B=1, 8, 32 for S2S / I2S / T2S. Use the project's fixed GPU device
set, confirm idle via `nvidia-smi`, wrap in `timeout`, and clean up per the
workspace conventions.

```bash
# 0) Correctness gate: parity mode must not raise on a real run.
MSTAR_TALKER_PENDING_QUEUE=parity timeout 30m \
  python -m mstar.<entrypoint> --task s2s --batch-size 1 --prompts <fixed_set>

# 1) RTF + throughput sweep, OFF vs ON, per modality and batch size.
for MODE in 0 1; do
  for TASK in s2s i2s t2s; do
    for BS in 1 8 32; do
      MSTAR_TALKER_PENDING_QUEUE=$MODE CUDA_VISIBLE_DEVICES=<fixed> timeout 30m \
        python -m mstar.<entrypoint> --task $TASK --batch-size $BS \
          --prompts <fixed_set> --report-rtf --report-throughput \
          --out benchmark-personal/talker-pending-queue/raw_${TASK}_bs${BS}_mode${MODE}.json
    done
  done
done
```

Expect: ON output byte-identical to OFF (decoded codec tokens / audio), with a
small RTF improvement concentrated at B=1 (where per-step Python churn is the
largest fraction of step time); diminishing at B=8/32 as GPU work dominates.
Record every datapoint per the raw-JSON + chart conventions before committing
any numbers.
