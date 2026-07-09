# Code2Wav vocoder — launch-overhead analysis (no-GPU code read)

Context: Code2Wav SP (cross-device frame sharding) was a decisive perf negative —
slower than the single-GPU baseline at every chunk size — because the vocoder is
launch/memory-bound on H200 and SP discards the CUDA graph the baseline relies on.
This note answers the lead's follow-up: does the production vocoder already get
CUDA-graph capture, and is there a launch-reduction lever independent of sharding?

## What the production path actually does (code facts)

- `make_audio_codec_config` (`engine/stateless_engine.py:84`): the vocoder engine
  runs with `apply_torch_compile=False` **but `cuda_graph_capable=True`**. It does
  NOT rely on torch.compile at the engine level — it relies on CUDA-graph replay.
  (The model also self-compiles its conv stack in `Code2Wav.consolidate()`; that
  compiled forward is what gets captured into the engine graph.)
- `Code2WavSubmodule.prepare_inputs` (`qwen3_omni/submodules.py:2067-2076`): **every
  chunk is zero-padded to `full_seqlen` = codec_left_context_frames + codec_chunk_frames
  (= 25+25 = 50 by default)** before the forward. Trimming happens later in
  `postprocess`. So the vocoder forward is ALWAYS a fixed `full_seqlen` shape.
- `get_cuda_graph_configs` captures the `code2wav_chunk` walk at
  `capture_batch_sizes=[1,2,4,8,16,32]`, input shaped `(num_quantizers, full_seqlen)`.
- `can_use_cuda_graphs` requires `codec_tokens.shape[1] == full_seqlen`.
- The runner **pads batch UP to the smallest captured size >= bs**
  (`cuda_graph_runner.py:805 _get_padded_batch_size`). So bs=3 -> graph for bs=4, etc.

### Answer to "does it get CUDA-graph at all chunk sizes?"
Effectively YES. Because the chunk dim is always padded to `full_seqlen`, and batch
is padded up to the next captured size, **every production vocoder call (bs<=32)
replays a captured CUDA graph** — per-call kernel-launch overhead is already
eliminated. This is precisely why the single-GPU baseline is hard to beat and why
SP (eager, cross-device, no graph) lost.

## Launch-reduction levers INDEPENDENT of sharding

1. **None cheap at the per-call level.** The CUDA graph already collapses the entire
   conv-stack launch sequence into one replay at `full_seqlen` x bs<=32. There is no
   low-hanging intra-call fusion (SnakeBeta/GELU/gamma chains, pad+conv) that beats an
   already-captured graph. Don't spend effort here.

2. **Fewer, larger calls = the real lever = Agent D's codec_chunk.** Per-call cost is
   ~fixed (graph replay of `full_seqlen` frames); total vocoder cost ~= per-call x
   (audio_frames / codec_chunk_frames). Larger `codec_chunk` -> larger `full_seqlen`
   -> fewer calls. Microbench (this dir, `vocoder_microbench/raw.json`) shows strongly
   sub-linear per-call scaling: 50 frames 11.5 ms vs 800 frames 46.8 ms — 16x frames
   for ~4x time, i.e. ~4x lower per-frame cost.

   **UPDATE (reconciled with Agent D's measured e2e A/B — codec_chunk is DISPROVEN):**
   The per-call saving above is real at the *vocoder-call* level but does NOT translate
   into an end-to-end serving win. D's fair paired on-graph A/B of a larger static
   codec_chunk is a NET NEGATIVE: S2S -18%, I2S +5-7% (below the 10% bar) -> default
   OFF / not landed (recorded DISPROVEN in the RESULTS.md ledger). Two reasons the
   per-call savings don't surface e2e: (a) a larger chunk kills talker->vocoder pipeline
   OVERLAP on short audio (S2S ~3-5s = only 1-2 chunks); (b) the vocoder is NOT the e2e
   bottleneck — post-GPU-mel the critical path is the talker AR loop, and the vocoder is
   already CUDA-graphed/cheap regardless of chunk size. So saved vocoder time is off the
   critical path. Net: neither vocoder lever (SP forward-shard NOR larger codec_chunk)
   is an e2e win; the vocoder is simply not where the I2S/S2S time goes.

3. **Capture-set tuning (cheap, concrete, helps the batched throughput path):**
   - Batch is padded UP to {1,2,4,8,16,32}; a continuous-batching vocoder step of
     bs=9 runs the bs=16 graph (7 dummy slots = wasted vocoder compute). Adding
     intermediate sizes (e.g. 3,6,12,24) to `capture_batch_sizes` reduces this padding
     waste at batch. Memory cost is small (vocoder graphs are cheap).
   - bs > 32 has NO captured graph -> falls back to eager (slow). If D's
     continuous batching drives the vocoder batch above 32, add larger captured sizes
     (48/64) or cap the vocoder batch, else those steps lose the graph.

4. **WARNING for Agent D — adaptive/variable chunk size breaks the graph gate.**
   `can_use_cuda_graphs` requires `shape[1] == full_seqlen` (a single value derived
   from config). A FIXED larger `codec_chunk` is fine — the graph is re-captured at the
   new `full_seqlen`. But a **runtime-ADAPTIVE** chunk size (different frame counts per
   step) will mismatch `full_seqlen` on the off-sizes and silently fall back to EAGER
   (no graph) — which is the slow path SP got stuck on. If D goes adaptive, either
   (a) capture a graph per chunk size used, or (b) keep padding every call up to a
   single max `full_seqlen` (trading some wasted frames for a guaranteed graph hit).

## Bottom line
The vocoder is already launch-optimized via CUDA graphs; sharding a single forward
across GPUs only adds copy + lost-graph overhead. Crucially, **the vocoder is not on
the I2S/S2S critical path** (post-GPU-mel the talker AR loop dominates), so per-call
vocoder savings — whether from SP or from a larger codec_chunk — do NOT convert to an
end-to-end win. Both vocoder levers are DISPROVEN e2e: SP (this work) and larger
codec_chunk (Agent D's A/B: S2S -18%, I2S +5-7%). The only items below that remain
genuinely useful are GRAPH-COVERAGE hygiene for the batched path (capture-set tuning;
avoid runtime-adaptive chunk sizes), which keep the already-cheap vocoder on its fast
graph path rather than promising an e2e speedup. Future vocoder-area effort is low ROI;
look to the talker AR loop instead. Remaining graph-coverage hygiene: avoid runtime-adaptive chunk
sizes that fall off the captured graph.
