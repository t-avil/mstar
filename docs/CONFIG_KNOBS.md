# Qwen3-Omni throughput config knobs

Three low-risk, near-free knobs for tuning throughput-at-batch on Qwen3-Omni.
Each is independent and A/B-able. All default to today's behavior, so turning
none of them on changes nothing.

| Knob | Type | Default | Effect when changed |
|------|------|---------|---------------------|
| `MSTAR_DECODE_BUCKET_64` | env var | `0` (off) | Adds a B=64 CUDA-graph decode capture bucket for Thinker / Talker / Code2Wav |
| `kv_cache.max_num_pages` | YAML | `2048` (code default) | Raises the KV page ceiling so high-batch decode is not admission-throttled |
| `MSTAR_NUM_SLOTS` | env var | `2` | Pipeline depth (double-buffer) for CUDA-graph plan/replay overlap |

## 1. B=64 decode capture bucket — `MSTAR_DECODE_BUCKET_64`

By default the Qwen3-Omni decode CUDA graphs capture batch sizes
`[1, 2, 4, 8, 16, 32]` for all three AR/decode stages:

- Thinker decode — `mstar/model/qwen3_omni/submodules.py` (`thinker_decode` config)
- Talker decode — `mstar/model/qwen3_omni/submodules.py` (`talker_decode` config)
- Code2Wav — `mstar/model/qwen3_omni/submodules.py` (`code2wav_chunk` config)

The engine default `DEFAULT_AR_CAPTURE_BATCH_SIZES`
(`mstar/engine/cuda_graph_runner.py`) already includes `64`. Above B=32 these
three stages fall off the captured graphs and run eager, which is slower.

Setting `MSTAR_DECODE_BUCKET_64=1` appends `64` to all three decode capture
lists (single shared helper `_decode_capture_batch_sizes()` so they stay
consistent). It is off by default because the extra bucket costs additional
capture memory and capture time at startup; turn it on only when you intend to
serve effective decode batches above 32.

```bash
# OFF (default): captures up to B=32, B>32 runs eager
mstar serve --config configs/qwen3omni_2gpu.yaml

# ON: also captures B=64
MSTAR_DECODE_BUCKET_64=1 mstar serve --config configs/qwen3omni_2gpu.yaml
```

A/B: compare decode/end-to-end throughput at B=32 vs 48 vs 64 with the env var
OFF then ON. The win should appear at B in (32, 64]; B<=32 should be unchanged
(same captured buckets), confirming no regression.

## 2. KV cache pages — `kv_cache.max_num_pages`

`KVCacheConfig.max_num_pages` (`mstar/engine/kv_store.py`) defaults to `2048`.
At high concurrency/batch the paged KV allocator can exhaust pages, forcing the
admission controller to throttle new requests even when GPU compute is free.

The conductor applies a YAML `kv_cache:` block over every AR node's
`KVCacheConfig` via `setattr` (`mstar/conductor/conductor.py`,
`_get_kv_config`), so the override lands on both Thinker and Talker. The
qwen3omni configs now set:

```yaml
kv_cache:
  max_num_pages: 8192   # tunable; size to available HBM (H200 starting point)
```

Applied in `configs/qwen3omni_2gpu.yaml`, `configs/qwen3omni_full_tp2.yaml`,
and `configs/qwen3omni_thinker_tp2.yaml`.

Do **not** set `cpu_offload_pages` here — CPU offload adds copy latency on the
decode path; this knob only raises the on-GPU page ceiling.

A/B: run a high-concurrency decode workload with `max_num_pages: 2048` vs
`8192` and watch for admission throttling / queue-wait dropping and throughput
rising. If HBM is tight, back the value off.

## 3. Pipeline depth — `MSTAR_NUM_SLOTS`

`CudaGraphRunner.NUM_SLOTS` (`mstar/engine/cuda_graph_runner.py`) reads
`MSTAR_NUM_SLOTS` (default `2`). It is the double-buffer depth: with N slots,
`plan(step+1)` on an inactive slot overlaps `replay(step)` on the active slot.

No code change is needed — the env var is already honored. Setting
`MSTAR_NUM_SLOTS=3` is worth an A/B: a deeper pipeline can increase plan/replay
overlap and throughput, at the cost of one more captured graph set + FlashInfer
wrapper set per (graph_walk, requires_cfg, bs) key (more capture memory and
capture time). Verify capture still succeeds before claiming the win.

```bash
MSTAR_NUM_SLOTS=3 mstar serve --config configs/qwen3omni_2gpu.yaml
```

A/B: throughput at NUM_SLOTS=2 vs 3 on the same workload; confirm HBM headroom
and that all captures complete.

## GPU validation commands

Run on the fixed project GPU set, confirming devices idle first (`nvidia-smi`),
wrapped in a hard timeout, with cleanup on every exit (see workspace GPU rules).

```bash
# Knob 1: B=64 decode bucket — throughput at B=32 / 48 / 64, OFF vs ON
for FLAG in 0 1; do
  for B in 32 48 64; do
    MSTAR_DECODE_BUCKET_64=$FLAG \
      <throughput-bench> --config configs/qwen3omni_2gpu.yaml --batch-size $B \
      --tag "bucket64=$FLAG.b=$B"
  done
done

# Knob 2: KV pages — 2048 vs 8192 under high concurrency
for PAGES in 2048 8192; do
  <throughput-bench> --config configs/qwen3omni_2gpu.yaml \
    --override kv_cache.max_num_pages=$PAGES --concurrency high \
    --tag "max_num_pages=$PAGES"
done

# Knob 3: pipeline depth — NUM_SLOTS 2 vs 3
for SLOTS in 2 3; do
  MSTAR_NUM_SLOTS=$SLOTS \
    <throughput-bench> --config configs/qwen3omni_2gpu.yaml \
    --tag "num_slots=$SLOTS"
done
```

Replace `<throughput-bench>` with the repo's throughput harness. Compare
decode/end-to-end tokens/s; for knob 1 expect parity at B<=32 and a win in
(32, 64].
