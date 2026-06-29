# Per-path architectural differences: vLLM-Omni vs M\*-new vs M\*-old

## Shared across all paths (system-wide knobs)

| Mechanism | vLLM-Omni | M\*-new | M\*-old |
|-----------|-----------|---------|---------|
| Continuous batching (prefill+decode in same step) | YES (default, v1 engine) | NO -- `micro_scheduler.py:78-105` enforces same `graph_walk` per batch; `MSTAR_MIXED_WALK=1` exists but is eager-only (-79% req/s at B=32) | NO -- same constraint |
| Chunked prefill (intra-prefill yields) | NO (prefill < 32768 token budget runs in one step) | YES -- 512-token chunks via `MSTAR_CHUNKED_PREFILL` | NO |
| KV cache block/page size | 16 (`vllm/config/cache.py:45`) | 128 (`mstar/engine/kv_store.py:71`) | 128 |
| Multi-model pipeline transport | 3 separate OS processes, `SharedMemoryConnector` | 1 process per worker, `StreamingGraphEdge` device-direct | 1 process per worker, `StreamingGraphEdge` device-direct |

## S2T (audio -> text)

Pipeline: audio bytes -> mel -> audio\_encoder -> Thinker.prefill\_audio -> Thinker.decode\_text -> text out

| Mechanism | vLLM-Omni | M\*-new | M\*-old |
|-----------|-----------|---------|---------|
| Audio mel computation | CPU | GPU (`MSTAR_GPU_MEL=1`) | CPU |
| Audio encoder placement | Stage 0 process (Thinker GPU) | In-process, Thinker GPU | In-process, Thinker GPU |
| Audio encoder batched across concurrent requests | NO | NO (~10-20ms, not worth batching) | NO |
| Audio encoder CUDA-graph capture | NO | YES (sized-bucket captures) | NO |
| Thinker -> text-out cross-process hop per token | NO (Stage 0 only) | NO (single process) | NO (single process) |
| Talker/Code2Wav touched | No | No | No |

## I2T (image -> text)

Pipeline: image bytes -> resize/patchify -> vision\_encoder -> Thinker.prefill\_vision -> Thinker.decode\_text -> text out

| Mechanism | vLLM-Omni | M\*-new | M\*-old |
|-----------|-----------|---------|---------|
| Image preprocess (resize + patchify) | CPU, marshaled to GPU | GPU (`MSTAR_GPU_IMAGE_PREPROCESS=1`) | CPU, marshaled to GPU |
| Vision encoder placement | Stage 0 process (Thinker GPU) | In-process, Thinker GPU | In-process, Thinker GPU |
| Vision encoder batched across concurrent requests | NO (per-request) | YES (`MSTAR_BATCH_VISION_PREFILL=1`) | NO |
| Vision encoder CUDA-graph buckets aligned to actual token counts | NO | YES (`MSTAR_VISION_GRAPH_ALIGN=1` -- intermediate buckets 128/192/256/320/384/512/768/1024/1536/2048/...) | NO (default coarse buckets) |
| Vision encoder torch.compile dynamic shape | NO | YES (`dynamic=True`, single shape-poly artifact) | NO |
| Vision encoder GPU<->CPU syncs in prefill prepare | YES | NO (sync-elim opt eliminates them) | YES |
| Thinker -> text-out cross-process hop per token | NO (Stage 0 only) | NO | NO |
| Talker/Code2Wav touched | No | No | No |

## S2S (audio -> speech)

Pipeline: audio -> mel -> audio\_encoder -> Thinker.prefill\_audio -> Thinker.decode\_text -> text tokens -> Talker.talker\_decode -> speech codes -> Code2Wav.chunked\_decode -> audio out

| Mechanism | vLLM-Omni | M\*-new | M\*-old |
|-----------|-----------|---------|---------|
| Audio mel computation | CPU | GPU (`MSTAR_GPU_MEL=1`) | CPU |
| Audio encoder placement | Stage 0 (Thinker GPU) | In-process, Thinker GPU | In-process, Thinker GPU |
| Thinker -> Talker handoff per token | Cross-process IPC (Stage 0 -> Stage 1, GPU->CPU->shmem->CPU->GPU per token in full-payload mode) | In-process, `StreamingGraphEdge` device-direct | In-process, device-direct |
| Talker MTP loop (per RVQ group, 8 codes/step) | Python `for pos in range(seq_len)` loop, eager (`qwen3_omni_moe_talker.py:125-189`) | One captured CUDA graph per batch size | One CUDA graph per batch size |
| Talker batch CUDA-graph buckets | Limited / eager fallback | bs in {1,2,4,8,16,32} | bs in {1,2,4,8,16,32} |
| Talker -> Code2Wav handoff per chunk | Cross-process IPC (Stage 1 -> Stage 2) | In-process, device-direct | In-process, device-direct |
| Code2Wav chunk size | 25 frames (`qwen3_omni_moe.yaml:14`) | 25 frames (default) | 25 frames |
| Code2Wav CUDA-graph capture | NO -- `enforce_eager: false`? Set TRUE for code2wav in vLLM's config (`yaml:55`) | YES (Code2Wav captured at fixed chunk sizes) | YES (limited) |

Per-token IPC cost for S2S = (Stage 0 -> Stage 1 hop) + (Stage 1 -> Stage 2 hop). Two cross-process boundaries per audio token.

## I2S (image -> speech)

Pipeline: image -> preprocess -> vision\_encoder -> Thinker.prefill\_vision -> Thinker.decode\_text -> text -> Talker -> speech codes -> Code2Wav -> audio out

| Mechanism | vLLM-Omni | M\*-new | M\*-old |
|-----------|-----------|---------|---------|
| Image preprocess (resize + patchify) | CPU, marshaled to GPU | GPU (`MSTAR_GPU_IMAGE_PREPROCESS=1`) | CPU |
| Vision encoder batched across requests | NO | YES (`MSTAR_BATCH_VISION_PREFILL=1`) | NO |
| Vision encoder CUDA-graph bucket alignment | NO | YES (`MSTAR_VISION_GRAPH_ALIGN=1`) | NO |
| Vision encoder torch.compile dynamic | NO | YES | NO |
| Vision encoder sync-elim | NO | YES | NO |
| Thinker -> Talker handoff per token | Cross-process IPC | In-process, device-direct | In-process, device-direct |
| Talker MTP loop | Python for-loop, eager | One captured CUDA graph per batch size | One CUDA graph per bs |
| Talker -> Code2Wav handoff per chunk | Cross-process IPC | In-process, device-direct | In-process, device-direct |
| Code2Wav CUDA-graph | Eager only | Captured | Captured |

Per-token IPC cost for I2S = same as S2S (Stage 0 -> Stage 1 -> Stage 2 = two cross-process boundaries per audio token). Image preprocess is also a major win for M\*-new here vs M\*-old; vLLM-Omni's CPU preprocess matters most at low batch.

## Takeaway

- **Throughput wins (every path)**: chunked prefill + KV page=128 + same-walk-only decode batching, plus on vision paths the trio (BATCH\_VISION\_PREFILL + VISION\_GRAPH\_ALIGN + dynamic-compile + sync-elim).
- **TTFT wins on S2T at low batch**: GPU mel + no IPC.
- **TTFT loss on I2T/S2T at high batch**: chunked prefill yields cost TTFT; mixed-position batching (the deferred fix) is implemented but eager-mode penalty makes it net-negative today.
- **Speech-path TTFT (S2S, I2S)**: M\* wins by ~30-50% at every batch because of the cumulative 2x cross-process IPC vLLM pays per audio token vs M\*'s in-process device-direct edges.
- **M\*-old**: same in-process pipeline as M\*-new but missing the chunked prefill / vision opts -- throughput ceilings at ~2-5 req/s due to whole-walk prefill serialization.
