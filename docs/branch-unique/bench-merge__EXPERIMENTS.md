# Qwen3-Omni M\* — Dedicated Experimentation Plan

Goal: (A) make the **native speech/audio encoder beat the HF wrapper**, and (B) improve
**all four new paths — I2S, S2S, I2T, S2T**. Grounded in the measured bottlenecks (see
`FINDINGS.md`): the encoder is <1% of B=1 E2E, B=1 RTF is ~98.6% Talker+Code2Wav, TTFT is
prefill + conductor round-trips + image preprocess.

**Rules for every experiment:** closed-loop max-concurrency, seed=42, ×50 reqs, report
RTF mean/p50/p95/p99 + TTFT + ITL + throughput + audio_dur. After any encoder/code change,
keep **`test_qwen3_omni_varlen_backend_parity.py` (18) + `qwen3_omni_encoder_parity.py`
(cos≈1.0)** green. Develop env-gated (default OFF) so the baseline stays byte-identical.
Isolate the encoder with **A2T/A2T-batch** (text out, no talker) so you see encoder/prefill,
not the vocoder.

---

## A. Speech (audio) encoder vs HF wrapper

Reality: native ≈ HF at **B=1** (encoder <1% of E2E → they tie). The native win is a **batch**
story — HF's dense O(n²) varlen degrades (M\*-old S2S → 2.0 RTF @ B=32), native varlen shouldn't.
So "beat HF" = **prove + widen the batch advantage**, plus squeeze the launch-bound B=1 path.

| ID | Experiment | Targets | Hypothesis | Measure | Expect |
|---|---|---|---|---|---|
| **A1** | **Audio-encoder batch sweep B=1→32**, native vs HF, via **A2T** (no talker) | encoder scaling | native varlen stays flat where HF dense blows up | A2T B=1..32 both servers; TTFT + encoder-forward (MSTAR_NODE_TIMING) | native ≪ HF at B≥8 — the #131 "superior" proof |
| **A2** | **Varlen-backend × batch matrix** — flash_attn / flashinfer / per_segment / padded / adaptive | encoder | the `adaptive` heuristic (τ=5e5) is miscalibrated for audio's many ~50-tok windows | isolated encoder-forward ms per backend per batch | a better default backend curve |
| **A3** | **Length-bucketed audio CUDA-graph** — pad `cu_seqlens` to fixed audio-length buckets so one graph replays | encoder batch | bucketing removes the per-length cache thrash that made live cudagraph *hurt* | capture bucketed graph; A2T-batch eager vs graph | graph reusable → wins at batch, tail tightens |
| **A4** | **torch.compile the `forward_batched` (B=1) path** (today eager) | encoder B=1 | cut Python/launch overhead on the launch-bound bs=1 loop | A2T B=1 ×50, compile on/off | small B=1 win, tighter p99 |
| **A5** | **GPU-native frontend** — kill `.tolist()/.item()`/per-element loops in `chunk_and_pad`/`get_valid_indices`/`get_audio_cu_seqlens` | encoder | removes CPU↔GPU serialization, esp. multi-segment/batch | encoder-forward ms at B≥4 | small win at batch |

---

## B. Per-path optimization (the real E2E levers)

### S2T (audio→text) — **TTFT-bound**
| B1 | **Merge `prefill_text`+`prefill_audio` into one Thinker walk** | drop the ~60 ms per-walk conductor round-trip → A2T TTFT | env-gated; A2T TTFT before/after |
| B2 | **PD-disaggregation** (`qwen3omni_pd_disaggregated.yaml`) | prefill stops contending decode → TTFT + tput | bench config head-to-head |
| B3 | Overlap audio mel-extract with prefill | hide the ~10 ms CPU mel | A2T TTFT |

### I2T (image→text) — **TTFT-bound, dominated by image preprocess**
| B4 | **Image `process_prompt` resize/patch → GPU / overlap** | kill the **up-to-175 ms CPU** cost (the single biggest I2T cost) | I2T TTFT mean+p99 before/after |
| B5 | **Merge `prefill_text`+`prefill_vision` walk** | drop the conductor round-trip | I2T TTFT |

### S2S (audio→speech) — **RTF: fixed startup (~40% of short wall) + Talker/Code2Wav**
| B6 | **Cut startup** = B1 (merge audio prefill walk) + B2 (PD-disagg) | shrink the fixed ~0.2 s that short audio can't amortize | S2S RTF mean/p50, vs the I2S-style margin |
| B7 | **`codec_chunk_frames` 25→15 sweep** | lower per-chunk ITL / time-to-first-audio | S2S ITL + TTFA sweep |
| B8 | **Talker TP2** (`full_tp2.yaml`) | shrink each of the 25 Talker AR steps/chunk | S2S ITL/RTF |

### I2S (image→speech) — **RTF: Talker+Code2Wav = 98.6%**
| B9 | **🎯 Code2Wav SEQUENCE PARALLELISM** — shard the vocoder frame-dim across both GPUs (stateless, non-AR, halo already exists) | ~halve the dominant vocoder cost on long audio | I2S RTF/tput, parity-check halo boundary; **M\*-only differentiator** |
| B10 | **`max_concurrent_requests=32` + PD-disagg** | saturate the bs-32 Talker decode graph | I2S throughput at B=8..32 |
| B11 | Image preprocess → GPU (= B4) | minor for I2S RTF (long audio), real for the prefill portion | I2S TTFT |

---

## C. Cross-cutting (all paths)
| C1 | **Config-only sweep**: `pd_disaggregated` vs `colocated` vs `full_tp2` vs `max_concurrent=32`, head-to-head on all 4 paths | **zero code, highest information** — sets the placement baseline before any code change |
| C2 | **Async pipelining at B=4→32** (the throughput story) — confirm component overlap sustains ~2–2.5× | throughput vs vLLM across batch |

---

## Recommended order (high info / low cost first)
1. **C1** — config sweep (no code). Establishes the best placement per path/metric.
2. **A1** — encoder batch sweep (the #131 "native > HF at batch" proof — the encoder's actual win).
3. **B4** — image preprocess → GPU (biggest single TTFT win, I2T).
4. **B1 / B6** — merge prefill walks (S2T TTFT + S2S RTF startup cut).
5. **B9** — Code2Wav sequence parallelism (the I2S/long-audio differentiator).
6. **A2 / A3** — encoder backend matrix + bucketed graph (widen the batch advantage).
7. **B7 / B8** — codec-chunk + Talker-TP2 (ITL).

Each row is independently A/B-able against the live baseline; gate behind an env flag, keep
parity green, and record the full metric set so we never compare across inconsistent runs.
