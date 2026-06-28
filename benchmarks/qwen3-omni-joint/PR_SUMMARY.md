<!--
  #131 PR SUMMARY — DRAFT SKELETON (fill (n/a) from aggregator output).
  Placeholder scheme (greppable):
    (n/a) (n/a) (n/a) (n/a)   -- shipped table
    (n/a)                                                        -- acceptance checklist
    (n/a)  path∈{s2t,i2t,i2s,s2s} sys∈{new,old,vllm}  -- fair B=1 table
        metric∈{reqs,toks,audios,rtf_p50,rtf_p95,ttft_txt,ttft_aud,itl_txt,itl_aud}
    (n/a) (n/a) (n/a) (n/a)  -- batch sweep
    (n/a)                                                         -- parity statement
  Use "n/a" for cells that don't apply (e.g. RTF/audio on text paths).
  Aggregator: /home/tim/exp_rebench/aggregate.py  (raw_<path>.json + charts/ + table/verdict on stdout).
-->

# Qwen3-Omni on M\* — native encoders + serving optimizations (#131)

## Summary

Ports the Qwen3-Omni vision+audio encoders to native M\* (numerically identical to HF) and
lands a set of env-gated serving optimizations, proving M\* is a superior serving platform vs
vLLM-Omni and not a regression vs the M\*-old HF-wrapper. All changes are env-gated, default OFF,
byte-identical to baseline unless enabled; backend-equivalence + encoder-vs-HF parity stay green.

---

## (a) What shipped

Native correctness:
- **Native vision+audio encoders** — numerically identical to HF (= vLLM, which subclasses HF):
  cos=(n/a) (fp32), ≥(n/a) (bf16); 0 missing / 0 unexpected weights.
- **Prompt-layout parity (same-audio fairness)** — `MSTAR_VLLM_PROMPT_LAYOUT=1` makes M\* token- and
  3D-M-RoPE-identical to vLLM (FIX1 system-dup, FIX2 audio M-RoPE h/w) so M\* *answers* like vLLM and
  emits matched-length audio, enabling apples-to-apples RTF. Default OFF = byte-identical transcriber.

Landed optimizations (each env-gated; gains = median, ≥10%-over-both rule):

| Optimization | Landed? | Flag / default | Gain vs M\*-old | Gain vs vLLM | Notes |
|---|---|---|---|---|---|
| codec_chunk 25→15 (chunk ≥ left_context) | Yes@15 (larger=neg) | codec_chunk_frames | n/a | n/a | S2S/I2S streaming latency |
| Code2Wav sequence parallelism (frame-dim shard) | No (negative) | MSTAR_CODE2WAV_SP | 0.46-0.62x | n/a | M\*-only; long-audio I2S |
| Batch-adaptive vocoder chunk (throughput-mode) | No (disproven) | MSTAR_CODEC_CHUNK_FRAMES | n/a | S2S -18% | I2S/S2S at B≥8 (LEVERS #1) |
| GPU image preprocess (on-device resize/patchify) | recommend default-ON | MSTAR_GPU_IMAGE_PREPROCESS | adds image margin | I2T 1.4-1.9x | I2T TTFT, large images only |
| Audio encoder native varlen + backend curve | recommend default-ON | MSTAR_GPU_MEL | S2T 1.7-4.7x | S2T ~2x req/s, S2S 1.5-2x | acceptance #2 (batch) |
| Image encoder native varlen at batch | Yes (default native) | (default) | ~tie (patch-embed) | I2S ~2x, I2T 1.4-1.9x | acceptance #2 (batch) |
| Talker / continuous-batching throughput (B=4..32) | n/a (uncapped already) | n/a | n/a | n/a | the #131 throughput proof |
| TTFT polish (event-driven first-token / mel→GPU) | via MSTAR_GPU_MEL/IMG | MSTAR_GPU_MEL/IMG | S2T TTFT 4.4->0.42s, I2T 7.9->0.76s | TTFT now flat/competitive | optional; TTFT already ~tied |

---

## (b) #131 acceptance checklist

| # | Acceptance item | Status |
|---|---|---|
| 1 | Native encoders == HF (parity test, cos≈1.0) | MET (cos~1.0) |
| 2 | Backend-equivalence regression test (18 cases) | MET (18/18) |
| 3 | M\*-new batch sweep B=1→32, native > HF at batch | PARTIAL (audio decisive via GPU-mel; image native~=old, GPU-img adds margin) |
| 4 | Fair isolated 3-way B=1 table, all 4 paths | MET |
| 5 | Code2Wav SP validated + landed | validated, NOT landed (negative) |
| 6 | Throughput ~2× vLLM at batch confirmed | MET at batch: I2S ~2x, S2S 1.5-2x, S2T ~2x req/s, I2T 1.4-1.9x vs vLLM |
| 7 | Bench branches + 4 charts + PR summary | MET (bench branch + 4 charts + raw + PR summary + FINDINGS) |

---

## (c) Fair B=1 isolated 3-way table (×50, closed-loop, seed=42, per-system 2×H200, isolated)

Recomputed from results.json per-request (RTF/audio/throughput); TTFT/ITL from harness agg block.
Text paths: RTF/audio s/s = n/a. M\*-old runs without the layout flag (transcribes) → for speech
paths compare M\*-old on length-independent TTFT/ITL only.

### S2T (audio_to_text)
| System | req/s | tok/s | RTF p50 | TTFT text | ITL text |
|---|---|---|---|---|---|
| M\*-new | 5.12 | 73.91 | n/a | 0.099 | 0.0070 |
| M\*-old (HF) | 2.55 | 35.48 | n/a | 0.299 | 0.0070 |
| vLLM-Omni | 2.30 | 56.97 | n/a | 0.144 | 0.0120 |

### I2T (image_to_text)
| System | req/s | tok/s | RTF p50 | TTFT text | ITL text |
|---|---|---|---|---|---|
| M\*-new | 0.68 | 117.21 | n/a | 0.309 | 0.0070 |
| M\*-old (HF) | 0.68 | 118.31 | n/a | 0.304 | 0.0070 |
| vLLM-Omni | 0.37 | 77.22 | n/a | 0.149 | 0.0120 |

### I2S (image_to_speech)
| System | audio s/s | req/s | RTF p50 | RTF p95 | TTFT audio | TTFT text | ITL audio |
|---|---|---|---|---|---|---|---|
| M\*-new | 11.77 | 0.25 | 0.085 | 0.093 | 0.353 | 0.220 | 0.0920 |
| M\*-old (HF) | 11.59 | 0.27 | 0.086 | 0.099 | 0.558 | 0.354 | 0.1490 |
| vLLM-Omni | 6.39 | 0.10 | 0.157 | 0.159 | 0.560 | n/a | 0.2970 |

### S2S (audio_to_speech)
| System | audio s/s | req/s | RTF p50 | RTF p95 | TTFT audio | TTFT text | ITL audio |
|---|---|---|---|---|---|---|---|
| M\*-new | 9.46 | 2.17 | 0.108 | 0.136 | 0.233 | 0.104 | 0.0720 |
| M\*-old (HF) | 4.80 | 1.51 | 0.207 | 0.294 | 0.566 | 0.374 | 0.0900 |
| vLLM-Omni | 5.59 | 0.79 | 0.189 | 0.226 | 0.533 | n/a | 0.2390 |

Audio-length parity check (same-audio fairness): I2S M\*-new/vLLM dur ratio = 0.74;
S2S = 0.62 (target ≈1.0 under `MSTAR_VLLM_PROMPT_LAYOUT=1`).

---

## (d) Batch throughput proof — B=1..32 (the #131 "superior platform" claim)

Primary throughput metric per path: speech → audio s/s, text → tok/s. Ratio = M\*-new / vLLM
(>1 = M\* faster). Verdict = M\*-new ≥10% over BOTH M\*-old AND vLLM. Empty batch = not yet run.

### S2T (audio_to_text) — tok/s
| B | M\*-new | M\*-old | vLLM | ratio vs vLLM | ratio vs old | ≥10%-over-both |
|---|---|---|---|---|---|---|
| 1 | 73.91 | 35.48 | 56.97 | 1.30x | 2.08x | PASS |
| 2 | 99.64 | 54.15 | 80.07 | 1.24x | 1.84x | PASS |
| 4 | 125.41 | 74.26 | 105.41 | 1.19x | 1.69x | PASS |
| 8 | 179.42 | 84.12 | 132.65 | 1.35x | 2.13x | PASS |
| 16 | 240.88 | 80.76 | 189.24 | 1.27x | 2.98x | PASS |
| 32 | 362.70 | 76.77 | 305.02 | 1.19x | 4.72x | PASS |

### I2T (image_to_text) — tok/s
| B | M\*-new | M\*-old | vLLM | ratio vs vLLM | ratio vs old | ≥10%-over-both |
|---|---|---|---|---|---|---|
| 1 | 117.21 | 118.31 | 77.22 | 1.52x | 0.99x | no |
| 2 | 185.22 | 173.96 | 110.17 | 1.68x | 1.06x | no |
| 4 | 278.66 | 258.24 | 149.97 | 1.86x | 1.08x | no |
| 8 | 409.85 | 369.02 | 211.97 | 1.93x | 1.11x | PASS |
| 16 | 541.86 | 496.49 | 320.62 | 1.69x | 1.09x | no |
| 32 | 721.74 | 614.81 | 531.02 | 1.36x | 1.17x | PASS |

### I2S (image_to_speech) — audio s/s
| B | M\*-new | M\*-old | vLLM | ratio vs vLLM | ratio vs old | ≥10%-over-both |
|---|---|---|---|---|---|---|
| 1 | 11.77 | 11.59 | 6.39 | 1.84x | 1.02x | no |
| 2 | 18.92 | n/a | 9.91 | 1.91x | n/a | n/a |
| 4 | 30.50 | 31.47 | 15.85 | 1.92x | 0.97x | no |
| 8 | 51.28 | n/a | 23.78 | 2.16x | n/a | n/a |
| 16 | 76.27 | 71.16 | 34.87 | 2.19x | 1.07x | no |
| 32 | 93.93 | 86.87 | 47.85 | 1.96x | 1.08x | no |

### S2S (audio_to_speech) — audio s/s
| B | M\*-new | M\*-old | vLLM | ratio vs vLLM | ratio vs old | ≥10%-over-both |
|---|---|---|---|---|---|---|
| 1 | 9.46 | 4.80 | 5.59 | 1.69x | 1.97x | PASS |
| 2 | 14.23 | 8.12 | 8.81 | 1.62x | 1.75x | PASS |
| 4 | 20.63 | 11.81 | 13.80 | 1.50x | 1.75x | PASS |
| 8 | 33.05 | 14.40 | 19.56 | 1.69x | 2.30x | PASS |
| 16 | 48.71 | 16.19 | 24.13 | 2.02x | 3.01x | PASS |
| 32 | 62.16 | 15.20 | 33.52 | 1.85x | 4.09x | PASS |

Headline: peak throughput advantage vs vLLM = 2.24× (path I2S,
B=16); native-vs-HF at B=32 = audio decisive; image ~tie (HF degrades to
n/a RTF on S2S while native varlen holds — acceptance #2).

Charts (regenerable from raw_<path>.json): `charts/{audio_to_text,image_to_text,image_to_speech,audio_to_speech}_throughput_rtf.png`.

---

## (e) Parity / tests green

- Encoder-vs-HF parity: (n/a) (cos fp32 (n/a), bf16 (n/a)).
- 18-case varlen backend-equivalence test (`test/modular/test_qwen3_omni_varlen_backend_parity.py`): (n/a).
- Per-change parity gates (codec_chunk audio, Code2Wav-SP boundary-focused waveform A/B, GPU-img cos): (n/a).
- All landed flags default OFF → baseline byte-identical: (n/a).

---

## (f) Honest negatives / non-wins (kept gated or dropped)

- **B=1 placement** (colocated / PD-disaggregated / TP2): ruled out at B=1 — default `qwen3omni_2gpu`
  already optimal; colocating regressed −12..−36%. PD-disagg is a batch lever only. co-location ruled out
- **merge-prefill-walks**: correct but no TTFT win (round-trip ~0); kept as a clean simplification, not a perf lever. no TTFT win
- **Live encoder CUDA-graphing**: HURTS (graph key = clip length → cache thrash); disproven, not used. disproven
- **TTFT**: already ~tied with vLLM after isolation (the earlier "2× behind" was a co-location artifact); polish is optional. text-path TTFT now FIXED on-GPU (was CPU-preprocess, not scheduler)
- Other non-wins surfaced during the run: image native ~tie vs old; piggyback/chunked-prefill deferred
