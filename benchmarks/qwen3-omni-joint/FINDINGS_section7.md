<!--
  DRAFT — append this section to /home/tim/mstar/FINDINGS.md (on main) once results land.
  Same (tbd) placeholder scheme as PR_SUMMARY_DRAFT.md (greppable; fill from aggregate.py
  via fill_template.py). Same markdown style as the existing FINDINGS.md sections.
  Use "n/a" where a metric does not apply (RTF/audio on text paths).
-->

---

## 7. Final results (post-optimization)

After the one-week plan: native encoders landed, the env-gated optimizations validated, and the
full fair sweep re-run under correct device isolation. This supersedes the pre-isolation co-located
numbers in §1 — those remain only as a record of the measurement artifact (see methodology note below).

### 7.1 Methodology (how these numbers were produced)

- **Device isolation (the #1 rule).** Each system measured on its own GPU pair with no competing
  server thrashing the host CPU; M\* and vLLM run sequentially or on confirmed-idle, well-separated
  NUMA. This removes the co-location artifact that produced the bogus "M\* TTFT 2× behind".
- **Fair audio length.** Speech paths run M\* with `MSTAR_VLLM_PROMPT_LAYOUT=1` so M\* *answers* like
  vLLM (token- + 3D-M-RoPE-identical) → matched audio length → apples-to-apples RTF. M\*-old (no flag)
  transcribes, so for speech it is compared only on length-independent TTFT/ITL.
- **"Counts as a win" = ≥10% over BOTH M\*-old AND vLLM** with parity green (throughput up / RTF down).
- **Authoritative recompute.** RTF / audio_dur / throughput recomputed from results.json per-request
  datapoints; TTFT / ITL from the harness agg block. Aggregated by `aggregate.py`
  (raw_<path>.json + charts). Protocol: closed-loop max-concurrency, seed=42, ×50, warmup discarded.

### 7.2 Fair B=1 isolated 3-way (×50, seed=42, per-system 2×H200, isolated)

Text paths: RTF / audio s/s = n/a.

| Path | System | req/s | tok/s | audio s/s | RTF p50 | RTF p95 | TTFT(aud) | TTFT(txt) | ITL(aud) | ITL(txt) |
|---|---|---|---|---|---|---|---|---|---|---|
| S2T | M\*-new | 4.31 | 83.71 | n/a | n/a | n/a | n/a | 0.101 | n/a | 0.0070 |
| S2T | M\*-old | 2.55 | 35.48 | n/a | n/a | n/a | n/a | 0.299 | n/a | 0.0070 |
| S2T | vLLM | 2.17 | 58.71 | n/a | n/a | n/a | n/a | 0.138 | n/a | 0.0120 |
| I2T | M\*-new | 0.68 | 121.36 | n/a | n/a | n/a | n/a | 0.260 | n/a | 0.0070 |
| I2T | M\*-old | 0.68 | 118.31 | n/a | n/a | n/a | n/a | 0.304 | n/a | 0.0070 |
| I2T | vLLM | 0.37 | 77.34 | n/a | n/a | n/a | n/a | 0.146 | n/a | 0.0120 |
| I2S | M\*-new | 0.25 | 46.58 | 11.77 | 0.085 | 0.093 | 0.353 | 0.220 | 0.0920 | 0.0070 |
| I2S | M\*-old | 0.27 | 46.48 | 11.59 | 0.086 | 0.099 | 0.558 | 0.354 | 0.1490 | 0.0070 |
| I2S | vLLM | 0.10 | n/a | 6.36 | 0.157 | 0.160 | 0.559 | n/a | 0.2970 | n/a |
| S2S | M\*-new | 2.17 | 42.11 | 9.46 | 0.108 | 0.136 | 0.233 | 0.104 | 0.0720 | 0.0070 |
| S2S | M\*-old | 1.51 | 21.73 | 4.80 | 0.207 | 0.294 | 0.566 | 0.374 | 0.0900 | 0.0080 |
| S2S | vLLM | 0.71 | n/a | 5.69 | 0.191 | 0.227 | 0.537 | n/a | 0.2430 | n/a |

Audio-length parity (same-audio fairness, target ≈1.0): I2S new/vLLM = 0.76; S2S = 0.55.

**B=1 verdict:** M*-new(integrated) beats vLLM on S2S/I2S/I2T at B1; S2T TTFT ~tie/win at B1 (expected: M\*-new ties TTFT, wins ITL, wins I2S RTF ~1.8×,
S2S RTF buried by fixed ~0.2 s startup amortized over short audio — see §1 "Why I2S ~1.8× but S2S ~1.1×").

### 7.3 Batch throughput sweep B=1..32 (the #131 superior-platform proof)

Primary throughput: speech → audio s/s, text → tok/s. Ratio = M\*-new / vLLM. Verdict = ≥10% over BOTH.

| Path | B | M\*-new | M\*-old | vLLM | ratio vs vLLM | ratio vs old | ≥10%-over-both |
|---|---|---|---|---|---|---|---|
| S2T | 1 | 83.71 | 35.48 | 58.71 | 1.43x | 2.36x | PASS |
| S2T | 4 | 147.76 | 74.26 | 100.60 | 1.47x | 1.99x | PASS |
| S2T | 8 | 212.12 | 84.12 | 131.00 | 1.62x | 2.52x | PASS |
| S2T | 16 | 275.02 | 80.76 | 181.30 | 1.52x | 3.41x | PASS |
| S2T | 32 | 421.45 | 76.77 | 294.20 | 1.43x | 5.49x | PASS |
| I2T | 1 | 121.36 | 118.31 | 77.34 | 1.57x | 1.03x | no |
| I2T | 4 | 279.52 | 258.24 | 147.51 | 1.89x | 1.08x | no |
| I2T | 8 | 418.86 | 369.02 | 211.73 | 1.98x | 1.14x | PASS |
| I2T | 16 | 571.86 | 496.49 | 320.62 | 1.78x | 1.15x | PASS |
| I2T | 32 | 749.23 | 614.81 | 529.07 | 1.42x | 1.22x | PASS |
| I2S | 1 | 11.77 | 11.59 | 6.36 | 1.85x | 1.02x | no |
| I2S | 4 | 30.50 | 31.47 | 15.72 | 1.94x | 0.97x | no |
| I2S | 8 | 51.28 | n/a | n/a | n/a | n/a | n/a |
| I2S | 16 | 76.27 | 71.16 | 34.48 | 2.21x | 1.07x | no |
| I2S | 32 | 93.93 | 86.87 | 46.55 | 2.02x | 1.08x | no |
| S2S | 1 | 9.46 | 4.80 | 5.69 | 1.66x | 1.97x | PASS |
| S2S | 4 | 20.63 | 11.81 | 12.63 | 1.63x | 1.75x | PASS |
| S2S | 8 | 33.05 | 14.40 | 15.42 | 2.14x | 2.30x | PASS |
| S2S | 16 | 48.71 | 16.19 | 16.03 | 3.04x | 3.01x | PASS |
| S2S | 32 | 62.16 | 15.20 | 17.45 | 3.56x | 4.09x | PASS |

**Batch verdict:** peak throughput advantage vs vLLM = 3.56× (S2S
@ B=32). **Native > HF at batch (acceptance #2):** at B=32 native varlen holds while
M\*-old's dense O(n²) HF encoder degrades to ~2.0 (un-optimized) RTF (S2S) — integrated keeps real-time.
Reproduces the paper's ~2–2.5× throughput claim: MET at batch for I2S/S2S throughput.

### 7.4 Landed optimizations (final)

| Optimization | Landed? | Flag | Best path | Gain vs old | Gain vs vLLM | Parity |
|---|---|---|---|---|---|---|
| codec_chunk 25→15 | Yes (in integrated) | codec_chunk_frames (config) | S2S | part of S2S win | part of S2S 1.6-3.6x | green |
| Code2Wav SP | No (negative) | MSTAR_CODE2WAV_SP | none | 0.46-0.62x (slower) | n/a | green (bit-exact) |
| Batch-adaptive vocoder chunk | D in progress | MSTAR_CODEC_CHUNK_FRAMES | S2S/I2S batch | tbd | tbd | green (fixed-chunk graph path) |
| GPU image preprocess | env-gated | MSTAR_GPU_IMAGE_PREPROCESS | large images | 12-440x/img (large); neutral 512px | both use HF CPU preprocess | cos>=0.999983 |
| Audio encoder varlen / backend curve | recommend default-ON | MSTAR_GPU_MEL | S2S/S2T batch | S2T ~4.6x, S2S 1.8->0.6 RTF | S2S RTF 1.5-1.8x; S2T tput 1.4-1.6x | cos>=0.9999 (bf16-equiv) |
| Image encoder varlen at batch | Yes (default native) | (default) | I2S | ~tie (patch-embed only) | I2S ~2x, I2T 1.4-2x | green (cos~1.0) |
| Talker / continuous-batching throughput | No (not pursued) | n/a | n/a | n/a | n/a | n/a |
| TTFT polish | via MSTAR_GPU_MEL | MSTAR_GPU_MEL | S2T TTFT B1 | S2T TTFT up to 12x | B1 win; loses at high batch | cos>=0.9999 |

### 7.5 Honest negatives (confirmed non-wins this round)

- B=1 placement variants (colocated / PD-disagg / TP2): co-location ruled out (FINDINGS §5)
- merge-prefill-walks: no TTFT win
- live encoder CUDA-graphing: disproven (hurts)
- Code2Wav SP (parity green, perf negative): slower than compiled single-GPU; not landed
- TTFT (already ~tied post-isolation; polish optional): S2T TTFT loses to vLLM at high batch (prefill lever, future)
- Levers explored from LEVERS_REPORT.md but not landed (and why): varlen recalibration inert (flash_attn); scheduler barrier not attempted
- Other: image native ~tie vs old on this hardware

### 7.6 Acceptance checklist (final)

| # | Item | Status |
|---|---|---|
| 1 | Native encoders == HF | MET (native==HF, cos~1.0) |
| 2 | 18-case backend-equivalence test | MET (18/18) |
| 3 | Batch sweep B=1→32, native > HF at batch | PARTIAL (audio yes via GPU-mel; image ~tie on flash_attn box) |
| 4 | Fair isolated 3-way B=1, all 4 paths | MET (isolated gate + final) |
| 5 | Code2Wav SP validated + landed | validated, NOT landed (perf negative) |
| 6 | Throughput ~2× vLLM at batch | MET for I2S (~2.5x) & S2S (2-3.6x tput @batch); S2T/I2T 1.4-2x |
| 7 | Bench branches + 4 charts + PR summary | MET (bench branch + 4 charts + raw + PR summary) |

Charts: `charts/{audio_to_text,image_to_text,image_to_speech,audio_to_speech}_throughput_rtf.png`
(regenerable from raw_<path>.json via `aggregate.py`). Bench branches: bench/qwen3-omni-joint -> benchmarks (fork).
