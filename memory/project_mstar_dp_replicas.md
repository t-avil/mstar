---
name: project_mstar_dp_replicas
description: "M* data-parallel Thinker replicas 2026-07 — DP wins decode/tok-s (ITL 3-6ms) but loses short-output req/s (encode-Thinker contention); TP=2 shards a model that fits (wrong)"
metadata:
  node_type: memory
  type: project
  originSessionId: e0c3d6dc-5207-40da-aea3-893c738981a0
---

The 30B-A3B Thinker FITS on one 80GB H200 (~62GB), so vLLM's TP=2 SHARDS a model
that doesn't need sharding — wrong tool. M* can run the Thinker as **data-parallel
replicas** instead (config-only, agent-verified): `Thinker: ranks:[0,1]` with NO
tp_size → `_tp_ranks` empty → conductor DP-picks worker_0/worker_1 per request
(conductor.py:468, coordinated by `_group_id` so prefill+decode+co-grouped-encoder
land on the SAME replica), any_tp=False → ZERO NCCL/all-reduce. Configs committed
mstar-moe 88569170: `qwen3omni_2gpu_dp2_txt.yaml` (Thinker-DP, single encoder rank0),
`qwen3omni_2gpu_dp2a_txt.yaml` (full pipeline [vision_encoder,audio_encoder,Thinker]
replicated per GPU). + MSTAR_DP_ROUND_ROBIN (conductor.py: deterministic 50/50 vs
default random np.random.randint — beware setdefault over-increment, call _pick once
per group). kv_cache.max_num_pages tuned down (1024) so 2 Thinkers fit.

**MEASURED B32 i2t (both replicas serve ~16 each):**
- **DECODE parallelizes SPECTACULARLY: ITL 3-6ms** (VarB 5, VarA 3) vs single-GPU 8,
  vLLM 15.4. At **len512 (decode-heavy) VarA = 2051 tok/s / 4.00 req/s vs single-GPU
  1771/3.45 and vLLM ~1769 → +16% tok/s WIN.** DP's decode ITL is 2.5x vLLM's.
- **But short/natural-output req/s REGRESSES: TTFT ~3.6-4.4s** (VarB 3624 6.46 req/s,
  VarA 4440 5.22) vs single-GPU 2011/7.76. Root cause FUNDAMENTAL to 2-GPU DP: each
  GPU must run encoder+Thinker, so vision-encode contends with prefill on every GPU —
  exactly what `encoff` (encoder on separate GPU) was built to avoid. Only 2 GPUs =
  can't both replicate Thinker AND isolate encode. B1 fine (1.01 req/s, TTFT 190ms).

**VERDICT: DP = a decode-throughput/tok-s win (esp. long-output + s2t where encode is
cheaper), but prefill-bound at short-output i2t B32.** For the tok/s-first goal DP is a
genuine partial win. To also win short-output req/s would need to hide encode behind
decode (side-stream) or a 3rd GPU. Testing MSTAR_DP_ROUND_ROBIN (dp2rr) for TTFT
recovery — but the contention is per-GPU, not imbalance, so RR likely only partial.
See [[project_mstar_ttft_root_cause]] (TP=2 net-negative even w/ symm-mem all-reduce).

**MSTAR_DP_ROUND_ROBIN RESULT (dp2rr, VarA+RR):** RR HELPS natural-length a lot (random
routing WAS causing bursty prefill imbalance): B32 natural **6.92 req/s / TTFT 3139 / ITL 4**
vs random 5.22/4440 (+33% req/s, −29% TTFT). But still < single-GPU 7.76/2011 at natural
(residual = encode-Thinker contention, not imbalance). len512 unchanged (decode-bound):
2041 tok/s / 3.98 req/s (RR doesn't touch decode regime). **NET: DP+RR wins fixed/long-output
tok/s+req/s (len512: 2041 tok/s / 3.98 req/s vs 1-GPU 1771/3.45, vLLM ~1769) but loses
short-natural-output req/s (6.92 vs 7.76). No single config wins BOTH regimes on 2 GPUs.**
Path to full win: hide per-replica encode behind decode (DP GPUs are ~idle during decode,
ITL 4ms) via encoder-async/side-stream — the encode-contention is the last blocker. Best DP
config = dp2a + MSTAR_DP_ROUND_ROBIN=1.

**FULL DP+RR SCOREBOARD (natural length, i2t, vs committed 1-GPU / vLLM):**
- B1 0.91 (1-GPU 0.98 / vLLM 0.88), B8 4.16 (4.20/3.67), **B16 6.49 (5.89/5.62 = WIN both)**,
  B32 6.92 (7.76/8.32 = worse than 1-GPU, loses vLLM). TTFT scales badly B16→B32 (1083→3139)
  = vision-encode contention worsens with batch.
- len512 decode-heavy: 2041 tok/s (WIN vs 1769). s2t B32: 35.31 req/s ≈ 1-GPU 36, both >> vLLM
  27 (s2t TTFT low 391ms — lighter audio encoder confirms i2t issue IS vision-encode contention).
**HONEST NET: DP wins i2t B16 + decode-heavy tok/s + ties s2t, but does NOT fix i2t B32 natural
(regresses vs single-GPU). Single-GPU still best for B32 i2t.** The one holdout across the WHOLE
campaign remains i2t B32 natural req/s (~7-17% behind vLLM, structural + within noise). DP's
GPUs are ~50% idle in decode (ITL 4ms) → hiding vision-encode behind decode (async/side-stream)
is the last lever that could make DP win B32 too — but SIDE_PREFILL is broken on flagship and
ENCODER_ASYNC is off-stack. Genuine wins banked: DP configs + round-robin + symm-allreduce.

**ENCODER-OVERLAP FALSIFIED (last B32 lever).** Implemented MSTAR_SIDE_ENCODER_ONLY
(worker.py:_is_side_eligible, commit 32d89781): route ONLY the stateless encoder to the
side stream (avoids thinker-prefill _ACTIVE_MANAGER race, parity-safe). Measured DP B32
natural **6.41 vs DP+RR 6.92 — REGRESSES**. Agent-predicted: closed-loop B32 encodes cluster
at admission with no live decode to overlap (_maybe_dispatch_side only fires inside an active
decode chain, worker.py:4641); side-dispatch overhead only. Would help open-loop/staggered
arrival, not closed-loop bench. Default off, kept.

**★★★ CAMPAIGN FINAL 2026-07-08: EVERY lever for i2t B32 natural exhausted** (single-GPU
prefill/mixed, TP=2+symm-mem, DP+RR, encoder-overlap — all neutral/regress). i2t B32 natural
req/s is the SOLE holdout (~7-17% vs vLLM, structural: vision-encode+prefill contention on 2
GPUs, within ±18% boot + protocol confound). **M* BEATS vLLM everywhere else: i2t B1-B16
(single-GPU or DP), s2t all batches (~35 vs 27 req/s), decode-heavy tok/s (len512 2041 vs
1769). Best B32-i2t config = single-GPU 7.76.** Committed artifacts: DP configs, round-robin,
symm-allreduce, encoder-only-side gate. Recommend owner re-measure vLLM at matched protocol.
