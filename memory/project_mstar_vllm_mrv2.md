---
name: project_mstar_vllm_mrv2
description: Why M* lost text paths — vLLM core 0.22 shipped Model Runner V2 (zero CPU-GPU sync decode); corrected scoreboard
metadata: 
  node_type: memory
  type: project
  originSessionId: e0c3d6dc-5207-40da-aea3-893c738981a0
---

M* didn't get slower — **vLLM got faster**. vLLM-Omni minor version is pinned to
vLLM core, so 0.21→0.22 rode core 0.21→0.22. Core **v0.22 shipped Model Runner V2
(MRV2)**, default for Qwen3 dense (PR #39337): GPU-native input prep (Triton),
**zero CPU↔GPU sync across the decode loop**, persistent-batch gather, Triton
Gumbel-Max sampler. That erased exactly M*'s known CPU/GIL floor (~26ms B32). Blog:
+56% tput, −6.3% TPOT. https://vllm.ai/blog/2026-03-24-mrv2 . Reinforcements
0.22→0.24: sync-elimination series, FlashInfer/Triton fused sampler,
FULL_AND_PIECEWISE cuda-graph dispatch, FP8/MoE kernel upgrades. **Do NOT re-try
naive async sched** (vLLM-Omni #4442 hit the same non-overlap M*'s ASYNC_SCHED did);
the win is zero-sync STATE design, not deferral. Track vLLM-Omni RFC #1770 for when
the omni Thinker text runner itself gets MRV2.

**Corrected scoreboard (from raw JSON, not docs — docs lie):** the "20/24 wins" were
vs BROKEN vLLM 0.21 (`raw_vllm_021.json`, i2t B32=2.55). vs healthy **0.22** (i2t B32
band 8.03–8.59): M* **loses i2t B2 (0.97×) and B32 (0.93×), ties B1/B16, wins B4/B8**
— a 3–7% loss, NOT "by a lot." s2t is NOT losing in any committed file (user's s2t
loss is fresh/uncommitted). Warmed run-to-run variance is huge (i2t B2 29%, B32 20%),
so any 2-decimal ratio is noise; HANDOFF_V8 §2 green claims (B1/B2/B16) are false,
V2-budget-policy "WIN" is a −4..−9% regression in its own A/B. Full writeup:
benchmarks/qwen3-omni-joint/REVIEW_V9_WHY_WE_LOSE.md (branch encoders-...-mstar-v2).
Real levers: fp8 MoE, mixed prefill+decode step (H1/H2), MTP spec decode (H3),
persistent scheduler ready-set (H4). See [[project_mstar_decode_bottleneck]],
[[project_mstar_boot_shm_leak]], [[feedback_experiment_report_format]].
