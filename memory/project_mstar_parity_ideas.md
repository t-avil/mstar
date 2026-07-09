---
name: project_mstar_parity_ideas
description: "M* 5 parity-safe throughput ideas (2026-07-08 code review) + CRITICAL: MSTAR_MOE_AUTOTUNE is NOT byte-identical (shipped parity bug)"
metadata:
  node_type: memory
  type: project
  originSessionId: e0c3d6dc-5207-40da-aea3-893c738981a0
---

4-agent M*-only code review (never vLLM) for PARITY-SAFE (byte-identical greedy) throughput
wins. Being A/B-tested in worktree+GPU-pair agents (baseline vs fix + greedy output diff).

**★ CRITICAL PARITY BUG FOUND: MSTAR_MOE_AUTOTUNE (ON in flagship stack) is NOT byte-identical.**
`utils/fused_moe/kernels.py:408` — the tuned `_DECODE_MOE_CONFIGS` varies BLOCK_SIZE_K per
M-bucket (64/128/256). The GEMM K-loop accumulates fp32 sequentially (kernels.py:113-134);
changing BLOCK_SIZE_K regroups the fp32 adds (non-associative) → low-bit drift → can flip a
greedy tie. The "numerically identical" header comment is WRONG. MSTAR_MOE_FP8 (also shipped)
is likewise not bit-exact (fp8 quant, cos≥0.99). ⇒ my whole session's "parity-safe" A/Bs ran
on a non-byte-identical build. FIX = Idea 4 below.

**THE 5 PARITY-SAFE IDEAS (ranked):**
1. **Denser thinker_decode CUDA-graph buckets** (submodules.py:1646). Decode never misses
   graphs but pads rows up (bs17→24 = ~29% wasted attn+MoE FLOPs). Byte-identical (padding =
   independent dummy rows). ENV-TESTABLE: MSTAR_DECODE_BUCKETS=1,2,4,8,12,16,20,24,28,32.
2. **fullgraph=True compiled decode** (cuda_graph_runner.py:618). With CUSTOM_OPS on, remaining
   breaks = per-layer os.environ reads (attention.py:84 custom_ops_enabled, moe.py:508
   _moe_fp8_flag) + trailing advance_seq_lens (thinker.py:261, DEAD on replay). Hoist env reads
   to constants + lift advance_seq_lens + flip fullgraph → Inductor fuses → fewer/larger kernels
   in the captured graph. Byte-identical (fusion boundary only moves). The "complexly compiled
   AR" ask.
3. **Incremental scheduler ready-set** (micro_scheduler.py:660-689 rebuilds the ready index
   EVERY step on the 26ms GIL floor). Maintain incrementally. Byte-identical IFF candidate
   iteration order preserved (else flips a tie → different walk → non-identical).
4. **Parity-RESTORING MoE tile table** (fused_moe/kernels.py:408): keep BLOCK_SIZE_K=64 (default)
   fixed, tune only num_warps/num_stages/BLOCK_M/N/GROUP_M (where most of the 1.03-1.12x lives).
   Recovers speedup AND makes MSTAR_MOE_AUTOTUNE byte-identical. Fixes the bug above.
5. **Finer prefill buckets + on-GPU vision metadata** (submodules.py:1468 VISION_GRAPH_ALIGN;
   kill .item()/.tolist() at vision_encoder.py:278/282, rope.py:333). Cuts B32 TTFT (a 258-tok
   image pads to 512 = ~99% slack). Byte-identical (zero-len padding rows, same integers).

**NOT parity-safe (agents flagged, exclude):** anything changing decode BATCH COMPOSITION
(PRIORITY sched, mixed-fold, multistep — FP non-assoc → non-identical, cf async-sched +9% len);
skip-the-FlashInfer-replan (split-KV schedule depends on KV len → reduction-order drift);
fused RoPE (tl.cos/sin ≠ torch low bits); fp8 MoE; atomic-accumulate MoE fusion.

Test scheme: 2 GPU pairs free (0 + 4,5 held by other users/naomi; use 2,3 and 6,7 only).
Round 1 testing ideas 1+2. Merge winners to opt/moe-autotune, repeat with 3,4,5.
See [[project_mstar_dp_replicas]], [[project_mstar_ttft_root_cause]].

**★★★ CRITICAL FINDING 2026-07-08: M* greedy output is NON-DETERMINISTIC run-to-run.**
Proven: SAME server, SAME food101 image, two runs → DIFFERENT generated text (same dish
identified, but wording diverges e.g. "visual information" vs "visual elements"). Cause:
closed-loop DYNAMIC BATCHING — which requests co-batch each step is timing-dependent, and
MoE/attention GEMMs are FP-non-associative, so a request's logits depend on its batch-mates.
CONSEQUENCES: (1) **byte-identical greedy parity is UNACHIEVABLE** — the baseline isn't
byte-identical to ITSELF, so diffing outputs across runs/configs is an INVALID parity test.
The real parity bar = no SYSTEMATIC change (speculative decoding, temperature, length shifts);
ULP-level FP-order differences (tiles, fusion, batching) are already present baseline-to-
baseline and are acceptable. (2) The Idea-4 "byte-identical MoE" justification is weaker than
stated — K=64 just picks the default's FP-ordering; committed anyway (6df89397, harmless +
keeps warps/stages tuning). 
**MEASUREMENT REALITY on this shared box:** perf A/B needs the 96-request protocol
(--num-requests 96 --num-warmup 2; the 48-req window under-measures: 6.4 vs true ~7.7) AND is
boot-lottery-limited (~5-18%/boot) — so few-% idea effects need MANY boots/condition to
resolve, or a dedicated quiet node + FIXED-batch-composition harness for determinism. Single
boot-static A/Bs (buckets/fullgraph/MoE-table via reboot) are confounded by lottery. Baseline
i2t B32 clean = 7.6-7.85 req/s (96-req). Ideas 1/2/3/5 remain implemented/reviewed but
UNVALIDATED for perf on this box. Idea 2 (fullgraph) edits: mstar-fullgraph worktree (removed;
edits were cuda_graph_runner.py fullgraph=True + moe/attention env-const hoists).
