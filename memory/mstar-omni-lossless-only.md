---
name: mstar-omni-lossless-only
description: For the M* Qwen3-Omni
metadata: 
  node_type: memory
  type: project
  originSessionId: 2a0ba7b0-a291-4fc9-8e08-0871e63bf616
---

For the M* Qwen3-Omni issue #131 effort (making M*-new beat vLLM-Omni and M*-old across
I2T/S2T/I2S/S2S/T2S), the deliverable PR must keep **M*-new ≡ M*-old** (ideally byte-identical
S2S). The user scoped optimizations to **parity-preserving / lossless** ones only.

**Pursue:** scheduler/batching + graph/overlap levers that don't change outputs —
batched vision-prefill, chunked prefill, encoder coalescing, adaptive vocoder chunk,
mixed-walk piggyback, talker batch-fill, precision toggles, parity mode, plus the two
lossless "big" levers: async-audio / encode↔prefill overlap and CUDA-graph token-budget
bucketing. Also lossless config knobs and the talker device-FIFO.

**Do NOT pursue** (explored on pushed branches but excluded from the deliverable): FP8
(weights/KV/MoE grouped-GEMM), MoE kernel surgery, speculative decoding, and multimodal
token reduction — they change numerics/outputs and carry quality risk.

**Why:** throughput already clears the ~2× vLLM bar; the remaining failing metrics are
TTFT/latency (I2T TTFT, S2T TTFT@batch), which the lossless scheduling/graph levers target
directly — so the lossy throughput levers aren't worth the parity/quality risk. See [[mstar-omni-131-status]].
