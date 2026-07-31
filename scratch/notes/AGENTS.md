# Experimental Discipline

Never stand still. The goal is maximum parallelism: always have multiple
experiments running, multiple code paths being explored, multiple hypotheses
being tested. Do not respond sequentially. Spawn agents immediately and let
them execute in parallel while you continue analysis.

## Core Mechanism: Agent Parallelism

When facing an unknown (why does M* underperform vLLM on I2T? Why does CUDA
graph capture fail in mixed decode-prefill?), do not investigate serially.
Spawn four agents in parallel immediately. They execute simultaneously;
results come back together.

### Agent Types

**The Idiot Agent** asks baseline questions: what have we already measured?
What are the commit hashes for prior versions? Which modalities regressed?
Cross-reference git history, remote branches, and existing benchmarks. This
agent prevents re-running old experiments. Output: inventory of what we know.

**The Research Agent** digs into papers, docs, and architecture specs. For M*
throughput gap, it reads Qwen3-Omni architecture (ViT/AuT encoder design,
Thinker/Talker, TM-RoPE), vLLM 0.22 continuous batching logic, CUDA graph
semantics for mixed decode-prefill buckets. Output: theory that explains the
observation.

**The Code Agent** identifies implementation points: Where does M* invoke
eager mode instead of capturing graphs? What is the branching logic for
decode vs. prefill? Can we instrument the code to measure prefill time,
decode time, encoder latency separately? Output: specific code locations and
instrumentation targets.

**The Propose Agent** generates testable branches: "Branch A: add run_mixed()
method in cuda_graph_runner.py for mixed buckets"; "Branch B: benchmark
encoder bottleneck in isolation"; "Branch C: compare M* vs vLLM kernel
timings for same batch"; "Branch D: validate NUMA co-scheduling overhead on
this config first." Output: ranked list of experiments to run.

All four agents deliver in parallel. You do not wait for Idiot Agent to
finish before starting Research Agent.

## Experiment Loop

1. Spawn all four agents with the unknown. Do not wait. Let them run in
   background.
2. While agents execute: Check GPU status. If capacity exists, schedule a
   minimal validation experiment on the most promising hypothesis from last
   iteration.
3. Agents return results. Idiot says: "M* I2T regression first appeared in
   commit abc123; vLLM 0.22 uses CUDA graphs for all decode paths; we have no
   prior mixed-mode benchmarks." Research says: "M*'s Talker uses
   decoder-only, vLLM uses auto-regressive generation with KV cache; M* may
   be failing to batch mixed prefill-decode correctly." Code says: "M* uses
   eager evaluation in MSTAR_MIXED_WALK mode; no cuda_graph_runner.run_mixed()
   method exists." Propose says: "Branch rank: (1) instrument mixed path,
   (2) compare single decode vs mixed decode latency, (3) validate NUMA
   baseline, (4) implement graph capture for mixed bucket."
4. Validate experiment completes. Data arrives: single batch size, single
   modality (I2T), WARMUP_RUNS + 1 validation run. Baseline isolation +
   co-scheduled variant measured. Overhead acceptable? Yes/no documented.
5. Reason through code and theory. Why did it work or fail? Compare benchmark
   results against theory from Research Agent. If validation passed and NUMA
   overhead ≤ NUMA_OVERHEAD_THRESHOLD, expand to additional batch sizes and
   spawn next round of real experiments. If validation failed (crash,
   regression, overhead > threshold), do not expand. Instead: spawn agents
   again with new unknown ("why did mixed decode fail at batch 64?"). Run
   code instrumentation branch. Measure encoder time vs decoder time
   separately. Identify bottleneck.
6. Never idle. Once you understand the failure mode, propose next branches
   immediately. Spawn agents on those branches before analysis is complete.
   Queue experiments to run while you write code or theory.

## Historical Data and Reuse

Before spawning any experiment, Idiot Agent must cross-reference commits and
remote branches. We already have data for M* iterations, POD Attention v2,
and all modalities (S2T, I2T, I2S, S2S). Do not re-run prior sweeps. Only
execute missing or invalidated datapoints.

## Benchmarking Configuration

```
WARMUP_RUNS: [configurable, typically 5-10]
VALIDATION_RUNS: 1
REAL_RUNS: [configurable, typically 5-10]
NUMA_OVERHEAD_THRESHOLD: [configurable, typically ~5%]
```

Validation phase runs WARMUP_RUNS + VALIDATION_RUNS on minimal config (single
batch size, single path, single modality). If validation passes and NUMA
overhead is acceptable, expand to additional batch sizes and run REAL_RUNS
per config. All runs must be from same commit/branch. If validation fails,
stop expansion; spawn agents with new hypothesis instead.

## Parallelism in Scheduling

Co-schedule validated configs in same NUMA cluster. Assign independent sweeps
to separate clusters in parallel. Batch similar configs (modality, adjacent
batch sizes) to amortize warmups. Always have experiments running on as many
GPUs as available. When idling, check GPU queue and propose new branches to
fill capacity.

## OWNER RULE (2026-07-05): do NOT benchmark/boot/race vLLM — the user does that
themselves. Compare against COMMITTED vLLM values (benchmarks branch, h2h_* raw;
i2t B32 fresh-boot band 8.03-8.50). M*-only cells and A/Bs are unrestricted.
