# encoders-implemented-md — wipe-safe documentation branch

All useful markdown from the M* Qwen3-Omni "beat vLLM-Omni" work, gathered into one
orphan branch so it survives an environment wipe. No code — docs only.

## Layout
- **`BRANCHES.md`** — index + description of every branch on the fork (126), with the
  headline/best-result branches called out. `-delta` branches hold local tips that had
  diverged from the remote (pushed under a suffix so nothing was overwritten).
- **`docs/benchmarks/`** — the authoritative benchmark doc set (REVIEW_V9_WHY_WE_LOSE,
  BENCH_V9_RESULTS, BEAT_VLLM_MASTERPLAN, AR_LOOP_10_REFACTORS, B32_LEVER_EXHAUSTION_LOG,
  EXPERIMENTS, FEATURES_SINCE_ENCODERS, HANDOFF_V5..V8, NUMBERS*, etc.).
- **`docs/home/`** — top-level project docs (CANONICAL_FLAGS, RESEARCH_*, MIXED_CG_GROUNDING,
  STORY, REVIEW_*, EXPERIMENTAL_DISCIPLINE via CLAUDE/AGENTS).
- **`docs/branch-unique/`** — markdown that existed only on individual worktrees/branches
  (design notes etc.), deduped by content, prefixed with the source worktree name.
- **`memory/`** — the agent's persistent findings (MEMORY.md index + per-topic notes:
  parity ideas + MOE_AUTOTUNE bug, DP replicas, TTFT root cause, decode profile, etc.).
  These live in ~/.claude and would be lost on a wipe — included here deliberately.

## Headline result (see BRANCHES.md + docs/benchmarks + memory for detail)
M* beats vLLM-Omni 0.22 on **i2t B1–B16**, **s2t every batch** (+27% B32), **ITL everywhere**
(2.5–3.8×), and **tok/s at matched length**. Lone holdout: **i2t B32** (prefill/TTFT, vision
encode serializing against decode). Current best build: `opt/moe-autotune`.
