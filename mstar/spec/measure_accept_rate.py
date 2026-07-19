"""No-boot accept-rate ceiling for n-gram spec decode on REAL M* output streams.

Tokenizes actual generated i2t / s2t completions with the real Qwen3-Omni
tokenizer and simulates the serving decode loop: at each committed position, draft
up to K tokens from the request's own history and count how many the greedy target
(the actual next tokens) would accept. This gives TOKENS-PER-STEP — the throughput
ceiling of the eager-verify approach. The wiring only pays off if tokens/step is
high enough to beat a captured single-token decode step against the host floor.

Usage: python measure_accept_rate.py <tokenizer_dir> <glob-of-req_txt> [K]
"""
from __future__ import annotations

import glob
import sys

from ngram_drafter import propose_draft, accept_prefix


def simulate(token_ids: list[int], k: int, min_n: int = 2, max_n: int = 4) -> dict:
    """Walk the decode loop over a known token stream; count steps vs tokens.

    Each step: draft from history[:t], accept the prefix the *actual* stream
    confirms, then commit accepted+1 tokens (the bonus is always the real next
    token). Steps = number of forward passes a spec run would take; tokens = len.
    tokens/step = mean speedup ceiling if an eager verify step ~ a decode step.
    """
    L = len(token_ids)
    t = 0
    steps = 0
    accepted_total = 0
    draft_offered = 0
    while t < L:
        history = token_ids[:t]
        draft = propose_draft(history, min_n=min_n, max_n=max_n, k=k)
        actual = token_ids[t:t + len(draft)]
        a = accept_prefix(draft, actual)
        draft_offered += len(draft)
        accepted_total += a
        commit = a + 1                      # accepted drafts + the guaranteed bonus
        t += commit
        steps += 1
    return {
        "tokens": L,
        "steps": steps,
        "tokens_per_step": L / steps if steps else 1.0,
        "accepted_total": accepted_total,
        "draft_offered": draft_offered,
        "draft_hit_rate": accepted_total / draft_offered if draft_offered else 0.0,
    }


def main():
    tok_dir, pattern = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no files match {pattern}"); return
    agg_tokens = agg_steps = 0
    per_file = []
    for f in files:
        text = open(f, encoding="utf-8", errors="ignore").read().strip()
        if not text:
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < 8:
            continue
        r = simulate(ids, k)
        agg_tokens += r["tokens"]; agg_steps += r["steps"]
        per_file.append((f, r))

    if not agg_steps:
        print("no usable streams"); return
    tps = agg_tokens / agg_steps
    # per-request tokens/step distribution
    tps_list = sorted(r["tokens_per_step"] for _, r in per_file)
    n = len(tps_list)
    p50 = tps_list[n // 2]
    p10 = tps_list[max(0, n // 10)]
    p90 = tps_list[min(n - 1, (9 * n) // 10)]
    hit = sum(r["accepted_total"] for _, r in per_file) / max(
        1, sum(r["draft_offered"] for _, r in per_file))
    print(f"K={k}  files={n}  total_tokens={agg_tokens}  total_steps={agg_steps}")
    print(f"  TOKENS/STEP (aggregate) = {tps:.3f}   "
          f"=> {(1-1/tps)*100:.1f}% fewer decode steps")
    print(f"  tokens/step  p10={p10:.2f}  p50={p50:.2f}  p90={p90:.2f}")
    print(f"  draft hit-rate (accepted/offered) = {hit*100:.1f}%")
    # break-even reference: eager verify step costs ~X vs captured decode ~26ms floor
    for ratio in (1.0, 1.3, 1.6):
        speedup = tps / ratio
        verdict = "WIN" if speedup > 1.02 else ("~wash" if speedup > 0.98 else "REGRESS")
        print(f"  if eager_step = {ratio:.1f}x captured_step -> net tok/s x{speedup:.2f}  [{verdict}]")


if __name__ == "__main__":
    main()
