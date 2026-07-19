# n-gram spec-decode accept-rate ceiling on REAL M* output — NO-GO for i2t/s2t

Measured no-boot by tokenizing REAL generated completions (Qwen3-Omni tokenizer)
and simulating the decode loop (mstar/spec/measure_accept_rate.py).

| path | source | tokens/step | draft hit-rate | verdict |
|------|--------|-------------|----------------|---------|
| i2t  | food101 captions (b1prof2, 60 reqs, 10358 tok) | **1.041** | 14-19% | ~4% fewer steps |
| s2t  | ASR transcripts (exp_b32_out, 128 reqs)         | **1.000** | 2.9%   | zero |

Simulator validated: repetitive synth=2.73, novel=1.00, exact-loop=4.35 (~K+1). Correct.

## Why NO-GO
- tokens/step 1.04 (i2t) / 1.00 (s2t) is the CEILING assuming an eager verify step
  costs the SAME as a captured decode step. It does NOT: eager verify runs OFF the
  captured decode graph (plan_attention .item() sync + full re-dispatch), so an
  eager step is ~1.3-1.6x a captured decode step against the ~26ms host/GIL floor.
- Net effect: i2t x0.80-0.65 (REGRESS), s2t x0.77-0.63 (REGRESS). Best case
  (eager==captured, unrealistic) is only x1.04 i2t / x1.00 s2t.
- Root cause: n-gram/prompt-lookup wins on self-repetitive or prompt-echoing output
  (RAG, code-edit, JSON with repeated keys). i2t captions and s2t transcripts are
  fluent NOVEL prose; the prompt is image/audio (no text tokens for output to copy),
  so prompt-seeding wouldn't rescue it either.

## Decision
DO NOT wire the full live integration for i2t/s2t — measured to regress. The no-boot
core (drafter/accept/rollback/orchestrator, 41 tests) + full integration map are
committed and correct, so if a self-repetitive workload (structured extraction,
code) ever becomes a target, the wiring is a known, de-risked step. For the CURRENT
text-parity goal, spec-decode is the wrong lever. Saved the multi-boot wiring cost
by measuring the payoff ceiling first.
