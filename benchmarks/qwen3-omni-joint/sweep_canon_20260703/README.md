# Canonical-pair (GPUs 6,7) runs of the final stack, 2026-07-03 19:27-19:59 UTC

Build: exp/overlap-sched @ ab75034, configs/qwen3omni_2gpu_encoff.yaml,
full final-stack flags incl. sidecar bundle (see EXPERIMENTS.md entries
"Sidecar bundle" and "Canonical-pair sweep").

- qb_canon_ab75/ — 8-cell sweep (i2t B1..B32, s2t B8/B32), bundle ON.
  Window degraded (see EXPERIMENTS caveat): i2t B32 5.978, B1 0.447.
- qb_canonoff/  — adjacent A/B leg, bundle OFF: i2t B32 6.378, B1 0.777.
- qb_canonon2/  — adjacent A/B leg, bundle ON:  i2t B32 6.962, B1 0.841.

Verdicts: sidecar bundle +9.2% B32 / +8.2% B1 (adjacent, canonical pair);
canonical i2t B32 honest band 0.73-0.85x vs vLLM 8.210; s2t B32 34.233
(1.11x, from the degraded window — likely understated).
