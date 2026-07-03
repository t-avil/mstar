# many-charts — consolidated comparison data + renderer

Self-contained: `data/*.json` hold every series, `render_charts.py` draws
per-path figures (throughput, TTFT p50, ITL mean, RTF p50 for speech),
every benchmarked batch for every system.

Series and provenance:
- `mstar_v3.json` — M* final stack (2026-07-03), pair 0,1 preview cells,
  sentinel-validated (cross-NUMA: ~10% understated).
- `mstar_v2.json` — M*-v2 canonical sweeps (2026-07-02, pair 6,7):
  sweep_text_w1 text + sweep_mstar_v2_final speech.
- `mstar_new.json` / `mstar_old.json` — committed aggregates
  (encoders-implemeneted 4c33b33 / ae7d173 eras).
- `vllm_022.json` — vLLM-Omni 0.22 rebenchmark (committed aggregates).
- `vllm_021.json` — pre-refresh 0.21-era vLLM, extracted from git history
  (5c27c12^); NOT re-benchmarked.

"M* new" (solid blue) = v3 req/s where measured else v2; latency from the
canonical v2 sweeps (v3 preview latency at B32 is a closed-loop harness
artifact — see EXPERIMENTS.md). vLLM speech charts have no TTFT/ITL —
their harness doesn't report text-latency on speech-out paths; nothing is
interpolated or fabricated.
