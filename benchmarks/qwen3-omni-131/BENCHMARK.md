# How to reproduce — Qwen3-Omni #131

2× H200 (whole GPUs, no co-location). Serving venv with this branch installed;
client venv with the in-repo `benchmark` package. See `env.txt` for the exact
software stack used for the committed numbers.

## 1. Boot the single config (serves all 5 paths, no topology swap)
```bash
CUDA_VISIBLE_DEVICES=6,7 \
  python -m mstar.cli.main serve qwen3_omni \
  --config configs/qwen3omni_2gpu_dpenc.yaml \
  --host 0.0.0.0 --port 8296 --tensor-comm-protocol SHM
```
The winning feature flags are all `MSTAR_*` env vars, default-OFF. Enable the stack
by exporting them before boot (full list + one-line rationale in the PR description).
Each is byte-identical when off; fp8/custom-ops are bounded-to-rounding.

## 2. Sweep (per path × batch, closed-loop, natural-EOS)
```bash
python -m benchmark.runner --url http://127.0.0.1:8296 --model qwen3omni \
  --request-type image_to_text --dataset food101 \
  --profiling-type closed_loop --max-concurrency <B> --num-requests <n> \
  --num-warmup 8 --inference-system ours --output-dir raw/this-work/image_to_text_b<B>
```
Request types: `image_to_text`, `audio_to_text`, `image_to_speech`,
`audio_to_speech`, `text_to_speech`. Batches: 1 2 4 8 16 32. Sample cadence
n: text 64/64/96/96/96/128, speech 12/20/24/40/64/96. See `command.txt`.

## 3. Charts
```bash
python scripts/gen_charts.py   # reads raw/, writes charts/final_*.png
```
Three series: this work (solid blue), m-star main upstream (grey dotted),
vLLM-Omni 0.24 (solid green). Deterministic — regenerates byte-identically from
`raw/`.

## Notes
- Speech B32 req/s is wave-lottery / shared-host sensitive at n=96; validate a
  tight speech cell at n≥256 before defending it.
- `vllm024` and `upmain` series are committed as reference; re-run vLLM-Omni 0.24
  through your own pipeline before publishing those cells.
