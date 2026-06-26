#!/usr/bin/env bash
# Split the current native-qwen3-omni-encoders work into two clean, stacked
# branches (handoff #13: keep the #131 encoder PR minimal/reviewable, move the
# cross-framework serving + benchmark infra onto its own branch).
#
#   Branch A  (off BASE)  = issue #131 encoder PR — native encoders, toggle,
#                           parity, encoder-only benchmarks. Minimal.
#   Branch B  (off A)     = serving/benchmark deliverable (figs 5/6 + I2T/S2T):
#                           serving_scripts, plot_serving, handoff/env docs,
#                           dataset/runner edits, serving artifacts.
#
# Non-destructive: creates two NEW branches, never force-pushes, leaves the
# original branch untouched. Review, then push yourself.
#
#   BASE=2e6465a FEATURE=native-qwen3-omni-encoders bash split_qwen3_omni_pr.sh
set -euo pipefail

BASE="${BASE:-2e6465a}"                 # last commit before this work
FEATURE="${FEATURE:-HEAD}"              # the branch holding all the work
BR_A="${BR_A:-qwen3-omni-native-encoders}"
BR_B="${BR_B:-qwen3-omni-serving-benchmark}"

# Resolve the feature ref to a concrete SHA NOW, before we switch branches
# (otherwise FEATURE=HEAD would follow us onto the new branch and point at BASE).
FEATURE="$(git rev-parse "$FEATURE")"
echo "splitting work at $FEATURE (base $BASE)"

# ---- file partition ------------------------------------------------------- #
A_FILES=(
  mstar/model/qwen3_omni/components/audio_encoder.py
  mstar/model/qwen3_omni/components/vision_encoder.py
  mstar/model/qwen3_omni/submodules.py
  mstar/model/qwen3_omni/config.py
  mstar/model/qwen3_omni/qwen3_omni_model.py
  test/modular/test_qwen3_omni_native_encoders.py
  test/modular/test_qwen3_omni_native_encoders_ci.py
  benchmark/qwen3_omni_encoder_parity.py
  benchmark/qwen3_omni_encoders.py
  benchmark/encoder_before_after.py
  benchmark/setup_and_run_qwen3_omni_encoders.sh
  benchmark/artifacts/README_qwen3_omni_encoders.md
  benchmark/artifacts/CODE_REVIEW_qwen3_omni_encoders.md
  benchmark/artifacts/encoder_before_after_chart.png
  benchmark/artifacts/qwen3_omni_latency_vs_batch_NVIDIA_H100_80GB_HBM3.png
  benchmark/artifacts/qwen3_omni_parity_depth_NVIDIA_H100_80GB_HBM3.png
  benchmark/artifacts/qwen3_omni_patch_embed_NVIDIA_H100_80GB_HBM3.png
)
B_FILES=(
  ENVIRONMENT.md
  TODO.md
  benchmark/HANDOFF_qwen3_omni_serving.md
  benchmark/artifacts/MSTAR_I2T_S2T_I2S_optimization_review.md
  benchmark/dataset.py
  benchmark/runner.py
  benchmark/run_omni_paths.sh
  benchmark/serving_scripts
  benchmark/artifacts/serving
)

echo ">>> Branch A ($BR_A) off $BASE — encoder PR"
git checkout -B "$BR_A" "$BASE"
git checkout "$FEATURE" -- "${A_FILES[@]}"
git commit -m "qwen3-omni: native audio/vision encoders (#131)

Native, batched audio + vision encoders behind a config/env toggle (HF wrapper
kept as fallback), HF-shard weight loading, parity tests (incl. a CI-runnable
CPU/SDPA structural check), and encoder-only before/after + patch-embed
benchmarks."

echo ">>> Branch B ($BR_B) off $BR_A — serving/benchmark deliverable"
git checkout -B "$BR_B" "$BR_A"
git checkout "$FEATURE" -- "${B_FILES[@]}"
git commit -m "qwen3-omni serving benchmark: figs 5/6 (I2S/T2S) + I2T/S2T (TTFT/ITL)

Cross-framework harness (M* native vs M* HF vs vLLM-Omni vs SGLang-Omni):
parametrized serve/bench scripts with a batch sweep and the native/HF encoder
variant split, plus plot_serving.py (reproducible TTFT/ITL + RTF/throughput
charts from results.json). Stacks on the encoder branch."

echo
echo ">>> done. Branches:"
echo "    $BR_A  (encoder PR, off $BASE)"
echo "    $BR_B  (serving, off $BR_A)"
echo "Review with: git diff $BASE..$BR_A   and   git diff $BR_A..$BR_B"
