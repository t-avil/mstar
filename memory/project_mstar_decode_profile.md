---
name: project_mstar_decode_profile
description: "Empirical py-spy floor of M* i2t B32 decode worker 2026-07 — _postprocess_batch 23%, check_stop D->H 18%, zmq 13%, sync 8%"
metadata:
  node_type: memory
  type: project
  originSessionId: e0c3d6dc-5207-40da-aea3-893c738981a0
---

Live py-spy profile (rate 300, 40s + 8 dumps) of the ACTUAL i2t B32 decode worker
(PID varies; the 62GB Thinker child on GPU7 — NOT the `serve` parent, whose threads
are all idle asyncio; NOT the 33GB GPU6 worker which sits at 0% GPU during decode).
Sustained load = `--ignore-eos --output-len 768 --max-concurrency 32` food101 i2t.
KEY: GPU6 0% while GPU7 decodes → the 30B Thinker decodes SINGLE-GPU, so the entire
per-step cost is ONE GIL-bound Python thread (~100% of one core). That IS the floor.

Self-time leaf ranking (mstar-new/mstar/worker/worker.py, current checkout anchors):
- **18% `_prematerialize_for_check_stop` (worker.py:1871)** — D->H materialize new
  tokens every step for CPU stop-check. #1 hotspot.
- **~13% `_send_outputs`->`send_pyobj`** (worker.py:946) — pickle+zmq per step.
- **8% `torch.cuda…synchronize`** (worker.py:1660 completion_event / :1664 default_stream)
  — hard per-step barrier before check_stop.
- **8% run() loop body** (worker.py:1949).
- **~5% plan replay/route** (tensors.py: _replay_populate_plan/_plan_matches/
  store_and_populate_graph_edges/_replay_route_plan).
- ~1.5% `[edge.clone()…]`, ~1.2% `uuid4` PER OUTPUT PER REQUEST PER STEP (waste).
Inclusive: `_postprocess_batch` (worker.py:1603-1762) = 23%, 100% serial per-request
Python for-loop (stop-loop zmq peer sends + store_and_populate_graph_edges + uuid
set-comp + edge clones + process_node_outputs).

Thesis: M* returns to Python + host-sync BETWEEN every decode step for 3 reasons —
re-plan FlashInfer, rebuild positions, postprocess/check-stop/route on engine thread.
vLLM does all 3 in-graph or off-thread. Fix = R1 exile postprocess (biggest immediate,
no numerics risk) + R2 capturable plan-advance (keystone; unblocks the already-byte-
identical MSTAR_DECODE_MULTISTEP). Full 10-refactor plan w/ vLLM file:line vs M*
file:line committed: benchmarks/qwen3-omni-joint/AR_LOOP_10_REFACTORS.md (bench branch
encoders-implemeneted-benchmarked-mstar-v2 @ff57e61f). Dumps in /home/tim/tmp/decode_prof3
(uncommitted, regenerable). See [[project_mstar_decode_bottleneck]],
[[project_mstar_sidecar_checkstop]], [[project_mstar_throughput_levers]],
[[project_mstar_vllm_mrv2]].
