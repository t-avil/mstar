# SMOKE — M* speech host-floor bundle (opt/speech-floor)

GPU A/B recipes for the two flag-gated items on this branch. **Code-only branch;
no GPU was used to produce it.** All commands assume the standard mstar bench
env (SHM protocol, `HF_HOME=/m-coriander/coriander/hf`, `ninja` on PATH,
`PYTHONPATH=<this worktree>` so spawned GPU workers load THIS worktree's code —
without it the A/B is invalid, see the worktree-PYTHONPATH trap).

Fixed setup for every cell below:

```bash
WT=/m-coriander/coriander/tim/mstar-speech
PY=/m-coriander/coriander/tim/mstar-new/.venv/bin/python   # or the bench venv
GPUS=${GPUS:-5,6}                 # one fixed pair for the whole session
PORT=${PORT:-8230}
SOCK=/home/tim/tmp/sk_speechfloor_${PORT}
LIBRI=/home/tim/tmp/libri_wavs    # reuse; do not re-download
export PYTHONPATH=$WT HF_HOME=/m-coriander/coriander/hf
# Confirm the devices are idle BEFORE launch (CLAUDE.md GPU-selection rule):
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -i $GPUS
```

Serve (base pattern; per-item env/flags differ):

```bash
setsid env CUDA_VISIBLE_DEVICES=$GPUS HF_HOME=/m-coriander/coriander/hf \
  PYTHONPATH=$WT <ITEM_FLAGS> \
  timeout 5400 $PY -m mstar.cli.main serve qwen3_omni \
    --gpus $GPUS --port $PORT --tensor-comm-protocol SHM \
    --socket-path-prefix $SOCK <--config ...> > $SERVER_LOG 2>&1 &
SERVER_PID=$!
```

Bench one cell (B=8; s2s = `audio_to_speech`, i2s = `image_to_speech`):

```bash
# request-type: audio_to_speech (s2s) | image_to_speech (i2s)
timeout 1800 $PY -m benchmark.runner \
  --url http://127.0.0.1:$PORT --model qwen3omni \
  --request-type <audio_to_speech|image_to_speech> \
  --dataset libri --profiling-type closed_loop \
  --max-concurrency 8 --num-requests 80 --num-warmup 5 \
  --inference-system ours --local-cache $LIBRI \
  --output-dir $ODIR
```

Always record the full metric set from `results.json` (TTFT/first-audio, ITL
audio, RTF, JCT, request + audio-seconds throughput) — not just the one metric
an item targets. Free `*.wav` after each cell to protect `/home`.

Hard rules from CLAUDE.md that apply to every cell: one fixed GPU pair, confirm
idle before launch, `timeout` wrapper on server + bench, kill the process group +
free devices on every exit path, poll `nvidia-smi` while running and kill on
freeze, commit only complete runs.

---

## Item A — MSTAR_FAST_CHECKSTOP_TALKER (batched talker stop check)

**What it does.** On `talker_decode` steps, replaces the per-request
`layer0_codes.item()` host read in `TalkerSubmodule.check_stop` with one batched
D→H of just the layer0 code per rid + pure-int compares against
`codec_eos_token_id`. Walk-gated to `talker_decode` and flag-gated; thinker paths
untouched. Off path is the byte-identical engine `check_stop`.

**A/B = two servers**, flag off vs on. Flag is dynflags-refreshable, but for a
clean measurement use two-server alternation, not a mid-run flip.

- OFF server: `<ITEM_FLAGS>` = `MSTAR_WALK_STATS=1`
- ON  server: `<ITEM_FLAGS>` = `MSTAR_FAST_CHECKSTOP_TALKER=1 MSTAR_WALK_STATS=1`

Run cells: **s2s B=8** and **i2s B=8** against each server.

### Proof the mechanism fired

`MSTAR_WALK_STATS=1` logs the per-walk step counters (incl. the new one) every
200 steps at WARNING into the server log. The new counter is
`talker_fast_checkstop_steps`, bumped once per talker_decode step that took the
fast path.

```bash
# ON server: the counter must appear and climb with talker_decode steps.
grep -o 'talker_fast_checkstop_steps[^,}]*' $SERVER_LOG_ON | tail -3
grep 'WALK_STATS' $SERVER_LOG_ON | tail -2

# OFF server: the counter must NEVER appear (fast path never entered).
grep -c 'talker_fast_checkstop_steps' $SERVER_LOG_OFF   # expect 0
```

Cross-check that the fast path covers the talker decode volume: on the ON server,
`talker_fast_checkstop_steps` should track the `talker_decode` step count in the
same WALK_STATS line (they should be equal when every talker step is uniform;
any shortfall = steps that fell back to the slow copy, which is correct but
un-accelerated — investigate if large).

### Proof of correctness (stop decisions unchanged)

The stop set must be identical off vs on — same audio, same length. Compare the
two runs' outputs:

```bash
# Same number of generated codec frames / audio duration per request off vs on.
# (audio_seconds_throughput and per-request output length must match within
#  sampling noise; the stop condition is deterministic given identical tokens.)
$PY - <<'PY'
import json
off=json.load(open("$ODIR_OFF/results.json")); on=json.load(open("$ODIR_ON/results.json"))
for k in ("audio_seconds_throughput","request_throughput"):
    print(k, "off", off.get(k), "on", on.get(k))
PY
```

If token dumps are enabled, diff the per-request layer0 code sequences off vs on
under a fixed seed — they must be identical (the fast path only changes HOW the
stop token is read, not WHICH token, so greedy/seeded decodes match exactly).

### Proof of win

Compare talker-side latency off vs on at B=8 (the host `.item()` per frame is a
CPU-floor cost, so expect the gap to show in ITL audio / RTF, largest at higher
talker batch):

```bash
$PY - <<'PY'
import json
def itl(p): 
    d=json.load(open(p)); a=d.get("itl",{}).get("audio") or {}
    return a.get("p50",0)*1000, a.get("mean",0)*1000, d.get("audio_seconds_throughput",0)
print("OFF itl_p50/mean(ms), audio_s/s:", itl("$ODIR_OFF/results.json"))
print("ON  itl_p50/mean(ms), audio_s/s:", itl("$ODIR_ON/results.json"))
PY
```

Expected: ON ≤ OFF on ITL audio / RTF, neutral-to-better throughput; thinker-only
paths (i2t/s2t) unaffected. If ON regresses, the batched D→H probe cost is
outweighing the saved `.item()`s at this batch — record and flag (mirrors the E4b
Talker-tax caution that motivated the walk gate).

---

## Item B — configs/qwen3omni_2gpu_taco.yaml (Talker+Code2Wav colocation)

**IMPORTANT — read the yaml header first.** Investigation on this branch found:

1. M* spawns ONE process per rank (`conductor._launch_workers`). The default
   `qwen3omni_2gpu.yaml` (the qwen3_omni default) already puts Talker AND
   Code2Wav on rank 0 → they already share one process (`worker_0`).
2. `taco.yaml` produces a **byte-identical worker-graph decomposition** to
   `qwen3omni_2gpu.yaml` (verified on CPU via `get_worker_graphs`). It only
   states the colocation as one node_group; it is not a new capability.
3. A config **cannot** fuse them into one worker graph: both are streaming
   consumers and `_divide_into_worker_graphs` forces every `consumes_stream` node
   into its own worker graph. The codec_tokens edge is therefore already
   intra-process on both configs.

So the honest Item B smoke has two parts.

### B.1 — Sanity: taco == default (no regression, no change)

Serve once with `--config configs/qwen3omni_2gpu_taco.yaml` and once with the
default (no `--config`). s2s B=8 + i2s B=8 on each. Results must match within
noise — this only confirms the pinned layout serves and is equivalent.

```bash
# taco
... serve qwen3_omni --config $WT/configs/qwen3omni_2gpu_taco.yaml ...
# default (qwen3omni_2gpu.yaml)
... serve qwen3_omni ...
```

Confirm placement in each server log (both should show Talker and Code2Wav on
worker_0 / rank 0):

```bash
grep -iE 'worker_0|rank 0|Talker|Code2Wav' $SERVER_LOG | grep -i 'rank\|worker' | head
```

### B.2 — Mechanism probe: does the cross-process codec edge cost anything?

To actually measure the Talker→Code2Wav IPC that colocation removes, force them
onto SEPARATE ranks/processes and compare against taco. Within 2 GPUs, put
Code2Wav on rank 1 (with Thinker+encoders); Talker stays rank 0:

```bash
cat > /tmp/qwen3omni_2gpu_split.yaml <<'YAML'
model: "qwen3_omni"
max_seq_len: 32768
# PROBE ONLY (not for shipping): Talker (rank 0) and Code2Wav (rank 1) on
# different ranks -> different processes -> codec_tokens crosses a real
# worker->worker boundary. Isolates the IPC that taco/2gpu.yaml already avoid.
node_groups:
  - node_names: [audio_encoder, vision_encoder]
    ranks: [1]
  - node_names: [Thinker]
    ranks: [1]
  - node_names: [Code2Wav]
    ranks: [1]
  - node_names: [Talker]
    ranks: [0]
YAML
# serve taco  (colocated, --config .../qwen3omni_2gpu_taco.yaml)
# serve split (--config /tmp/qwen3omni_2gpu_split.yaml)
# s2s B=8 + i2s B=8 on each; compare ITL audio / RTF / first-audio latency.
```

Interpretation:
- taco ≈ split → the codec edge is not a bottleneck at this batch; colocation
  buys nothing measurable and the default already has it. Report as such.
- taco < split (better) → quantifies the cross-process codec IPC. Since the
  default already colocates, the takeaway is a guardrail ("keep Talker+Code2Wav
  co-ranked"), plus a pointer that squeezing the *residual intra-process* edge
  needs a code change (fused in-graph Talker→Code2Wav, or non-streaming
  Code2Wav), not a yaml.

VRAM: taco's rank 0 holds only Talker + Code2Wav (small) — never tight on H200
(143GB); rank 1 (30B-A3B Thinker MoE + encoders) is the heavy GPU. The split
probe adds Code2Wav to the already-heavy rank 1 — watch rank 1 memory there.

---

## Item C — MSTAR_CODEC_CHUNK_EMIT (chunk-batched codec edge handoff)

**What it does.** The Talker emits one `[16]`-code frame per AR step onto the
`codec_tokens` StreamingGraphEdge, so the colocated Code2Wav `StreamBuffer` takes
~25 individual `put`s + id→tensor dict churn per `LeftContextChunkPolicy(chunk=25,
left_context=25)` window. When ON, local-route codec frames are STAGED and
written into the buffer in ONE batched put per chunk boundary
(`StreamBuffer.stage` + `flush_pending`). Registration (`pre_read_register`) stays
per-frame, so the buffered item sequence — and every popped window — is byte
identical, and coalescing at the policy's `chunk` granularity means a chunk
becomes ready at the same frame count (first-audio timing preserved).

Gating: default OFF, dynflags-refreshable, EDGE-gated via `policy.coalesce_size()>1`
— only the codec edge (LeftContextChunkPolicy) opts in; `thinker_states`/
`thinker_mask` (FixedChunkPolicy chunk=1) and every other stream are untouched.
Only the LOCAL (colocated) route is coalesced; a split/remote codec edge is
unaffected by the flag.

**A/B = two servers**, flag off vs on. Focus s2s B=8 (audio path; ITL/RTF).

- OFF server: `<FLAGS>` = `MSTAR_WALK_STATS=1`
- ON  server: `<FLAGS>` = `MSTAR_CODEC_CHUNK_EMIT=1 MSTAR_WALK_STATS=1`

Via the lab harness (dynflags file): add `MSTAR_CODEC_CHUNK_EMIT=1` to the ON
lab's `$FLAGS_FILE`, run `lab_ab.sh` with the `s2s:8` cell (i2s:8 optional). Use
the DEFAULT config (colocated) — this optimization only fires on the local codec
edge.

### Proof the mechanism fired

WALK_STATS counter `codec_chunk_emits` is bumped once per batched flush (~one per
25 codec frames per request). It must appear and climb on the ON server, never on
OFF:

```bash
grep -o 'codec_chunk_emits[^,}]*' $SERVER_LOG_ON | tail -3
grep -c 'codec_chunk_emits' $SERVER_LOG_OFF     # expect 0
```

Sanity on the rate: `codec_chunk_emits` should be roughly
`(total codec frames) / 25` — i.e. far fewer than the per-frame `put` count. If it
tracks the frame count 1:1, coalescing isn't engaging (check the edge policy /
that the codec edge is local).

### Proof of correctness (identical audio)

Windows are byte-identical by construction (covered by
`test/modular/test_codec_chunk_emit_parity.py`, which drives the real
StreamBuffer + LeftContextChunkPolicy per-frame vs coalesced and also checks an
independent HF-style slicing oracle). On GPU, confirm the produced audio matches
off vs on: same `audio_seconds_throughput` / per-request audio length within
sampling noise, and if token dumps are on, identical codec token sequences under
a fixed seed.

### Proof of win

```bash
$PY - <<'PY'
import json
def m(p):
    d=json.load(open(p)); a=d.get("itl",{}).get("audio") or {}
    return (a.get("p50",0)*1000, a.get("mean",0)*1000,
            d.get("audio_seconds_throughput",0), d.get("jct_mean_ms",0))
print("OFF itl_p50/mean(ms), audio_s/s, jct:", m("$ODIR_OFF/results.json"))
print("ON  itl_p50/mean(ms), audio_s/s, jct:", m("$ODIR_ON/results.json"))
PY
```

Honest expectation: this cuts the buffer-side per-frame `put` + dict churn (25→1
buffer writes per chunk), NOT the per-frame edge routing/registration
(`pre_read_register` + `get_tensor` + `clone` + `dereference` still run per frame
at arrival). So the win is bounded by how much of the audio-path CPU floor is the
buffer write vs the routing. If ITL/RTF barely moves, that's a real result: the
buffer churn wasn't the bottleneck, and the larger lever is producer emit-batching
(collapse the N routed edges per chunk into one) — a bigger change, flagged as
follow-up, not done here. Neutral-or-better is the pass bar; any regression →
record and flag.
