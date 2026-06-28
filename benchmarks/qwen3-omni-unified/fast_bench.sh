#!/usr/bin/env bash
#
# fast_bench.sh — FAST TARGETED Qwen3-Omni go/no-go runner (DRAFT, do not run on GPU yet).
#
# Point-targeted: ONE path, ONE batch, ONE system. Minimum wall-clock. Not a sweep.
# Drives the SINGLE shared configuration entrypoint (`python -m benchmark.runner`,
# selected purely by --request-type) for all five paths via one PATHS table.
#
# Complies with CLAUDE.md: hard timeout, cleanup trap on every exit path, GPU
# monitoring, warmup-then-capture, env capture, phase-tagged raw datapoints with
# units + seed, only-complete-runs status. It does NOT commit (prototyping tool).
#
# Usage:
#   fast_bench.sh --path I2T --batch 1 [--warmup 3] [--measure 3] [--seed 0] \
#                 [--system ours] [--url http://127.0.0.1:8000] [--gpus 0,2,3] \
#                 [--out <dir>] [--max-wall 1200]
set -uo pipefail

# ---- arg parsing ------------------------------------------------------------
PATH_SHORT=""; BATCH=1; WARMUP=3; MEASURE=3; SEED=0
SYSTEM="ours"; URL="${URL:-http://127.0.0.1:8000}"; GPUS="${CUDA_VISIBLE_DEVICES:-0,2,3}"
OUT=""; MAX_WALL="${MAX_WALL:-1200}"
while [ $# -gt 0 ]; do
  case "$1" in
    --path)    PATH_SHORT="$2"; shift 2;;
    --batch)   BATCH="$2"; shift 2;;
    --warmup)  WARMUP="$2"; shift 2;;
    --measure) MEASURE="$2"; shift 2;;
    --seed)    SEED="$2"; shift 2;;
    --system)  SYSTEM="$2"; shift 2;;
    --url)     URL="$2"; shift 2;;
    --gpus)    GPUS="$2"; shift 2;;
    --out)     OUT="$2"; shift 2;;
    --max-wall) MAX_WALL="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[ -n "$PATH_SHORT" ] || { echo "ERROR: --path is required (I2T|S2T|I2S|S2S|T2S)" >&2; exit 2; }

# ---- the ONE shared path table (short -> request-type / dataset / modality) --
declare -A REQTYPE=( [I2T]=image_to_text [S2T]=audio_to_text [I2S]=image_to_speech [S2S]=audio_to_speech [T2S]=text_to_speech )
declare -A DATASET=( [I2T]=food101 [S2T]=libri [I2S]=food101 [S2S]=libri [T2S]=text )
declare -A MODALITY=( [I2T]=text [S2T]=text [I2S]=speech [S2S]=speech [T2S]=speech )
RT="${REQTYPE[$PATH_SHORT]:-}"; DS="${DATASET[$PATH_SHORT]}"; MOD="${MODALITY[$PATH_SHORT]}"
[ -n "$RT" ] || { echo "ERROR: bad --path '$PATH_SHORT' (I2T|S2T|I2S|S2S|T2S)" >&2; exit 2; }

OUT="${OUT:-/home/tim/bench-wt/benchmarks/qwen3-omni-joint/fast/${PATH_SHORT}_B${BATCH}}"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPUS"
# Speech paths need a non-greedy thinker or the talker gets empty embeds and crashes.
if [ "$MOD" = "speech" ]; then export BENCH_SPEECH_THINKER_TEMPERATURE="${BENCH_SPEECH_THINKER_TEMPERATURE:-0.7}"; fi

# ---- env capture (automated, never hand-written) ----------------------------
{
  echo "=== date (UTC) ==="; date -u +%Y%m%dT%H%M%SZ
  echo "=== uname ==="; uname -a
  echo "=== CUDA_VISIBLE_DEVICES ==="; echo "$CUDA_VISIBLE_DEVICES"
  echo "=== seed ==="; echo "$SEED"
  echo "=== fast_bench args ==="; echo "path=$PATH_SHORT req_type=$RT dataset=$DS modality=$MOD batch=$BATCH warmup=$WARMUP measure=$MEASURE system=$SYSTEM url=$URL"
  echo "=== nvidia-smi query ==="
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm,persistence_mode --format=csv 2>/dev/null || echo "no nvidia-smi"
  echo "=== nvcc ==="; nvcc --version 2>/dev/null || echo "no nvcc"
  echo "=== torch cuda ==="
  python -c "import torch;print('torch',torch.__version__);print('cuda',torch.version.cuda)" 2>/dev/null || echo "no torch"
  echo "=== git ==="; git -C /home/tim/ttft-wt rev-parse HEAD 2>/dev/null || true
} > "$OUT/env.txt" 2>&1

# ---- precondition: devices idle (CLAUDE.md: never co-locate) -----------------
# Confirm nothing else is on the chosen GPUs before launch; stop+report if busy.
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null | tee "$OUT/preflight_gpu.txt" >&2 || true
# (Agent: inspect preflight_gpu.txt; abort if any foreign PID sits on $GPUS.)

# ---- clock teardown (mandatory if ever locked; idempotent post-check) --------
LOCKED=0   # this runner does not lock clocks; teardown stays a safe no-op.
teardown() {
  if [ "${LOCKED:-0}" -eq 1 ]; then
    nvidia-smi -rgc >/dev/null 2>&1 || true
    nvidia-smi -rmc >/dev/null 2>&1 || true
    nvidia-smi -pm 0 >/dev/null 2>&1 || true
  fi
  # idempotent post-check regardless of whether we locked
  nvidia-smi -rgc >/dev/null 2>&1 || true
}

# ---- GPU monitor (poll cadence scaled to modality) --------------------------
POLL=$([ "$MOD" = "speech" ] && echo 60 || echo 30)
MONLOG="$OUT/gpu_monitor.csv"
( while true; do
    echo "$(date -u +%H:%M:%SZ),$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ';')" >> "$MONLOG"
    sleep "$POLL"
  done ) &
MONPID=$!

# ---- cleanup trap: fires on success, failure, timeout, signal ---------------
JOBPID=""
cleanup() {
  [ -n "$JOBPID" ] && kill -- -"$JOBPID" 2>/dev/null || true   # kill job's whole process group
  kill "$MONPID" 2>/dev/null || true
  teardown
}
trap cleanup EXIT INT TERM

# ---- warmup + measure via the SINGLE entrypoint -----------------------------
# Warmup is run by the runner itself (--num-warmup); per_request holds only
# measured requests. Closed-loop, concurrency == batch (point latency at that B).
echo "[fast] $PATH_SHORT ($RT) B=$BATCH system=$SYSTEM warmup=$WARMUP measure=$MEASURE seed=$SEED -> $OUT" >&2

# ============================ TODO (env-specific) ============================
# Fill in the venv python, model-cache env, and confirm the server at $URL is up.
# Everything else (timeout, process group, args) is wired. Example shape:
#
#   PY=/home/tim/vllm-layout-wt/.venv/bin/python
#   export HF_HOME=/m-coriander/coriander/hf HF_HUB_OFFLINE=1 OPENAI_API_KEY=EMPTY
#   # health check: curl -fsS "$URL/health" || { echo "server down"; exit 3; }
#
PY="${PY:-python}"   # TODO: replace with the real venv python
RUN_CMD=( "$PY" -m benchmark.runner
  --url "$URL" --model qwen3omni --inference-system "$SYSTEM"
  --request-type "$RT" --dataset "$DS"
  --profiling-type closed_loop --batch-size "$BATCH" --max-concurrency "$BATCH"
  --num-warmup "$WARMUP" --num-requests "$MEASURE"
  --output-len-seed "$SEED"
  --output-dir "$OUT" )
# ============================================================================

# Hard timeout + own process group so the ceiling fires even if the agent dies.
timeout "$MAX_WALL" setsid "${RUN_CMD[@]}" > "$OUT/run.log" 2>&1 &
JOBPID=$!
wait "$JOBPID"; RC=$?

# ---- build tiny phase-tagged raw.json + headline from results.json ----------
python - "$OUT" "$PATH_SHORT" "$RT" "$MOD" "$SYSTEM" "$BATCH" "$WARMUP" "$MEASURE" "$SEED" "$RC" <<'PY'
import json, os, sys, datetime
out, short, rt, mod, system, batch, warmup, measure, seed, rc = sys.argv[1:11]
batch=int(batch); warmup=int(warmup); measure=int(measure); seed=int(seed); rc=int(rc)
rj = os.path.join(out, "results.json")
status = "complete" if (rc == 0 and os.path.exists(rj)) else "incomplete"
SR = 24000
datapoints=[]; agg={}
if os.path.exists(rj):
    r = json.load(open(rj))
    agg = {"ttft": r.get("ttft"), "itl": r.get("itl"), "rtf": r.get("rtf"),
           "request_throughput": r.get("request_throughput"),
           "text_token_throughput": r.get("text_token_throughput"),
           "audio_seconds_throughput": r.get("audio_seconds_throughput"),
           "wall_time_s": r.get("wall_time_s"), "completed": r.get("completed")}
    for pr in r.get("per_request", []):
        ob = pr.get("output_bytes", {}) or {}
        ab = ob.get("audio_bytes", 0) or 0
        asec = ab / (SR * 2) if ab else 0.0
        jct = pr.get("jct_ms", 0.0)
        rtf = (jct/1000.0)/asec if asec else None
        datapoints.append({"system": system, "batch": batch, "phase": "measure",
            "request_id": pr.get("request_id"), "jct_ms": jct, "audio_seconds": asec,
            "rtf": rtf, "text_bytes": ob.get("text_bytes", 0), "sample_rate": SR,
            "audio_seconds_method": "output_bytes.audio_bytes / (sample_rate * 2)  # 24kHz int16 mono"})
    # completion guard: need all measured requests present
    if len(datapoints) < measure:
        status = "incomplete"

raw = {"benchmark": f"qwen3-omni-fast-{short.lower()}-B{batch}", "path": rt,
       "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
       "timestamp_utc": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
       "seed": seed, "warmup_iters": [warmup], "status": status,
       "units": {"jct_ms": "ms", "rtf": "ratio (wall/audio)", "audio_seconds": "s",
                 "ttft": "s", "itl": "s", "throughput_metric": "req/s | tok/s | audio-s/s",
                 "sample_rate": SR},
       "datapoints": datapoints, "aggregate": agg}
json.dump(raw, open(os.path.join(out, "raw.json"), "w"), indent=2)

# headline (one line, go/no-go)
def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict): return None
        d = d.get(k)
    return d
if status != "complete":
    print(f"HEADLINE {short} B{batch}: INCOMPLETE (rc={rc}) — not a valid run, do not use")
elif mod == "text":
    print(f"HEADLINE {short} B{batch}: TTFT_p50={g(agg,'ttft','p50')}s "
          f"ITL_mean={g(agg,'itl','mean')}s req/s={agg.get('request_throughput')} "
          f"tok/s={agg.get('text_token_throughput')} [seed={seed} n={len(datapoints)}]")
else:
    rtfs=sorted(d['rtf'] for d in datapoints if d.get('rtf') is not None)
    rtf_p50 = rtfs[len(rtfs)//2] if rtfs else None
    print(f"HEADLINE {short} B{batch}: RTF_p50={rtf_p50} audio-s/s={agg.get('audio_seconds_throughput')} "
          f"TTFT_p50={g(agg,'ttft','p50')}s [seed={seed} n={len(datapoints)}]")
PY

echo "[fast] done rc=$RC out=$OUT" >&2
exit "$RC"
