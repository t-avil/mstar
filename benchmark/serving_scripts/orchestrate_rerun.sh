#!/usr/bin/env bash
# orchestrate_rerun.sh -- fan out the Qwen3-Omni headline re-run across ALL idle
# H200 pairs in parallel, one ISOLATED unified-build server per pair (#131).
#
# This is the re-run driver for the UNIFIED entrypoint: every serving config is
# a named OptimizationConfig preset (baseline|native|full|...), launched via the
# same MSTAR_* switchboard the production server reads (see serve_mstar.sh /
# mstar/model/qwen3_omni/config.py). One code path for prod + ablations.
#
# Guarantees (per /home/tim/CLAUDE.md "GPU runtime hygiene"):
#   * device isolation  : whole GPU pairs, CUDA_VISIBLE_DEVICES pinned, never
#                         co-located -- a pair is confirmed idle before launch.
#   * hard ceiling      : each measured run is wrapped in `timeout`.
#   * PID-only cleanup  : servers are killed by their UNIQUE socket-prefix match
#                         (process-scoped), never a broad pkill, on every exit.
#   * auto-recorded     : per preset we capture env.txt + requirements.txt, and
#                         per (path,batch) a command.txt with the exact command.
#   * resumable         : any (path,batch) whose results.json exists is skipped;
#                         a preset whose whole matrix exists never starts a server.
#
# NO GPU EXECUTION here unless you run it; this file just encodes the procedure.
#
# Usage:
#   bash orchestrate_rerun.sh                       # auto-discover idle pairs
#   GPU_PAIRS="1,0 3,2" PRESETS="baseline full" bash orchestrate_rerun.sh
#   DRY_RUN=1 bash orchestrate_rerun.sh             # print the plan, launch nothing
set -uo pipefail

# --------------------------------------------------------------------------- #
# Config (all env-overridable; defaults match this workspace).
# --------------------------------------------------------------------------- #
MSTAR_WT="${MSTAR_WT:-/home/tim/qwen3-omni-unified-wt}"
LAUNCH="${LAUNCH:-/home/tim/launch_mstar_wt.sh}"
CONFIG="${CONFIG:-configs/qwen3omni_2gpu.yaml}"   # 2 GPUs per pair
BENCH_PY="${BENCH_PY:-/home/tim/vllm-layout-wt/.venv/bin/python}"  # client venv
OUT_ROOT="${OUT_ROOT:-/home/tim/exp_unified/rerun}"

PRESETS="${PRESETS:-baseline native full}"
# S2T=audio_to_text I2T=image_to_text I2S=image_to_speech S2S=audio_to_speech
PATHS="${PATHS:-audio_to_text image_to_text image_to_speech audio_to_speech}"
BATCHES="${BATCHES:-1 2 4 8 16 32}"
NWARM="${NWARM:-5}"
NREQ="${NREQ:-10}"
INFER_SYS="${INFER_SYS:-ours}"

# Per-path dataset. S2* -> libri, I2* -> food101.
declare -A DS=( [audio_to_text]=libri [audio_to_speech]=libri \
                [image_to_text]=food101 [image_to_speech]=food101 )

# Device discovery / isolation.
IDLE_MEM_MIB="${IDLE_MEM_MIB:-100}"   # a GPU with < this MiB used is "idle"
EXCLUDE_GPUS="${EXCLUDE_GPUS:-7}"     # GPU 7 belongs to another user (protocol)
PORT_BASE="${PORT_BASE:-8100}"
SOCK_ROOT="${SOCK_ROOT:-/home/tim/tmp}"

# Timeouts / polling.
READY_TIMEOUT="${READY_TIMEOUT:-900}"   # s to wait for a server to come up
RUN_TIMEOUT="${RUN_TIMEOUT:-1800}"      # s hard ceiling per (path,batch) run
POLL_SEC="${POLL_SEC:-30}"              # nvidia-smi monitor cadence

# HF / dataset caches (client side).
export HF_HOME="${HF_HOME:-/m-coriander/coriander/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/tim/hf_datasets}"
LOCAL_CACHE="${LOCAL_CACHE:-/home/tim/tmp/libri_wavs}"
# Speech paths need a non-empty thinker temperature to avoid the empty-thinker
# crash (AGENT_PROTOCOL). Applied to the CLIENT request env.
export BENCH_SPEECH_THINKER_TEMPERATURE="${BENCH_SPEECH_THINKER_TEMPERATURE:-0.7}"

DRY_RUN="${DRY_RUN:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT_ROOT" "$SOCK_ROOT"
SERVERS_FILE="$(mktemp "$SOCK_ROOT/rerun_servers.XXXXXX")"   # unique socks to clean

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

numa_of() { [ "$1" -lt 4 ] && echo 0 || echo 1; }   # GPUs 0-3 numa0, 4-7 numa1

gpu_mem_used() {  # MiB used on a single GPU index
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1" 2>/dev/null \
    | tr -d ' '
}

is_excluded() { case " $EXCLUDE_GPUS " in *" $1 "*) return 0;; *) return 1;; esac; }

# Echo space-separated idle GPU indices (mem < IDLE_MEM_MIB, not excluded).
idle_gpus() {
  local idx used out=""
  for idx in $(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null); do
    is_excluded "$idx" && continue
    used="$(gpu_mem_used "$idx")"; [ -z "$used" ] && continue
    if [ "$used" -lt "$IDLE_MEM_MIB" ]; then out+="$idx "; fi
  done
  echo "$out"
}

# Pair idle GPUs WITHIN the same NUMA node (never straddle nodes), echo "a,b".
discover_pairs() {
  local g prev n0="" n1=""
  for g in $(idle_gpus | tr ' ' '\n' | sort -n); do
    if [ "$(numa_of "$g")" = 0 ]; then n0+="$g "; else n1+="$g "; fi
  done
  local node arr i
  for node in "$n0" "$n1"; do
    arr=($node); i=0
    while [ $((i+1)) -lt ${#arr[@]} ]; do
      echo "${arr[i]},${arr[i+1]}"; i=$((i+2))
    done
  done
}

# preset -> unified MSTAR_* flag list (ENV=VAL ...), passed to launch_mstar_wt.sh.
preset_flags() {
  case "$1" in
    baseline) echo "MSTAR_QWEN3_NATIVE_AUDIO_ENCODER=0 MSTAR_QWEN3_NATIVE_VISION_ENCODER=0 MSTAR_GPU_MEL=0 MSTAR_GPU_IMAGE_PREPROCESS=0 MSTAR_VLLM_PROMPT_LAYOUT=0 MSTAR_VLLM_AUDIO_SENTINELS=0" ;;
    native)   echo "MSTAR_QWEN3_NATIVE_AUDIO_ENCODER=1 MSTAR_QWEN3_NATIVE_VISION_ENCODER=1 MSTAR_GPU_MEL=0 MSTAR_GPU_IMAGE_PREPROCESS=0 MSTAR_VLLM_PROMPT_LAYOUT=0" ;;
    full)     echo "MSTAR_QWEN3_NATIVE_AUDIO_ENCODER=1 MSTAR_QWEN3_NATIVE_VISION_ENCODER=1 MSTAR_GPU_MEL=1 MSTAR_GPU_IMAGE_PREPROCESS=1 MSTAR_VLLM_PROMPT_LAYOUT=1" ;;
    *) echo "__UNKNOWN__" ;;
  esac
}

# Does a preset already have every (path,batch) result? (skip serving if so)
preset_complete() {
  local preset="$1" path bs
  for path in $PATHS; do for bs in $BATCHES; do
    [ -f "$OUT_ROOT/$preset/$path/bs$bs/results.json" ] || return 1
  done; done
  return 0
}

# Kill a server by its UNIQUE socket prefix (PID-scoped, never broad pkill).
cleanup_server() {
  local sock="$1" pid
  for pid in $(pgrep -f "$sock" 2>/dev/null); do
    kill -- -"$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ')" 2>/dev/null || kill "$pid" 2>/dev/null || true
  done
}

# Wait until the server answers on its port (or time out).
wait_ready() {
  local port="$1" deadline=$(( $(date +%s) + READY_TIMEOUT ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -s -o /dev/null "http://127.0.0.1:$port/v1/models" 2>/dev/null \
       || (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
      return 0
    fi
    sleep 5
  done
  return 1
}

# --------------------------------------------------------------------------- #
# Per-pair worker: serve each assigned preset, sweep path x batch, tear down.
# Args: pair("a,b")  port  preset...
# --------------------------------------------------------------------------- #
run_pair() {
  local pair="$1"; shift
  local port="$1"; shift
  local gpus="$pair" numa; numa="$(numa_of "${pair%%,*}")"
  local a="${pair%%,*}" b="${pair##*,}"

  for preset in "$@"; do
    local pdir="$OUT_ROOT/$preset"
    mkdir -p "$pdir"
    if preset_complete "$preset"; then
      log "pair $pair: preset '$preset' already complete -> skip"; continue
    fi

    # Re-confirm BOTH GPUs idle right before launch (never co-locate).
    local ua ub; ua="$(gpu_mem_used "$a")"; ub="$(gpu_mem_used "$b")"
    if [ -z "$ua" ] || [ -z "$ub" ] || [ "$ua" -ge "$IDLE_MEM_MIB" ] || [ "$ub" -ge "$IDLE_MEM_MIB" ]; then
      log "pair $pair: NOT idle (used ${ua}/${ub} MiB) -> SKIP preset '$preset' (reporting, no fallback)"
      continue
    fi

    local sockn="rerun-${preset}-${port}" sock="$SOCK_ROOT/rerun-${preset}-${port}"
    local slog="$pdir/server.log"
    local flags; flags="$(preset_flags "$preset")"
    [ "$flags" = "__UNKNOWN__" ] && { log "unknown preset '$preset' -> skip"; continue; }

    echo "$sock" >> "$SERVERS_FILE"
    capture_env_for "$pdir" "$gpus"

    log "pair $pair numa$numa: launch preset='$preset' port=$port sock=$sock"
    log "  flags: $flags"
    if [ "$DRY_RUN" = 1 ]; then
      log "  [dry-run] $LAUNCH $MSTAR_WT $gpus $numa $port $sockn $slog $flags"
    else
      # launch_mstar_wt.sh pins the 2-GPU config (configs/qwen3omni_2gpu.yaml)
      # and CUDA_VISIBLE_DEVICES/NUMA; we only pass the unified MSTAR_* flags.
      bash "$LAUNCH" "$MSTAR_WT" "$gpus" "$numa" "$port" "$sockn" "$slog" $flags
      if ! wait_ready "$port"; then
        log "pair $pair: server '$preset' did NOT become ready in ${READY_TIMEOUT}s -> cleanup+skip"
        cleanup_server "$sock"; continue
      fi
      log "pair $pair: server '$preset' ready on :$port"
    fi

    # ---- sweep path x batch (5 warmup + n=10 measured each) ----
    for path in $PATHS; do
      local ds="${DS[$path]:-libri}"
      for bs in $BATCHES; do
        local od="$pdir/$path/bs$bs"
        if [ -f "$od/results.json" ]; then
          log "  skip (done): $preset/$path/bs$bs"; continue
        fi
        mkdir -p "$od"
        local cmd=( "$BENCH_PY" -m benchmark.runner
          --url "http://127.0.0.1:$port" --model qwen3omni
          --inference-system "$INFER_SYS"
          --request-type "$path" --dataset "$ds"
          --num-requests "$NREQ" --num-warmup "$NWARM" --batch-size "$bs"
          --profiling-type closed_loop --max-concurrency "$bs"
          --local-cache "$LOCAL_CACHE" --output-dir "$od" )
        # AUTO-RECORD the exact command + context for this datapoint set.
        { echo "# preset=$preset path=$path bs=$bs gpus=$gpus port=$port"; echo "# utc=$(date -u +%Y%m%dT%H%M%SZ)";
          printf '%q ' "${cmd[@]}"; echo; } > "$od/command.txt"
        log "  >>> $preset $path bs=$bs (warmup=$NWARM n=$NREQ)"
        if [ "$DRY_RUN" = 1 ]; then log "  [dry-run] skip exec"; continue; fi
        if timeout "$RUN_TIMEOUT" "${cmd[@]}" > "$od/run.log" 2>&1; then
          log "  OK  $preset $path bs=$bs"
        else
          log "  FAIL/timeout $preset $path bs=$bs (see $od/run.log) -> not committed"
        fi
      done
    done

    # ---- teardown: free the pair the instant this preset's sweep ends ----
    log "pair $pair: teardown server '$preset'"
    cleanup_server "$sock"
    sleep 3
  done
}

capture_env_for() {  # <dir> <gpus>
  local dir="$1" gpus="$2"
  if [ "$DRY_RUN" = 1 ]; then return 0; fi
  CUDA_VISIBLE_DEVICES="$gpus" bash "$HERE/capture_env.sh" "$dir" "$MSTAR_WT" || true
}

# --------------------------------------------------------------------------- #
# Plan: discover pairs, distribute presets round-robin, fan out.
# --------------------------------------------------------------------------- #
PAIRS="${GPU_PAIRS:-$(discover_pairs)}"
if [ -z "${PAIRS// }" ]; then
  log "no idle GPU pairs found (excluded: $EXCLUDE_GPUS). nvidia-smi:"; nvidia-smi 2>/dev/null | sed 's/^/   /'
  exit 1
fi
PAIRS_ARR=($PAIRS); PRESETS_ARR=($PRESETS)
log "idle pairs: ${PAIRS_ARR[*]}"
log "presets   : ${PRESETS_ARR[*]}"
log "paths     : $PATHS"
log "batches   : $BATCHES   (warmup=$NWARM measured=$NREQ)"
log "out root  : $OUT_ROOT"

# Round-robin presets onto pairs -> one queue per pair.
declare -A QUEUE
for i in "${!PRESETS_ARR[@]}"; do
  p=$(( i % ${#PAIRS_ARR[@]} ))
  QUEUE[$p]+="${PRESETS_ARR[i]} "
done

# Global cleanup: kill every server we launched, on any exit path.
cleanup_all() {
  log "cleanup_all: tearing down launched servers"
  if [ -f "$SERVERS_FILE" ]; then
    while read -r s; do [ -n "$s" ] && cleanup_server "$s"; done < "$SERVERS_FILE"
    rm -f "$SERVERS_FILE"
  fi
}
trap cleanup_all EXIT INT TERM

# Launch one background worker per pair (parallel fan-out).
declare -a WPIDS=()
for p in "${!PAIRS_ARR[@]}"; do
  pair="${PAIRS_ARR[p]}"; port=$(( PORT_BASE + p ))
  presets_for_pair="${QUEUE[$p]:-}"
  [ -z "${presets_for_pair// }" ] && continue
  log "ASSIGN pair=$pair port=$port presets=[ $presets_for_pair]"
  run_pair "$pair" "$port" $presets_for_pair &
  WPIDS+=("$!")
done

# Monitor: poll nvidia-smi for the active pairs while workers run (observability;
# the durable ceiling is the per-run `timeout`, not this loop).
( while kill -0 "${WPIDS[0]}" 2>/dev/null; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null \
      | sed "s/^/[$(date -u +%H:%M:%S) gpu] /" >> "$OUT_ROOT/monitor.log"
    sleep "$POLL_SEC"
  done ) &
MON=$!

rc=0
for pid in "${WPIDS[@]}"; do wait "$pid" || rc=1; done
kill "$MON" 2>/dev/null || true
cleanup_all; trap - EXIT INT TERM

log "ALL PAIRS DONE (rc=$rc). Artifacts under $OUT_ROOT/<preset>/<path>/bs<N>/"
log "Each valid run = results.json + raw datapoints + command.txt; env.txt per preset."
exit "$rc"
