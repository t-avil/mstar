#!/usr/bin/env bash
# Clean apples-to-apples re-bench: S2S + S2T(A2T) B=1 x50, seed=42, 3 systems in parallel.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"; export HF_HOME=/m-coriander/coriander/hf HF_DATASETS_CACHE=/home/tim/hf_datasets
cd /home/tim/mstar; source .venv/bin/activate
R=/home/tim/exp_rebench; ts(){ date -u +%Y%m%dT%H%M%SZ; }
echo "[$(ts)] waiting for 8093(vLLM) 8103(M*-new) 8104(M*-old)"
for i in $(seq 1 120); do
  a=$( (exec 3<>/dev/tcp/127.0.0.1/8093) 2>/dev/null&&echo 1||echo 0)
  b=$( (exec 3<>/dev/tcp/127.0.0.1/8103) 2>/dev/null&&echo 1||echo 0)
  c=$( (exec 3<>/dev/tcp/127.0.0.1/8104) 2>/dev/null&&echo 1||echo 0)
  [ "$a" = 1 ]&&[ "$b" = 1 ]&&[ "$c" = 1 ]&&{ echo "[$(ts)] all up";break;}; sleep 10
done
run_sys(){ local tag=$1 url=$2 isys=$3 numa=$4
  for path in audio_to_speech audio_to_text; do
    local OUT="$R/${tag}_${path}"; mkdir -p "$OUT"
    echo "[$(ts)] $tag $path B=1 x50"
    numactl --cpunodebind=$numa --membind=$numa timeout 1800 python -m benchmark.runner --url "$url" --model qwen3omni \
      --request-type "$path" --dataset libri --profiling-type closed_loop --max-concurrency 1 --num-requests 50 --num-warmup 5 \
      --inference-system "$isys" --local-cache /home/tim/tmp/libri_wavs --output-dir "$OUT" > "$OUT/stdout.txt" 2>&1
    mkdir -p "$OUT/samples"; ls "$OUT"/*.wav 2>/dev/null|sort -V|head -5|while read f; do mv "$f" "$OUT/samples/"; done
    rm -f "$OUT"/*.wav "$OUT"/req_*.txt 2>/dev/null
  done; echo "[$(ts)] $tag DONE"
}
run_sys mstarNEW http://127.0.0.1:8103 ours 1 &
run_sys mstarOLD http://127.0.0.1:8104 ours 0 &
run_sys vllm     http://127.0.0.1:8093 vllm_omni 0 &
wait
echo "[$(ts)] ===== AGGREGATE ====="
python3 - <<'PY'
import re,os
R="/home/tim/exp_rebench"
def g(f,pat):
    try:t=open(f).read()
    except:return None
    m=re.search(pat,t);return m.group(1) if m else None
print(f"{'sys':9} | {'path':4} | {'RTF mean/p50/p95/p99':24} | {'TTFT':7} | {'ITL':7} | {'E2E':7} | {'tput':6} | {'audio_s':7} | {'toks':5} | reqs")
for tag in ['mstarNEW','mstarOLD','vllm']:
  for path,short in [('audio_to_speech','S2S'),('audio_to_text','S2T')]:
    f=f"{R}/{tag}_{path}/stdout.txt"
    if not os.path.isfile(f): print(f"{tag:9} | {short:4} | (no file)");continue
    rtf=g(f,r'^RTF.*?mean=([0-9.]+).*?p50=([0-9.]+).*?p95=([0-9.]+).*?p99=([0-9.]+)')
    rtfs=re.search(r'^RTF.*?mean=([0-9.]+) +p50=([0-9.]+) +p95=([0-9.]+) +p99=([0-9.]+)',open(f).read(),re.M)
    rtfstr="/".join(rtfs.groups()) if rtfs else "-"
    ttft=g(f,r'TTFT.*?mean=([0-9.]+)s');itl=g(f,r'^ITL.*?mean=([0-9.]+)s');e2e=g(f,r'^E2E.*?mean=([0-9.]+)s')
    tput=g(f,r'([0-9.]+) audio sec/s');adur=g(f,r'Audio dur.*?mean=([0-9.]+)s');tok=g(f,r'Text tokens: ([0-9]+)');req=g(f,r'Requests : ([0-9]+/[0-9]+)')
    print(f"{tag:9} | {short:4} | {rtfstr:24} | {str(ttft):7} | {str(itl):7} | {str(e2e):7} | {str(tput):6} | {str(adur):7} | {str(tok):5} | {req}")
PY
echo "[$(ts)] REBENCH DONE"
