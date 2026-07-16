#!/usr/bin/env bash
# b32_gpuprobe.sh <port> — run i2t B32 while sampling GPU6/7 util at 100ms to
# classify the B32 TTFT window: compute-bound (both GPUs ~high) vs wave/idle-bound.
set -uo pipefail
PORT="${1:-8346}"
OUT=/m-coriander/coriander/tim/campaign_i2t/results/pd/b32_probe
mkdir -p "$OUT"
# start high-freq GPU sampler in background
( for i in $(seq 1 120); do
    ts=$(date +%s.%N)
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits \
      | awk -F',' -v t="$ts" '($1+0==6||$1+0==7){print t,$0}'
    sleep 0.1
  done ) > "$OUT/gpuutil.log" 2>&1 &
SP=$!
# fire a single B32 wave (small n so it's a clean burst)
bash /m-coriander/coriander/tim/campaign_i2t/bench_cell.sh "$PORT" i2t 32 32 0 "$OUT" 2>&1 | tail -2
kill $SP 2>/dev/null || true
echo "=== GPU6/7 util distribution during run ==="
awk '{print $2","$3}' "$OUT/gpuutil.log" | sort -t, -k1,1n | \
 awk -F',' '{u[$1]=u[$1]" "$2; c[$1]++; s[$1]+=$2; if($2>m[$1])m[$1]=$2} END{for(g in s) printf "GPU%s: mean_util=%.0f%% max=%.0f%% samples=%d\n",g,s[g]/c[g],m[g],c[g]}'
echo "(GPU6=rank0 prefill/encode, GPU7=rank1 decode in PD)"
