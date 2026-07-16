#!/usr/bin/env python3
"""4-in-1 head-to-head chart: M* (clean matched-protocol) vs vLLM-Omni 0.22.
Panels 2x2: tok/s | req/s | TTFT p50 (ms) | ITL mean (ms). Reads a sweep dir of
results.json (campaign matched-protocol, greedy natural-EOS). vLLM = committed 0.22.

Usage: gen_charts.py <sweep_results_dir> <path:i2t|s2t> <out.png> "<title>"
  sweep_results_dir contains <path>_B<b>/results.json
Shared style: chartstyle.mplstyle (workspace convention). M*=BLUE, vLLM=GREEN.
"""
import glob, json, os, sys, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(HERE, "chartstyle.mplstyle"))
BLUE, GREEN = "#1f77b4", "#2ca02c"

# committed vLLM-Omni 0.22 (authoritative; benchmarks branch chart_v10_h2h_data.json[vllm022])
VLLM = {
 'i2t': {1:(0.876,194.6,86.8,4.76),2:(1.514,321.8,92.4,5.58),4:(2.189,483.2,129.5,6.88),
         8:(3.665,770.0,100.0,9.56),16:(5.623,1185.1,168.1,11.88),32:(8.322,1768.8,178.6,15.42)},
 's2t': {1:(3.831,95.0,66.0,5.00),4:(13.225,328.0,60.0,9.00),8:(15.791,381.6,143.0,15.00),16:(19.798,472.4,191.0,23.00),32:(30.850,746.1,217.0,29.00)},
}  # tuple = (req_s, tok_s, ttft_ms, itl_ms)

def sec(d,m,k):
    return ((d.get(m) or {}).get("text") or {}).get(k)

def load_cell(f):
    d=json.load(open(f))
    return dict(req=d.get('request_throughput'), tok=d.get('text_token_throughput'),
               ttft=(sec(d,'ttft','p50') or 0)*1000, itl=(sec(d,'itl','mean') or 0)*1000)

def main():
    sweepdir, pth, out, title = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    cells={}
    for f in sorted(glob.glob(os.path.join(sweepdir, f"{pth}_B*/results.json"))):
        b=int(os.path.basename(os.path.dirname(f)).split("_B")[1])
        cells[b]=load_cell(f)
    batches=sorted(cells)
    vll=VLLM.get(pth,{})
    panels=[("tok",  "text token throughput (tok/s)  — higher better", 1),
            ("req",  "request throughput (req/s)  — higher better", 0),
            ("ttft", "TTFT p50 (ms)  — lower better", 2),
            ("itl",  "ITL mean (ms)  — lower better", 3)]
    fig,axes=plt.subplots(2,2,figsize=(11.0,8.0))
    for ax,(key,label,vi) in zip(axes.flat,panels):
        mstar=[cells[b][key] for b in batches]
        vllm=[vll.get(b,(None,)*4)[vi] for b in batches]
        ax.plot(batches, mstar, color=BLUE, marker='o', label='M* (this campaign)')
        if any(v is not None for v in vllm):
            ax.plot(batches, [v for v in vllm], color=GREEN, marker='s', ls='-', label='vLLM-Omni 0.22')
        ax.set_title(label); ax.set_xlabel("batch (max concurrency)")
        ax.set_xscale('log', base=2); ax.set_xticks(batches)
        ax.set_xticklabels([str(b) for b in batches])
        ax.legend()
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(out)
    print("wrote", out, "batches", batches)

if __name__=="__main__": main()
