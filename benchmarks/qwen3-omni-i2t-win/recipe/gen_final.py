#!/usr/bin/env python3
"""Aggregate repeat cells -> median (+min/max) per batch, print table + 4-in-1 chart.
Usage: gen_final.py <repdir> <path:i2t|s2t> <out.png> "<title>"
 repdir contains <path>_B<b>_r*/results.json  (multiple repeats per batch)
M*=BLUE, vLLM=GREEN. Panels: tok/s | req/s | TTFT p50 ms | ITL mean ms.
Error bars = min..max across repeats (shows run-to-run variance / load sensitivity)."""
import glob, json, os, sys, collections, statistics
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(HERE,"chartstyle.mplstyle"))
BLUE,GREEN="#1f77b4","#2ca02c"
VLLM={'i2t':{1:(0.876,194.6,86.8,4.76),2:(1.514,321.8,92.4,5.58),4:(2.189,483.2,129.5,6.88),
             8:(3.665,770.0,100.0,9.56),16:(5.623,1185.1,168.1,11.88),32:(8.322,1768.8,178.6,15.42)},
      's2t':{1:(3.831,95.0,66.0,5.00),4:(13.225,328.0,60.0,9.00),8:(15.791,381.6,143.0,15.00),16:(19.798,472.4,191.0,23.00),32:(30.850,746.1,217.0,29.00)}}
def sec(d,m,k): return ((d.get(m) or {}).get("text") or {}).get(k)
def cell(f):
    d=json.load(open(f))
    return dict(req=d.get('request_throughput'),tok=d.get('text_token_throughput'),
                ttft=(sec(d,'ttft','p50') or 0)*1000, itl=(sec(d,'itl','mean') or 0)*1000,
                outb=statistics.mean([ (r.get('output_bytes') or {}).get('text',0) if isinstance(r.get('output_bytes'),dict) else (r.get('output_bytes') or 0) for r in d.get('per_request',[]) ] or [0]))

def main():
    repdir,pth,out,title=sys.argv[1:5]
    agg=collections.defaultdict(lambda: collections.defaultdict(list))
    seen=set()
    for f in sorted(set(glob.glob(os.path.join(repdir,f"{pth}_B*_r*/results.json"))+glob.glob(os.path.join(repdir,f"{pth}_B*/results.json")))):
        if os.path.realpath(f) in seen: continue
        seen.add(os.path.realpath(f))
        base=os.path.basename(os.path.dirname(f))
        b=int(base.split("_B")[1].split("_")[0])
        c=cell(f)
        for k,v in c.items(): agg[b][k].append(v)
    batches=sorted(agg)
    med={b:{k:statistics.median(agg[b][k]) for k in agg[b]} for b in batches}
    lo={b:{k:min(agg[b][k]) for k in agg[b]} for b in batches}
    hi={b:{k:max(agg[b][k]) for k in agg[b]} for b in batches}
    vll=VLLM.get(pth,{})
    # print table
    print(f"\n=== {title} — median of repeats (n per batch below) ===")
    print(f"{'B':>3} {'reps':>4} | {'req/s':>7} {'Δ%':>6} | {'tok/s':>7} {'Δ%':>6} | {'TTFT':>6} {'vLLM':>5} | {'ITL':>5} {'vLLM':>5} | outB")
    for b in batches:
        m=med[b]; v=vll.get(b)
        nrep=len(agg[b]['req'])
        dr=dt=None
        if v:
            dr=(m['req']/v[0]-1)*100; dt=(m['tok']/v[1]-1)*100
            print(f"{b:>3} {nrep:>4} | {m['req']:>7.2f} {dr:>+6.1f} | {m['tok']:>7.0f} {dt:>+6.1f} | {m['ttft']:>6.0f} {v[2]:>5.0f} | {m['itl']:>5.1f} {v[3]:>5.1f} | {m['outb']:.0f}")
        else:
            print(f"{b:>3} {nrep:>4} | {m['req']:>7.2f} {'':>6} | {m['tok']:>7.0f} {'':>6} | {m['ttft']:>6.0f} {'':>5} | {m['itl']:>5.1f} {'':>5} | {m['outb']:.0f}")
    # chart
    panels=[("tok","text token throughput (tok/s)  — higher is better",1),
            ("req","request throughput (req/s)  — higher is better",0),
            ("ttft","TTFT p50 (ms)  — lower is better",2),
            ("itl","ITL mean (ms)  — lower is better",3)]
    fig,axes=plt.subplots(2,2,figsize=(11.0,8.2))
    for ax,(key,label,vi) in zip(axes.flat,panels):
        y=[med[b][key] for b in batches]
        yerr=[[med[b][key]-lo[b][key] for b in batches],[hi[b][key]-med[b][key] for b in batches]]
        ax.errorbar(batches,y,yerr=yerr,color=BLUE,marker='o',capsize=3,label='M* PD (this campaign)')
        vv=[vll.get(b,(None,)*4)[vi] for b in batches]
        if any(x is not None for x in vv):
            ax.plot(batches,vv,color=GREEN,marker='s',ls='-',label='vLLM-Omni 0.22')
        ax.set_title(label); ax.set_xlabel("batch (max concurrency)")
        ax.set_xscale('log',base=2); ax.set_xticks(batches); ax.set_xticklabels([str(b) for b in batches])
        ax.legend()
    fig.suptitle(title,fontsize=13,y=0.995)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(out); print("\nwrote",out)

if __name__=="__main__": main()
