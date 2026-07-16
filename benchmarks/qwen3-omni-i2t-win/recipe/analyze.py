#!/usr/bin/env python3
"""analyze.py <results.json> [i2t|s2t] [batch] — print clean metrics + vLLM delta.
Handles nested ttft/itl dicts. tok/s is the length-invariant primary comparator."""
import json, sys

# committed vLLM-Omni 0.22 reference (authoritative, benchmarks branch chart_v10_h2h_data.json)
VLLM = {
 'i2t': {1:(0.876,194.6,86.8,4.76),4:(2.189,483.2,129.5,6.88),8:(3.665,770.0,100.0,9.56),
         16:(5.623,1185.1,168.1,11.88),32:(8.322,1768.8,178.6,15.42)},
 's2t': {1:(3.831,95.0,66.0,5.00),4:(13.225,328.0,60.0,9.00),8:(15.791,381.6,143.0,15.00),16:(19.798,472.4,191.0,23.00),32:(30.850,746.1,217.0,29.00)},
}

def getn(d,*p):
    x=d
    for k in p:
        if isinstance(x,dict): x=x.get(k)
        else: return None
    return x

def main():
    f=sys.argv[1]; pth=sys.argv[2] if len(sys.argv)>2 else 'i2t'; B=int(sys.argv[3]) if len(sys.argv)>3 else 0
    d=json.load(open(f))
    req=d.get('request_throughput'); tok=d.get('text_token_throughput')
    ttft=getn(d,'ttft','text','p50'); ttft_m=getn(d,'ttft','text','mean')
    itl=getn(d,'itl','text','mean')
    jm=d.get('jct_median_ms')
    pr=d.get('per_request',[])
    obs=[]
    for r in pr:
        ob=r.get('output_bytes')
        if isinstance(ob,dict): ob=ob.get('text',0)
        obs.append(ob or 0)
    avgob=sum(obs)/len(obs) if obs else 0
    ttms=ttft*1000 if isinstance(ttft,(int,float)) else None
    ttmm=ttft_m*1000 if isinstance(ttft_m,(int,float)) else None
    itms=itl*1000 if isinstance(itl,(int,float)) else None
    line=f"{pth} B{B}: req/s={req:.2f} tok/s={tok:.0f}"
    if ttms is not None: line+=f" ttft_p50={ttms:.0f}ms ttft_mean={ttmm:.0f}ms"
    if itms is not None: line+=f" itl_mean={itms:.2f}ms"
    line+=f" jct_med={jm:.0f}ms n={len(pr)} out_bytes={avgob:.0f}"
    print(line)
    ref=VLLM.get(pth,{}).get(B)
    if ref and ref[0] is not None:
        vreq,vtok,vttft,vitl=ref
        dr=(req/vreq-1)*100 if vreq else None
        dt=(tok/vtok-1)*100 if vtok else None
        cmp=f"  vs vLLM: req/s {dr:+.1f}%"
        if dt is not None: cmp+=f" | tok/s {dt:+.1f}%  (vLLM {vtok:.0f})"
        if itms is not None and vitl: cmp+=f" | ITL M* {itms:.1f} vs {vitl:.1f}"
        print(cmp)

if __name__=='__main__': main()
