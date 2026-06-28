#!/usr/bin/env python3
"""Fold the repeated I2S B1/B2 recheck runs (M*-new + M*-old, multiple reps) into the committed
raw_image_to_speech.json aggregates — fills the missing M*-old B=2 and refreshes B1/B2 with
rep-averaged values. Then the charts regenerate to show the corrected datapoints."""
import json, glob, re, statistics, os
BPS = 24000*2  # 24kHz int16
RAW = "/home/tim/bench-wt/benchmarks/qwen3-omni-joint/raw_image_to_speech.json"
RECHECK = "/home/tim/exp_rebench/recheck"

def parse_stdout(p):
    out={}
    try: t=open(p,encoding="utf-8",errors="replace").read()
    except OSError: return out
    for mod in ("audio","text"):
        m=re.search(rf"TTFT \({mod}\)\s*:\s*mean=([\d.]+)s\s+p50=([\d.]+)s\s+p95=([\d.]+)s\s+p99=([\d.]+)s",t)
        if m: out[f"ttft_{mod}"]=dict(zip(("mean","p50","p95","p99"),map(float,m.groups())))
        m=re.search(rf"ITL  \({mod}\)\s*:\s*mean=([\d.]+)s\s+p50=([\d.]+)s\s+p95=([\d.]+)s\s+p99=([\d.]+)s",t)
        if m: out[f"itl_{mod}"]=dict(zip(("mean","p50","p95","p99"),map(float,m.groups())))
    return out

def cell_from_reps(tag,B):
    rtfs=[]; auds=[]; reqs=[]; ttfts=[]; itls=[]; ndp=0; nreps=0
    for d in sorted(glob.glob(f"{RECHECK}/{tag}/image_to_speech_B{B}_r*")):
        rj=os.path.join(d,"results.json")
        if not os.path.exists(rj): continue
        dd=json.load(open(rj)); pr=dd.get("per_request") or []
        wall=dd.get("wall_time_s") or 0; tot_aud=0
        for r in pr:
            ab=(r.get("output_bytes") or {}).get("audio",0);
            if ab<=0: continue
            dur=ab/BPS; tot_aud+=dur; rtfs.append((r.get("jct_ms",0)/1000.0)/dur); ndp+=1
        if wall>0: auds.append(tot_aud/wall); reqs.append((dd.get("completed") or len(pr))/wall)
        so=parse_stdout(os.path.join(d,"out.txt"))
        if so.get("ttft_audio"): ttfts.append(so["ttft_audio"]["p50"])
        if so.get("itl_audio"): itls.append(so["itl_audio"]["mean"])
        nreps+=1
    if not rtfs: return None
    return {
      "recomputed":{"n":ndp,"n_with_audio":ndp,
        "rtf_p50":statistics.median(rtfs),"rtf_mean":statistics.mean(rtfs),
        "rtf_p95":statistics.quantiles(rtfs,n=20)[18] if len(rtfs)>=20 else max(rtfs),
        "audio_throughput":statistics.mean(auds) if auds else None,
        "request_throughput":statistics.mean(reqs) if reqs else None,
        "text_token_throughput":None},
      "harness":{"ttft_audio":{"p50":statistics.mean(ttfts)} if ttfts else {},
                 "itl_audio":{"mean":statistics.mean(itls)} if itls else {}},
      "source":f"recheck ({nreps} reps, n={ndp})","completed":ndp,"num_requests":ndp}

raw=json.load(open(RAW)); agg=raw["aggregates"]
changed=[]
for tag in ("mstar_new","mstar_old"):
    for B in (1,2):
        c=cell_from_reps(tag,B)
        if c:
            agg.setdefault(f"B{B}",{})[tag]=c
            changed.append(f"B{B}/{tag}: rtf_p50={c['recomputed']['rtf_p50']:.4f} aud/s={c['recomputed']['audio_throughput']:.2f} ({c['source']})")
json.dump(raw,open(RAW,"w"),indent=2)
print("Updated", RAW)
for ch in changed: print("  ",ch)
