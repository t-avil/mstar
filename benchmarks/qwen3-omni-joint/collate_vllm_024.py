#!/usr/bin/env python3
"""Collate vLLM-Omni 0.24 full-sweep results.json -> chart_vllm_024_data.json.
Text paths (i2t,s2t): req_s, tok, ttft(p50, ms), itl(mean, ms).
Speech paths (i2s,s2s): req_s, audio_thr (audio-sec/s), rtf (p50), ttft(p50, ms)."""
import json, os

OUT = "/m-coriander/coriander/tim/vllm024_out"
BATCHES = [1, 2, 4, 8, 16, 32]
data = {"note": "vLLM-Omni 0.24.0+cu129 GPUs 6,7 full sweep (text N=committed-0.22, speech lighter)",
        "i2t": {}, "s2t": {}, "i2s": {}, "s2s": {}}

def g(dd, *path, default=None):
    for k in path:
        if not isinstance(dd, dict) or dd.get(k) is None: return default
        dd = dd[k]
    return dd

for tag in ("i2t", "s2t", "i2s", "s2s"):
    speech = tag in ("i2s", "s2s")
    for B in BATCHES:
        f = os.path.join(OUT, f"{tag}_B{B}", "results.json")
        if not os.path.exists(f):
            print(f"MISSING {f}"); continue
        r = json.load(open(f))
        # speech: TTFT is time-to-first-AUDIO (ttft.audio); text: ttft.text
        ttft = g(r, "ttft", "audio", "p50") if speech else g(r, "ttft", "text", "p50")
        row = {"req_s": g(r, "request_throughput", default=0.0),
               "ttft": ttft * 1000.0 if (ttft is not None and ttft < 5) else ttft,
               "completed": g(r, "completed", default=0)}
        if speech:
            row["audio_thr"] = g(r, "audio_seconds_throughput", default=0.0)
            rtf = g(r, "rtf", "p50")
            row["rtf"] = rtf
        else:
            row["tok"] = g(r, "text_token_throughput", default=0.0)
            itl = g(r, "itl", "text", "mean")
            row["itl"] = itl * 1000.0 if (itl is not None and itl < 2) else itl
        data[tag][str(B)] = row

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chart_vllm_024_data.json")
json.dump(data, open(out, "w"), indent=2)
print("wrote", out)
for tag in data:
    if isinstance(data[tag], dict) and data[tag]:
        print(tag, "B32:", data[tag].get("32"))
