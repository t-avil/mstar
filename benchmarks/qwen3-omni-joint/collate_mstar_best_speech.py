#!/usr/bin/env python3
"""Collate newest/best M* (godv9/E1) speech sweep -> chart_mstar_best_speech_data.json.
Speech metrics: req_s, audio_thr (audio-sec/s), rtf (p50), ttft (audio p50, ms)."""
import json, os
OUT = "/m-coriander/coriander/tim/mstar_best_speech_out"
BATCHES = [1, 2, 4, 8, 16, 32]
data = {"note": "newest/best M* (godv9 E1) speech full-N sweep", "i2s": {}, "s2s": {}}
def g(dd, *p, default=None):
    for k in p:
        if not isinstance(dd, dict) or dd.get(k) is None: return default
        dd = dd[k]
    return dd
for tag in ("i2s", "s2s"):
    for B in BATCHES:
        f = os.path.join(OUT, f"{tag}_B{B}", "results.json")
        if not os.path.exists(f): print("MISSING", f); continue
        r = json.load(open(f))
        ttft = g(r, "ttft", "audio", "p50")
        data[tag][str(B)] = {
            "req_s": g(r, "request_throughput", default=0.0),
            "audio_thr": g(r, "audio_seconds_throughput", default=0.0),
            "rtf": g(r, "rtf", "p50"),
            "ttft": ttft * 1000.0 if (ttft is not None and ttft < 5) else ttft,
            "completed": g(r, "completed", default=0)}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chart_mstar_best_speech_data.json")
json.dump(data, open(out, "w"), indent=2); print("wrote", out)
for t in ("i2s", "s2s"): print(t, "B32:", data[t].get("32"))
