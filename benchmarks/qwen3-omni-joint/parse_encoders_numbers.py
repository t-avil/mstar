#!/usr/bin/env python3
"""Parse the authoritative encoders-implemented NUMBERS.md (M*-new = 1f66ce6, native
encoders) from the encoders-implemeneted-benchmarked branch into chart_encoders_impl_data.json.
Covers all 4 paths. TTFT/ITL/RTF are seconds in the table -> ttft/itl converted to ms."""
import re, json, os

SRC = "/m-coriander/coriander/tim/enc_numbers.md"
HDR = {"## S2T": "s2t", "## I2T": "i2t", "## I2S": "i2s", "## S2S": "s2s"}
# metric label in table -> our key (+ whether to *1000 for ms)
TEXT = {"throughput": ("tok", 1), "req_per_s": ("req_s", 1), "TTFT_p50": ("ttft", 1000), "ITL_mean": ("itl", 1000)}
SPEECH = {"audio_s_per_s": ("audio_thr", 1), "req_per_s": ("req_s", 1),
          "RTF_p50": ("rtf", 1), "TTFT_p50": ("ttft", 1000), "ITL_mean": ("itl", 1000)}

data = {"note": "encoders-implemented = M*-new 1f66ce6 (native encoders) from "
        "encoders-implemeneted-benchmarked:NUMBERS.md", "i2t": {}, "s2t": {}, "i2s": {}, "s2s": {}}
cur = None
for ln in open(SRC):
    for h, tag in HDR.items():
        if ln.startswith(h): cur = tag
    m = re.match(r"\|\s*(\d+)\s*\|\s*([A-Za-z0-9_]+)\s*\|\s*([-\d.]+)\s*\|", ln)
    if not m or cur is None: continue
    B, metric, mstar_new = m.group(1), m.group(2), float(m.group(3))
    table = SPEECH if cur in ("i2s", "s2s") else TEXT
    if metric not in table: continue
    key, mul = table[metric]
    data[cur].setdefault(B, {})[key] = mstar_new * mul

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chart_encoders_impl_data.json")
json.dump(data, open(out, "w"), indent=2)
print("wrote", out)
for t in ("i2t", "s2t", "i2s", "s2s"):
    print(t, "B32:", data[t].get("32"))
