#!/usr/bin/env python3
"""Rebuild raw_<path>.json aggregates from the encoders-implemeneted-benchmarked
branch's committed NUMBERS.md, then inject our piecewise cells.

Why not the benchmarks-branch raw_<path>.json: its `vllm` system is the LATER
0.22-era refresh. The charts on encoders-implemeneted-benchmarked were drawn
against vLLM 0.21 — verified: that branch's S2T B1 vLLM req_per_s = 2.3284
matches raw_vllm_021.json's 2.32839... exactly. Sourcing from NUMBERS.md keeps
M*-new / M*-old / vLLM byte-identical to what that branch documented.

NUMBERS.md rows are: | B | metric | M*-new | M*-old | vLLM | new/vLLM | new/old |
"""
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
BRANCH = "fork/encoders-implemeneted-benchmarked"
MD = "benchmarks/qwen3-omni-native-encoders/NUMBERS.md"
NEW_ROOT = os.environ.get(
    "PIECEWISE_RAW", "/m-coriander/coriander/tim/sweep_piecewise_clean")
# benchmark/sweep.sh writes <short>/B<n>/results.json (short = s2t/i2t/s2s/i2s),
# not the <modality>_b<n>/ layout the qwen3-omni-131 runner uses. Accept either
# so the same script works against a sweep dir or a benchmarks/raw dir.
SHORT = {"audio_to_text": "s2t", "image_to_text": "i2t",
         "image_to_speech": "i2s", "audio_to_speech": "s2s"}


def cell_paths(path, b):
    return [
        os.path.join(NEW_ROOT, SHORT[path], f"B{b}", "results.json"),
        os.path.join(NEW_ROOT, f"{path}_b{b}", "results.json"),
    ]
NEW_KEY = "mstar_piecewise"
BATCHES = [1, 2, 4, 8, 16, 32]

SECTION = re.compile(r"^##\s+\S+\s+--\s+(\w+)\s*$")
ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\w+)\s*\|\s*([\d.eE+-]+)\s*\|"
                 r"\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)\s*\|")

# NUMBERS.md metric name -> (aggregates block, field)
FIELD = {
    "throughput":    ("recomputed", "text_token_throughput"),
    "audio_s_per_s": ("recomputed", "audio_throughput"),
    "req_per_s":     ("recomputed", "request_throughput"),
    "RTF_p50":       ("recomputed", "rtf_p50"),
    "TTFT_p50":      ("harness", "ttft"),      # modality suffix added below
    "ITL_mean":      ("harness", "itl"),
}
SYS_COL = {2: "mstar_new", 3: "mstar_old", 4: "vllm"}


def parse():
    md = subprocess.run(["git", "show", f"{BRANCH}:{MD}"], cwd="/home/tim/mstar",
                        capture_output=True, text=True).stdout
    out, cur = {}, None
    for line in md.splitlines():
        m = SECTION.match(line)
        if m:
            cur = m.group(1)
            out[cur] = {}
            continue
        if not cur:
            continue
        r = ROW.match(line)
        if not r:
            continue
        b, metric = int(r.group(1)), r.group(2)
        if metric not in FIELD:
            continue
        modality = "audio" if cur.endswith("speech") else "text"
        block, field = FIELD[metric]
        if block == "harness":
            field = f"{field}_{modality}"
        vals = {SYS_COL[i]: float(r.group(i + 1)) for i in (2, 3, 4)}
        cell = out[cur].setdefault(f"B{b}", {})
        for sysname, v in vals.items():
            d = cell.setdefault(sysname, {"recomputed": {}, "harness": {}})
            if block == "harness":
                stat = "p50" if metric.endswith("p50") else "mean"
                d["harness"].setdefault(field, {})[stat] = v
            else:
                d["recomputed"][field] = v
    return out


def add_piecewise(agg, path):
    modality = "audio" if path.endswith("speech") else "text"
    n = 0
    for b in BATCHES:
        fp = next((p for p in cell_paths(path, b) if os.path.exists(p)), None)
        if fp is None:
            continue
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        if (d.get("completed") or 0) <= 0:
            continue
        tt = ((d.get("ttft") or {}).get(modality) or {}).get("p50")
        it = ((d.get("itl") or {}).get(modality) or {}).get("mean")
        agg.setdefault(f"B{b}", {})[NEW_KEY] = {
            "recomputed": {
                "text_token_throughput": d.get("text_token_throughput"),
                "request_throughput": d.get("request_throughput"),
                "audio_throughput": d.get("audio_seconds_throughput"),
                "rtf_p50": (d.get("rtf") or {}).get("p50"),
            },
            "harness": {
                f"ttft_{modality}": {"p50": tt},
                f"itl_{modality}": {"mean": it},
            },
        }
        n += 1
    return n


parsed = parse()
for path, agg in parsed.items():
    n = add_piecewise(agg, path)
    json.dump({"_source": f"{BRANCH}:{MD} (vLLM = 0.21-era) + {n} piecewise cells",
               "aggregates": agg},
              open(os.path.join(ROOT, f"raw_{path}.json"), "w"), indent=1)
    sysset = sorted({s for c in agg.values() for s in c})
    print(f"{path}: batches={len(agg)} systems={sysset} piecewise_cells={n}")
