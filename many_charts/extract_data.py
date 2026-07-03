#!/usr/bin/env python3
"""Extract every chart series into data/*.json from committed sources.

Run from many_charts/ inside the repo. Deterministic; re-run any time.
Sources (all committed — nothing is re-benchmarked):
  vllm_022 / mstar_new / mstar_old : ../benchmarks/qwen3-omni-joint/raw_<path>.json
      EXCEPT s2t mstar_old, which comes from git d2d1983 (the clean
      single-build sweep — the joint-dir current copy carries the later
      degenerate-ITL-drop era; see bench commit 254f5c1 rationale).
  vllm_021 : git 5c27c12^ raw_<path>.json (pre-0.22-refresh vLLM)
  mstar_v2 : canonical v2 sweep results.json trees (text + speech sweeps)
  mstar_v3 : final-stack preview sweep tree
Speech-path latency = audio stream (complete everywhere); text = text stream.
"""
import glob
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
JOINT = "benchmarks/qwen3-omni-joint"
PATHS = ["audio_to_text", "image_to_text", "audio_to_speech", "image_to_speech"]
SHORT = {"audio_to_text": "s2t", "image_to_text": "i2t",
         "audio_to_speech": "s2s", "image_to_speech": "i2s"}
V2_SWEEPS = ["/m-coriander/coriander/tim/sweep_text_w1",
             "/m-coriander/coriander/tim/sweep_mstar_v2_final"]
V3_SWEEP = "/m-coriander/coriander/tim/sweep_mstar_v3"
OLD_S2T_COMMIT = "d2d1983"          # clean single-build sweep (mstar_old s2t)
OLD_VLLM_COMMIT = "5c27c12^"        # pre-0.22 vLLM refresh


def _g(h, k, sub):
    x = (h or {}).get(k)
    return x.get(sub) if isinstance(x, dict) else x


def _cell(blk, speech):
    rec, har = blk.get("recomputed") or {}, blk.get("harness") or {}
    stream = "audio" if speech else "text"
    return {
        "req_s": rec.get("request_throughput") or har.get("request_throughput"),
        "tok_s": rec.get("text_token_throughput"),
        "ttft_p50": _g(har, f"ttft_{stream}", "p50"),
        "itl_mean": _g(har, f"itl_{stream}", "mean"),
        "rtf_p50": rec.get("rtf_p50"),
    }


def from_aggregates(raw_by_path, sysname, prov):
    out = {"_provenance": prov}
    for p in PATHS:
        speech = p.endswith("speech")
        out[p] = {b: _cell(sm.get(sysname) or {}, speech)
                  for b, sm in raw_by_path[p].get("aggregates", {}).items()}
    return out


def git_json(commit, path):
    blob = subprocess.run(["git", "show", f"{commit}:{path}"],
                          capture_output=True, text=True, cwd=REPO)
    return json.loads(blob.stdout)


def from_sweeps(bases, prov):
    out = {"_provenance": prov}
    for p in PATHS:
        s, speech = SHORT[p], p.endswith("speech")
        out[p] = {}
        for base in bases:
            for bdir in sorted(glob.glob(f"{base}/{s}/B*")):
                try:
                    res = json.load(open(f"{bdir}/results.json"))
                except FileNotFoundError:
                    continue
                stream = "audio" if speech else "text"
                t = ((res.get("ttft") or {}).get(stream) or {})
                i = res.get("itl") or {}
                iv = (i.get(stream) or {}).get("mean") if isinstance(i.get(stream), dict) else None
                per_req = res.get("per_request", [])
                rtfs = sorted((q["jct_ms"] / 1000.0) / (q["output_bytes"]["audio"] / 48000.0)
                              for q in per_req
                              if q.get("output_bytes", {}).get("audio", 0) > 0 and q.get("jct_ms"))
                out[p][os.path.basename(bdir)] = {
                    "req_s": res.get("request_throughput"),
                    "tok_s": res.get("text_token_throughput"),
                    "ttft_p50": t.get("p50"), "itl_mean": iv,
                    "rtf_p50": rtfs[len(rtfs) // 2] if rtfs else None,
                }
    return out


def main():
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    current = {p: json.load(open(os.path.join(REPO, JOINT, f"raw_{p}.json")))
               for p in PATHS}

    for sysname, outname in [("vllm", "vllm_022"), ("mstar_new", "mstar_new")]:
        d = from_aggregates(current, sysname,
                            f"committed {JOINT}/raw_<path>.json '{sysname}'; speech latency = audio stream")
        json.dump(d, open(os.path.join(HERE, f"data/{outname}.json"), "w"), indent=1)

    old = from_aggregates(current, "mstar_old",
                          f"committed {JOINT} 'mstar_old'; s2t OVERRIDDEN from git {OLD_S2T_COMMIT} "
                          "(clean single-build sweep; later copies dropped degenerate ITL cells)")
    d2d = git_json(OLD_S2T_COMMIT, f"{JOINT}/raw_audio_to_text.json")
    old["audio_to_text"] = {b: _cell(sm.get("mstar_old") or {}, False)
                            for b, sm in d2d.get("aggregates", {}).items()}
    json.dump(old, open(os.path.join(HERE, "data/mstar_old.json"), "w"), indent=1)

    prev = {p: git_json(OLD_VLLM_COMMIT, f"{JOINT}/raw_{p}.json") for p in PATHS}
    d = from_aggregates(prev, "vllm",
                        f"vLLM pre-0.22 refresh (git {OLD_VLLM_COMMIT}); speech latency = audio stream")
    json.dump(d, open(os.path.join(HERE, "data/vllm_021.json"), "w"), indent=1)

    json.dump(from_sweeps(V2_SWEEPS, "mstar_v2 canonical sweeps (pair 6,7, 2026-07-02); speech latency = audio stream"),
              open(os.path.join(HERE, "data/mstar_v2.json"), "w"), indent=1)
    json.dump(from_sweeps([V3_SWEEP], "mstar_v3 final-stack preview (pair 0,1 cross-NUMA, 2026-07-03)"),
              open(os.path.join(HERE, "data/mstar_v3.json"), "w"), indent=1)
    for f in sorted(os.listdir(os.path.join(HERE, "data"))):
        print("wrote data/" + f)


if __name__ == "__main__":
    main()
