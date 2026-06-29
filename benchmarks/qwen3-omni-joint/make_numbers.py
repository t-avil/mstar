#!/usr/bin/env python3
"""Regenerate NUMBERS.md from raw_*.json aggregates.

Usage: python make_numbers.py [raw_dir]
"""
import json, os, sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/qwen3-omni-joint"
BATCHES = [1, 2, 4, 8, 16, 32]
SYSTEMS = ["mstar_new", "mstar_old", "vllm"]

PATHS = [
    ("audio_to_text",   "S2T", "text"),
    ("image_to_text",   "I2T", "text"),
    ("image_to_speech", "I2S", "speech"),
    ("audio_to_speech", "S2S", "speech"),
]


def val(agg, b, s, kind, modality):
    c = agg.get(f"B{b}", {}).get(s)
    if not c:
        return None
    rec = c.get("recomputed", {})
    har = c.get("harness", {})
    if kind == "tok":  return rec.get("text_token_throughput")
    if kind == "reqs": return rec.get("request_throughput")
    if kind == "aud":  return rec.get("audio_throughput")
    if kind == "rtf":  return rec.get("rtf_p50")
    if kind == "ttft":
        d = har.get("ttft_audio" if modality == "speech" else "ttft_text")
        return d.get("p50") if isinstance(d, dict) else None
    if kind == "itl":
        d = har.get("itl_audio" if modality == "speech" else "itl_text")
        return d.get("mean") if isinstance(d, dict) else None
    return None


def ratio(new_v, other_v, lower_better=False):
    if new_v is None or other_v is None or other_v == 0 or new_v == 0:
        return "—"
    r = other_v / new_v if lower_better else new_v / other_v
    return f"{r:.2f}x"


def fmt(v):
    return "—" if v is None else f"{v:.4f}"


lines = []
lines.append("# NUMBERS.md -- headline numbers (auto-generated)")
lines.append("")
lines.append("Ratios: new/vLLM and new/old; for lower-is-better metrics the ratio "
             "is inverted so **>1.00x always means M*-new is faster**.")
lines.append("")

for fname, label, modality in PATHS:
    fp = os.path.join(ROOT, f"raw_{fname}.json")
    if not os.path.exists(fp):
        continue
    agg = json.load(open(fp))["aggregates"]

    prov = None
    for b in BATCHES:
        p = agg.get(f"B{b}", {}).get("mstar_new", {}).get("provenance")
        if p:
            prov = p
            break

    if modality == "text":
        metrics = [
            ("throughput", "tok", False),
            ("req_per_s", "reqs", False),
            ("TTFT_p50", "ttft", True),
            ("ITL_mean", "itl", True),
        ]
        unit_note = "throughput unit = **tok/s**; req/s / TTFT(text) p50 / ITL(text) mean"
    else:
        metrics = [
            ("audio_s_per_s", "aud", False),
            ("req_per_s", "reqs", False),
            ("RTF_p50", "rtf", True),
            ("TTFT_p50", "ttft", True),
            ("ITL_mean", "itl", True),
        ]
        unit_note = "throughput unit = **audio s/s**; req/s / RTF p50 / TTFT(audio) p50 / ITL(audio) mean"

    lines.append(f"## {label} -- {fname}")
    lines.append("")
    if prov:
        lines.append(f"M*-new provenance: `{prov.get('git_commit', '?')}`, "
                     f"flags: `{prov.get('flags', '?')}`")
        lines.append("")
    lines.append(unit_note)
    lines.append("")
    lines.append("| B | metric | M*-new | M*-old | vLLM | new/vLLM | new/old |")
    lines.append("|---|---|---|---|---|---|---|")

    for b in BATCHES:
        for mname, kind, lower_better in metrics:
            vals = {s: val(agg, b, s, kind, modality) for s in SYSTEMS}
            r_vllm = ratio(vals["mstar_new"], vals["vllm"], lower_better)
            r_old = ratio(vals["mstar_new"], vals["mstar_old"], lower_better)
            lines.append(f"| {b} | {mname} | {fmt(vals['mstar_new'])} | "
                         f"{fmt(vals['mstar_old'])} | {fmt(vals['vllm'])} | "
                         f"{r_vllm} | {r_old} |")
    lines.append("")

out = os.path.join(ROOT, "NUMBERS.md")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {out}")
