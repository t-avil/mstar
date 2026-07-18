#!/usr/bin/env python3
"""Dual-goal verdict: current build (natural-EOS sweep) vs
  (1) older-M* text tok/s parity target, (2) committed vLLM speech 2-3x.
Usage: compare.py <label>   (reads /m-coriander/coriander/tim/exp_nateos_out/<label>/)
"""
import json, os, sys, glob

LABEL = sys.argv[1] if len(sys.argv) > 1 else "encoff"
ROOT = f"/m-coriander/coriander/tim/exp_nateos_out/{LABEL}"
BASE = json.load(open("/m-coriander/coriander/tim/deliverable/baselines.json"))
BATCHES = [1, 2, 4, 8, 16, 32]

def load(rt, b):
    f = f"{ROOT}/{rt}_b{b}/results.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    return d

def g(d, k):
    v = d.get(k) if d else None
    return v

print(f"\n===== DUAL-GOAL VERDICT — build label: {LABEL} (natural-EOS) =====\n")

# ---- TEXT tok/s parity vs older M* ----
for rt, older_key, older2 in [("image_to_text","i2t_E1v9","i2t_encoders"),
                              ("audio_to_text","s2t_v9sess","s2t_encoders")]:
    tgt = BASE["older_mstar_text_tok"][older_key]
    tgt2 = BASE["older_mstar_text_tok"][older2]
    vll = BASE["vllm022_text_tok"]["i2t" if rt=="image_to_text" else "s2t"]
    print(f"--- {rt} tok/s : current vs older-M* [{older_key}] / vs vLLM0.22 ---")
    print(f"{'B':>3} {'cur':>8} {'older':>8} {'ratio':>6} {'PARITY':>7} | {'vLLM':>7} {'vs_vLLM':>7}")
    allpar = True
    for b in BATCHES:
        d = load(rt, b)
        cur = g(d, "text_token_throughput")
        o = tgt[str(b)]; v = vll[str(b)]
        if cur is None:
            print(f"{b:>3} {'--':>8} {o:>8.0f}"); allpar=False; continue
        r = cur/o; par = "OK" if r >= 0.99 else "LOW"
        if r < 0.99: allpar=False
        print(f"{b:>3} {cur:>8.1f} {o:>8.0f} {r:>5.2f}x {par:>7} | {v:>7.0f} {cur/v:>6.2f}x")
    print(f"    => {rt} parity vs older-M*: {'PASS' if allpar else 'CHECK'}\n")

# ---- SPEECH 2-3x vs committed vLLM 0.22 (natural-EOS) ----
for rt, aud_key, req_key in [("image_to_speech","i2s_audio_sps","i2s_req_s"),
                             ("audio_to_speech","s2s_audio_sps","s2s_req_s")]:
    va = BASE["vllm022_speech_natEOS"][aud_key]
    vq = BASE["vllm022_speech_natEOS"][req_key]
    ma = BASE["mstar_new_speech_natEOS"][aud_key]
    print(f"--- {rt} : current vs committed vLLM0.22 (natural-EOS) ---")
    print(f"{'B':>3} {'cur_aud':>8} {'vLLM':>7} {'aud_x':>6} | {'cur_req':>8} {'vLLM_req':>8} {'req_x':>6} | {'committed_aud':>7}")
    for b in BATCHES:
        d = load(rt, b)
        ca = g(d, "audio_seconds_throughput"); cq = g(d, "request_throughput")
        a = va[str(b)]; q = vq[str(b)]; cm = ma[str(b)]
        if ca is None:
            print(f"{b:>3} {'--':>8}"); continue
        print(f"{b:>3} {ca:>8.1f} {a:>7.1f} {ca/a:>5.2f}x | {cq:>8.2f} {q:>8.3f} {cq/q:>5.2f}x | {cm:>7.1f}")
    print()
