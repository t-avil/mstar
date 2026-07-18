#!/usr/bin/env python3
"""Median encoff/base text tok/s ratio across paired A/B rounds (load-robust).
Reads exp_paired_out/<topo>_r<r>_<rt>_b<B>/stdout.txt ("N text tok/s")."""
import glob, os, re, statistics as st
ROOT = "/m-coriander/coriander/tim/exp_paired_out"
def tok(p):
    try:
        s = open(p).read()
        m = re.search(r"([0-9.]+) text tok/s", s)
        return float(m.group(1)) if m else None
    except Exception: return None
data = {}  # (rt,B) -> {'encoff':[...], 'base':[...]}
for d in glob.glob(f"{ROOT}/*_r*_*_b*"):
    b = os.path.basename(d)
    m = re.match(r"(encoff|base)_r(\d+)_([a-z_]+)_b(\d+)$", b)
    if not m: continue
    topo, r, rt, B = m.group(1), m.group(2), m.group(3), int(m.group(4))
    v = tok(os.path.join(d, "stdout.txt"))
    if v is None: continue
    data.setdefault((rt, B), {}).setdefault(topo, []).append(v)
print(f"{'path':>14} {'B':>3} {'encoff(med)':>11} {'base(med)':>10} {'encoff/base':>11}  {'n':>3}")
for (rt, B) in sorted(data, key=lambda k: (k[0], k[1])):
    e = data[(rt,B)].get("encoff", []); bs = data[(rt,B)].get("base", [])
    if not e or not bs: continue
    em, bm = st.median(e), st.median(bs)
    print(f"{rt:>14} {B:>3} {em:>11.1f} {bm:>10.1f} {em/bm:>10.3f}x  {min(len(e),len(bs)):>3}")
