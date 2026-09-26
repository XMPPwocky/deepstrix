#!/usr/bin/env python3
"""Summarise the reviewer's 3 process runs: per (shape, b) median-of-runs for base and tB, ratio,
and the engineer's r3/r4/r5 medians alongside. Reads KBJSON lines."""
import json, re, statistics, sys, glob, os
here = os.path.dirname(os.path.abspath(__file__))
eng = os.path.join(here, "..", "..", "results")

def load(files):
    d = {}
    for f in files:
        for line in open(f, errors="replace"):
            if not line.startswith("KBJSON {\"tag\""):
                continue
            j = json.loads(line[7:])
            tag = j["tag"].split(" cold(")[0].split(" WARM")[0]
            d.setdefault((tag, j["variant"]), []).append(j["med_us"])
    return d

mine = load(sorted(glob.glob(os.path.join(here, "results", "gemv_sweep_v*.txt")) + glob.glob(os.path.join(here, "results", "grouped_v*.txt"))))
theirs = load([os.path.join(eng, f"gemv_sweep_r{i}.txt") for i in (3, 4, 5)] + [os.path.join(eng, f"small_r{i}.txt") for i in (3, 4, 5)])

def pick(d, tag, pat):
    for (t, v), xs in d.items():
        if t == tag and re.match(pat, v):
            return xs
    return None

tags = sorted({t for (t, v) in mine}, key=lambda s: (s.split(" b=")[0], int(re.search(r" b=(\d+)", s).group(1))))
print(f"{'shape/b':52s} {'base(me)':>9s} {'tB(me)':>8s} {'ratio':>6s} | {'base(eng)':>9s} {'tB(eng)':>8s} {'ratio':>6s} | runs")
hl = None
for t in tags:
    b_me = pick(mine, t, r"^base q8_0_(gemv_bpack_warp8|grouped_gemv_bpack)$")
    c_me = pick(mine, t, r"^cand q8_0_(gemv_bpack_tB\d+|grouped_gemv_gemv_bpack_tB\d+|grouped_gemv_bpack_tB\d+)$")
    b_th = pick(theirs, t, r"^base q8_0_(gemv_bpack_warp8|grouped_gemv_bpack)$")
    c_th = pick(theirs, t, r"^cand q8_0_(gemv_bpack_tB\d+|grouped_gemv_bpack_tB\d+)$")
    if not (b_me and c_me):
        continue
    bm, cm = statistics.median(b_me), statistics.median(c_me)
    bt = statistics.median(b_th) if b_th else float("nan")
    ct = statistics.median(c_th) if c_th else float("nan")
    short = t.replace("gemv ", "").replace(" copies=", " c")
    print(f"{short:52s} {bm:9.2f} {cm:8.2f} {cm/bm:6.3f} | {bt:9.2f} {ct:8.2f} {ct/bt:6.3f} | {len(b_me)}")
    if t.startswith("gemv q_a") and " b=4 " in t + " ":
        hl = (bm, cm, b_me, c_me)
if hl:
    bm, cm, b_me, c_me = hl
    print(f"\nHEADLINE q_a b=4: base med-of-{len(b_me)} = {bm:.2f} us {b_me}; tB4 = {cm:.2f} us {c_me}; speedup = {bm/cm:.3f}x (claim 1.24x)")
