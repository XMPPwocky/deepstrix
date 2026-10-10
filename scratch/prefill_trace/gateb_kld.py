#!/usr/bin/env python3
"""Gate B (statistical): pairwise KLD / argmax / top-5 overlap between the per-arm last-token prefill logits that
arms_p1p2a.py saved (<dir>/<arm>.f32, N_VOCAB f32 LE each). PASS = KLD(a0 || x) within the null KLD(a0 || a0n) spread
(and same argmax). Usage: gateb_kld.py [dir]"""
import math, os, struct, sys

D = sys.argv[1] if len(sys.argv) > 1 else "/home/claude-code/gateb_20261010"
NV = 129280
recs = {}
for fn in sorted(os.listdir(D)):
    if fn.endswith(".f32"):
        with open(os.path.join(D, fn), "rb") as f:
            recs[fn[:-4]] = struct.unpack(f"<{NV}f", f.read(NV * 4))


def lsm(x):
    m = max(x)
    s = math.log(sum(math.exp(v - m) for v in x)) + m
    return [v - s for v in x]


ls = {k: lsm(v) for k, v in recs.items()}
top = {k: sorted(range(NV), key=lambda i: -v[i])[:5] for k, v in recs.items()}
names = list(recs)
print(f"arms: {names}")
for k in names:
    print(f"  {k:9} argmax {top[k][0]:6d} p={math.exp(ls[k][top[k][0]]):.4f} top5 {top[k]}")
print("\nKLD(row || col) nats")
print(" " * 10 + "".join(f"{c:>11}" for c in names))
for r in names:
    row = []
    for c in names:
        if r == c:
            row.append(f"{'-':>11}")
        else:
            kl = sum(math.exp(a) * (a - b) for a, b in zip(ls[r], ls[c]))
            row.append(f"{kl:11.3e}")
    print(f"{r:10}" + "".join(row))
for r in names:
    maxabs = max(abs(a - b) for a, b in zip(recs["a0"], recs[r])) if "a0" in recs else float("nan")
    print(f"  max |logit(a0) - logit({r})| = {maxabs:.4e}")
