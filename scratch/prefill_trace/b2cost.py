#!/usr/bin/env python3
"""Box-2 PREFILL request cost model from b2win.py output: is the per-request GPU time weight-read bound (flat in rows,
proportional to distinct experts) or compute bound (proportional to rows x picks)?

  b2cost.py B2WIN_TXT [arm labels for the ranges, comma list]

Prefill requests = b >= MIN_B. Per arm: count, rows, distinct experts, n_miss, exec_gpu_us / compute_us / server_us
quantiles; then a least-squares fit on HIT-ONLY requests (n_miss == 0):
    exec_gpu_us ~ c0 + c_e * n_distinct + c_r * (b * picks share)
and the implied per-window time if a window's U units were ONE request per layer."""
import math, sys

MIN_B = 64
path = sys.argv[1]
labels = sys.argv[2].split(",") if len(sys.argv) > 2 else None
ranges, fields, rows = [], None, []
for line in open(path):
    if line.startswith("# ranges raw "):
        ranges = [tuple(int(x) for x in r.split(":")) for r in line.split()[3].split(",")]
    elif line.startswith("#F b2_req "):
        fields = line.split()[2:]
    elif line.startswith("b2_req ") and fields:
        v = [float(x) for x in line.split()[1:]]
        rows.append(dict(zip(fields, v)))
labels = labels or [f"r{i}" for i in range(len(ranges))]


def arm(t):
    for (lo, hi), lab in zip(ranges, labels):
        if lo <= t <= hi:
            return lab
    return None


def q(v, p):
    v = sorted(x for x in v if x == x)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


pre = [r for r in rows if r["b"] >= MIN_B]
print(f"{len(rows)} b2_req records, {len(pre)} with b >= {MIN_B} (prefill)")
by = {}
for r in pre:
    by.setdefault(arm(r["t_frame"]), []).append(r)
print(f"{'arm':12} {'n':>6} {'b p50':>6} {'dist p50':>8} {'miss>0':>7} {'gpu p50':>8} {'gpu p90':>8} {'comp p50':>9} {'srv p50':>8} {'page p50':>9}")
for lab in labels:
    v = by.get(lab, [])
    if not v:
        continue
    print(f"{lab:12} {len(v):6d} {q([r['b'] for r in v], .5):6.0f} {q([r['n_distinct'] for r in v], .5):8.0f} "
          f"{sum(1 for r in v if r['n_miss'] > 0) / len(v):7.2f} {q([r['exec_gpu_us'] for r in v], .5):8.0f} "
          f"{q([r['exec_gpu_us'] for r in v], .9):8.0f} {q([r['compute_us'] for r in v], .5):9.0f} "
          f"{q([r['server_us'] for r in v], .5):8.0f} {q([r['page_us'] for r in v], .5):9.0f}")

# least squares on hit-only requests: gpu = c0 + ce * distinct + cr * n_sel
hit = [r for r in pre if r["n_miss"] == 0 and r["exec_gpu_us"] == r["exec_gpu_us"] and r["exec_gpu_us"] > 0]
print(f"\nhit-only prefill requests: {len(hit)}")
X = [(1.0, r["n_distinct"], r["n_sel"]) for r in hit]
y = [r["exec_gpu_us"] for r in hit]
# normal equations 3x3
A = [[sum(x[i] * x[j] for x in X) for j in range(3)] for i in range(3)]
bvec = [sum(x[i] * yy for x, yy in zip(X, y)) for i in range(3)]


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[c][c]:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


if len(hit) > 10:
    c0, ce, cs = solve(A, bvec)
    pred = [c0 + ce * x[1] + cs * x[2] for x in X]
    ss_res = sum((a - b) ** 2 for a, b in zip(y, pred))
    mean = sum(y) / len(y)
    ss_tot = sum((a - mean) ** 2 for a in y)
    print(f"fit exec_gpu_us = {c0:.0f} + {ce:.1f} * n_distinct + {cs:.3f} * n_sel   (R^2 {1 - ss_res / ss_tot:.3f})")
    print(f"  ranges: n_distinct {q([x[1] for x in X], .1):.0f}..{q([x[1] for x in X], .9):.0f}, "
          f"n_sel {q([x[2] for x in X], .1):.0f}..{q([x[2] for x in X], .9):.0f}")
    d50, s50 = q([x[1] for x in X], .5), q([x[2] for x in X], .5)
    one = c0 + ce * d50 + cs * s50
    print(f"  typical request (distinct {d50:.0f}, picks {s50:.0f}): {one:.0f} us = fixed {c0:.0f} + experts {ce * d50:.0f} + picks {cs * s50:.0f}")
    for U in (2, 4, 8):
        # U units of the same layer merged: picks add, distinct grows sub-linearly; bound it by min(U*d50, 384*share)
        dU = min(U * d50, 245.0)
        merged = c0 + ce * dU + cs * U * s50
        print(f"  U={U}: {U} separate = {U * one / 1000:.1f} ms vs merged (distinct <= {dU:.0f}) = {merged / 1000:.1f} ms  -> x{U * one / merged:.2f}")
