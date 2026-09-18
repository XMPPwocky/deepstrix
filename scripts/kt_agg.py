#!/usr/bin/env python3
"""Aggregate a rocprofv3 --kernel-trace CSV into per-kernel GPU wall time,
split into prefill (everything up to and including the first sampler
dispatch) and decode (everything after), and per device (agent id).

usage: kt_agg.py <kt_kernel_trace.csv> [--top N] [--decode-tokens T]
"""
import csv, sys
from collections import defaultdict

def main():
    path = sys.argv[1]
    top = 60
    ntok = None
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])
    if "--decode-tokens" in sys.argv:
        ntok = int(sys.argv[sys.argv.index("--decode-tokens") + 1])
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"], r.get("Agent_Id", "?")))
    rows.sort()
    # split at first sampler kernel (end of prefill = head + sample)
    split_idx = None
    for i, (_, _, n, _) in enumerate(rows):
        if n.startswith("softmax_sample_one") or n.startswith("argmax_one") or n.startswith("topp_bracket_step"):
            split_idx = i
            break
    if split_idx is None:
        split_idx = len(rows)
    # decode tokens: count sampler dispatches after the split
    samplers = [i for i, (_, _, n, _) in enumerate(rows) if n.startswith("softmax_sample_one") or n.startswith("argmax_one")]
    n_dec = max(len(samplers) - 1, 1) if ntok is None else ntok
    phases = {"prefill": rows[: split_idx + 1], "decode": rows[split_idx + 1 :]}
    for ph, rs in phases.items():
        agg = defaultdict(lambda: [0, 0, 0])
        for s, e, n, a in rs:
            d = e - s
            agg[(n, a)][0] += d
            agg[(n, a)][1] += 1
            agg[(n, a)][2] = max(agg[(n, a)][2], d)
        tot = sum(v[0] for v in agg.values())
        wall = (rs[-1][1] - rs[0][0]) if rs else 0
        div = n_dec if ph == "decode" else 1
        per = "per token" if ph == "decode" else "whole prefill"
        print(f"\n== {ph}: {len(rs)} dispatches, kernel-busy sum {tot/1e6:.1f} ms, span {wall/1e6:.1f} ms; showing {per} (÷{div}) ==")
        print(f"{'calls':>7} {'total ms':>10} {'us/call':>9} {'max us':>9}  agent  kernel")
        for (n, a), (t, c, mx) in sorted(agg.items(), key=lambda x: -x[1][0])[:top]:
            print(f"{c/div:>7.0f} {t/1e6/div:>10.3f} {t/c/1e3:>9.1f} {mx/1e3:>9.1f}  {a:<5}  {n}")
        per_agent = defaultdict(int)
        for (n, a), (t, c, mx) in agg.items():
            per_agent[a] += t
        for a, t in per_agent.items():
            print(f"   agent {a}: {t/1e6/div:.2f} ms busy {per}")

if __name__ == "__main__":
    main()
