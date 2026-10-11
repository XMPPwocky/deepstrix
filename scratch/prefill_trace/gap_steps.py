#!/usr/bin/env python3
"""Do decode steps fill the prefill gaps? For every gap >= MIN_MS with no box-2 prefill request outstanding (hub_req
b >= 64), the decode steps (hub_step) overlapping it: count, rows, summed step time. Totals by the next request's layer.
  gap_steps.py HUBREQ_TXT HUBSTEP_TXT [MIN_MS=50]"""
import sys


def load(path, kind):
    fields, out = None, []
    for line in open(path):
        if line.startswith(f"#F {kind} "):
            fields = line.split()[2:]
        elif line.startswith(kind + " ") and fields:
            out.append(dict(zip(fields, (float(x) for x in line.split()[1:]))))
    return out


reqs = sorted((r["t_submit"] / 1e6, r["t_wait_exit"] / 1e6, int(r["layer"])) for r in load(sys.argv[1], "hub_req") if r["b"] >= 64)
steps = [(s["t_start"] / 1e6, s["t_end"] / 1e6, s["rows"], s["step_ms"]) for s in load(sys.argv[2], "hub_step")]
min_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
tot, cur_end, last_l = {}, None, None
for s, e, l in reqs:
    if cur_end is not None and s - cur_end >= min_ms:
        g0, g1 = cur_end, s
        ov = [(a, b, r) for a, b, r, _ in steps if b > g0 and a < g1]
        cov = sum(min(b, g1) - max(a, g0) for a, b, _ in ov)
        k = f"L{last_l}->L{l}"
        t = tot.setdefault(k, [0, 0.0, 0.0, 0])
        t[0] += 1; t[1] += g1 - g0; t[2] += cov; t[3] += len(ov)
        if g1 - g0 >= 100 and g1 - g0 < 5000:
            print(f"gap {g1 - g0:6.0f} ms {k:9}: {len(ov):2d} decode steps covering {cov:6.0f} ms, rows {[int(r) for _, _, r in ov][:8]}")
    if cur_end is None or e > cur_end:
        cur_end, last_l = e, l
print("\nby boundary: n gaps, gap s, of which decode-step s, steps")
for k, (n, g, c, ns) in sorted(tot.items(), key=lambda kv: -kv[1][1]):
    print(f"  {k:9} {n:3d} {g / 1e3:7.2f} {c / 1e3:7.2f} {ns:4d}")
