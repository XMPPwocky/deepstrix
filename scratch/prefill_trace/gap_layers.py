#!/usr/bin/env python3
"""Which layers the per-window gaps sit at: for every gap >= MIN_MS with no box-2 prefill request outstanding (union of
[t_submit, t_wait_exit] of hub_req b >= 64), print the (layer, lane) of the request that ended last before the gap and
of the first one submitted after it, plus a histogram of the gap time by the next request's layer.
  gap_layers.py HUBREQ_TXT [MIN_MS=20]"""
import sys
fields, reqs = None, []
for line in open(sys.argv[1]):
    if line.startswith("#F hub_req "):
        fields = line.split()[2:]
    elif line.startswith("hub_req ") and fields:
        r = dict(zip(fields, (float(x) for x in line.split()[1:])))
        if r["b"] >= 64:
            reqs.append((r["t_submit"] / 1e6, r["t_wait_exit"] / 1e6, int(r["layer"]), int(r["lane"]), r["b"]))
min_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
reqs.sort()
by_layer, cur_end, last = {}, None, None
for s, e, l, ln, b in reqs:
    if cur_end is not None and s - cur_end >= min_ms:
        g = s - cur_end
        by_layer[l] = by_layer.get(l, 0) + g
        if g >= 100:
            print(f"gap {g:7.1f} ms: after L{last[2]} lane {last[3]} -> next L{l} lane {ln} ({b:.0f} rows)")
    if cur_end is None or e > cur_end:
        cur_end, last = e, (s, e, l, ln)
print("\ngap time (>= %.0f ms) by the NEXT request's layer:" % min_ms)
for l in sorted(by_layer, key=lambda k: -by_layer[k])[:12]:
    print(f"  L{l:2d}: {by_layer[l] / 1e3:6.2f} s")
