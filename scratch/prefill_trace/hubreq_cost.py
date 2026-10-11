#!/usr/bin/env python3
"""Hub-side per-request timing for PREFILL box-2 requests (hub_req, b >= 64) from b2win/evtfilt text output.
Per request (ms): submit duration (t_submit -> t_submit_end), round trip (t1 -> t4 = send -> reply landed), the blocking
wait in post (t_wait_enter -> t_wait_exit), and per lane the gap between consecutive submits (the hub's cycle per unit).
  hubreq_cost.py TXT"""
import sys
fields, rows = None, []
for line in open(sys.argv[1]):
    if line.startswith("#F hub_req "):
        fields = line.split()[2:]
    elif line.startswith("hub_req ") and fields:
        rows.append(dict(zip(fields, (float(x) for x in line.split()[1:]))))
rows = [r for r in rows if r["b"] >= 64]
rows.sort(key=lambda r: r["t_submit"])


def q(v, p):
    v = sorted(x for x in v if x == x)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


ms = lambda a, b: [(r[b] - r[a]) / 1e6 for r in rows]
print(f"prefill requests {len(rows)}; rows p50 {q([r['b'] for r in rows], .5):.0f}")
for name, a, b in [("submit (host)", "t_submit", "t_submit_end"), ("round trip t1->t4", "t1", "t4"),
                   ("post wait (blocked)", "t_wait_enter", "t_wait_exit"), ("submit -> wait exit", "t_submit", "t_wait_exit")]:
    v = ms(a, b)
    print(f"  {name:22} p10 {q(v, .1):7.2f}  p50 {q(v, .5):7.2f}  p90 {q(v, .9):7.2f}")
print(f"  rtt_us (hub field) p50 {q([r['rtt_us'] / 1e3 for r in rows], .5):.2f}; srv_us p50 {q([r['srv_us'] / 1e3 for r in rows], .5):.2f}; "
      f"bytes_in p50 {q([r['bytes_in'] for r in rows], .5) / 1e6:.1f} MB, bytes_out p50 {q([r['bytes_out'] for r in rows], .5) / 1e6:.1f} MB")
for lane in sorted({r["lane"] for r in rows}):
    lr = [r for r in rows if r["lane"] == lane]
    gaps = [(b["t_submit"] - a["t_submit"]) / 1e6 for a, b in zip(lr, lr[1:])]
    print(f"  lane {lane:.0f}: {len(lr)} requests, submit-to-submit p50 {q(gaps, .5):.1f} ms p90 {q(gaps, .9):.1f}")
allgaps = [(b["t_submit"] - a["t_submit"]) / 1e6 for a, b in zip(rows, rows[1:])]
print(f"  any lane: submit-to-submit p50 {q(allgaps, .5):.1f} ms")
# where does a cycle go: from a request's reply landing (t4) to its post wait exit, and wait-exit -> next submit on that lane
w2n = []
for lane in sorted({r["lane"] for r in rows}):
    lr = [r for r in rows if r["lane"] == lane]
    w2n += [(b["t_submit"] - a["t_wait_exit"]) / 1e6 for a, b in zip(lr, lr[1:])]
print(f"  same lane: post-wait exit -> next submit p50 {q(w2n, .5):.1f} ms p90 {q(w2n, .9):.1f}")
