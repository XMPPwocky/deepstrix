#!/usr/bin/env python3
"""Decode-step substitution + box-2 miss series from b2win/evtfilt text of hub_step (hub clock): per BUCKET-second
bucket (wall, via the '# ranges raw' line + the span start): steps, mean rows, sub_picks_swapped per step and per row,
b2_misses per step, remote_wait_ms / step_ms p50.
  sub_series.py TXT SPAN0_ISO BUCKET_S"""
import collections, datetime, sys
path, s0, bs = sys.argv[1], sys.argv[2], float(sys.argv[3])
lo = int([l for l in open(path) if l.startswith("# ranges raw ")][0].split()[-1].split(":")[0])
t0 = datetime.datetime.fromisoformat(s0).replace(tzinfo=datetime.timezone.utc).timestamp()
F, B = None, collections.defaultdict(list)
for line in open(path):
    if line.startswith("#F hub_step "):
        F = line.split()[2:]
    elif line.startswith("hub_step ") and F:
        r = dict(zip(F, (float(x) for x in line.split()[1:])))
        w = t0 + (r["t_start"] - lo) / 1e9
        B[int(w // bs) * bs].append(r)
for k in sorted(B):
    v = B[k]
    n = len(v)
    rows = sum(r["rows"] for r in v)
    sw = sum(r["sub_picks_swapped"] for r in v)
    mi = sum(r["b2_misses"] for r in v)
    sm = sorted(r["step_ms"] for r in v)[n // 2]
    ts = datetime.datetime.fromtimestamp(k, datetime.timezone.utc).strftime("%H:%M")
    print(f"{ts}  steps {n:5d}  rows/step {rows / n:5.1f}  swapped/row {sw / max(rows, 1):6.3f}  b2_misses/step {mi / n:6.2f}  step_ms p50 {sm:6.1f}")
