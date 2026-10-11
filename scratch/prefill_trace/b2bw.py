#!/usr/bin/env python3
"""Box-2 disk read bandwidth inside a window, from b2win text (b2_read): per source (src), per route: count, read
latency (t_read_start -> t_read_end) p10/p50/p90, the three part reads (r0/r1/r2) latency, achieved concurrency (time-
weighted mean reads in flight, from the [t_read_start, t_read_end] intervals) and GB/s at 18.8 MB per expert read; plus
the queue wait (t_hint/t_pop -> t_read_start) and land time.
  b2bw.py TXT W0_ISO W1_ISO SPAN0_ISO   (box-2 wall; SPAN0 = the b2win range start)"""
import collections, datetime, sys

path, w0, w1, s0 = sys.argv[1:5]
F, reads = {}, []
for line in open(path):
    if line.startswith("#F "):
        p = line.split()
        F[p[1]] = p[2:]
        continue
    if line.startswith("b2_read ") and "b2_read" in F:
        reads.append(dict(zip(F["b2_read"], (float(x) for x in line.split()[1:]))))
lo = int([l for l in open(path) if l.startswith("# ranges raw ")][0].split()[-1].split(":")[0])
iso = lambda s: datetime.datetime.fromisoformat(s).timestamp()
W0 = lo + (iso(w0) - iso(s0)) * 1e9
W1 = lo + (iso(w1) - iso(s0)) * 1e9
R = [r for r in reads if r["t_read_start"] > 0 and r["t_read_end"] > r["t_read_start"] and W0 <= r["t_read_start"] <= W1]
MB = 18.8e6


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def conc(iv, a, b):
    ev = sorted([(max(s, a), 1) for s, e in iv if e > a and s < b] + [(min(e, b), -1) for s, e in iv if e > a and s < b])
    cur, last, acc, busy = 0, a, 0.0, 0.0
    for t, d in ev:
        acc += cur * (t - last)
        busy += (t - last) if cur > 0 else 0
        cur += d
        last = t
    return acc / (b - a), busy / (b - a)


dur = (W1 - W0) / 1e9
print(f"window {dur:.2f} s: {len(R)} reads = {len(R) * MB / 1e9:.1f} GB -> {len(R) * MB / 1e9 / dur:.2f} GB/s overall")
for key in ("src", "route"):
    for v, sub in sorted(collections.defaultdict(list, {k: [r for r in R if r[key] == k] for k in {r[key] for r in R}}).items()):
        lat = [(r["t_read_end"] - r["t_read_start"]) / 1e6 for r in sub]
        c, busy = conc([(r["t_read_start"], r["t_read_end"]) for r in sub], W0, W1)
        qw = [(r["t_read_start"] - r["t_pop"]) / 1e6 for r in sub if r["t_pop"] > 0]
        print(f"  {key}={v:.0f}: n {len(sub):5d}  read ms p10 {q(lat, .1):6.1f} p50 {q(lat, .5):6.1f} p90 {q(lat, .9):6.1f}"
              f"  in flight mean {c:5.2f} (busy {busy:.0%})  {len(sub) * MB / 1e9 / dur:5.2f} GB/s  pop->start p50 {q(qw, .5):.1f} ms"
              f"  per-read {MB / 1e6 / max(q(lat, .5), 1e-9):.2f} GB/s")
parts = [[(r[f"r{i}_end"] - r[f"r{i}_start"]) / 1e6 for r in R if r[f"r{i}_end"] > r[f"r{i}_start"] > 0] for i in range(3)]
print("  part reads r0/r1/r2 ms p50:", " / ".join(f"{q(p, .5):.1f}" for p in parts), " n", [len(p) for p in parts])
c, busy = conc([(r["t_read_start"], r["t_read_end"]) for r in R], W0, W1)
print(f"  all reads: in flight mean {c:.2f}, any read in flight {busy:.0%} of the window")
print("  demand_reads_at_start p50", q([r["demand_reads_at_start"] for r in R], .5), " run_certain p50", q([r["run_certain_at_start"] for r in R], .5),
      " run_spec p50", q([r["run_spec_at_start"] for r in R], .5), " wanted share", f"{sum(1 for r in R if r['wanted'] > 0) / max(len(R), 1):.0%}",
      " repack ms p50", f"{q([r['repack_ns'] / 1e6 for r in R], .5):.2f}", " land ms p50", f"{q([(r['t_land_end'] - r['t_land_start']) / 1e6 for r in R if r['t_land_end'] > 0], .5):.2f}")
