#!/usr/bin/env python3
"""Box-1 (hub) disk + memory per layer-major window of one job: from `evtrace.py csv -k sys` of the hub's .evt
(CLOCK_MONOTONIC_RAW ns, mapped to wall time with THIS box's current raw-vs-wall offset -- run on box 1) and the hub
log's prefill_lm_window open/done lines. Per window: seconds, d1 (YMTC, model disk) read GB + GB/s + mean IO KB,
d0 read GB, min MemAvailable, page-cache GB at the end.
  b1_win_disk.py SYS_CSV SERVER_LOG JOB_TOKENS_TOTAL [N_WINDOWS=16]"""
import csv, datetime, re, sys, time
A = re.compile(r"\x1b\[[0-9;]*m")
off = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) / 1e9 - time.time()
iso = lambda s: datetime.datetime.fromisoformat(s.rstrip("Z")).replace(tzinfo=datetime.timezone.utc).timestamp()
rows = [r for r in csv.DictReader(open(sys.argv[1]))]
ts = [float(r["t"]) / 1e9 - off for r in rows]
job, nwin = sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 16
wins, t_open = [], None
for l in open(sys.argv[2], errors="replace"):
    l = A.sub("", l)
    if "prefill_lm_window" not in l or f"tokens_total={job}" not in l:
        continue
    t = iso(l.split()[0])
    if " open " in l:
        t_open = t
    elif " done " in l and t_open is not None:
        wins.append((t_open, t))
        t_open = None


def at(t):
    i = min(range(len(ts)), key=lambda k: abs(ts[k] - t))
    return rows[i]


for k, (a, b) in enumerate(wins[:nwin]):
    ra, rb = at(a), at(b)
    d = lambda f: float(rb[f]) - float(ra[f])
    gb1, n1, gb0 = d("d1_rsect") * 512 / 1e9, d("d1_reads"), d("d0_rsect") * 512 / 1e9
    inwin = [r for r, t in zip(rows, ts) if a <= t <= b]
    mem = min(float(r["mem_avail_kb"]) for r in inwin) / 1e6 if inwin else float("nan")
    print(f"w{k + 1:2d} {b - a:5.1f} s  YMTC read {gb1:5.2f} GB ({gb1 / (b - a):4.2f} GB/s, {gb1 * 1e6 / max(n1, 1):5.0f} KB/IO)  "
          f"E100 read {gb0:5.2f} GB  MemAvail min {mem:4.1f} GB  cached {float(rb['cached_kb']) / 1e6:4.1f} GB")
