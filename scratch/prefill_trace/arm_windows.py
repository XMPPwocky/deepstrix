#!/usr/bin/env python3
"""Per-arm layer-major WINDOW times from the hub log, for arms_p1p2a.py runs (its summary.json gives each arm's t0 and wall).

Window time = `prefill_lm_window open` -> matching `done` (same tokens_total, next done). Wall time per request swings with
other traffic (decode bursts interleave between windows); a window is one atomic pass sequence, so its time is the cleaner
per-arm number. Reports median / p25 / p75 / mean over the arm's 82K-token job's steady windows (the first window, which
includes job start-up and a cold pool, is reported separately), plus rows/s at the median.

  arm_windows.py [summary.json] [server.log]
"""
import json, re, statistics, sys
from datetime import datetime, timezone

SUMMARY = sys.argv[1] if len(sys.argv) > 1 else "/home/claude-code/gateb_20261010/summary.json"
LOG = sys.argv[2] if len(sys.argv) > 2 else "/home/claude-code/logs/v41-server.log"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RE = re.compile(r"^(\S+)Z\s+INFO .*prefill_lm_window (open|done) .*?rows=(\d+).*?tokens_total=(\d+)")
LO = re.compile(r"layer_outer=(\w+)")


def ts(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()


events = []
with open(LOG, errors="replace") as f:
    for line in f:
        line = ANSI.sub("", line)
        m = RE.match(line)
        if m:
            lo = LO.search(line)
            events.append((ts(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)), lo.group(1) if lo else None))

runs = json.load(open(SUMMARY))["runs"]
print(f"{'arm':9} {'wall_s':>7} {'n':>3} {'first_s':>7} {'med_s':>6} {'p25':>6} {'p75':>6} {'mean':>6} {'rows/s@med':>10}  layer_outer")
for r in runs:
    t0, t1 = r["t0"], r["t0"] + r["wall_s"]
    n_prompt = r["usage"].get("prompt_tokens", 0)
    ev = [e for e in events if t0 <= e[0] <= t1 + 1 and abs(e[3] - n_prompt) <= 2]
    durs, lo, open_t = [], set(), None
    for t, kind, rows, total, l in ev:
        if kind == "open":
            open_t = t
            if l:
                lo.add(l)
        elif open_t is not None:
            durs.append((t - open_t, rows))
            open_t = None
    if len(durs) < 3:
        print(f"{r['arm']:9} {r['wall_s']:7.1f}  too few windows ({len(durs)})")
        continue
    first, steady = durs[0][0], [d for d, rows in durs[1:] if rows == durs[1][1]]
    q = statistics.quantiles(steady, n=4)
    med = statistics.median(steady)
    print(f"{r['arm']:9} {r['wall_s']:7.1f} {len(durs):3d} {first:7.2f} {med:6.2f} {q[0]:6.2f} {q[2]:6.2f} "
          f"{statistics.mean(steady):6.2f} {durs[1][1] / med:10.0f}  {sorted(lo)}")
