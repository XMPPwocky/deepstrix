#!/usr/bin/env python3
"""Per FIRST layer-major window (hub log, tokens_done=0): each box's disk reads during it, from disksample.sh logs
(box-2 times shifted by B2_SKEW_S, its clock lead): per device GB read, GB/s, and busy share (ms_reading delta /
window). Answers: while box 2 pages a cold window, is box 1's disk idle (room to shift the split toward box 1)?
  cold_disks.py HUB_LOG DISK_B1 DISK_B2 B2_SKEW_S [SINCE_ISO]"""
import bisect, collections, datetime, re, sys
A = re.compile(r"\x1b\[[0-9;]*m")
hub, d1, d2, skew = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
since = sys.argv[5] if len(sys.argv) > 5 else ""
iso = lambda s: datetime.datetime.fromisoformat(s.rstrip("Z")).replace(tzinfo=datetime.timezone.utc).timestamp()


def load(path, shift):
    s = collections.defaultdict(list)  # dev -> [(t, sectors, ms_reading)]
    for l in open(path):
        p = l.split()
        if len(p) == 5:
            s[p[1]].append((float(p[0]) - shift, int(p[2]), int(p[3])))
    return s


B = {"box1": load(d1, 0.0), "box2": load(d2, skew)}
NAMES = {"box1": {"nvme0n1": "E100", "nvme1n1": "YMTC"}, "box2": {"nvme0n1": "SN5000", "nvme1n1": "990Pro"}}


def at(series, t):
    ts = [x[0] for x in series]
    i = bisect.bisect_left(ts, t)
    return series[min(i, len(series) - 1)] if series else None


op, wins = {}, []
for l in open(hub, errors="replace"):
    l = A.sub("", l)
    if "prefill_lm_window" not in l or l[:19] < since:
        continue
    g = dict(re.findall(r"(\w+)=([0-9]+)", l))
    t = iso(l.split()[0])
    if " open " in l:
        op[g["tokens_total"]] = (t, g)
    elif " done " in l and g.get("tokens_total") in op:
        t0, go = op.pop(g["tokens_total"])
        if go["tokens_done"] == "0":
            wins.append((t0, t, int(go["rows"])))
agg = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
for t0, t1, rows in wins:
    d = t1 - t0
    cells = []
    for box in ("box1", "box2"):
        for dev, ser in sorted(B[box].items()):
            a, b = at(ser, t0), at(ser, t1)
            if not a or not b or b[0] <= a[0]:
                continue
            gb = (b[1] - a[1]) * 512 / 1e9
            busy = (b[2] - a[2]) / 1e3 / (b[0] - a[0])
            name = f"{box}:{NAMES[box].get(dev, dev)}"
            cells.append(f"{name} {gb:5.1f} GB {gb / d:4.2f} GB/s busy {busy:4.0%}")
            x = agg[name]
            x[0] += gb
            x[1] += d
            x[2] += busy * d
    print(f"{datetime.datetime.fromtimestamp(t0, datetime.timezone.utc):%H:%M:%S} rows {rows:5d} {d:5.1f} s | " + " | ".join(cells))
if wins:
    print("\nall first windows:")
    for k, (gb, s, b) in sorted(agg.items()):
        print(f"  {k:14} {gb:7.1f} GB over {s:6.1f} s = {gb / s:4.2f} GB/s, busy {b / s:4.0%}")
