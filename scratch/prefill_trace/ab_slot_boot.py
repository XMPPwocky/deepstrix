#!/usr/bin/env python3
"""Slot-level block bootstrap for a two-arm knob A/B (switch log as ab_mirror_eval.py): every metric sample is tagged
with its SLOT (switch interval); resampling whole slots per arm keeps the job-mix correlation within a slot. Metrics:
first windows (s per 1K rows), steady windows >= 3584 rows (s), replay (s). Prints per arm n / median, and the
bootstrap 95% CI of median(B) - median(A) (B = the second arm value seen).
  ab_slot_boot.py SWITCH_LOG [SETTLE_S=30] [ITERS=4000] LOG [LOG...]"""
import bisect, collections, datetime, random, re, statistics, sys
A = re.compile(r"\x1b\[[0-9;]*m")
iso = lambda s: datetime.datetime.fromisoformat(s.rstrip("Z")).timestamp()
sw = []
for l in open(sys.argv[1]):
    m = re.match(r"^(\S+) ([A-Za-z0-9_]+)=([0-9.]+)$", l.strip())
    if m:
        sw.append((iso(m.group(1)), m.group(3)))
end = [iso(l.split()[0]) for l in open(sys.argv[1]) if " END " in l]
t_end = end[-1] if end else float("inf")
settle, iters = float(sys.argv[2]), int(sys.argv[3])
ts = [s[0] for s in sw]


def slot(t):
    i = bisect.bisect_right(ts, t) - 1
    if i < 0 or t > t_end or t - ts[i] < settle:
        return None
    return i


M = collections.defaultdict(lambda: collections.defaultdict(list))  # metric -> slot -> values
op = {}
for p in sys.argv[4:]:
    for l in open(p, errors="replace"):
        l = A.sub("", l)
        if "prefill_lm_window" in l:
            g = dict(re.findall(r"(\w+)=([0-9]+)", l))
            t = iso(l.split()[0])
            if " open " in l:
                op[g["tokens_total"]] = (t, g)
            elif " done " in l and g.get("tokens_total") in op:
                t0, go = op.pop(g["tokens_total"])
                i = slot(t0)
                if i is None or slot(t) != i:
                    continue
                rows, d = int(go["rows"]), t - t0
                if int(go["tokens_done"]) == 0:
                    M["first s/1K rows"][i].append(d * 1000 / rows)
                elif rows >= 3584:
                    M["steady s"][i].append(d)
        elif "prefill_job_replay" in l:
            i = slot(iso(l.split()[0]))
            m = re.search(r'elapsed_s="([0-9.]+)"', l)
            if i is not None and m:
                M["replay s"][i].append(float(m.group(1)))
arms = []
for _, v in sw:
    if v not in arms:
        arms.append(v)
a0, a1 = arms[0], arms[1]
rng = random.Random(1)
for name, by in M.items():
    sl = {a: [i for i in by if sw[i][1] == a and by[i]] for a in (a0, a1)}
    med = {a: statistics.median([x for i in sl[a] for x in by[i]]) if sl[a] else float("nan") for a in (a0, a1)}
    n = {a: sum(len(by[i]) for i in sl[a]) for a in (a0, a1)}
    diffs = []
    for _ in range(iters):
        res = {}
        for a in (a0, a1):
            pick = [rng.choice(sl[a]) for _ in sl[a]]
            vals = [x for i in pick for x in by[i]]
            res[a] = statistics.median(vals)
        diffs.append(res[a1] - res[a0])
    diffs.sort()
    lo, hi = diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]
    print(f"{name:16s} {a0}: n {n[a0]:4d} ({len(sl[a0])} slots) med {med[a0]:6.3f} | {a1}: n {n[a1]:4d} ({len(sl[a1])} slots) "
          f"med {med[a1]:6.3f} | {a1}-{a0} {med[a1] - med[a0]:+.3f} 95% CI [{lo:+.3f}, {hi:+.3f}]")
