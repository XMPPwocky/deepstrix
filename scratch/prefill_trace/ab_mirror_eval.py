#!/usr/bin/env python3
"""Score the mirror_frac A/B: each layer-major window (hub log) and each DSpark blocks line gets the arm in force at its
time (switch log), skipping the first SETTLE_S seconds after a switch. Per arm: non-first windows >= 3584 rows (n, p50,
mean), first windows (n, p50 s and rows/s), decode tok/s and step_ms (DSpark blocks lines, mean).
  ab_mirror_eval.py SWITCH_LOG [SETTLE_S=30] LOG [LOG...]"""
import bisect, collections, datetime, re, sys
A = re.compile(r"\x1b\[[0-9;]*m")
iso = lambda s: datetime.datetime.fromisoformat(s.rstrip("Z")).timestamp()
sw = []
KNOB = ["knob"]
for l in open(sys.argv[1]):
    m = re.match(r"^(\S+) ([A-Za-z0-9_]+)=([0-9.]+)$", l.strip())
    if m:
        KNOB[0] = m.group(2)
        sw.append((iso(m.group(1)), m.group(3)))
end = [iso(l.split()[0]) for l in open(sys.argv[1]) if " END " in l]
t_end = end[-1] if end else float("inf")
settle = float(sys.argv[2])
ts = [s[0] for s in sw]


def arm(t):
    i = bisect.bisect_right(ts, t) - 1
    if i < 0 or t > t_end or t - ts[i] < settle:
        return None
    return sw[i][1]


op, W, D = {}, collections.defaultdict(lambda: {"steady": [], "first": []}), collections.defaultdict(list)
REP = collections.defaultdict(list)
for p in sys.argv[3:]:
    for l in open(p, errors="replace"):
        l = A.sub("", l)
        if "prefill_lm_window" in l:
            g = dict(re.findall(r"(\w+)=([0-9]+)", l))
            t = iso(l.split()[0])
            if " open " in l:
                op[g["tokens_total"]] = (t, g)
            elif " done " in l and g.get("tokens_total") in op:
                t0, go = op.pop(g["tokens_total"])
                a = arm(t0)
                if a is None or arm(t) != a:
                    continue
                rows, d = int(go["rows"]), t - t0
                if int(go["tokens_done"]) == 0:
                    W[a]["first"].append((d, rows))
                elif rows >= 3584:
                    W[a]["steady"].append(d)
        elif "prefill_job_replay" in l:
            a = arm(iso(l.split()[0]))
            m = re.search(r'elapsed_s="([0-9.]+)"', l)
            if a is not None and m:
                REP[a].append(float(m.group(1)))
        elif "ms dspark: blocks blocks=" in l:
            a = arm(iso(l.split()[0]))
            if a is not None:
                g = lambda k: float(re.search(k + r'="([0-9.]+)"', l).group(1))
                D[a].append((g("tok_per_s"), g("step_ms"), g("accept_rate")))
q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
for a in sorted(set(W) | set(D) | set(REP)):
    s, f, d = W[a]["steady"], W[a]["first"], D[a]
    print(f"{KNOB[0]}={a}: steady windows n {len(s):3d} p50 {q(s, .5):5.2f} s mean {sum(s) / max(len(s), 1):5.2f} | "
          f"first windows n {len(f):3d} p50 {q([x for x, _ in f], .5):5.2f} s, rows/s p50 {q([r / x for x, r in f], .5):5.0f} | "
          f"decode lines {len(d):4d} tok/s {sum(x[0] for x in d) / max(len(d), 1):5.1f} step_ms {sum(x[1] for x in d) / max(len(d), 1):5.1f} "
          f"accept {sum(x[2] for x in d) / max(len(d), 1):.3f} | replay n {len(REP[a]):3d} p50 {q(REP[a], .5):5.2f} s")
