#!/usr/bin/env python3
"""Per-window GAPS of a layer-major prefill job: inside each window (hub log `prefill_lm_window open/done`), the time
with NO box-2 request outstanding on any lane (union of [t_submit, t_wait_exit] of the window's hub_req records,
b >= 64). Prints per window: duration, request-covered time, gap total, the largest gaps with their offset from the
window open, and a histogram of gap positions. Run on box 1 (hub clocks).
  window_gaps.py HUBREQ_TXT SERVER_LOG TOKENS_TOTAL WALL_FROM_ISO WALL_TO_ISO"""
import datetime, re, sys, time

hub_txt, log, total, a_iso, b_iso = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
now_raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
now_wall = datetime.datetime.now(datetime.timezone.utc).timestamp()
to_wall = lambda raw: now_wall - (now_raw - raw) / 1e9
iso = lambda s: datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc).timestamp()

A = re.compile(r"\x1b\[[0-9;]*m")
wins, o = [], None
for l in open(log, errors="replace"):
    l = A.sub("", l)
    m = re.match(r"^(\S+)Z .*prefill_lm_window (open|done) .*?tokens_total=(\d+)", l)
    if not m or int(m.group(3)) != total:
        continue
    t = iso(m.group(1))
    if not (iso(a_iso) <= t <= iso(b_iso)):
        continue
    if m.group(2) == "open":
        o = t
    elif o is not None:
        wins.append((o, t))
        o = None

fields, reqs = None, []
for line in open(hub_txt):
    if line.startswith("#F hub_req "):
        fields = line.split()[2:]
    elif line.startswith("hub_req ") and fields:
        r = dict(zip(fields, (float(x) for x in line.split()[1:])))
        if r["b"] >= 64:
            reqs.append((to_wall(r["t_submit"]), to_wall(r["t_wait_exit"]), int(r["layer"]), int(r["lane"])))


def union(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


tot_d = tot_gap = 0.0
hist = {}
for k, (ws, we) in enumerate(wins):
    iv = [(max(s, ws), min(e, we)) for s, e, _, _ in reqs if e > ws and s < we]
    u = union(iv)
    cov = sum(e - s for s, e in u)
    gaps, cur = [], ws
    for s, e in u:
        if s > cur:
            gaps.append((cur - ws, s - cur))
        cur = max(cur, e)
    if we > cur:
        gaps.append((cur - ws, we - cur))
    d = we - ws
    g = sum(x for _, x in gaps)
    if 0 < k < len(wins) - 1:
        tot_d += d
        tot_gap += g
        for off, x in gaps:
            b = int(off / d * 10)
            hist[b] = hist.get(b, 0) + x
    big = sorted(gaps, key=lambda x: -x[1])[:5]
    print(f"win {k:2d}: {d:5.2f} s, requests cover {cov:5.2f} s, gaps {g:5.2f} s in {len(gaps):3d}; largest "
          + ", ".join(f"{x * 1000:.0f} ms @+{off:.2f}s" for off, x in big))
if tot_d:
    print(f"\nsteady windows: {tot_d:.1f} s, gaps {tot_gap:.1f} s = {tot_gap / tot_d:.0%}")
    print("gap time by position in the window (tenths):", " ".join(f"{b / 10:.1f}:{hist.get(b, 0):.2f}s" for b in range(10)))
