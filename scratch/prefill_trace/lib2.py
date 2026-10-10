import sys, json, re, datetime, math, bisect
from collections import defaultdict, Counter
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from load import load, isnan, pct

ansi = re.compile(r'\x1b\[[0-9;]*m')
import os
DATE = os.environ.get('PF_DATE', '2026-10-10')


def utc_s(s):
    return datetime.datetime.fromisoformat(DATE + 'T' + s + '+00:00').timestamp()


def union(iv):
    iv = sorted(iv)
    tot, cs, ce = 0, None, None
    for s, e in iv:
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None:
        tot += ce - cs
    return tot


def merge(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


class Hub:
    """Hub evtrace extract + its server log; log<->raw drift corrected from hub_phase."""
    def __init__(self, fn, log, kinds=('hub_req', 'hub_phase', 'sys')):
        self.hdr, self.R = load(fn, set(kinds))
        self.r0, self.e0 = self.hdr['t_mono_raw_at_open'], self.hdr['t_realtime_at_open']
        logp = []
        if log:
            for line in open(log, errors='replace'):
                if 'ms.phase' in line:
                    line = ansi.sub('', line)
                    logp.append(datetime.datetime.fromisoformat(line[:26] + '+00:00').timestamp())
        pts = []
        for p in self.R['hub_phase']:
            tu = (p['t'] - self.r0 + self.e0) / 1e9
            if not logp:
                break
            b = min(logp, key=lambda x: abs(x - tu))
            if abs(b - tu) < float(os.environ.get('PF_TOL', '1.0')):
                pts.append((tu, b - tu))
        n = len(pts)
        self.drift_pts = pts
        if n >= 2:
            mx = sum(p[0] for p in pts) / n
            my = sum(p[1] for p in pts) / n
            sxx = sum((p[0] - mx) ** 2 for p in pts)
            self.slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx if sxx > 0 else 0.0
            self.icpt = my - self.slope * mx
        elif n == 1:
            self.slope, self.icpt = 0.0, pts[0][1]
        else:
            self.slope, self.icpt = 0.0, 0.0
        req = self.R['hub_req']
        self.pre = sorted((r for r in req if isnan(r['step'])), key=lambda r: r['t_submit'])
        self.dec = sorted((r for r in req if not isnan(r['step'])), key=lambda r: r['t_submit'])
        best = {}
        for h in self.pre + self.dec[::5]:
            if isnan(h['clock_offset_ns']):
                continue
            k = int(h['t1'] // 2e9)
            if k not in best or h['clock_delay_ns'] < best[k][1]:
                best[k] = (h['t1'], h['clock_delay_ns'], h['clock_offset_ns'])
        self.opts = sorted((t, o) for t, _, o in best.values())
        self.ot = [p[0] for p in self.opts]

    def raw(self, s):
        lu = utc_s(s)
        tu = lu - (self.icpt + self.slope * lu)
        return self.r0 + tu * 1e9 - self.e0

    def utc(self, t):
        tu = (t - self.r0 + self.e0) / 1e9
        lu = tu + self.icpt + self.slope * tu
        return datetime.datetime.fromtimestamp(lu, datetime.UTC).strftime('%H:%M:%S.%f')[:-3]

    def off(self, t):
        pts, i = self.opts, bisect.bisect_left(self.ot, t)
        if i <= 0:
            return pts[0][1]
        if i >= len(pts):
            return pts[-1][1]
        (t0, o0), (t1, o1) = pts[i - 1], pts[i]
        return o0 + (o1 - o0) * (t - t0) / (t1 - t0)


def windows(path):
    out = []
    for line in open(path):
        if not line.strip() or line.startswith('#'):
            continue
        lab, a, b, rows = line.split()
        out.append((lab, a, b, int(rows)))
    return out


GROUPS = [(0, 2), (2, 8), (8, 14), (14, 20), (20, 21)]


def gid(r):
    return next((i for i, (s, e) in enumerate(GROUPS) if s <= r['layer'] < e), 9)


def units_of(rs):
    """split prefill requests (time-sorted) into units (group, sub-chunk)."""
    if not rs:
        return []
    units, cur = [], [rs[0]]
    for p, q in zip(rs, rs[1:]):
        if gid(q) != gid(p) or q['layer'] < p['layer'] or (q['layer'] == p['layer'] and q['lane'] == p['lane']):
            units.append(cur)
            cur = []
        cur.append(q)
    units.append(cur)
    return units
