import sys, json, re, datetime, math, bisect
from collections import defaultdict
sys.path.insert(0, '/home/claude-code/.claude/jobs/fe82922a/tmp/pf')
from load import load, isnan, pct

D = '/home/claude-code/.claude/jobs/fe82922a/tmp/pf/'
HUBF = {'0906': (D + 'hub_0906.txt', '/home/claude-code/logs/v41-server.log.20261001-193833'),
        '1939': (D + 'hub_1939.txt', '/home/claude-code/logs/v41-server.log'),
        '1703': (D + 'hub_1703.txt', '/home/claude-code/logs/v41-server.log.20261001-190550')}
ansi = re.compile(r'\x1b\[[0-9;]*m')


def utc_s(s):
    return datetime.datetime.fromisoformat('2026-10-01T' + s + '+00:00').timestamp()


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
    """One hub evtrace extract + its server log; log<->raw drift corrected from hub_phase."""
    def __init__(self, key, kinds=('hub_req', 'hub_phase', 'sys')):
        fn, log = HUBF[key]
        self.hdr, self.R = load(fn, set(kinds))
        self.r0, self.e0 = self.hdr['t_mono_raw_at_open'], self.hdr['t_realtime_at_open']
        # drift fit: log_utc - evt_utc vs evt_utc
        logp = []
        for line in open(log, errors='replace'):
            if 'ms.phase' in line:
                line = ansi.sub('', line)
                logp.append(datetime.datetime.fromisoformat(line[:26] + '+00:00').timestamp())
        pts = []
        for p in self.R['hub_phase']:
            tu = (p['t'] - self.r0 + self.e0) / 1e9
            b = min(logp, key=lambda x: abs(x - tu))
            if abs(b - tu) < 1.0:
                pts.append((tu, b - tu))
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        self.slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx if sxx > 0 else 0.0
        self.icpt = my - self.slope * mx
        self.drift_pts = pts
        req = self.R['hub_req']
        self.pre = sorted((r for r in req if isnan(r['step'])), key=lambda r: r['t_submit'])
        self.dec = sorted((r for r in req if not isnan(r['step'])), key=lambda r: r['t_submit'])
        # smoothed clock offset (b2 raw - hub raw): min delay per 2 s bucket
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


def windows(path=D + 'allwin_now.txt'):
    out = []
    for line in open(path):
        if not line.strip() or line.startswith('#'):
            continue
        lab, a, b, rows = line.split()
        out.append((lab, a, b, int(rows)))
    return out


def hubkey(a):
    if a < '19:05:00':
        return '1703'
    return '0906' if a < '19:39:00' else '1939'


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


def load_b2(files=('b2_000.txt', 'b2_001.txt', 'b2_002.txt'), kinds=('b2_req', 'b2_read', 'b2_ensure', 'sys', 'b2_write')):
    B, W = {}, {}
    out = defaultdict(list)
    for fn in files:
        _, RB = load(D + fn, set(kinds))
        for b in RB.get('b2_req', []):
            B[(int(b['seq']), int(b['t_frame']))] = b
        for w in RB.get('b2_write', []):
            W[(int(w['seq']), int(w['t3']))] = w
        for k in kinds:
            if k not in ('b2_req', 'b2_write'):
                out[k] += RB.get(k, [])
    for k in out:
        tf = {'sys': 't', 'b2_read': 't_read_start', 'b2_ensure': 't_start'}[k]
        out[k].sort(key=lambda r: r[tf] if not isnan(r[tf]) else 0)
    return B, W, out


COLD = {'J1921', 'J2023', 'J2037', 'J2045', 'J2105', 'J1731', 'J1754', 'J1809', 'J1813'}
AFTER_BURST = {'J2105w9', 'J2105w22', 'J2105w34', 'J2105w47'}


def wclass(lab, rows):
    j = re.match(r'(J\d+)', lab).group(1)
    if lab.endswith('R'):
        return 'replay'
    if lab.endswith('D'):
        return 'burst'
    if j in COLD:
        if lab.endswith('w1') or lab in AFTER_BURST:
            return 'first'
        return 'steady' if rows == 4096 else 'lastpart'
    return 'cont'
