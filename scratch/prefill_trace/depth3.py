"""Pipeline depth occupancy + turnaround/period per class.
usage: depth3.py HUBEXTRACT LOG WINFILE SPLITJSON MAXS MINH [B2FILES]"""
import sys, json
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from lib2 import *

H = Hub(sys.argv[1], sys.argv[2])
S = {r['lab']: r for r in json.load(open(sys.argv[4]))}
MAXS, MINH = int(sys.argv[5]), int(sys.argv[6])
B = {}
if len(sys.argv) > 7:
    for fn in sys.argv[7].split(','):
        _, RB = load(fn, {'b2_req'})
        for b in RB['b2_req']:
            B[(int(b['seq']), int(b['t_frame']))] = b
T = defaultdict(lambda: defaultdict(float))
WAITCTX = defaultdict(Counter)
TURN = defaultdict(list)
PERIOD = defaultdict(list)
UNITD = defaultdict(lambda: defaultdict(list))
UL = defaultdict(list)        # unit-layer pair: first submit -> last wait_exit
ULP = defaultdict(list)       # unit-layer period: first submit of layer L -> first submit of L+1
RTTL = defaultdict(list)
NW = Counter()
B2B = defaultdict(float)      # box-2 GPU busy (union run) seconds
B2R = defaultdict(float)
LINK = defaultdict(lambda: [0.0, 0.0])
WALL = defaultdict(float)
for lab, a, b, rows in windows(sys.argv[3]):
    s = S.get(lab)
    if not s or s['kind'] != 'W' or s['ndec'] > 0 or rows != 4096:
        continue
    c = 'steady' if s['nmiss'] <= MAXS else ('heavy' if s['nmiss'] >= MINH else 'mid')
    NW[c] += 1
    lo, hi = H.raw(a), H.raw(b)
    WALL[c] += (hi - lo) / 1e9
    rs = [r for r in H.pre if lo <= r['t_submit'] <= hi]
    us = units_of(rs)
    ev = []
    for r in rs:
        ev.append((r['t_submit'], 1, 'out'))
        ev.append((r['t4'], -1, 'out'))
        if r['t_wait_exit'] > r['t_wait_enter']:
            ev.append((r['t_wait_enter'], 1, 'blk'))
            ev.append((r['t_wait_exit'], -1, 'blk'))
    for u in us:
        ev.append((min(r['t_submit'] for r in u) - 1, 1, 'unit'))
        ev.append((max(r['t_wait_exit'] for r in u), -1, 'unit'))
    ev.append((lo, 0, 'x'))
    ev.append((hi, 0, 'x'))
    ev.sort(key=lambda e: (e[0], e[1]))
    st = {'out': 0, 'blk': 0, 'unit': 0}
    prev = lo
    for t, d, k in ev:
        t = min(max(t, lo), hi)
        if t > prev:
            hs = 'blocked' if st['blk'] > 0 else ('busy' if st['unit'] > 0 else 'gap')
            T[c][(min(st['out'], 3), hs)] += (t - prev) / 1e9
            prev = t
        if k in st:
            st[k] += d
    by = {(int(r['layer']), int(r['lane']), u_i): r for u_i, u in enumerate(us) for r in u}
    for u_i, u in enumerate(us):
        for r in u:
            if r['t_wait_exit'] - r['t_wait_enter'] < 0.2e6:
                continue
            other = by.get((int(r['layer']), 1 - int(r['lane']), u_i))
            nout = sum(1 for q in u if q is not r and q['t_submit'] <= r['t_wait_enter'] < q['t4'])
            if other is None:
                WAITCTX[c]['no other lane req (same layer)'] += 1
            elif other['t_submit'] <= r['t_wait_enter']:
                WAITCTX[c]['other lane L submitted before wait'] += 1
            else:
                WAITCTX[c]['other lane L NOT yet submitted'] += 1
            WAITCTX[c][f'other reqs outstanding during wait: {nout}'] += 1
    for u in us:
        g = gid(u[0])
        for ln in (0, 1):
            v = [r for r in u if int(r['lane']) == ln]
            for p, q in zip(v, v[1:]):
                if q['layer'] == p['layer'] + 1:
                    TURN[(c, ln)].append((q['t_submit'] - p['t_wait_exit']) / 1e6)
                    PERIOD[(c, ln)].append((q['t_submit'] - p['t_submit']) / 1e6)
        nl = len(set(int(r['layer']) for r in u))
        UNITD[c][f'G{g} ({nl} layers)'].append((max(r['t_wait_exit'] for r in u) - min(r['t_submit'] for r in u)) / 1e6)
        # per layer pairs
        bl = defaultdict(list)
        for r in u:
            bl[int(r['layer'])].append(r)
        ls = sorted(bl)
        for L in ls:
            UL[c].append((max(r['t_wait_exit'] for r in bl[L]) - min(r['t_submit'] for r in bl[L])) / 1e6)
        for L0, L1 in zip(ls, ls[1:]):
            if L1 == L0 + 1:
                ULP[c].append((min(r['t_submit'] for r in bl[L1]) - min(r['t_submit'] for r in bl[L0])) / 1e6)
    # box-2 GPU busy and link
    iv, ivr, ivo = [], [], []
    for r in rs:
        bb = B.get((int(r['seq']), int(r['t2_b2'])))
        if bb:
            o = H.off(r['t1'])
            iv.append(((bb['t_run_start'] - o), (bb['t_run_end'] - o)))
            ivr.append((r['t3_b2'] - o, r['t4']))     # wire back (approx incl. write)
            ivo.append((r['t1'], r['t2_b2'] - o))
    B2B[c] += union([(max(x, lo), min(y, hi)) for x, y in iv if y > lo and x < hi]) / 1e9
    LINK[c][0] += union([(max(x, lo), min(y, hi)) for x, y in ivo if y > lo and x < hi]) / 1e9
    LINK[c][1] += union([(max(x, lo), min(y, hi)) for x, y in ivr if y > lo and x < hi]) / 1e9
for c in ('steady', 'mid', 'heavy'):
    if c not in T:
        continue
    tot = sum(T[c].values())
    print(f'== {c}: {NW[c]} windows, {tot:.1f} s of window time ({WALL[c] / NW[c]:.2f} s/window)')
    for hs in ('blocked', 'busy', 'gap'):
        row = [T[c][(n, hs)] for n in range(4)]
        print(f"   hub {hs:<8} " + '  '.join(f"out={n}{'+' if n == 3 else ''}: {100 * row[n] / tot:5.1f}%" for n in range(4)) + f"   (sum {100 * sum(row) / tot:5.1f}%)")
    tn = [sum(T[c][(n, hs)] for hs in ('blocked', 'busy', 'gap')) for n in range(4)]
    print('   outstanding histogram (time share): ' + '  '.join(f"{n}{'+' if n == 3 else ''}: {100 * tn[n] / tot:5.1f}%" for n in range(4)))
    print('   waits (>0.2 ms):', dict(WAITCTX[c]))
    if B:
        print(f"   box-2 GPU run-busy {100 * B2B[c] / tot:.1f}%  link out-busy {100 * LINK[c][0] / tot:.1f}%  link back-busy {100 * LINK[c][1] / tot:.1f}% (wire-leg intervals, union)")
    for ln in (0, 1):
        print(f"   lane {ln}: turnaround wait_exit(L)->submit(L+1) p10/p50/p90 {pct(TURN[(c, ln)], .1):.1f}/{pct(TURN[(c, ln)], .5):.1f}/{pct(TURN[(c, ln)], .9):.1f} ms;"
              f" period submit(L)->submit(L+1) p10/p50/p90 {pct(PERIOD[(c, ln)], .1):.1f}/{pct(PERIOD[(c, ln)], .5):.1f}/{pct(PERIOD[(c, ln)], .9):.1f} ms mean {sum(PERIOD[(c, ln)]) / len(PERIOD[(c, ln)]):.1f} (n={len(PERIOD[(c, ln)])})")
    print(f"   unit-layer (both lanes) first submit -> last reply: p10/p50/p90 {pct(UL[c], .1):.1f}/{pct(UL[c], .5):.1f}/{pct(UL[c], .9):.1f} mean {sum(UL[c]) / len(UL[c]):.1f} n={len(UL[c])}")
    print(f"   unit-layer period (first submit L -> first submit L+1): p10/p50/p90 {pct(ULP[c], .1):.1f}/{pct(ULP[c], .5):.1f}/{pct(ULP[c], .9):.1f} mean {sum(ULP[c]) / len(ULP[c]):.1f} n={len(ULP[c])}")
    print('   unit span ms p50 (p10/p90):', {k: f"{pct(v, .5):.0f} ({pct(v, .1):.0f}/{pct(v, .9):.0f}) n={len(v)}" for k, v in sorted(UNITD[c].items())})
