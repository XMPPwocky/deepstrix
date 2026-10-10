"""Box-2 b2_read stats inside steady windows. usage: reads.py HUB LOG WIN SPLIT B2FILES MAXS"""
import sys, json
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from lib2 import *

H = Hub(sys.argv[1], sys.argv[2])
S = {r['lab']: r for r in json.load(open(sys.argv[4]))}
MAXS = int(sys.argv[6])
RD = []
for fn in sys.argv[5].split(','):
    _, RB = load(fn, {'b2_read'})
    RD += RB['b2_read']
print('reads total', len(RD), 'fields src values', Counter(int(r['src']) for r in RD), 'route', Counter(int(r['route']) for r in RD if not isnan(r['route'])))
RD.sort(key=lambda r: r['t_read_start'])
ts = [r['t_read_start'] for r in RD]
agg = defaultdict(list)
wins = 0
wall = 0.0
infl = 0.0
for lab, a, b, rows in windows(sys.argv[3]):
    s = S.get(lab)
    if not s or s['kind'] != 'W' or s['ndec'] > 0 or rows != 4096 or s['nmiss'] > MAXS:
        continue
    lo, hi = H.raw(a), H.raw(b)
    o = H.off(lo)
    blo, bhi = lo + o, hi + o
    i0 = bisect.bisect_left(ts, blo)
    i1 = bisect.bisect_left(ts, bhi)
    wins += 1
    wall += (hi - lo) / 1e9
    for r in RD[i0:i1]:
        d = (r['t_read_end'] - r['t_read_start']) / 1e6
        q = (r['t_read_start'] - r['t_pop']) / 1e6 if not isnan(r['t_pop']) else float('nan')
        h2r = (r['t_read_end'] - r['t_hint']) / 1e6 if not isnan(r['t_hint']) else float('nan')
        k = int(r['src'])
        agg[(k, 'dur')].append(d)
        agg[(k, 'q')].append(q)
        agg[(k, 'hint2end')].append(h2r)
        agg[(k, 'land')].append((r['t_land_end'] - r['t_land_start']) / 1e6 if not isnan(r['t_land_end']) else float('nan'))
        infl += d / 1e3
print('windows', wins, 'wall', wall)
for k in sorted(set(k for k, _ in agg)):
    n = len(agg[(k, 'dur')])
    print(f"src {k}: n={n} ({n / wins:.0f}/window)  read dur ms p10/p50/p90 {pct(agg[(k, 'dur')], .1):.2f}/{pct(agg[(k, 'dur')], .5):.2f}/{pct(agg[(k, 'dur')], .9):.2f}  pop->start p50 {pct(agg[(k, 'q')], .5):.2f}  hint->end p50 {pct(agg[(k, 'hint2end')], .5):.2f}  land p50 {pct(agg[(k, 'land')], .5):.2f}")
print('mean reads in flight (sum of read durations / wall):', infl / wall)
