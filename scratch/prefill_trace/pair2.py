"""Unit-layer period / span binned by total n_missing of the pair (both lanes). Windows: no decode, 4096 rows.
usage: pair2.py HUBEXTRACT LOG WINFILE SPLITJSON B2FILES"""
import sys, json
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from lib2 import *

H = Hub(sys.argv[1], sys.argv[2])
S = {r['lab']: r for r in json.load(open(sys.argv[4]))}
B = {}
for fn in sys.argv[5].split(','):
    _, RB = load(fn, {'b2_req'})
    for b in RB['b2_req']:
        B[(int(b['seq']), int(b['t_frame']))] = b
bins = [(0, 2), (3, 6), (7, 12), (13, 20), (21, 40), (41, 999)]
R = defaultdict(lambda: defaultdict(list))
tot = Counter()
for lab, a, b, rows in windows(sys.argv[3]):
    s = S.get(lab)
    if not s or s['kind'] != 'W' or s['ndec'] > 0 or rows != 4096:
        continue
    lo, hi = H.raw(a), H.raw(b)
    rs = [r for r in H.pre if lo <= r['t_submit'] <= hi]
    for u in units_of(rs):
        bl = defaultdict(dict)
        for r in u:
            bl[int(r['layer'])][int(r['lane'])] = r
        ls = sorted(bl)
        for L in ls:
            d = bl[L]
            if 0 not in d or 1 not in d:
                continue
            b0 = B.get((int(d[0]['seq']), int(d[0]['t2_b2'])))
            b1 = B.get((int(d[1]['seq']), int(d[1]['t2_b2'])))
            if not b0 or not b1:
                continue
            nm = b0['n_missing'] + b1['n_missing']
            k = next(i for i, (x, y) in enumerate(bins) if x <= nm <= y)
            span = (max(r['t_wait_exit'] for r in d.values()) - min(r['t_submit'] for r in d.values())) / 1e6
            R[k]['span'].append(span)
            R[k]['nm'].append(nm)
            R[k]['run0'].append((b0['t_run_end'] - b0['t_run_start']) / 1e6)
            R[k]['run1'].append((b1['t_run_end'] - b1['t_run_start']) / 1e6)
            R[k]['q1'].append((b1['t_dequeue'] - b1['t_frame']) / 1e6)
            R[k]['l1sub'].append((d[1]['t_submit'] - d[0]['t_submit']) / 1e6)
            if L + 1 in bl and 0 in bl[L + 1]:
                R[k]['period'].append((bl[L + 1][0]['t_submit'] - d[0]['t_submit']) / 1e6)
            tot[k] += 1
n = sum(tot.values())
print(f"unit-layers N={n}")
print(f"{'sum n_missing':<14}{'N':>6}{'share':>7} | {'span p50':>8}{'mean':>7} | {'period p50':>10}{'mean':>7} | run0 p50 run1 p50 | l1 queue p50 | l1 submit offset p50")
for k, (x, y) in enumerate(bins):
    v = R[k]
    if not v['span']:
        continue
    m = lambda key: sum(v[key]) / len(v[key])
    print(f"{x:>3}-{y:<10}{tot[k]:6d}{100 * tot[k] / n:6.1f}% | {pct(v['span'], .5):8.1f}{m('span'):7.1f} | {pct(v['period'], .5):10.1f}{m('period'):7.1f} | {pct(v['run0'], .5):7.1f} {pct(v['run1'], .5):7.1f} | {pct(v['q1'], .5):8.1f} | {pct(v['l1sub'], .5):6.1f}")
allp = [x for k in R for x in R[k]['period']]
alls = [x for k in R for x in R[k]['span']]
print('overall period p50/mean', pct(allp, .5), sum(allp) / len(allp), 'span p50/mean', pct(alls, .5), sum(alls) / len(alls))
