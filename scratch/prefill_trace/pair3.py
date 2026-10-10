"""Unit-layer pair timeline (steady windows): relative ms from first hub submit, medians; plus park/served_under.
usage: pair.py HUBEXTRACT LOG WINFILE SPLITJSON MAXS B2FILES [LAYERFILTER]"""
import sys, json
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from lib2 import *

H = Hub(sys.argv[1], sys.argv[2])
S = {r['lab']: r for r in json.load(open(sys.argv[4]))}
MAXS = int(sys.argv[5])
B = {}
for fn in sys.argv[6].split(','):
    _, RB = load(fn, {'b2_req'})
    for b in RB['b2_req']:
        B[(int(b['seq']), int(b['t_frame']))] = b
P = defaultdict(list)
cnt = Counter()
for lab, a, b, rows in windows(sys.argv[3]):
    s = S.get(lab)
    if not s or s['kind'] != 'W' or s['ndec'] > 0 or rows != 4096 :
        continue
    lo, hi = H.raw(a), H.raw(b)
    rs = [r for r in H.pre if lo <= r['t_submit'] <= hi]
    for u in units_of(rs):
        bl = defaultdict(dict)
        for r in u:
            bl[int(r['layer'])][int(r['lane'])] = r
        for L, d in bl.items():
            if 0 in d and 1 in d:
                h0, h1 = d[0], d[1]
                b0 = B.get((int(h0['seq']), int(h0['t2_b2'])))
                b1 = B.get((int(h1['seq']), int(h1['t2_b2'])))
                if not b0 or not b1:
                    continue
                nmm = b0['n_missing'] + b1['n_missing']
                if not (int(sys.argv[5]) <= nmm <= int(sys.argv[7])):
                    continue
                o = H.off(h0['t1'])
                t0 = h0['t_submit']
                f = lambda x: (x - o - t0) / 1e6      # box-2 stamp -> ms since lane-0 submit
                g = lambda x: (x - t0) / 1e6
                P['l0 submit'].append(0.0)
                P['l1 submit'].append(g(h1['t_submit']))
                P['l0 arrive b2'].append(f(h0['t2_b2']))
                P['l1 arrive b2'].append(f(h1['t2_b2']))
                P['l0 run_start'].append(f(b0['t_run_start']))
                P['l0 run_end'].append(f(b0['t_run_end']))
                P['l1 run_start'].append(f(b1['t_run_start']))
                P['l1 run_end'].append(f(b1['t_run_end']))
                P['l0 t3 (send)'].append(f(h0['t3_b2']))
                P['l1 t3 (send)'].append(f(h1['t3_b2']))
                P['l0 t4 (hub recv)'].append(g(h0['t4']))
                P['l1 t4 (hub recv)'].append(g(h1['t4']))
                P['l0 wait_exit'].append(g(h0['t_wait_exit']))
                P['l1 wait_exit'].append(g(h1['t_wait_exit']))
                P['l0 next submit'].append(None)
                P['l0 park_ms'].append(b0['park_wait_ns'] / 1e6 if not isnan(b0['park_wait_ns']) else 0)
                P['l1 park_ms'].append(b1['park_wait_ns'] / 1e6 if not isnan(b1['park_wait_ns']) else 0)
                P['l0 served_under'].append(0 if isnan(b0['served_under']) else 1)
                P['l1 served_under'].append(0 if isnan(b1['served_under']) else 1)
                P['l0 n_missing'].append(b0['n_missing'])
                P['l1 n_missing'].append(b1['n_missing'])
                P['l0 n_paged'].append(b0['n_paged'])
                P['l1 n_paged'].append(b1['n_paged'])
                P['l0 compute_us'].append(b0['compute_us'] / 1e3)
                P['l1 compute_us'].append(b1['compute_us'] / 1e3)
                P['l0 page_us'].append(b0['page_us'] / 1e3)
                P['l1 page_us'].append(b1['page_us'] / 1e3)
                P['l1 merged'].append(b1['merged'])
                cnt['pairs'] += 1
print('pairs', cnt['pairs'])
for k, v in P.items():
    v = [x for x in v if x is not None]
    if v:
        print(f"  {k:<18} p10 {pct(v, .1):8.2f}  p50 {pct(v, .5):8.2f}  mean {sum(v) / len(v):8.2f}  p90 {pct(v, .9):8.2f}")
