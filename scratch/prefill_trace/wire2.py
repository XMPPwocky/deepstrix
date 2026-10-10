"""Wire legs of bulk prefill requests using a clock offset fitted ONLY on small decode frames (min delay), linear drift.
usage: wire2.py HUB LOG WIN SPLIT B2FILES"""
import sys, json
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from lib2 import *

H = Hub(sys.argv[1], sys.argv[2])
S = {r['lab']: r for r in json.load(open(sys.argv[4]))}
best = {}
for h in H.dec:
    if isnan(h['clock_offset_ns']) or isnan(h['clock_delay_ns']):
        continue
    if h['bytes_out'] > 200000 or h['bytes_in'] > 200000:
        continue
    k = int(h['t1'] // 5e9)
    if k not in best or h['clock_delay_ns'] < best[k][1]:
        best[k] = (h['t1'], h['clock_delay_ns'], h['clock_offset_ns'])
pts = sorted(best.values())
print('decode min-delay offset samples', len(pts), 'delay ns min/med', pts and (min(p[1] for p in pts), sorted(p[1] for p in pts)[len(pts) // 2]))
x = [p[0] / 1e9 for p in pts]
y = [p[2] for p in pts]
n = len(x)
mx, my = sum(x) / n, sum(y) / n
b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sum((a - mx) ** 2 for a in x)
a0 = my - b * mx
res = [c - (a0 + b * a) for a, c in zip(x, y)]
print('drift ppm', b / 1e9 * 1e6 * 1e3 if False else b * 1e-9 * 1e6, 'residual std us', (sum(r * r for r in res) / n) ** .5 / 1e3, 'max abs us', max(abs(r) for r in res) / 1e3)
off2 = lambda t: a0 + b * (t / 1e9)
B, W = {}, {}
for fn in sys.argv[5].split(','):
    _, RB = load(fn, {'b2_req'})
    for bb in RB['b2_req']:
        B[(int(bb['seq']), int(bb['t_frame']))] = bb
P = defaultdict(list)
for lab, a, bq, rows in windows(sys.argv[3]):
    s = S.get(lab)
    if not s or s['kind'] != 'W' or s['ndec'] > 0 or rows != 4096:
        continue
    lo, hi = H.raw(a), H.raw(bq)
    for h in H.pre:
        if lo <= h['t_submit'] <= hi:
            o = off2(h['t1'])
            wo = (h['t2_b2'] - h['t1'] - o) / 1e6
            wb = (h['t4'] - h['t3_b2'] + o) / 1e6
            P['out'].append(wo)
            P['back'].append(wb)
            P['sum'].append(wo + wb)
            o1 = H.off(h['t1'])
            P['sum_old'].append((h['t2_b2'] - h['t1'] - o1 + h['t4'] - h['t3_b2'] + o1) / 1e6)
for k in ('out', 'back', 'sum'):
    v = P[k]
    print(f"wire {k:<5} N={len(v)} p10 {pct(v, .1):6.2f} p50 {pct(v, .5):6.2f} mean {sum(v) / len(v):6.2f} p90 {pct(v, .9):6.2f}")
print('MB/s implied by p50: out', 3.01e6 / (pct(P['out'], .5) * 1e-3) / 1e9, 'GB/s; back', 5.24e6 / (pct(P['back'], .5) * 1e-3) / 1e9, 'GB/s; combined', 8.25e6 / (pct(P['sum'], .5) * 1e-3) / 1e9, 'GB/s')
