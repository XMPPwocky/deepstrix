import sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from lib2 import *

H = Hub(sys.argv[1], sys.argv[2])
lab = sys.argv[4]
for l, a, b, rows in windows(sys.argv[3]):
    if l != lab:
        continue
    lo, hi = H.raw(a), H.raw(b)
    rs = [r for r in H.pre if lo <= r['t_submit'] <= hi]
    print(len(rs), 'layers', sorted(set(int(r['layer']) for r in rs)), 'lanes', Counter(int(r['lane']) for r in rs), 'b', Counter(int(r['b']) for r in rs), 'flags', Counter(int(r['flags']) for r in rs))
    t0 = rs[0]['t_submit']
    for r in rs[:44]:
        print('%7.1f L%2d l%d b%4d seq%d  sub->t1 %5.1f  rtt %6.1f srv %6.1f pg %6.1f comp %6.1f miss %3d out %.2f in %.2f blk %d wait %6.1f exit-sub %6.1f' % (
            (r['t_submit'] - t0) / 1e6, r['layer'], r['lane'], r['b'], r['seq'], (r['t1'] - r['t_submit']) / 1e6, r['rtt_us'] / 1e3, r['srv_us'] / 1e3, r['page_us'] / 1e3, r['compute_us'] / 1e3,
            r['n_miss'], r['bytes_out'] / 1e6, r['bytes_in'] / 1e6, r['blocked'], (r['t_wait_exit'] - r['t_wait_enter']) / 1e6, (r['t_wait_exit'] - r['t_submit']) / 1e6))
