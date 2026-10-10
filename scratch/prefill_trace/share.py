"""Box-2 share of picks and experts per 512-row prefill request (16:44 job, steady)."""
import sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from load import load, isnan, pct
_, R = load(sys.argv[1], {'b2_req'})
rows = [r for r in R['b2_req'] if r['b'] > 16 and not isnan(r['n_sel'])]
print('prefill b2_req', len(rows))
for f in ('b', 'n_sel', 'n_distinct', 'n_miss'):
    v = [r[f] for r in rows if not isnan(r[f])]
    print('%-10s p10 %8.1f p50 %8.1f p90 %8.1f mean %8.1f' % (f, pct(v, .1), pct(v, .5), pct(v, .9), sum(v) / len(v)))
sh = [r['n_sel'] / (r['b'] * 6.0) for r in rows if r['b'] > 0]
print('box-2 share of the lane\'s 6*b picks: p10 %.3f p50 %.3f p90 %.3f' % (pct(sh, .1), pct(sh, .5), pct(sh, .9)))
mem = [r['n_sel'] / r['n_distinct'] for r in rows if r['n_distinct'] > 0]
print('members per distinct expert on box 2: p50 %.1f mean %.1f' % (pct(mem, .5), sum(mem) / len(mem)))
run = [(r['t_run_end'] - r['t_run_start']) / 1e6 for r in rows if r.get('n_miss', 1) == 0 or True]
for lo, hi in ((0, 120), (120, 180), (180, 240), (240, 400)):
    g = [(r['t_run_end'] - r['t_run_start']) / 1e6 for r in rows if lo <= r['n_distinct'] < hi and r.get('n_miss', 0) == 0]
    if g:
        print('n_distinct %3d-%3d (no hub-visible miss): n=%5d run p50 %.1f ms' % (lo, hi, len(g), pct(g, .5)))
