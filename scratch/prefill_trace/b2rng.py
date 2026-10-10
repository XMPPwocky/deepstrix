import sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from lib2 import *

H = Hub(sys.argv[1], sys.argv[2])
wins = windows(sys.argv[3])
lo = H.raw(wins[0][1])
hi = H.raw(wins[-1][2])
rs = [r for r in H.pre if lo <= r['t_submit'] <= hi]
t2 = [r['t2_b2'] for r in rs if not isnan(r['t2_b2'])]
t3 = [r['t3_b2'] for r in rs if not isnan(r['t3_b2'])]
print(len(rs), '%d:%d' % (min(t2) - 3e9, max(t3) + 3e9))
print('offset sample', H.opts[:2], H.opts[-1], len(H.opts))
