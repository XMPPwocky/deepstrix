import sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from lib2 import *
c = Counter()
for fn in sys.argv[1].split(','):
    _, RB = load(fn, {'b2_req'})
    for b in RB['b2_req']:
        big = b['b'] >= 256
        c[('prefill' if big else 'decode', 'served_under_set' if not isnan(b['served_under']) else 'nan')] += 1
        c[('prefill' if big else 'decode', 'park>1ms')] += 1 if (not isnan(b['park_wait_ns']) and b['park_wait_ns'] > 1e6) else 0
        c[('prefill' if big else 'decode', 'merged')] += 1 if b['merged'] and b['merged'] > 0 else 0
print(dict(c))
