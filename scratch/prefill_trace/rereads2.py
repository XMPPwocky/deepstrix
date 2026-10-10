"""Which experts does box 2 read during the 16:44 cold prefill, and what do they evict?
usage: rereads2.py B2_TEXT [B2_TEXT...]"""
import sys
from collections import Counter, defaultdict
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from load import load, isnan, pct

R = []
for fn in sys.argv[1:]:
    _, RB = load(fn, {'b2_read'})
    R += RB['b2_read']
R.sort(key=lambda r: r['t_read_start'])
print('b2_read records', len(R))
src = Counter(int(r['src']) for r in R)
print('src (0 demand, 1-3 background):', dict(src))
exp = [int(r['expert']) for r in R]
print('expert id min/max', min(exp), max(exp), ' share >=265:', sum(1 for e in exp if e >= 265) / len(exp))
lay = Counter(int(r['layer']) for r in R)
print('reads by layer (target):', sorted(lay.items()))
# victims: which layer's experts are evicted to make room
vl = Counter(int(r['victim_layer']) for r in R if not isnan(r['victim_layer']) and r['victim_layer'] >= 0)
tot = sum(vl.values())
enc = sum(c for l, c in vl.items() if l <= 20)
print('victim layers: encoder(<=20) %.1f%%  decoder(>20) %.1f%%  (n=%d)' % (100 * enc / max(tot, 1), 100 * (tot - enc) / max(tot, 1), tot))
ve = [int(r['victim_expert']) for r in R if not isnan(r['victim_expert']) and r['victim_expert'] >= 0]
if ve:
    print('victim expert id: share >=265:', sum(1 for e in ve if e >= 265) / len(ve))
# distinct (layer, expert) read vs repeats
seen = Counter((int(r['layer']), int(r['expert'])) for r in R)
print('distinct (layer,expert) read:', len(seen), ' total reads:', len(R), ' mean reads per distinct:', len(R) / len(seen))
print('distribution of reads per distinct expert:', sorted(Counter(seen.values()).items())[:12])
# distinct per layer within encoder layers (<=20) for experts >=265 and <265
by_l = defaultdict(set)
for (l, e) in seen:
    by_l[l].add(e)
print('distinct experts read per layer (encoder layers):', [(l, len(by_l[l]), sum(1 for e in by_l[l] if e >= 265)) for l in sorted(by_l) if l <= 20])
