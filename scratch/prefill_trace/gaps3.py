import sys, json
from collections import defaultdict
R = json.load(open(sys.argv[1]))
MAXS = int(sys.argv[2])
acc = defaultdict(list)
n = 0
for r in R:
    if r['kind'] != 'W' or r['rows'] != 4096 or r['ndec'] > 0 or r['nmiss'] > MAXS:
        continue
    n += 1
    for k, v in r['gl'].items():
        acc[k].append(v)
print('windows', n)
tot = 0
for k in sorted(acc):
    m = sum(acc[k]) / n
    tot += m
    print(f"  gap {k}: mean per window {m:.3f} s  (x{len(acc[k]) / n:.0f} per window -> {m / (len(acc[k]) / n) * 1e3:.0f} ms each)")
print('  total', round(tot, 3))
