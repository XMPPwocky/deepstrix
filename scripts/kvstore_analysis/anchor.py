import json, os
import numpy as np
ROOT = '/home/claude-code/.cache/deepstrix/snapshots-v41'
CK = {int(x) for x in open('/home/claude-code/.claude/jobs/16b63e08/tmp/ckpt_counts.txt').read().split()}
USER, ASSIST = 128803, 128804
seen = {}
for d in os.listdir(ROOT):
    try:
        m = json.load(open(os.path.join(ROOT, d, 'meta.json')))
        t = np.fromfile(os.path.join(ROOT, d, 'tokens.bin'), dtype='<i4')
    except Exception:
        continue
    if len(t) in CK:
        continue
    a_idx = np.nonzero(t == ASSIST)[0]
    if len(a_idx) == 0:
        continue
    fa = int(a_idx[0])
    u_idx = np.nonzero(t[:fa] == USER)[0]
    if len(u_idx) == 0:
        continue
    A = int(u_idx[-1])
    # one record per distinct first-turn (prefix through first <Assistant>)
    key = hash(t[:fa + 1].tobytes())
    seen[key] = (A, fa + 1 - A)
vals = sorted(s for _, s in seen.values())
anchors = sorted(a for a, _ in seen.values())
n = len(vals)
print('distinct first turns', n)
print('anchor A p10/p50/p90', anchors[n // 10], anchors[n // 2], anchors[9 * n // 10])
print('first-turn suffix after A (incl <User>..<Assistant>) p10/p25/p50/p90:', vals[n // 10], vals[n // 4], vals[n // 2], vals[9 * n // 10])
print('share with suffix <= 128:', sum(1 for v in vals if v <= 128) / n)
print('share with suffix <= 128 and A >= 1024:', sum(1 for a, s in seen.values() if s <= 128 and a >= 1024) / max(1, sum(1 for a, s in seen.values() if a >= 1024)))
