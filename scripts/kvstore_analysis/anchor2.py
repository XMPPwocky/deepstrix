import json, os, collections
import numpy as np
ROOT = '/home/claude-code/.cache/deepstrix/snapshots-v41'
CK = {int(x) for x in open('/home/claude-code/.claude/jobs/16b63e08/tmp/ckpt_counts.txt').read().split()}
USER, ASSIST = 128803, 128804
snaps = []
for d in os.listdir(ROOT):
    try:
        m = json.load(open(os.path.join(ROOT, d, 'meta.json')))
        t = np.fromfile(os.path.join(ROOT, d, 'tokens.bin'), dtype='<i4')
    except Exception:
        continue
    if len(t) in CK:
        continue
    snaps.append((m.get('created_at_unix', 0), t))
snaps.sort(key=lambda s: s[0])
first_turns = {}
for _, t in snaps:
    a_idx = np.nonzero(t == ASSIST)[0]
    if len(a_idx) == 0:
        continue
    fa = int(a_idx[0])
    u_idx = np.nonzero(t[:fa] == USER)[0]
    if len(u_idx) == 0:
        continue
    A = int(u_idx[-1])
    if A < 1024:
        continue
    ft = hash(t[:fa + 1].tobytes())
    anc = hash(t[:A].tobytes())
    first_turns[ft] = anc
by_anchor = collections.Counter(first_turns.values())
print('distinct first turns with A>=1024:', len(first_turns))
print('distinct anchor prefixes:', len(by_anchor))
print('first turns per anchor (sorted):', sorted(by_anchor.values(), reverse=True))
print('anchors used by only one first turn:', sum(1 for v in by_anchor.values() if v == 1))
