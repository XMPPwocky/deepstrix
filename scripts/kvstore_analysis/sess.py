import json, os, collections
ROOT = '/home/claude-code/.cache/deepstrix/snapshots-v41'
CK = {int(x) for x in open('/home/claude-code/.claude/jobs/16b63e08/tmp/ckpt_counts.txt').read().split()}
n = 0; with_sid = 0; sids = collections.Counter(); sizes = []; created = []
odd = 0; lens = []
for d in os.listdir(ROOT):
    try:
        m = json.load(open(os.path.join(ROOT, d, 'meta.json')))
    except Exception:
        continue
    n += 1
    tc = m['token_count']
    lens.append(tc)
    if tc % 2: odd += 1
    created.append(m.get('created_at_unix', 0))
    sizes.append(m.get('disk_bytes', 0))
    if m.get('session_id'):
        with_sid += 1
        sids[m['session_id']] += 1
print('entries', n, 'with session_id', with_sid, 'distinct sessions', len(sids), 'top', sids.most_common(5))
print('total GB %.1f' % (sum(sizes) / 1e9))
import time
print('oldest entry age h %.1f, newest %.1f' % ((time.time() - min(created)) / 3600, (time.time() - max(created)) / 3600))
print('odd token counts', odd, 'of', n)
# bytes per token check on the largest
big = max(range(n), key=lambda i: lens[i])
print('largest', lens[big], 'tokens', sizes[big] / 1e6, 'MB', 'per token', sizes[big] / lens[big])
