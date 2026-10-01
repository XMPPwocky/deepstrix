#!/usr/bin/env python3
"""For the large snapshots with no reusable prefix: who is their best partner
(longest common prefix among OLDER snapshots), and what text sits at the
divergence. Prompt-end snapshots only (checkpoints excluded)."""
import json, os, sys
import numpy as np
sys.path.insert(0, '/home/claude-code/.claude/jobs/16b63e08/tmp')
ROOT = '/home/claude-code/.cache/deepstrix/snapshots-v41'
CK = {int(x) for x in open('/home/claude-code/.claude/jobs/16b63e08/tmp/ckpt_counts.txt').read().split()}
TOK = '/persist/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277/tokenizer.json'
tj = json.load(open(TOK))
inv = {v: k for k, v in tj['model']['vocab'].items()}
for t in tj.get('added_tokens', []):
    inv[t['id']] = t['content']
bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
cs = bs[:]; n = 0
for b in range(256):
    if b not in bs:
        bs.append(b); cs.append(256 + n); n += 1
bd = {chr(c): b for b, c in zip(bs, cs)}
def text(ids):
    out = bytearray()
    for i in ids:
        s = inv.get(int(i), f'<{i}>')
        out += bytes(bd[ch] for ch in s) if all(ch in bd for ch in s) else s.encode()
    return out.decode('utf-8', 'replace')

snaps = []
for d in os.listdir(ROOT):
    try:
        meta = json.load(open(os.path.join(ROOT, d, 'meta.json')))
        toks = np.fromfile(os.path.join(ROOT, d, 'tokens.bin'), dtype='<i4')
    except Exception:
        continue
    if len(toks) in CK:
        continue
    snaps.append((meta.get('created_at_unix', 0), toks))
snaps.sort(key=lambda s: s[0])
def lcp(a, b):
    m = min(len(a), len(b)); ne = np.nonzero(a[:m] != b[:m])[0]
    return int(ne[0]) if len(ne) else m
# common opening across ALL prompt snapshots
starts = {}
for _, t in snaps:
    key = tuple(t[:2000].tolist())
    starts.setdefault(key[:1], []).append(t)
first = [t for _, t in snaps]
allcommon = min(lcp(first[0], t) for t in first[1:])
print(f'{len(snaps)} prompt snapshots; prefix common to ALL of them: {allcommon} tokens: {text(first[0][:allcommon])!r}')
# distinct "openings": cluster by the first 2000 tokens
heads = {}
for _, t in snaps:
    h = tuple(t[:4000].tolist())
    heads[h] = heads.get(h, 0) + 1
print(f'distinct 4000-token openings: {len(heads)}  (sizes {sorted(heads.values(), reverse=True)[:12]})')
cold = []
for i, (_, t) in enumerate(snaps):
    if len(t) < 20000:
        continue
    best, bj = 0, None
    for j in range(i):
        l = lcp(t, snaps[j][1])
        if l > best:
            best, bj = l, j
    if best < 2000:
        cold.append((len(t), best, i, bj))
print(f'{len(cold)} large prompt snapshots sharing < 2000 tokens with any older one')
for ln, best, i, bj in sorted(cold, reverse=True)[:6]:
    a = snaps[i][1]
    print(f'--- len {ln:,}, best lcp {best}')
    print('   common :', repr(text(a[max(0, best - 30):best])[-160:]))
    print('   this   :', repr(text(a[best:best + 40])[:200]))
    if bj is not None:
        print('   partner:', repr(text(snaps[bj][1][best:best + 40])[:200]))
