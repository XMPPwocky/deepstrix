#!/usr/bin/env python3
"""Find snapshots by token_count, report where two of them diverge (token
index, byte context around it). Usage: snap_diverge.py COUNT_A COUNT_B"""
import json, os, sys
ROOT = '/home/claude-code/.cache/deepstrix/snapshots-v41'
TOK = '/persist/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277/tokenizer.json'

def find(count):
    out = []
    for d in os.listdir(ROOT):
        m = os.path.join(ROOT, d, 'meta.json')
        try:
            meta = json.load(open(m))
        except Exception:
            continue
        if meta.get('token_count') == count:
            out.append((d, meta))
    return out

def tokens(d):
    raw = open(os.path.join(ROOT, d, 'tokens.bin'), 'rb').read()
    return [int.from_bytes(raw[i:i + 4], 'little', signed=True) for i in range(0, len(raw), 4)]

def byte_decoder():
    bs = list(range(ord('!'), ord('~') + 1)) + list(range(ord('¡'), ord('¬') + 1)) + list(range(ord('®'), ord('ÿ') + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

tj = json.load(open(TOK))
inv = {v: k for k, v in tj['model']['vocab'].items()}
for t in tj.get('added_tokens', []):
    inv[t['id']] = t['content']
bd = byte_decoder()
def text(ids):
    out = bytearray()
    for i in ids:
        s = inv.get(i, f'<{i}>')
        if all(ch in bd for ch in s):
            out += bytes(bd[ch] for ch in s)
        else:
            out += s.encode()
    return out.decode('utf-8', 'replace')

a_count, b_count = int(sys.argv[1]), int(sys.argv[2])
A, B = find(a_count), find(b_count)
print(f'{a_count}: {[(d[:12], m.get("session_id"), m.get("disk_bytes")) for d, m in A]}')
print(f'{b_count}: {[(d[:12], m.get("session_id"), m.get("disk_bytes")) for d, m in B]}')
if not A or not B:
    sys.exit('missing snapshot')
ta, tb = tokens(A[0][0]), tokens(B[0][0])
n = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
print(f'common token prefix: {n} of {len(ta)} / {len(tb)}')
print('--- before ---'); print(repr(text(ta[max(0, n - 40):n])))
print('--- A after ---'); print(repr(text(ta[n:n + 60])))
print('--- B after ---'); print(repr(text(tb[n:n + 60])))
