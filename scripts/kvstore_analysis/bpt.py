import json, os
ROOT = '/home/claude-code/.cache/deepstrix/snapshots-v41'
CK = {int(x) for x in open('/home/claude-code/.claude/jobs/16b63e08/tmp/ckpt_counts.txt').read().split()}
rows = []
for d in os.listdir(ROOT):
    p = os.path.join(ROOT, d)
    try:
        m = json.load(open(os.path.join(p, 'meta.json')))
    except Exception:
        continue
    tc = m['token_count']
    sizes = {f: os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)}
    rows.append((tc, tc in CK, sizes, m))
rows.sort(key=lambda r: -r[0])
for tc, ck, sizes, m in rows[:6]:
    comp = sizes.get('comp_kv.bin', 0); ik = sizes.get('index_k.bin', 0); kv = sizes.get('kv.bin', 0)
    ncomp = [l['n_comp'] for l in m['layers'] if l['has_compressor']]
    print(tc, 'ckpt' if ck else 'prompt', 'comp+keys per pos %.1f' % ((comp + ik) / tc), 'kv.bin %d' % kv, 'n_comp', ncomp, 'n_raw L0/L20', m['layers'][0]['n_raw'], m['layers'][20]['n_raw'])
# any prompt-end snapshot whose ratio-2 n_comp != floor(tc/2)?
bad = 0
for tc, ck, sizes, m in rows:
    nc = [l['n_comp'] for l in m['layers'] if l['has_compressor']]
    if not ck and nc and (nc[0] != tc // 2 or nc[-1] != tc):
        bad += 1
print('prompt snapshots with n_comp != positional:', bad, 'of', sum(1 for r in rows if not r[1]))
dec0 = sum(1 for tc, ck, s, m in rows if not ck and m['layers'][20]['n_raw'] < min(tc, 128))
print('prompt snapshots with decoder n_raw < min(T,128):', dec0)
