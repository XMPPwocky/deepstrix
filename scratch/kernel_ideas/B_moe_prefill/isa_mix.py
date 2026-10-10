"""Static instruction mix per kernel from an llvm-objdump .s (see ../_infra/isa.sh --dis).
usage: python3 -I isa_mix.py FILE.s [kernel-substring ...]"""
import re
import sys

path = sys.argv[1]
pats = sys.argv[2:] or ['wmma']
cur = None
mix = {}
for line in open(path):
    m = re.match(r'^[0-9a-f]+ <(.*)>:$', line)
    if m:
        cur = m.group(1)
        mix[cur] = dict(wmma=0, valu=0, ds=0, gload=0, other=0)
        continue
    if cur is None:
        continue
    m = re.match(r'^\s+([a-z_0-9]+)\s', line)
    if not m:
        continue
    op = m.group(1)
    d = mix[cur]
    if op.startswith('v_wmma'):
        d['wmma'] += 1
    elif op.startswith('v_'):
        d['valu'] += 1
    elif op.startswith('ds_'):
        d['ds'] += 1
    elif op.startswith('global_load'):
        d['gload'] += 1
    else:
        d['other'] += 1
for k, d in mix.items():
    if any(p in k for p in pats) and d['wmma']:
        print('%-52s wmma=%4d valu=%5d ds=%4d gload=%4d  valu/wmma=%.1f' % (k, d['wmma'], d['valu'], d['ds'], d['gload'], d['valu'] / d['wmma']))
