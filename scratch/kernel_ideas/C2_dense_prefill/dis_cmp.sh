#!/usr/bin/env bash
# Full per-kernel disassembly (from isa.sh's full .s next to the hsaco) + instruction histograms.
set -u
cd "$(dirname "$0")"
bash ../_infra/isa.sh cand_gfx1201.hsaco gfx1201 --dis q8_0_gemm_wmma_i8x > /dev/null 2>&1
bash ../_infra/isa.sh cand_r1_gfx1201.hsaco gfx1201 --dis q8_0_gemm_wmma_i8x > /dev/null 2>&1
python3 - <<'EOF'
import re, collections
def extract(path, kernel):
    out = []; on = False
    for l in open(path):
        m = re.match(r'^[0-9a-f]+ <(.*)>:$', l.rstrip())
        if m: on = (m.group(1) == kernel); continue
        if on and l.strip(): out.append(l.rstrip())
    return out
def hist(lines):
    c = collections.Counter()
    for l in lines:
        m = re.match(r'^\s*([a-z_0-9]+)', l)
        if m: c[m.group(1)] += 1
    return c
sets = {'r1': ('cand_r1_gfx1201.s', 'q8_0_gemm_wmma_i8x'), 'cur': ('cand_gfx1201.s', 'q8_0_gemm_wmma_i8x'),
        'grd_aon': ('cand_gfx1201.s', 'q8_0_gemm_wmma_i8x_grd_aon'), 'db': ('cand_gfx1201.s', 'q8_0_gemm_wmma_i8x_db')}
H = {}
for k, (p, kn) in sets.items():
    L = extract(p, kn); open(f'i8x_{k}_full.s', 'w').write('\n'.join(L) + '\n'); H[k] = hist(L)
    print(f'== {k}: {len(L)} instrs, wmma={H[k]["v_wmma_f32_16x16x16_f16"]}')
keys = sorted(set().union(*[set(h) for h in H.values()]), key=lambda x: -max(h.get(x, 0) for h in H.values()))
print(f'{"insn":34s}' + ''.join(f'{k:>9s}' for k in H))
for x in keys[:40]:
    row = [H[k].get(x, 0) for k in H]
    if max(row) - min(row) >= 2 or x.startswith('s_wait') or x.startswith('v_wmma'):
        print(f'{x:34s}' + ''.join(f'{v:9d}' for v in row))
EOF
