#!/usr/bin/env python3
# Compare instruction classes (calls, LDS forms, scratch) of fast vs slow i8x builds.
import re, collections, subprocess
subprocess.run(['bash', '../_infra/isa.sh', 'cand_small_gfx1201.hsaco', 'gfx1201', '--dis', 'r1c_i8x'], capture_output=True)
def extract(path, kernel):
    out = []; on = False
    for l in open(path):
        m = re.match(r'^[0-9a-f]+ <(.*)>:$', l.rstrip())
        if m: on = (m.group(1) == kernel); continue
        if on and l.strip(): out.append(re.sub(r'\s*//.*$', '', l.rstrip()).strip())
    return out
S = {'r1 i8x (fast 629)': ('cand_r1_gfx1201.s', 'q8_0_gemm_wmma_i8x'),
     'small r1c_i8x (988)': ('cand_small_gfx1201.s', 'r1c_i8x'),
     'cur i8x (1078)': ('cand_gfx1201.s', 'q8_0_gemm_wmma_i8x'),
     'cur i8x_db (485)': ('cand_gfx1201.s', 'q8_0_gemm_wmma_i8x_db'),
     'cur f16x_pf2': ('cand_gfx1201.s', 'q8_0_gemm_wmma_f16x_pf2'),
     'r1 f16x_pf2': ('cand_r1_gfx1201.s', 'q8_0_gemm_wmma_f16x_pf2')}
H = {}
for tag, (p, k) in S.items():
    L = extract(p, k); h = collections.Counter(x.split()[0] for x in L); H[tag] = h
    calls = sum(h[x] for x in h if x in ('s_swappc_b64', 's_setpc_b64', 's_call_b64'))
    ds_other = {x: h[x] for x in h if x.startswith('ds_') and x not in ('ds_load_b128', 'ds_store_b128')}
    scratch = sum(v for x, v in h.items() if x.startswith('scratch_'))
    print(f'{tag:22s} instrs={len(L):4d} calls={calls} ds_load_b128={h["ds_load_b128"]} ds_store_b128={h["ds_store_b128"]} ds_other={ds_other} scratch={scratch} s_nop={h["s_nop"]} v_wmma={h["v_wmma_f32_16x16x16_f16"]}')
# loop-body diff r1 vs r1c: print the two B-stage regions (between the two s_barrier_signal of the stage)
def region(L):
    idx = [i for i, x in enumerate(L) if x.startswith('v_cvt_f32_i32')]
    return L[idx[0] - 30: idx[-1] + 12] if idx else []
for tag in ('r1 i8x (fast 629)', 'small r1c_i8x (988)'):
    p, k = S[tag]; L = extract(p, k); R = region(L)
    print(f'\n---- {tag}: B-stage region ({len(R)} instrs) ----')
    print('\n'.join(R))
