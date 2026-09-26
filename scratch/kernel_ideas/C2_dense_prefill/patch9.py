#!/usr/bin/env python3
# Patch 9: module-composition test. cand_small.hip = cand.hip with only a few kernels kept;
# harness pseudo-prefix "small:<kernel>" loads that kernel from cand_small_gfx1201.hsaco.
import re
KEEP = {'q8_0_gemm_wmma_i8x', 'q8_0_gemm_wmma_i8x_db', 'r1c_i8x', 'q8_0_gemm_wmma_f16x_pf2',
        'q8_0_gemm_wmma_f16x_db_bn64', 'q8_0_gemm_wmma_f16x_256x128', 'q8_0_gemm_wmma_f16x_db', 'q8_0_gemm_wmma_f16x_lb'}
s = open('cand.hip').read()
# 1. macro-instantiated kernels: F16X_KERNEL(name, ...) / F16X_DB(...) / F16X_ABL(...) / R1C_I8X(...)
def keep_macro(m):
    name = m.group(2)
    return m.group(0) if name in KEEP else '// [small] dropped: ' + m.group(0)
s = re.sub(r'^(F16X_KERNEL|F16X_DB|F16X_ABL|R1C_I8X)\((\w+),[^\n]*\)\s*(//[^\n]*)?$', keep_macro, s, flags=re.M)
# 2. extern "C" kernels written out by hand: drop those not in KEEP (block = from 'extern "C"' to the next line that is exactly '}')
out = []; i = 0; lines = s.split('\n')
while i < len(lines):
    l = lines[i]
    m = re.match(r'extern "C" __global__ void (?:__launch_bounds__\(\d+\) )?(\w+)\(', l)
    if m and m.group(1) not in KEEP and m.group(1) != 'NAME' and not l.rstrip().endswith('\\'):
        j = i
        while lines[j].rstrip() != '}': j += 1
        out.append('// [small] dropped kernel ' + m.group(1)); i = j + 1; continue
    out.append(l); i += 1
open('cand_small.hip', 'w').write('\n'.join(out)); print('cand_small ok')

h = open('harness.cpp').read()
old = '''            if (c == "f16x_at_b") {'''
new = '''            if (c.rfind("small:", 0) == 0) { static kb::Module sm(g_dir + "/cand_small_" + g_arch + ".hsaco"); cfs.push_back({c, sm.fn(c.c_str() + 6)}); }
            else if (c == "f16x_at_b") {'''
assert old in h; h = h.replace(old, new)
old = '''static bool cand_launch(const std::string& name, hipFunction_t f, const Shape& s, const Bufs& B, unsigned b, float* out, hipStream_t st) {
    unsigned blocks = s.K / 32;'''
new = '''static bool cand_launch(const std::string& name_in, hipFunction_t f, const Shape& s, const Bufs& B, unsigned b, float* out, hipStream_t st) {
    const std::string name = name_in.rfind("small:", 0) == 0 ? name_in.substr(6) : name_in;
    unsigned blocks = s.K / 32;'''
assert old in h; h = h.replace(old, new)
open('harness.cpp', 'w').write(h); print('harness ok')
