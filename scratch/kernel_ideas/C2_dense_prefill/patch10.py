#!/usr/bin/env python3
# Patch 10: isolate MODULE SIZE from source edits. cand_r1plus.hip = cand_r1.hip (verbatim) + ~30 extra
# instantiations of ITS OWN template (dummy kernels, never launched). If i8x from that module is slow,
# module size alone explains the r1 gap. Harness: generic pseudo-prefix "mod:<basename>:<kernel>".
s = open('cand_r1.hip').read()
extra = ['\n// ---- patch10: dummy instantiations to grow the module (never launched) ----',
         '#if defined(__gfx1200__) || defined(__gfx1201__)']
n = 0
for (bm, bn, wm, wn) in [(128, 128, 2, 4), (128, 64, 4, 2), (64, 64, 2, 2)]:
    for pf in (2, 3, 4):
        for lb in ('true', 'false'):
            n += 1
            extra.append(f'F16X_KERNEL(dummy_f16x_{n}, {bm}, {bn}, {wm}, {wn}, {pf}, {lb})')
for pf in (2, 3):
    for (bm, bn, wm, wn) in [(128, 128, 2, 4), (128, 64, 4, 2)]:
        n += 1
        extra.append(f'''extern "C" __global__ void __launch_bounds__({wm*wn*32}) dummy_i8x_{n}(
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks)
{{ f16x_core<{bm}, {bn}, {wm}, {wn}, {pf}, true, true>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, K); }}''')
extra.append('#endif\n')
open('cand_r1plus.hip', 'w').write(s + '\n'.join(extra)); print('cand_r1plus ok', n, 'dummies')

h = open('harness.cpp').read()
old = '''            if (c.rfind("small:", 0) == 0) {'''
new = '''            if (c.rfind("mod:", 0) == 0) {   // mod:<basename>:<kernel> -> <dir>/<basename>_<arch>.hsaco
                size_t p = c.find(':', 4);
                static std::map<std::string, kb::Module*> mods;
                std::string base = c.substr(4, p - 4);
                if (!mods.count(base)) mods[base] = new kb::Module(g_dir + "/" + base + "_" + g_arch + ".hsaco");
                cfs.push_back({c, mods[base]->fn(c.c_str() + p + 1)});
            }
            else if (c.rfind("small:", 0) == 0) {'''
assert old in h; h = h.replace(old, new)
old = '''    const std::string name = name_in.rfind("small:", 0) == 0 ? name_in.substr(6) : name_in;'''
new = '''    std::string name = name_in.rfind("small:", 0) == 0 ? name_in.substr(6) : name_in;
    if (name.rfind("mod:", 0) == 0) name = name.substr(name.find(':', 4) + 1);'''
assert old in h; h = h.replace(old, new)
open('harness.cpp', 'w').write(h); print('harness ok')
