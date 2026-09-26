#!/usr/bin/env python3
# Patch 5: pseudo-candidate "i8x_r1" = q8_0_gemm_wmma_i8x from the first attempt's cand.hip
# (git show aab5a67 -> cand_r1.hip, built as cand_r1_gfx1201.hsaco) to reproduce the r1 625 us number.
p='harness.cpp'; s=open(p).read()
old='''            else if (c == "f16x_bn64_at_b") cfs.push_back({c, cm->fn("q8_0_gemm_wmma_f16x_bn64")});'''
new='''            else if (c == "f16x_bn64_at_b") cfs.push_back({c, cm->fn("q8_0_gemm_wmma_f16x_bn64")});
            else if (c == "i8x_r1") { static kb::Module r1(g_dir + "/cand_r1_" + g_arch + ".hsaco"); cfs.push_back({c, r1.fn("q8_0_gemm_wmma_i8x")}); }'''
assert old in s; s=s.replace(old,new)
old='''    if ((name == "q8_0_gemm_wmma_i8x" || name == "q8_0_gemm_wmma_i8x_pf2" || name == "q8_0_gemm_wmma_i8x_pf3" || name == "q8_0_gemm_wmma_i8x_grd"'''
new='''    if ((name == "q8_0_gemm_wmma_i8x" || name == "i8x_r1" || name == "q8_0_gemm_wmma_i8x_pf2" || name == "q8_0_gemm_wmma_i8x_pf3" || name == "q8_0_gemm_wmma_i8x_grd"'''
assert old in s; s=s.replace(old,new)
open(p,'w').write(s); print("harness ok")
