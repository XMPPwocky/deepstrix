#!/usr/bin/env python3
# Patch 4: big tiles (256x128 / 256x256) to cut L2->CU tile traffic per flop; + DB flavour.
p='cand.hip'; s=open(p).read()
old='''F16X_ABL(q8_0_gemm_wmma_f16x_abl_noglobal, 1, 1)'''
new='''// Big tiles: BM=256 halves the activation-tile re-reads (B is re-fetched once per M-block; for qb
// that is 335 of 492 MB of tile traffic), BN=256 halves the weight-tile re-reads. Wave tile 64x64
// (MT=4, NTL=4, 16 accumulators = 128 VGPRs). LDS: 256x128 -> 30 KB (2 WGs/WGP), 256x256 -> 40 KB
// (1 WG/WGP, 16 waves), 256x128 DB -> 60 KB (1 WG/WGP, 8 waves).
F16X_KERNEL(q8_0_gemm_wmma_f16x_256x128,  256, 128, 4, 2, 1, true)
F16X_KERNEL(q8_0_gemm_wmma_f16x_256x128_pf2, 256, 128, 4, 2, 2, true)
F16X_KERNEL(q8_0_gemm_wmma_f16x_256x256,  256, 256, 4, 4, 1, true)
F16X_KERNEL(q8_0_gemm_wmma_f16x_256x256_pf2, 256, 256, 4, 4, 2, true)
F16X_DB(q8_0_gemm_wmma_f16x_256x128_db, 256, 128, 4, 2, 2)
F16X_ABL(q8_0_gemm_wmma_f16x_abl_noglobal, 1, 1)'''
assert old in s; s=s.replace(old,new)
open(p,'w').write(s); print("cand ok")

p='harness.cpp'; s=open(p).read()
old='''    // f16x 64x128 tile (BM=64 weight rows, BN=128 tokens): grid (ceil(b/128), M/64, G) x 256'''
new='''    // f16x big tiles: grid (ceil(b/BN), M/256, G) x (waves*32)
    if (name == "q8_0_gemm_wmma_f16x_256x128" || name == "q8_0_gemm_wmma_f16x_256x128_pf2" || name == "q8_0_gemm_wmma_f16x_256x128_db") {
        if (s.M % 256) { fprintf(stderr, "M %% 256\\n"); return false; }
        kb::launch(f, dim3((b + 127) / 128, s.M / 256, s.G), dim3(256), 0, st,
                   out, (const unsigned char*)B.w, (const uint16_t*)B.x16, s.K, s.M, s.G, b, blocks, s.ldx);
        return true;
    }
    if (name == "q8_0_gemm_wmma_f16x_256x256" || name == "q8_0_gemm_wmma_f16x_256x256_pf2") {
        if (s.M % 256) { fprintf(stderr, "M %% 256\\n"); return false; }
        kb::launch(f, dim3((b + 255) / 256, s.M / 256, s.G), dim3(512), 0, st,
                   out, (const unsigned char*)B.w, (const uint16_t*)B.x16, s.K, s.M, s.G, b, blocks, s.ldx);
        return true;
    }
    // f16x 64x128 tile (BM=64 weight rows, BN=128 tokens): grid (ceil(b/128), M/64, G) x 256'''
assert old in s; s=s.replace(old,new)
open(p,'w').write(s); print("harness ok")
