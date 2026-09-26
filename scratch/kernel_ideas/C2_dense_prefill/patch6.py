#!/usr/bin/env python3
# Patch 6: BRF loads keep the RAW registers; the b_on select moves into stage() (after the barrier),
# so the compiler no longer emits s_wait_loadcnt right after the load (patch2's select-at-load
# turned the B prefetch into a synchronous load in every variant measured in r2-r4).
p='cand.hip'; s=open(p).read()
old='''            vb[p][i].x = b_on[i] ? t.x : 0u; vb[p][i].y = b_on[i] ? t.y : 0u;
            vb[p][i].z = b_on[i] ? t.z : 0u; vb[p][i].w = b_on[i] ? t.w : 0u;
            if (BI8) xs[p][i] = b_on[i] ? ts : 0.f;
        }
    };'''
new='''            vb[p][i] = t;                 // raw; masked at stage time (see patch6 note)
            if (BI8) xs[p][i] = ts;
        }
    };'''
assert old in s; s=s.replace(old,new)
old='''            if (BI8) {
                // (f16)((float)q * xscale), the lds_tiled expression, 16 values -> 2 x b128
                int8_t q[16]; __builtin_memcpy(q, &vb[p][i], 16);
                const float sc = xs[p][i];'''
new='''            uint4 vraw = vb[p][i];
            if (BRF && !b_on[i]) vraw = make_uint4(0u, 0u, 0u, 0u);   // select AFTER the barrier
            if (BI8) {
                // (f16)((float)q * xscale), the lds_tiled expression, 16 values -> 2 x b128
                int8_t q[16]; __builtin_memcpy(q, &vraw, 16);
                const float sc = (BRF && !b_on[i]) ? 0.f : xs[p][i];'''
assert old in s; s=s.replace(old,new)
old='''                uint4* dst = (uint4*)(&Bt[b_row[i] * LDS_STRIDE + b_koff[i]]);
                dst[0] = vb[p][i];'''
new='''                uint4* dst = (uint4*)(&Bt[b_row[i] * LDS_STRIDE + b_koff[i]]);
                dst[0] = vraw;'''
assert old in s; s=s.replace(old,new)
open(p,'w').write(s); print("cand ok")
