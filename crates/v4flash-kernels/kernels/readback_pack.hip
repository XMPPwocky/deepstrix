// Router readback pack (2026-09-25).
//
// Gathers the host-bound router outputs -- up to RP_MAX_SEG device segments of
// 32-bit words -- back to back into ONE host-mapped pinned buffer. Launched on
// the compute stream right before the router-done event, so the event's
// system-scope release makes every word host-visible when it completes: the
// host reads them after its event wait with no copy and no second stream.
//
// Segment k covers dst words [end_{k-1}, end_k) (end_{-1} = 0) and reads
// src_k[0 .. end_k - end_{k-1}). An empty segment has end_k == end_{k-1} and a
// src that is never dereferenced.

#include <hip/hip_runtime.h>

#define RP_BLOCK 256u

extern "C" __global__ void __launch_bounds__(RP_BLOCK)
readback_pack(unsigned* __restrict__ dst,
              const unsigned* __restrict__ s0, const unsigned* __restrict__ s1,
              const unsigned* __restrict__ s2, const unsigned* __restrict__ s3,
              const unsigned* __restrict__ s4, const unsigned* __restrict__ s5,
              const unsigned* __restrict__ s6, const unsigned* __restrict__ s7,
              const unsigned* __restrict__ s8, const unsigned* __restrict__ s9,
              unsigned e0, unsigned e1, unsigned e2, unsigned e3, unsigned e4,
              unsigned e5, unsigned e6, unsigned e7, unsigned e8, unsigned e9)
{
    const unsigned i = blockIdx.x * RP_BLOCK + threadIdx.x;
    if (i >= e9) return;
    const unsigned* src;
    unsigned base;
    if      (i < e0) { src = s0; base = 0u; }
    else if (i < e1) { src = s1; base = e0; }
    else if (i < e2) { src = s2; base = e1; }
    else if (i < e3) { src = s3; base = e2; }
    else if (i < e4) { src = s4; base = e3; }
    else if (i < e5) { src = s5; base = e4; }
    else if (i < e6) { src = s6; base = e5; }
    else if (i < e7) { src = s7; base = e6; }
    else if (i < e8) { src = s8; base = e7; }
    else             { src = s9; base = e8; }
    dst[i] = src[i - base];
}
