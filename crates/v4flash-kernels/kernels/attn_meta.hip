// attn_meta_fill -- the attention stage's per-row metadata (n_raw_per,
// n_raw_offset_per, n_comp_per, n_index_comp_per_b: a few int32 per row) as
// ONE kernel launch whose values travel in the kernel ARGUMENTS, instead of up
// to five tiny pageable hipMemcpyAsync H2D copies per lane-layer. Each of those
// is its own copy command on the stream: ~4.1 us of device timeline apiece at
// decode shapes (tests/bench_decode_latency_breakdown.rs, attn section: x4 =
// 16.4 us), for 4-12 bytes. A kernel costs one dispatch (~2.8 us).
//
// Grid (1), block 32. d_k may be null (not written).

#include <hip/hip_runtime.h>

#define AMF_MAX_ROWS 16u
#define AMF_N_DST 4u

struct AmfArgs {
    int v[AMF_N_DST][AMF_MAX_ROWS];
};

extern "C" __global__ void attn_meta_fill(
        int *d0, int *d1, int *d2, int *d3, unsigned int n, AmfArgs a) {
    const unsigned int t = threadIdx.x;
    if (t >= n || t >= AMF_MAX_ROWS) return;
    if (d0) d0[t] = a.v[0][t];
    if (d1) d1[t] = a.v[1][t];
    if (d2) d2[t] = a.v[2][t];
    if (d3) d3[t] = a.v[3][t];
}
