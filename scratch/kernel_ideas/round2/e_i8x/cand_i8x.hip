// Round 2 item (e): int8-activation WMMA GEMMs for the dGPU dense prefill sites, on the in-tree
// f16x_core (the kernel file is included verbatim so the template is shared, not copied).
// Activations: Q8_0 xq [B, ldx] int8 + xscale [B, ldx/32] f32 (quantize_input_batched's output),
// dequantised at stage exactly as q8_0_gemm_wmma_lds_tiled / i8x_db ((f16)((float)q * xs)).
// NOT bit-exact vs the f16x arm (f16 activations): numerics = the dp4a arm's int8 activations.
#include KERNELS_WMMA_HIP

#if defined(__gfx1200__) || defined(__gfx1201__)
#define I8X_ENTRY(NAME, ...)                                                                                   \
extern "C" __global__ void __launch_bounds__(256) NAME(                                                       \
    float* __restrict__ out, const unsigned char* __restrict__ w, const int8_t* __restrict__ xq,              \
    const float* __restrict__ xscale, uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch,              \
    uint32_t blocks, uint32_t ldx) {                                                                           \
    f16x_core<__VA_ARGS__>(out, w, nullptr, xq, xscale, K, M, n_groups, batch, blocks, ldx);                   \
}
#else
#define I8X_ENTRY(NAME, ...)                                                                                   \
extern "C" __global__ void NAME(float* out, const unsigned char* w, const int8_t* xq, const float* xscale,    \
    uint32_t K, uint32_t M, uint32_t n_groups, uint32_t batch, uint32_t blocks, uint32_t ldx) {                \
    (void)out; (void)w; (void)xq; (void)xscale; (void)K; (void)M; (void)n_groups; (void)batch;                \
    (void)blocks; (void)ldx; }
#endif

// 128x128, DB, PF2 (the Engram i8x_db geometry) with an explicit activation pitch (groups)
I8X_ENTRY(i8x_db_ld, 128, 128, 2, 4, 2, true, true, 0, true, true)
// 256x128 (the q_b / wo_a f16x geometry), single-buffered, PF1
I8X_ENTRY(i8x_256x128_ld, 256, 128, 4, 2, 1, true, true)
// 256x128 double-buffered
I8X_ENTRY(i8x_256x128_db_ld, 256, 128, 4, 2, 1, true, true, 0, true, true)
// 128x64 DB PF2 (the kv / q_a f16x geometry)
I8X_ENTRY(i8x_bn64_ld, 128, 64, 4, 2, 2, true, true, 0, true, true)
// 128x128 single-buffered PF2 (the base f16x geometry)
I8X_ENTRY(i8x_128_ld, 128, 128, 2, 4, 2, true, true)
