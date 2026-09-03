/* Cross-check harness: upstream llama.cpp tables (iq3s_grid, kmask_iq2xs
 * from ggml-common.h) + VERBATIM copies of ggml_vec_dot_iq3_s_q8_K_generic
 * (ggml/src/ggml-cpu/quants.c) and dequantize_row_iq3_s
 * (ggml/src/ggml-quants.c), run on deterministic LCG blocks.
 * The Rust `v4flash_core::iq3_s_ref::{dot_iq3_s_q8_k, dequant_row_iq3_s}`
 * must reproduce these values (dot: rel < 1e-6; dequant: bit-exact).
 * Independence is over TRANSCRIPTION: this side is upstream's own code.
 *
 * Build/run (inside the dev shell, which has cc):
 *   cc -O2 -I ~/llama.cpp/ggml/src -I ~/llama.cpp/ggml/include \
 *      -o /tmp/iq3_s_gen crates/v4flash-kernels/tests/ref/iq3_s_gen.c -lm && /tmp/iq3_s_gen
 * Paste the printed values into crates/v4flash-core/src/iq3_s_ref.rs
 * (tests::matches_llama_cpp_reference) and
 * crates/v4flash-kernels/tests/iq3_s_cpu_ref.rs.
 *
 * NOTE: uses llama.cpp's block_q8_K field order (d | qs[256] | bsums[16]),
 * which is also ours — see q8_k_quantize.hip.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

typedef uint16_t ggml_half;
static float half_to_float(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
    if (e == 0)       { if (!m) bits = s; else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3ff; bits = s | (e << 23) | (m << 13); } }
    else if (e == 31) bits = s | 0x7f800000u | (m << 13);
    else              bits = s | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &bits, 4); return f;
}
#define QK_K 256
#define IQ3S_N_SCALE QK_K/64
typedef struct {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t signs[QK_K/8];
    uint8_t scales[IQ3S_N_SCALE];
} block_iq3_s;
_Static_assert(sizeof(block_iq3_s) == 110, "iq3_s block size");

static uint32_t lcg_state;
static uint8_t lcg(void) { lcg_state = lcg_state * 1103515245u + 12345u; return (uint8_t)(lcg_state >> 16); }

/* verbatim: ggml/src/ggml-quants.c dequantize_row_iq3_s (GGML_FP16_TO_FP32 -> half_to_float) */
static void dequantize_row_iq3_s(const block_iq3_s * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = half_to_float(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = x[i].signs;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const float db1 = d * (1 + 2*(x[i].scales[ib32/2] & 0xf));
            const float db2 = d * (1 + 2*(x[i].scales[ib32/2] >>  4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db1 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db1 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db2 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db2 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}

/* verbatim: ggml/src/ggml-cpu/quants.c ggml_vec_dot_iq3_s_q8_K_generic body
 * (block_q8_K accessed as raw bytes: f32 d | int8 qs[256] | int16 bsums[16]) */
static float vec_dot_iq3_s_q8_K(int nb, const block_iq3_s * x, const uint8_t * y) {
    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        float yd; memcpy(&yd, y + i*292, 4);
        const float d = half_to_float(x[i].d) * yd;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = x[i].signs;
        const int8_t  * q8 = (const int8_t *)(y + i*292 + 4);
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const uint32_t ls1 = 2*(x[i].scales[ib32/2] & 0xf) + 1;
            const uint32_t ls2 = 2*(x[i].scales[ib32/2] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls2;
        }
        sumf += d * bsum;
    }
    return sumf;
}

/* Per-block d: one of four sane f16 values picked by block index, so a
 * uniform-d bug in the transcription cannot hide (the tile8 oracle trap). */
static const uint16_t D_TABLE[4] = { 0x2e66 /* ~0.1 */, 0x3266 /* ~0.2 */, 0x2a66 /* ~0.05 */, 0x3466 /* ~0.275 */ };

int main(void) {
    const int nb = 4;
    for (int seed = 1; seed <= 3; seed++) {
        lcg_state = seed * 7919u;
        uint8_t w[4 * 110]; for (size_t i = 0; i < sizeof w; i++) w[i] = lcg();
        for (int b = 0; b < nb; b++) { uint16_t dd = D_TABLE[(b + seed) & 3]; w[b*110+0] = dd & 0xff; w[b*110+1] = dd >> 8; }
        uint8_t y[4 * 292];
        for (int b = 0; b < nb; b++) {
            float yd = 0.05f * (float)(b + 1); memcpy(y + b*292, &yd, 4);
            for (int j = 0; j < 256; j++) y[b*292 + 4 + j] = lcg();
            for (int j = 0; j < 32; j++)  y[b*292 + 260 + j] = 0;
        }
        const block_iq3_s *x = (const block_iq3_s *)w;
        float dot = vec_dot_iq3_s_q8_K(nb, x, y);
        float deq[4 * 256];
        dequantize_row_iq3_s(x, deq, nb * 256);
        double sum = 0, asum = 0;
        for (int i = 0; i < nb * 256; i++) { sum += deq[i]; asum += deq[i] < 0 ? -deq[i] : deq[i]; }
        uint32_t b0, b37, b255, b1023;
        memcpy(&b0, &deq[0], 4); memcpy(&b37, &deq[37], 4); memcpy(&b255, &deq[255], 4); memcpy(&b1023, &deq[1023], 4);
        printf("seed %d -> dot %.9e sum %.12e asum %.12e y0 %08x y37 %08x y255 %08x y1023 %08x\n",
               seed, dot, sum, asum, b0, b37, b255, b1023);
    }
    return 0;
}
