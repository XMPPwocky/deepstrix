/* Cross-check harness: upstream llama.cpp tables + a VERBATIM copy of
 * ggml_vec_dot_iq2_xs_q8_K_generic, run on deterministic LCG blocks.
 * The Rust cpu_dot_iq2_xs_q8_k must reproduce these values exactly.
 * Independence is over TRANSCRIPTION: this side is upstream's own code.
 *
 * Build/run:
 *   cc -O2 -I ~/llama.cpp/ggml/src -I ~/llama.cpp/ggml/include \
 *      -o /tmp/iq2_xs_gen tests/ref/iq2_xs_gen.c -lm && /tmp/iq2_xs_gen
 * Paste the printed values into tests/iq2_xs_cpu_ref.rs EXPECTED.
 *
 * NOTE: this harness uses llama.cpp's block_q8_K field order
 * (d | qs[256] | bsums[16]), which is also ours — see q8_k_quantize.hip.
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
typedef struct { ggml_half d; uint16_t qs[256/8]; uint8_t scales[256/32]; } blk_iq2_xs;

static uint32_t lcg_state;
static uint8_t lcg(void) { lcg_state = lcg_state * 1103515245u + 12345u; return (uint8_t)(lcg_state >> 16); }

int main(void) {
    const int nb = 4;
    for (int seed = 1; seed <= 3; seed++) {
        lcg_state = seed * 7919u;
        uint8_t w[4 * 74]; for (size_t i = 0; i < sizeof w; i++) w[i] = lcg();
        /* force a sane d per block: 0x2e66 ~= 0.1 */
        for (int b = 0; b < nb; b++) { w[b*74+0] = 0x66; w[b*74+1] = 0x2e; }
        /* q8_K block: f32 d | int8 qs[256] | int16 bsums[16]  (292 B) */
        uint8_t y[4 * 292];
        for (int b = 0; b < nb; b++) {
            float yd = 0.05f; memcpy(y + b*292, &yd, 4);
            for (int j = 0; j < 256; j++) y[b*292 + 4 + j] = lcg();
            for (int j = 0; j < 32; j++)  y[b*292 + 260 + j] = 0;
        }
        /* verbatim from ggml/src/ggml-cpu/quants.c */
        const blk_iq2_xs *x = (const blk_iq2_xs *)w;
        float sumf = 0.f;
        for (int i = 0; i < nb; ++i) {
            float yd; memcpy(&yd, y + i*292, 4);
            const float d = half_to_float(x[i].d) * yd;
            const uint16_t *q2 = x[i].qs; const uint8_t *sc = x[i].scales;
            const int8_t *q8 = (const int8_t *)(y + i*292 + 4);
            int32_t bsum = 0;
            for (int ib32 = 0; ib32 < 256/32; ++ib32) {
                const uint16_t ls1 = 2*(sc[ib32] & 0xf) + 1;
                const uint16_t ls2 = 2*(sc[ib32] >>  4) + 1;
                int32_t sumi = 0;
                for (int l = 0; l < 2; ++l) {
                    const uint8_t *grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                    const uint8_t signs = ksigns_iq2xs[q2[l] >> 9];
                    for (int j = 0; j < 8; ++j) sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                    q8 += 8;
                }
                bsum += sumi * ls1; sumi = 0;
                for (int l = 2; l < 4; ++l) {
                    const uint8_t *grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                    const uint8_t signs = ksigns_iq2xs[q2[l] >> 9];
                    for (int j = 0; j < 8; ++j) sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                    q8 += 8;
                }
                bsum += sumi * ls2; q2 += 4;
            }
            sumf += d * bsum;
        }
        printf("seed %d -> %.9e\n", seed, 0.125f * sumf);
    }
    return 0;
}
