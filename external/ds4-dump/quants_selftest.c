/* quants_selftest — validates ds4's Track R quant ports (IQ3_XXS, IQ2_S,
 * MXFP4) against llama.cpp's reference implementations.
 *
 * Two checks per type:
 *
 * 1. DEQUANT EXACTNESS: deterministic synthetic blocks (LCG bytes, fixed
 *    seeds, sane forced scales) are dequantized through ds4_ref_dequant_row
 *    and compared BIT-FOR-BIT against quants_selftest_expected.inc — f32
 *    bit patterns produced by llama.cpp's actual dequantize_row_* scalar
 *    code (generator: gen_expected.c harness compiled against
 *    llama.cpp/ggml/src/ggml-quants.c; same LCG, same seeds).
 *
 * 2. VEC_DOT CONSISTENCY: random weight rows (4 × 256 elements) dotted
 *    with a ds4-quantized Q8_K activation must equal the double-precision
 *    dot of dequantize(weights) · dequantize(q8_k) within 1e-4 relative.
 *
 * Block-byte reconstruction MUST match the generator: fill the whole
 * struct-sized region with LCG bytes, then force d (f16 le at the type's
 * offset) or the MXFP4 e bytes.
 *
 * Usage: ./quants-selftest   (exit 0 = pass)
 */

#include "../ds4/ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "quants_selftest_expected.inc"

#define QK_K 256

/* Byte sizes / layouts of the tested block formats (static-asserted
 * against the structs inside ds4.c). */
#define IQ3_XXS_BLOCK_BYTES 98  /* u16 d | qs[96] */
#define IQ2_S_BLOCK_BYTES   82  /* u16 d | qs[64] | qh[8] | scales[8] */
#define MXFP4_BLOCK_BYTES   17  /* u8 e | qs[16], 32 elements */

/* Q8_K block layout (ds4.c block_q8_K, static-asserted 292 B):
 * f32 d | i8 qs[256] | i16 bsums[16] */
#define Q8_K_BLOCK_BYTES 292

static uint32_t lcg_state = 0;
static uint8_t lcg_u8(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (uint8_t)(lcg_state >> 24);
}
static void fill(uint8_t *p, size_t n, uint32_t seed) {
    lcg_state = seed;
    for (size_t i = 0; i < n; i++) p[i] = lcg_u8();
}

static int failures = 0;

static void check_bits(const char *name, const float *got, const uint32_t *want, int n) {
    for (int i = 0; i < n; i++) {
        uint32_t g;
        memcpy(&g, &got[i], 4);
        if (g != want[i]) {
            fprintf(stderr, "FAIL %s: elem %d got 0x%08x want 0x%08x\n", name, i, g, want[i]);
            failures++;
            return;
        }
    }
    printf("ok   %s dequant matches llama.cpp bit-for-bit (%d values)\n", name, n);
}

/* d16 forced into every block: 0x2e66 (~0.1), same as the generator. */
static const uint16_t d16 = 0x2e66;

static void build_iq3_xxs(uint8_t *blk, uint32_t seed) {
    fill(blk, IQ3_XXS_BLOCK_BYTES, seed);
    memcpy(blk, &d16, 2);
}
static void build_iq2_s(uint8_t *blk, uint32_t seed) {
    fill(blk, IQ2_S_BLOCK_BYTES, seed);
    memcpy(blk, &d16, 2);
}
/* 8 consecutive MXFP4 blocks (256 elements), e forced to sane exponents. */
static void build_mxfp4_run(uint8_t *blks, uint32_t seed, int n_blocks, int e_base) {
    fill(blks, (size_t)n_blocks * MXFP4_BLOCK_BYTES, seed);
    for (int i = 0; i < n_blocks; i++) {
        blks[(size_t)i * MXFP4_BLOCK_BYTES] = (uint8_t)(e_base + i);
    }
}

static void test_dequant_exact(void) {
    float y[QK_K];

    uint8_t b3[IQ3_XXS_BLOCK_BYTES];
    build_iq3_xxs(b3, 0x1111u);
    if (ds4_ref_dequant_row(18, b3, y, QK_K) != 0) { fprintf(stderr, "FAIL iq3_xxs dispatch\n"); failures++; }
    else check_bits("iq3_xxs", y, expected_iq3_xxs, QK_K);

    uint8_t b2s[IQ2_S_BLOCK_BYTES];
    build_iq2_s(b2s, 0x2222u);
    if (ds4_ref_dequant_row(22, b2s, y, QK_K) != 0) { fprintf(stderr, "FAIL iq2_s dispatch\n"); failures++; }
    else check_bits("iq2_s", y, expected_iq2_s, QK_K);

    uint8_t bm[8 * MXFP4_BLOCK_BYTES];
    build_mxfp4_run(bm, 0x3333u, 8, 120);
    if (ds4_ref_dequant_row(39, bm, y, QK_K) != 0) { fprintf(stderr, "FAIL mxfp4 dispatch\n"); failures++; }
    else check_bits("mxfp4", y, expected_mxfp4, QK_K);
}

/* Dequantize a ds4 Q8_K activation row from raw block bytes. */
static void dequant_q8_k(const uint8_t *q8k, float *y, int n) {
    const int nb = n / QK_K;
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = q8k + (size_t)b * Q8_K_BLOCK_BYTES;
        float d;
        memcpy(&d, blk, 4);
        const int8_t *qs = (const int8_t *)(blk + 4);
        for (int i = 0; i < QK_K; i++) y[b * QK_K + i] = d * (float)qs[i];
    }
}

static void test_vec_dot(uint32_t type, const char *name, size_t block_bytes,
                         int elems_per_block, uint32_t seed) {
    const int n = 4 * QK_K;
    const int n_blocks = n / elems_per_block;
    uint8_t *w = malloc((size_t)n_blocks * block_bytes);
    float *wf = malloc((size_t)n * sizeof(float));
    float *x = malloc((size_t)n * sizeof(float));
    float *xf = malloc((size_t)n * sizeof(float));
    uint8_t *xq = malloc(ds4_ref_q8_K_row_bytes(n));

    /* weight row */
    if (type == 39) {
        build_mxfp4_run(w, seed, n_blocks, 118);
    } else {
        for (int b = 0; b < n_blocks; b++) {
            uint8_t *blk = w + (size_t)b * block_bytes;
            fill(blk, block_bytes, seed + (uint32_t)b * 0x9e3779b9u);
            memcpy(blk, &d16, 2);
        }
    }
    if (ds4_ref_dequant_row(type, w, wf, n) != 0) {
        fprintf(stderr, "FAIL %s dequant dispatch\n", name);
        failures++;
        goto done;
    }

    /* activation: deterministic pseudo-random floats in [-2, 2) */
    lcg_state = seed ^ 0xabcdef01u;
    for (int i = 0; i < n; i++) {
        x[i] = ((float)lcg_u8() - 127.5f) / 64.0f;
    }
    ds4_ref_quantize_row_q8_K(x, xq, n);
    dequant_q8_k(xq, xf, n);

    float got = 0.0f;
    if (ds4_ref_vec_dot_q8k(type, n, &got, w, xq) != 0) {
        fprintf(stderr, "FAIL %s vec_dot dispatch\n", name);
        failures++;
        goto done;
    }

    double want = 0.0;
    for (int i = 0; i < n; i++) want += (double)wf[i] * (double)xf[i];

    const double denom = fabs(want) > 1e-6 ? fabs(want) : 1e-6;
    const double rel = fabs((double)got - want) / denom;
    if (rel > 1e-4) {
        fprintf(stderr, "FAIL %s vec_dot: got %.9g want %.9g (rel %.3g)\n", name, got, want, rel);
        failures++;
    } else {
        printf("ok   %s vec_dot == dequant-dot (got %.6g, rel err %.2g)\n", name, got, rel);
    }

done:
    free(xq);
    free(xf);
    free(x);
    free(wf);
    free(w);
}

int main(void) {
    test_dequant_exact();
    test_vec_dot(18, "iq3_xxs", IQ3_XXS_BLOCK_BYTES, QK_K, 0x51u);
    test_vec_dot(22, "iq2_s", IQ2_S_BLOCK_BYTES, QK_K, 0x52u);
    test_vec_dot(39, "mxfp4", MXFP4_BLOCK_BYTES, 32, 0x53u);

    if (failures != 0) {
        fprintf(stderr, "quants_selftest: %d FAILURE(S)\n", failures);
        return 1;
    }
    printf("quants_selftest: all checks passed\n");
    return 0;
}
