//! Cross-check `cpu_dot_iq2_xs_q8_k` against llama.cpp's own
//! `ggml_vec_dot_iq2_xs_q8_K_generic`.
//!
//! Expected values come from a C harness that pairs upstream's tables
//! (ggml-common.h, GGML_COMMON_IMPL_C) with a verbatim copy of the
//! upstream function, driven by the same LCG stream reproduced below.
//! What this guards is TRANSCRIPTION: the reference side is upstream's
//! code, so a misread of the 74-byte layout, the 9-bit grid index, the
//! 7-bit sign index or the per-16 nibble scales fails here.
//!
//! Harness: scratchpad xscheck/gen.c (see git history / docs).

use v4flash_kernels::iq2_xs_tables::{cpu_dot_iq2_xs_q8_k, BLOCK_IQ2_XS_BYTES};

struct Lcg(u32);
impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) as u8
    }
}

#[test]
fn matches_llama_cpp_reference() {
    // seed -> value printed by the C harness
    const EXPECTED: [(u32, f32); 3] = [
        (1, -5.558305664e+02),
        (2, 7.960100098e+02),
        (3, -6.713171387e+01),
    ];
    let nb = 4usize;
    for (seed, want) in EXPECTED {
        let mut lcg = Lcg(seed * 7919);
        let mut w = vec![0u8; nb * BLOCK_IQ2_XS_BYTES];
        for b in w.iter_mut() {
            *b = lcg.next_u8();
        }
        // force d = 0x2e66 (~0.1) per block, matching the harness
        for b in 0..nb {
            w[b * BLOCK_IQ2_XS_BYTES] = 0x66;
            w[b * BLOCK_IQ2_XS_BYTES + 1] = 0x2e;
        }
        let mut y = vec![0u8; nb * 292];
        for b in 0..nb {
            y[b * 292..b * 292 + 4].copy_from_slice(&0.05f32.to_le_bytes());
            for j in 0..256 {
                y[b * 292 + 4 + j] = lcg.next_u8();
            }
            // bsums stay zero (unused by this dot)
        }
        let got = cpu_dot_iq2_xs_q8_k(nb, &w, &y);
        let rel = (got - want).abs() / want.abs().max(1e-6);
        assert!(
            rel < 1e-6,
            "seed {seed}: got {got:e}, llama.cpp {want:e} (rel {rel:e})"
        );
    }
}
