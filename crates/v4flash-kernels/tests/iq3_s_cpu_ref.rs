//! Cross-check `cpu_dot_iq3_s_q8_k` against llama.cpp's own
//! `ggml_vec_dot_iq3_s_q8_K_generic`.
//!
//! Expected values come from `tests/ref/iq3_s_gen.c`, which pairs
//! upstream's tables (ggml-common.h, GGML_COMMON_IMPL_C) with a verbatim
//! copy of the upstream function, driven by the same LCG stream reproduced
//! below. What this guards is TRANSCRIPTION: the reference side is
//! upstream's code, so a misread of the 110-byte layout, the 9-bit grid
//! index (8 from qs + 1 from qh), the raw sign bytes, the per-64 nibble
//! scales or a stray 0.125/0.25 prefactor fails here.

use v4flash_kernels::iq3_s_tables::{cpu_dot_iq3_s_q8_k, BLOCK_IQ3_S_BYTES};

struct Lcg(u32);
impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        (self.0 >> 16) as u8
    }
}

/// Per-block d table of the harness (varied so a uniform-d bug can't hide).
const D_TABLE: [u16; 4] = [0x2e66, 0x3266, 0x2a66, 0x3466];

#[test]
fn matches_llama_cpp_reference() {
    // seed -> value printed by the C harness
    const EXPECTED: [(u32, f32); 3] = [
        (1, -2.962029053e+03),
        (2, 9.427104492e+03),
        (3, -5.965603027e+02),
    ];
    let nb = 4usize;
    for (seed, want) in EXPECTED {
        let mut lcg = Lcg(seed * 7919);
        let mut w = vec![0u8; nb * BLOCK_IQ3_S_BYTES];
        for b in w.iter_mut() {
            *b = lcg.next_u8();
        }
        for b in 0..nb {
            let dd = D_TABLE[(b + seed as usize) & 3];
            w[b * BLOCK_IQ3_S_BYTES] = (dd & 0xff) as u8;
            w[b * BLOCK_IQ3_S_BYTES + 1] = (dd >> 8) as u8;
        }
        let mut y = vec![0u8; nb * 292];
        for b in 0..nb {
            let yd = 0.05f32 * (b + 1) as f32;
            y[b * 292..b * 292 + 4].copy_from_slice(&yd.to_le_bytes());
            for j in 0..256 {
                y[b * 292 + 4 + j] = lcg.next_u8();
            }
            // bsums stay zero (unused by this dot)
        }
        let got = cpu_dot_iq3_s_q8_k(nb, &w, &y);
        let rel = (got - want).abs() / want.abs().max(1e-6);
        assert!(
            rel < 1e-6,
            "seed {seed}: got {got:e}, llama.cpp {want:e} (rel {rel:e})"
        );
    }
}
