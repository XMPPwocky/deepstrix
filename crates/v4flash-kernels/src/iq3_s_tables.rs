//! IQ3_S codebook + scalar CPU dot reference for the `iq3_s_pair_matvec`
//! kernel oracles. The tables and math live in
//! [`v4flash_core::iq3_s_ref`] (pinned to llama.cpp by
//! `tests/ref/iq3_s_gen.c`); this module keeps the same surface the
//! sibling `iq2_s_tables` / `iq3_xxs_tables` expose so the oracle tests
//! read alike.

pub use v4flash_core::iq3_s_ref::{
    dequant_block_iq3_s, dequant_row_iq3_s, IQ3S_GRID, IQ3S_OFF_D, IQ3S_OFF_QH, IQ3S_OFF_QS,
    IQ3S_OFF_SCALES, IQ3S_OFF_SIGNS, KMASK_IQ2XS,
};

/// Block size of one IQ3_S super-block: f16 d + 64 grid-index bytes +
/// 8 qh high-bit bytes + 32 raw sign bytes + 4 scale-nibble bytes.
pub const BLOCK_IQ3_S_BYTES: usize = v4flash_core::iq3_s_ref::BLOCK_IQ3_S_BYTES;

/// Scalar CPU reference mirroring llama.cpp's
/// `ggml_vec_dot_iq3_s_q8_K_generic`: per super-block
/// `d * y.d * Σ_ib32 ls * sumi` with `ls = 1 + 2*nibble` — no prefactor
/// (iq2_s: 0.125, iq3_xxs: 0.25).
pub fn cpu_dot_iq3_s_q8_k(n_blocks: usize, w_bytes: &[u8], y_bytes: &[u8]) -> f32 {
    v4flash_core::iq3_s_ref::dot_iq3_s_q8_k(n_blocks, w_bytes, y_bytes)
}
