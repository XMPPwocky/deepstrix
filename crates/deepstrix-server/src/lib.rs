//! deepstrix-server — OpenAI-compatible HTTP frontend over the V4-Flash
//! heterogeneous inference engine. See `/home/claude-code/.claude/plans/
//! shimmering-wandering-blanket.md` for the full phased design.

pub mod dsml;
pub mod embed;
pub mod engine_worker;
pub mod openai;
pub mod prompt;
pub mod tokens;

use color_eyre::eyre;
use v4flash_kernels::config::COMPRESS_RATIOS;
use v4flash_kernels::RopeParams;

// V4-Flash RoPE constants (copied from chat.rs:222-227 to avoid a
// cross-crate dep; if RoPE config ever changes upstream, keep these in
// sync).
const ROPE_FREQ_BASE_DENSE: f32 = 10000.0;
const ROPE_FREQ_BASE_COMP: f32 = 160000.0;
const ROPE_SCALE_FACTOR: f32 = 16.0;
const ROPE_ORIG_CTX: u64 = 65536;
const ROPE_BETA_FAST: f32 = 32.0;
const ROPE_BETA_SLOW: f32 = 1.0;

pub fn rope_for_layer(layer: i32) -> RopeParams {
    let ratio = COMPRESS_RATIOS[layer as usize];
    let compressed = ratio != 0;
    let freq_base = if compressed {
        ROPE_FREQ_BASE_COMP
    } else {
        ROPE_FREQ_BASE_DENSE
    };
    let freq_scale = if compressed {
        1.0 / ROPE_SCALE_FACTOR
    } else {
        1.0
    };
    let ext_factor = if compressed && ROPE_SCALE_FACTOR > 1.0 {
        1.0
    } else {
        0.0
    };
    let mut attn_factor = 1.0f32;
    if ext_factor != 0.0 && freq_scale > 0.0 {
        attn_factor /= 1.0 + 0.1 * (1.0 / freq_scale).ln();
    }
    let n_ctx_orig = if compressed { ROPE_ORIG_CTX } else { 0 };
    let floats = [
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        ROPE_BETA_FAST,
        ROPE_BETA_SLOW,
    ];
    RopeParams::from_dump_blob(&floats, n_ctx_orig).expect("valid rope params")
}

/// Suppress unused-import lint at the crate root.
const _: fn() -> eyre::Result<()> = || Ok(());
