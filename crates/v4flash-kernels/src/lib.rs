//! v4flash-kernels — HIP kernels for V4 Flash inference + a per-kernel
//! oracle-based validation framework.
//!
//! Layout:
//! - [`oracle`]   — loads the M2 activation dump tree (manifest + binary blobs)
//!                  produced by `external/ds4-dump/ds4-dump-activations`
//! - [`rms_norm`] — first ported kernel, `rms_norm_weighted`
//!
//! Each ported kernel ships a Rust wrapper around its HIP `.hip` source
//! (compiled to per-arch `.hsaco` by `build.rs`) plus an `#[ignore]`-gated
//! oracle test under `tests/`. The test loads the relevant tag slices
//! from the activation dump and asserts `max_abs_diff < threshold`.

/// Decimal parse for build-script-provided bounds, usable in `const` items.
///
/// `build.rs` emits the kwide `*_KW_MAX_CHUNK` values as `rustc-env`
/// strings so the launch guards track the macro the kernels were actually
/// compiled with; this turns one back into a `u32` at compile time.
pub(crate) const fn parse_u32_dec(s: &str) -> u32 {
    let b = s.as_bytes();
    assert!(!b.is_empty(), "expected a decimal integer");
    let mut i = 0;
    let mut v: u32 = 0;
    while i < b.len() {
        assert!(b[i] >= b'0' && b[i] <= b'9', "expected a decimal integer");
        v = v * 10 + (b[i] - b'0') as u32;
        i += 1;
    }
    v
}

pub mod attention;
pub mod attn_meta;
pub mod attention_dec;
pub mod b2_fast_chain;
pub mod broadcast;
pub mod candidate_blocks;
pub mod comp_kv_append;
pub mod comp_kv_fp8;
pub mod index_kv_e2m1;
pub mod compressor;
pub mod config;
pub mod f16;
pub mod ffn;
pub mod gqa_attention;
pub mod head;
pub mod model_weights;
pub mod routing;
pub mod het;
pub mod expert_sel_count;
pub mod indexer;
pub mod iq2_xxs;
pub mod iq2_xxs_tables;
pub mod dense_gemm;
pub mod device_ceilings;
pub mod embed;
pub mod iq2_s;
pub mod iq2_s_tables;
pub mod iq2_xs;
pub mod iq2_xs_tables;
pub mod iq3_xxs;
pub mod iq3_xxs_pair;
pub mod iq3_xxs_tables;
pub mod iq3_s;
pub mod iq3_s_tables;
pub mod mxfp4;
pub mod mxfp4_pair;
pub mod mxfp4_repack;
pub mod engram_gate;
pub mod fp4_kv;
pub mod mxfp4_tables;
pub mod kv_cache_append;
pub mod laguna;
pub mod laguna_het;
pub mod laguna_het_moe;
pub mod laguna_moe_tiled;
pub mod moe_group_builder;
pub mod oracle;
pub mod router_topk;
pub mod weight_contract;
/// `V41_GRID_PAD` (default ON; `0` = the exact grids): ONE extra, idle
/// work-group in grid.x for the short dGPU launches whose production grid lands
/// in gfx1201's slow-dispatch window. Every padded kernel already guards x (the
/// extra WG exits / is a sentinel row), so outputs are byte-identical
/// (tests/grid_pad_bitexact.rs).
///
/// DO NOT REMOVE THE PAD. It is real gfx1201 dispatch behaviour (the same one
/// `V41_Q8_QUANT_GRID_PAD` works around): a short kernel whose total wave count
/// is (at or a few WGs below) a multiple of 2048 -- 1-wave WGs at 2040..2048,
/// 8-wave WGs at 255/256, 511/512, 768, 1023/1024, 16-wave WGs at 127/128,
/// 255/256, 384, 512 -- cannot finish in under ~17 us, whatever its work
/// (synthetic probe: 3.7 us at 2049 WGs vs 17 us at 2048; a kernel whose natural
/// time is >= ~20 us hides it). gfx1151 does not have it (probe + real kernels
/// measured identical), and the idle WG costs nothing measurable there.
/// Measured 2026-09-27 (round 2, a_gridpad; warm, graph, 5 runs, med / p10
/// ratio padded vs exact): fp8_act_quant_inplace b=128/256/512 (prefill window
/// KV) 0.28/0.34/0.42; indexer_fp4 32*b rows b=16/32/64/128 0.26/0.30/0.40/0.69
/// (replay b=64); rope_tail_batched q (64,1,32) 0.29, idx-q (32,1,64) 0.29
/// (replay); f32_to_f16_cast_2d b=16 x 32768 and b=64 x 8192 (replay heads /
/// low) 0.26. Neutral (+-1%) at every other b measured (1..512).
/// NOT a general rule: the same pad is 1.28x SLOWER on f16_matvec_batched
/// (4,1,64) and neutral on hc_post / f16_matvec_narrow / the router matvec,
/// so it is applied per launch site, never globally.
pub fn grid_pad() -> u32 {
    static D: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_GRID_PAD").as_deref() != Ok("0"));
    if *D { 1 } else { 0 }
}

pub mod q2_k;
pub mod q4_k;
pub mod q4_k_dense;
pub mod q5_k_dense;
pub mod q6_k;
pub mod q6_k_dense;
pub mod q8_0;
pub mod q8_k;
pub mod mhc_pre_fused;
pub mod mhc_arena;
pub mod readback_pack;
pub mod rms_norm;
pub mod rope;
pub mod sampler;
pub mod weights;
pub mod wmma_probe;
pub mod wmma_wsum;

pub use attention::{AttentionMixed, AttentionSwa, ATTN_MIXED_MAX_KEYS, ATTN_SCORES_STRIDE, ATTN_SWA_BATCHED_MAX_KV, ATTN_SWA_MAX_KV};
pub use broadcast::BroadcastToHc;
pub use comp_kv_append::CompKvAppend;
pub use comp_kv_fp8::{CompKvFp8, FP8_KV_HEAD_ROWS, FP8_KV_ROW_BYTES};
pub use index_kv_e2m1::{IndexKvE2m1, E2M1_KEY_ROW_BYTES};
pub use compressor::{
    CompressorPool, CompressorStateShuffleR4, CompressorStateSnapshot, CompressorStateWrite,
    F16Roundtrip, Fp8E4m3fnQuantize,
};
pub use expert_sel_count::ExpertSelCount;
pub use f16::F16Matvec;
pub use ffn::{Swiglu, SwigluClampWeighted, VecAddInplace};
pub use gqa_attention::{GqaAttention, FLASH_HEAD_DIM, GQA_HEAD_DIM_MAX};
pub use head::{HcPost, HcSigmoidBias, HcSinkhorn, HcWeightedSum};
pub use indexer::{
    IndexerBitpack, IndexerGather, IndexerQat, IndexerScore, IndexerScoreWmma, IndexerTopk,
    IndexerTopkBitonic, VecScaleInplace, INDEXER_HEAD_DIM, INDEXER_N_HEAD, INDEXER_TOP_K,
};
pub use iq2_xxs::{Iq2XxsPairMatvec, BLOCK_IQ2_XXS_BYTES};
pub use kv_cache_append::KvCacheAppend;
pub use laguna::{LagunaHparams, LagunaModel, LagunaOps};
pub use laguna_moe_tiled::LagunaMoeTiled;
pub use moe_group_builder::MoeGroupBuilder;
// `oracle::{ActivationDump, Dtype, TensorEntry}` is intentionally NOT
// re-exported at the crate root — it's a test-fixture loader for the
// M2-era activation dumps, not part of the production API. Tests import
// it as `v4flash_kernels::oracle::ActivationDump`.
pub use router_topk::{RouterTopk, ROUTER_MAX_EXPERTS, ROUTER_MAX_USED};
pub use q2_k::{Q2KAccumulateMatvec, BLOCK_Q2_K_BYTES};
pub use q4_k::{Q4KMatvec, BLOCK_Q4_K_BYTES};
pub use q4_k_dense::{Q4_KDenseMatvec, Q4_K_DENSE_BLOCK_BYTES, Q4_K_DENSE_BLOCK_ELEMS};
pub use q6_k::Q6KMatvec;
pub use q6_k_dense::{Q6_KDenseMatvec, Q6_K_DENSE_BLOCK_BYTES, Q6_K_DENSE_BLOCK_ELEMS};
pub use q8_0::{Q8_0GroupedMatvec, Q8_0Matvec, Q8_0MatvecWmma};
pub use q8_k::{Q8KQuantize, BLOCK_Q8_K_BYTES, QK_K};
pub use mhc_pre_fused::MhcPreFused;
pub use mhc_arena::MhcArena;
pub use readback_pack::ReadbackPack;
pub use rms_norm::{RmsNorm, RmsNormNoWeight, RmsNormNoWeightMultiWG};
pub use rope::{RopeParams, RopeTail};
pub use sampler::{
    sampler_topp_mass_len, top_p_cutoff, top_p_min_p_threshold, Sampler, SamplerRng,
    SAMPLER_N_WG, SAMPLER_TOPP_LEVELS, SAMPLER_TOPP_LOG_RANGE, SAMPLER_TOPP_NBINS,
    SAMPLER_TOPP_NEDGE,
};
pub use weights::{load_to_device, DeviceWeight};
