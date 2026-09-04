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
pub mod broadcast;
pub mod comp_kv_append;
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
pub mod q2_k;
pub mod q4_k;
pub mod q4_k_dense;
pub mod q5_k_dense;
pub mod q6_k;
pub mod q6_k_dense;
pub mod q8_0;
pub mod q8_k;
pub mod mhc_pre_fused;
pub mod rms_norm;
pub mod rope;
pub mod sampler;
pub mod weights;
pub mod wmma_probe;
pub mod wmma_wsum;

pub use attention::{AttentionMixed, AttentionSwa, ATTN_MIXED_MAX_KEYS, ATTN_SCORES_STRIDE, ATTN_SWA_BATCHED_MAX_KV, ATTN_SWA_MAX_KV};
pub use broadcast::BroadcastToHc;
pub use comp_kv_append::CompKvAppend;
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
pub use rms_norm::{RmsNorm, RmsNormNoWeight, RmsNormNoWeightMultiWG};
pub use rope::{RopeParams, RopeTail};
pub use sampler::{Sampler, SamplerRng, SAMPLER_N_WG};
pub use weights::{load_to_device, DeviceWeight};
