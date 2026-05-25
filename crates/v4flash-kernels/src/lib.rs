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

pub mod attention;
pub mod broadcast;
pub mod comp_kv_append;
pub mod compressor;
pub mod f16;
pub mod ffn;
pub mod forward;
pub mod head;
pub mod het;
pub mod indexer;
pub mod iq2_xxs;
pub mod iq2_xxs_tables;
pub mod keep_alive;
pub mod kv_cache_append;
pub mod oracle;
pub mod router_topk;
pub mod q2_k;
pub mod q4_k;
pub mod q8_0;
pub mod q8_k;
pub mod rms_norm;
pub mod rope;
pub mod weights;

pub use attention::{AttentionMixed, AttentionSwa, ATTN_MIXED_MAX_KEYS, ATTN_SWA_MAX_KV};
pub use broadcast::BroadcastToHc;
pub use comp_kv_append::CompKvAppend;
pub use compressor::{
    CompressorPool, CompressorStateShuffleR4, CompressorStateWrite, F16Roundtrip,
    Fp8E4m3fnQuantize,
};
pub use f16::F16Matvec;
pub use ffn::{Swiglu, SwigluClampWeighted, VecAddInplace};
pub use head::{HcPost, HcSigmoidBias, HcSinkhorn, HcWeightedSum};
pub use indexer::{IndexerScore, INDEXER_HEAD_DIM, INDEXER_N_HEAD, INDEXER_TOP_K};
pub use iq2_xxs::{Iq2XxsPairMatvec, BLOCK_IQ2_XXS_BYTES};
pub use kv_cache_append::KvCacheAppend;
pub use oracle::{ActivationDump, Dtype, TensorEntry};
pub use router_topk::{RouterTopk, ROUTER_MAX_EXPERTS, ROUTER_MAX_USED};
pub use q2_k::{Q2KAccumulateMatvec, BLOCK_Q2_K_BYTES};
pub use q4_k::{Q4KMatvec, BLOCK_Q4_K_BYTES};
pub use q8_0::{Q8_0GroupedMatvec, Q8_0Matvec};
pub use q8_k::{Q8KQuantize, BLOCK_Q8_K_BYTES, QK_K};
pub use rms_norm::{RmsNorm, RmsNormNoWeight};
pub use rope::{RopeParams, RopeTail};
pub use weights::{load_to_device, DeviceWeight};
