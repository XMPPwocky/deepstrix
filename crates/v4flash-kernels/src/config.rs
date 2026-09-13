//! V4 Flash model architecture constants.
//!
//! Pure compile-time facts derived from the GGUF metadata; no logic.
//! Imported by every layer-shape-aware module — keep this minimal so
//! reading the file teaches you the model's shape.

// === Embedding / projection dims ===

#[cfg(not(feature = "v41"))]
pub const N_EMBD: u32 = 4096;
#[cfg(feature = "v41")]
pub const N_EMBD: u32 = 5120;
pub const N_HC: u32 = 4;
pub const HC_DIM: u32 = N_EMBD * N_HC; // 16384
pub const HC_MIX_DIM: u32 = 2 * N_HC + N_HC * N_HC; // 24

// === Attention dims ===

pub const N_HEAD: u32 = 64;
pub const N_HEAD_DIM: u32 = 512;
pub const N_ROT: u32 = 64;
#[cfg(not(feature = "v41"))]
pub const N_LORA_Q: u32 = 1024;
#[cfg(feature = "v41")]
pub const N_LORA_Q: u32 = 1280;
pub const Q_FLAT: u32 = N_HEAD * N_HEAD_DIM; // 32768
/// V4.1 Engram (ARCH_SPEC §1.7): 24 gathered rows × 256 → wkv → (N_HC + 1) × N_EMBD
/// (N_HC keys + one value). `ENGRAM_CHUNK` = prefill rows per wkv/gate pass.
pub const ENGRAM_IN: u32 = 24 * 256;
pub const ENGRAM_OUT: u32 = (N_HC + 1) * N_EMBD;
pub const ENGRAM_CHUNK: u32 = 64;

/// V4.1 CSA2 reuse: the layer whose compressed-KV store layer `layer` attends
/// over. A KV-source layer owns its store; every other compressed layer reads
/// the most recent source's. V4-Flash layers all own their own (`None`).
#[cfg(feature = "v41")]
pub fn kv_source_of(layer: usize) -> Option<usize> {
    if COMPRESS_RATIOS[layer] == 0 {
        return None;
    }
    let mut src = None;
    for &s in KV_SOURCE_LAYERS {
        if (s as usize) <= layer {
            src = Some(s as usize);
        }
    }
    match src {
        Some(s) if s != layer => Some(s),
        _ => None,
    }
}

#[cfg(not(feature = "v41"))]
pub fn kv_source_of(_layer: usize) -> Option<usize> {
    None
}
pub const N_GROUPS: u32 = 8;
pub const GROUP_DIM: u32 = 4096;
pub const RANK: u32 = 1024;
pub const OUT_LOW: u32 = N_GROUPS * RANK; // 8192

// === FFN dims ===

#[cfg(not(feature = "v41"))]
pub const N_FF_SHARED: u32 = 2048;
#[cfg(feature = "v41")]
pub const N_FF_SHARED: u32 = 2304;
#[cfg(not(feature = "v41"))]
pub const N_FF_EXP: u32 = 2048;
#[cfg(feature = "v41")]
pub const N_FF_EXP: u32 = 2304;
#[cfg(not(feature = "v41"))]
pub const N_EXPERT: u32 = 256;
#[cfg(feature = "v41")]
pub const N_EXPERT: u32 = 384;
pub const N_EXPERT_USED: usize = 6;

// === Output / layers ===

pub const N_VOCAB: u32 = 129280;
#[cfg(not(feature = "v41"))]
pub const N_LAYER: i32 = 43;
#[cfg(feature = "v41")]
pub const N_LAYER: i32 = 40;

// === Quant block-count helpers ===

pub const BLOCKS_N_EMBD: u32 = N_EMBD / 32;
pub const BLOCKS_OUT_LOW: u32 = OUT_LOW / 32;
pub const BLOCKS_GROUPED_OUT: u32 = (GROUP_DIM / 32) * N_GROUPS; // 1024
pub const BLOCKS_N_LORA_Q: u32 = N_LORA_Q / 32;
pub const BLOCKS_N_FF_SHARED: u32 = N_FF_SHARED / 32;
pub const BLOCKS_Q8K_GATE_IN: u32 = N_EMBD / 256; // 16
pub const BLOCKS_Q8K_DOWN_IN: u32 = N_FF_EXP / 256; // 8

// === Numerical / sentinel constants ===

#[cfg(not(feature = "v41"))]
pub const RMS_EPS: f32 = 1.0e-6;
#[cfg(feature = "v41")]
pub const RMS_EPS: f32 = 1.0e-20; // config `norm_eps`; a launch argument everywhere
pub const SINKHORN_EPS: f32 = 1.0e-6;
pub const SINKHORN_ITERS: u32 = 20;
pub const SWIGLU_CLAMP_EXP: f32 = 10.0;
pub const EXPERT_WEIGHT_SCALE: f32 = 1.5;
pub const ROPE_ORIG_CTX: u64 = 65536;
pub const NEG_INF: f32 = -3.4028235e38;

// === Indexer (ratio=4 layers only) ===

#[cfg(not(feature = "v41"))]
pub const N_INDEXER_HEAD: u32 = 64;
#[cfg(feature = "v41")]
pub const N_INDEXER_HEAD: u32 = 32;
pub const N_INDEXER_HEAD_DIM: u32 = 128;
pub const INDEXER_TOP_K: u32 = 512;
/// Indexer compressor width — coff * N_INDEXER_HEAD_DIM where coff = 2 at
/// ratio=4 (the only ratio the indexer fires on). Mirrors the main
/// compressor's `comp_width` formula, just at the indexer's head_dim.
pub const INDEXER_COMP_WIDTH: u32 = 2 * N_INDEXER_HEAD_DIM;

// === Attention / routing topology ===

/// SWA window: hard cap on `n_raw` in attention. Forward orchestrator
/// memmove-evicts beyond this.
pub const SWA_WINDOW: u32 = 128;

/// Per-layer compressor ratio. 0 = dense (no compression), N = compress
/// every N tokens into one comp row.
#[cfg(not(feature = "v41"))]
pub const COMPRESS_RATIOS: [u32; N_LAYER as usize] = [
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
];
/// V4.1-Flash CSA2: layers 0–1 Full (no compression), 2–19 ratio 2, 20–39 ratio 1
/// (`inference/config.json` `compress_ratios`, first 40 entries).
#[cfg(feature = "v41")]
pub const COMPRESS_RATIOS: [u32; N_LAYER as usize] = [
    0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
];
/// V4.1-Flash CSA2 source tables (`kv_source_layers` / `index_source_layers`): a
/// KV-source layer owns its compressor + window/main KV; an index-source layer
/// owns indexer keys; every other layer reuses the nearest source below it.
#[cfg(feature = "v41")]
pub const KV_SOURCE_LAYERS: &[i32] = &[2, 8, 14, 20];
#[cfg(feature = "v41")]
pub const INDEX_SOURCE_LAYERS: &[i32] = &[2, 8, 14, 20, 24, 28, 32, 36];
#[cfg(feature = "v41")]
pub const ENGRAM_LAYERS: &[i32] = &[1, 14];
/// V4.1 Causal Encoder-Decoder split (tech report §2.2): layers below are the
/// causal encoder; this layer is the decoder's Full-mode layer whose global KV
/// is projected from the final encoder hidden state. CED prefill runs the
/// encoder over every prompt token and the decoder over the last `SWA_WINDOW`
/// only (`het::forward_prefill::CedMode`, docs/v41/ENGINE_PORT.md "M7 CED").
#[cfg(feature = "v41")]
pub const CED_DECODER_START: usize = 20;
#[cfg(not(feature = "v41"))]
pub const CED_DECODER_START: usize = N_LAYER as usize;
/// V4-Flash has no Engram. Defined as EMPTY (rather than cfg-gating every use) so the shared
/// forward/engine code compiles for both models: the Engram sites are already runtime-guarded
/// (`dgpu_layers[l].engram.is_some()`, `rows: Option<..>`), and an empty slice makes the
/// `position()` lookups return None, so V4-Flash behaviour is unchanged.
#[cfg(not(feature = "v41"))]
pub const ENGRAM_LAYERS: &[i32] = &[];
#[cfg(feature = "v41")]
pub const CANDIDATE_SOURCE_LAYER: i32 = 20;
#[cfg(feature = "v41")]
pub const CANDIDATE_TOPK_BLOCKS: u32 = 2048;
#[cfg(feature = "v41")]
pub const CANDIDATE_BLOCK_SIZE: u32 = 8;
/// Which model this binary was compiled for (see docs/v41/ENGINE_PORT.md §0).
pub const MODEL_NAME: &str = if cfg!(feature = "v41") { "DeepSeek-V4.1-Flash" } else { "DeepSeek-V4-Flash" };

/// First N_HASH_LAYERS layers use the hash router (bootstrap). The rest
/// use the learned router.
#[cfg(not(feature = "v41"))]
pub const N_HASH_LAYERS: i32 = 3;
#[cfg(feature = "v41")]
pub const N_HASH_LAYERS: i32 = 0;
