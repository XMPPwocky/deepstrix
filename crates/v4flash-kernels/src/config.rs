//! V4 Flash model architecture constants.
//!
//! Pure compile-time facts derived from the GGUF metadata; no logic.
//! Imported by every layer-shape-aware module — keep this minimal so
//! reading the file teaches you the model's shape.

// === Embedding / projection dims ===

pub const N_EMBD: u32 = 5120;
pub const N_HC: u32 = 4;
pub const HC_DIM: u32 = N_EMBD * N_HC; // 16384
pub const HC_MIX_DIM: u32 = 2 * N_HC + N_HC * N_HC; // 24

// === Attention dims ===

pub const N_HEAD: u32 = 64;
pub const N_HEAD_DIM: u32 = 512;
pub const N_ROT: u32 = 64;
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

pub const N_GROUPS: u32 = 8;
pub const GROUP_DIM: u32 = 4096;
pub const RANK: u32 = 1024;
pub const OUT_LOW: u32 = N_GROUPS * RANK; // 8192

// === FFN dims ===

pub const N_FF_SHARED: u32 = 2304;
pub const N_FF_EXP: u32 = 2304;
pub const N_EXPERT: u32 = 384;
pub const N_EXPERT_USED: usize = 6;

// === Output / layers ===

pub const N_VOCAB: u32 = 129280;
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

pub const RMS_EPS: f32 = 1.0e-20; // config `norm_eps`; a launch argument everywhere
pub const SINKHORN_EPS: f32 = 1.0e-6;
pub const SINKHORN_ITERS: u32 = 20;
pub const SWIGLU_CLAMP_EXP: f32 = 10.0;
pub const EXPERT_WEIGHT_SCALE: f32 = 1.5;
pub const ROPE_ORIG_CTX: u64 = 65536;
pub const NEG_INF: f32 = -3.4028235e38;

// === Indexer (ratio=4 layers only) ===

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

/// V4.1-Flash CSA2: layers 0–1 Full (no compression), 2–19 ratio 2, 20–39 ratio 1
/// (`inference/config.json` `compress_ratios`, first 40 entries).
pub const COMPRESS_RATIOS: [u32; N_LAYER as usize] = [
    0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
];
/// V4.1-Flash CSA2 source tables (`kv_source_layers` / `index_source_layers`): a
/// KV-source layer owns its compressor + window/main KV; an index-source layer
/// owns indexer keys; every other layer reuses the nearest source below it.
pub const KV_SOURCE_LAYERS: &[i32] = &[2, 8, 14, 20];
pub const INDEX_SOURCE_LAYERS: &[i32] = &[2, 8, 14, 20, 24, 28, 32, 36];
pub const ENGRAM_LAYERS: &[i32] = &[1, 14];

/// The index source whose top-k selection `layer` attends to: the nearest
/// index-source layer at or below it. Reference `model.py::_compress_topk_idxs`
/// — "index sources run their own indexer; the layers in between reuse the
/// result their source published", i.e. the MOST RECENT source, not the KV one.
///
/// Distinct from [`kv_source_of`] on purpose. Layers 20/24/28/32/36 are all
/// index sources but share KV source 20, so keying a published selection on the
/// KV source cannot tell them apart and a reuse layer could silently consume an
/// older source's selection.
pub fn index_source_of(layer: usize) -> Option<usize> {
    if COMPRESS_RATIOS[layer] == 0 {
        return None;
    }
    INDEX_SOURCE_LAYERS
        .iter()
        .copied()
        .filter(|&s| (s as usize) <= layer)
        .max()
        .map(|s| s as usize)
}
/// V4.1 Causal Encoder-Decoder split (tech report §2.2): layers below are the
/// causal encoder; this layer is the decoder's Full-mode layer whose global KV
/// is projected from the final encoder hidden state. CED prefill runs the
/// encoder over every prompt token and the decoder over the last `SWA_WINDOW`
/// only (`het::forward_prefill::CedMode`, docs/v41/ENGINE_PORT.md "M7 CED").
pub const CED_DECODER_START: usize = 20;
pub const CANDIDATE_SOURCE_LAYER: i32 = 20;
pub const CANDIDATE_TOPK_BLOCKS: u32 = 2048;
pub const CANDIDATE_BLOCK_SIZE: u32 = 8;
/// Which model this binary was compiled for (see docs/v41/ENGINE_PORT.md §0).
pub const MODEL_NAME: &str = "DeepSeek-V4.1-Flash";

pub const N_HASH_LAYERS: i32 = 0;
