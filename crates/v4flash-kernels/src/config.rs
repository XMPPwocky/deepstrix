//! V4 Flash model architecture constants.
//!
//! Pure compile-time facts derived from the GGUF metadata; no logic.
//! Imported by every layer-shape-aware module — keep this minimal so
//! reading the file teaches you the model's shape.

// === Embedding / projection dims ===

pub const N_EMBD: u32 = 4096;
pub const N_HC: u32 = 4;
pub const HC_DIM: u32 = N_EMBD * N_HC; // 16384
pub const HC_MIX_DIM: u32 = 2 * N_HC + N_HC * N_HC; // 24

// === Attention dims ===

pub const N_HEAD: u32 = 64;
pub const N_HEAD_DIM: u32 = 512;
pub const N_ROT: u32 = 64;
pub const N_LORA_Q: u32 = 1024;
pub const Q_FLAT: u32 = N_HEAD * N_HEAD_DIM; // 32768
pub const N_GROUPS: u32 = 8;
pub const GROUP_DIM: u32 = 4096;
pub const RANK: u32 = 1024;
pub const OUT_LOW: u32 = N_GROUPS * RANK; // 8192

// === FFN dims ===

pub const N_FF_SHARED: u32 = 2048;
pub const N_FF_EXP: u32 = 2048;
pub const N_EXPERT: u32 = 256;
pub const N_EXPERT_USED: usize = 6;

// === Output / layers ===

pub const N_VOCAB: u32 = 129280;
pub const N_LAYER: i32 = 43;

// === Quant block-count helpers ===

pub const BLOCKS_N_EMBD: u32 = N_EMBD / 32;
pub const BLOCKS_OUT_LOW: u32 = OUT_LOW / 32;
pub const BLOCKS_GROUPED_OUT: u32 = (GROUP_DIM / 32) * N_GROUPS; // 1024
pub const BLOCKS_N_LORA_Q: u32 = N_LORA_Q / 32;
pub const BLOCKS_N_FF_SHARED: u32 = N_FF_SHARED / 32;
pub const BLOCKS_Q8K_GATE_IN: u32 = N_EMBD / 256; // 16
pub const BLOCKS_Q8K_DOWN_IN: u32 = N_FF_EXP / 256; // 8

// === Numerical / sentinel constants ===

pub const RMS_EPS: f32 = 1.0e-6;
pub const SINKHORN_EPS: f32 = 1.0e-6;
pub const SINKHORN_ITERS: u32 = 20;
pub const SWIGLU_CLAMP_EXP: f32 = 10.0;
pub const EXPERT_WEIGHT_SCALE: f32 = 1.5;
pub const ROPE_ORIG_CTX: u64 = 65536;
pub const NEG_INF: f32 = -3.4028235e38;

// === Indexer (ratio=4 layers only) ===

pub const N_INDEXER_HEAD: u32 = 64;
pub const N_INDEXER_HEAD_DIM: u32 = 128;
pub const INDEXER_TOP_K: u32 = 512;

// === Attention / routing topology ===

/// SWA window: hard cap on `n_raw` in attention. Forward orchestrator
/// memmove-evicts beyond this.
pub const SWA_WINDOW: u32 = 128;

/// Per-layer compressor ratio. 0 = dense (no compression), N = compress
/// every N tokens into one comp row.
pub const COMPRESS_RATIOS: [u32; 43] = [
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
];

/// First N_HASH_LAYERS layers use the hash router (bootstrap). The rest
/// use the learned router.
pub const N_HASH_LAYERS: i32 = 3;
