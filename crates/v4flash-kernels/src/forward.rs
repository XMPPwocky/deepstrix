//! V4 Flash architecture constants + shared per-layer weight data types
//! and helpers used by the het orchestrator (`het/`).
//!
//! The legacy single-iGPU `Engine` / `ModelWeights` / `ModelState` /
//! `Scratch` runtime path was removed 2026-05-27 — see git history if
//! you need the original. The het path in `het/` is the only runtime
//! forward implementation now.

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::DeviceBuffer;

use crate::weights::DeviceWeight;

// === Architecture constants (V4 Flash) ===

pub const N_EMBD: u32 = 4096;
pub const N_HC: u32 = 4;
pub const HC_DIM: u32 = N_EMBD * N_HC; // 16384
pub const HC_MIX_DIM: u32 = 2 * N_HC + N_HC * N_HC; // 24
pub const N_HEAD: u32 = 64;
pub const N_HEAD_DIM: u32 = 512;
pub const N_ROT: u32 = 64;
pub const N_LORA_Q: u32 = 1024;
pub const Q_FLAT: u32 = N_HEAD * N_HEAD_DIM; // 32768
pub const N_GROUPS: u32 = 8;
pub const GROUP_DIM: u32 = 4096;
pub const RANK: u32 = 1024;
pub const OUT_LOW: u32 = N_GROUPS * RANK; // 8192
pub const N_FF_SHARED: u32 = 2048;
pub const N_FF_EXP: u32 = 2048;
pub const N_EXPERT: u32 = 256;
pub const N_EXPERT_USED: usize = 6;
pub const N_VOCAB: u32 = 129280;
pub const N_LAYER: i32 = 43;
pub const BLOCKS_N_EMBD: u32 = N_EMBD / 32;
pub const BLOCKS_OUT_LOW: u32 = OUT_LOW / 32;
pub const BLOCKS_GROUPED_OUT: u32 = (GROUP_DIM / 32) * N_GROUPS; // 1024
pub const BLOCKS_N_LORA_Q: u32 = N_LORA_Q / 32;
pub const BLOCKS_N_FF_SHARED: u32 = N_FF_SHARED / 32;
pub const BLOCKS_Q8K_GATE_IN: u32 = N_EMBD / 256; // 16
pub const BLOCKS_Q8K_DOWN_IN: u32 = N_FF_EXP / 256; // 8

pub const RMS_EPS: f32 = 1.0e-6;
pub const SINKHORN_EPS: f32 = 1.0e-6;
pub const SINKHORN_ITERS: u32 = 20;
pub const SWIGLU_CLAMP_EXP: f32 = 10.0;
pub const EXPERT_WEIGHT_SCALE: f32 = 1.5;
pub const ROPE_ORIG_CTX: u64 = 65536;
pub const NEG_INF: f32 = -3.4028235e38;
pub const N_INDEXER_HEAD: u32 = 64;
pub const N_INDEXER_HEAD_DIM: u32 = 128;
pub const INDEXER_TOP_K: u32 = 512;

/// SWA window: hard cap on `n_raw` in attention. Forward orchestrator
/// memmove-evicts beyond this.
pub const SWA_WINDOW: u32 = 128;

/// Per-layer compressor ratio. 0 = dense (no compression), N = compress
/// every N tokens into one comp row.
pub const COMPRESS_RATIOS: [u32; 43] = [
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
];

pub const N_HASH_LAYERS: i32 = 3;

// === Shared per-layer weight data types ===
//
// Used by the het loader in `het/weights.rs`. The single-iGPU
// `LayerWeights` / `GlobalWeights` / `ModelWeights` aggregates were
// retired with the legacy Engine; these per-component structs remain
// because the het split (`DgpuLayerWeights`, `IgpuLayerWeights`)
// reuses them.

pub struct CompressorWeights {
    pub wkv: DeviceWeight,    // F16 [n_embd, comp_width]
    pub wgate: DeviceWeight,  // F16 [n_embd, comp_width]
    pub ape: DeviceWeight,    // F16 [comp_width, ratio]
    pub norm: DeviceBuffer<f32>, // F32 [head_dim]
    pub width: u32,
    pub head_dim: u32,
}

/// IndexerWeights (only on ratio==4 layers). Indexer scoring path is
/// skipped for our short-prompt validation; we load `q_b` / `proj` only
/// when the indexer is actually wired.
pub struct IndexerWeights {
    pub q_b: DeviceWeight,
    pub proj: DeviceWeight,
}

pub struct SharedExpertWeights {
    pub gate: DeviceWeight,
    pub up: DeviceWeight,
    pub down: DeviceWeight,
}

/// Per-layer routed-expert weights, iGPU-resident. ~1.2 GiB/layer × 43
/// layers = ~52 GiB total — fits the 88 GiB budget enabled by
/// `amdgpu.no_system_mem_limit=1`. The per-expert byte slice is
/// addressed via pointer-offset on the kernel launch (see
/// `Iq2XxsPairMatvec::launch_with_offsets` and
/// `Q2KAccumulateMatvec::launch_with_offset`); no per-token host→device
/// upload (that was too slow — see feedback memory).
pub struct RoutedExpertWeights {
    pub gate: DeviceWeight,
    pub up: DeviceWeight,
    pub down: DeviceWeight,
    pub gate_bytes_per_expert: usize,
    pub up_bytes_per_expert: usize,
    pub down_bytes_per_expert: usize,
}

// === Weight loading helpers ===

pub fn load_f32_weight(
    gguf: &MappedGguf,
    name: &str,
    device_id: i32,
    expected_len: usize,
) -> eyre::Result<DeviceBuffer<f32>> {
    let t = gguf
        .gguf()
        .tensor(name)
        .ok_or_else(|| eyre!("tensor `{name}` missing"))?;
    if t.dtype != GgufType::F32 {
        return Err(eyre!("{name}: dtype {:?} != F32", t.dtype));
    }
    let bytes = gguf.read_tensor(t)?;
    if bytes.len() != expected_len * 4 {
        return Err(eyre!(
            "{name}: have {} bytes, expected {}",
            bytes.len(),
            expected_len * 4
        ));
    }
    let mut v = vec![0f32; expected_len];
    for (i, c) in bytes.chunks_exact(4).enumerate() {
        v[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    let mut buf: DeviceBuffer<f32> = DeviceBuffer::new(device_id, expected_len)?;
    buf.copy_from_host(&v)?;
    Ok(buf)
}

pub fn load_i32_tensor(gguf: &MappedGguf, name: &str) -> eyre::Result<Vec<i32>> {
    let t = gguf
        .gguf()
        .tensor(name)
        .ok_or_else(|| eyre!("tensor `{name}` missing"))?;
    if t.dtype != GgufType::I32 {
        return Err(eyre!("{name}: dtype {:?} != I32", t.dtype));
    }
    let bytes = gguf.read_tensor(t)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

// === Router helpers ===

fn softplus_stable(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0f32 + x.exp()).ln()
    }
}

/// Hash-router selection from `tid2eid[token_id * 6 + slot]`. Returns
/// `(selected[6], weights[6])`. Mirrors `layer_hash_selected_experts` +
/// `layer_hash_router_weights_one` (ds4.c:5209, 5260).
pub fn hash_router_select(
    tid2eid: &[i32],
    token_id: i32,
    logits_host: &[f32],
) -> ([i32; 6], [f32; 6]) {
    let mut selected = [0i32; N_EXPERT_USED];
    for i in 0..N_EXPERT_USED {
        selected[i] = tid2eid[(token_id as usize) * N_EXPERT_USED + i];
    }
    let mut w = [0f32; N_EXPERT_USED];
    let mut sum = 0f32;
    for i in 0..N_EXPERT_USED {
        let p = softplus_stable(logits_host[selected[i] as usize]).sqrt();
        w[i] = p;
        sum += p;
    }
    if sum < 6.103515625e-5 {
        sum = 6.103515625e-5;
    }
    for i in 0..N_EXPERT_USED {
        w[i] = w[i] / sum * EXPERT_WEIGHT_SCALE;
    }
    (selected, w)
}
