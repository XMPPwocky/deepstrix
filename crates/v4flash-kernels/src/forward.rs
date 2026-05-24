//! V4 Flash per-token forward orchestrator (M11.6).
//!
//! Composes every validated kernel + chain into a single token loop. The
//! engine owns:
//! - All HIP kernel modules
//! - Per-token scratch buffers (re-used across tokens)
//!
//! [`ModelWeights`] owns GPU-resident weights for everything *except* the
//! routed-expert IQ2_XXS / Q2_K weights (those stay host-mmap'd and are
//! sliced per-token by [`Engine::forward_token`] for the 6 selected
//! experts — V4 Flash's MoE selects 6 of 256 per token so streaming a few
//! MB per layer per token costs ~µs on UMA).
//!
//! [`ModelState`] owns the KV cache, compressor state, and comp-KV cache
//! per layer. Allocated once per inference session.
//!
//! Per-layer composition (mirrors `mhc_chain` + `attention_setup_chain` +
//! `attention_compute_chain` or `hca_chain` + `routed_moe` + `shared_expert`):
//!
//!   1. rms_norm_no_weight(residual_hc) → flat
//!   2. F16Matvec(hc_attn_fn, flat) → mix
//!   3. HcSinkhorn(mix, hc_attn_scale, hc_attn_base) → split = [w, post, comb]
//!   4. HcWeightedSum(residual_hc, w[0..4]) → attn_cur
//!   5. rms_norm_weighted(attn_cur, attn_norm) → attn_input_norm
//!   6. Q LoRA chain (Q8_0 q_a + RMS + Q8_0 q_b + head-RMS + RoPE) → q_post_rope
//!   7. KV chain (Q8_0 kv + RMS + RoPE) → kv_post_rope; append to KV cache
//!   8. (ratio>0) compressor step on attn_input_norm; emit comp_kv on boundary
//!   9. AttentionSwa (ratio==0) or AttentionMixed (ratio>0) → heads
//!  10. inverse RoPE + Q8_0Grouped + Q8_0 → attn_out
//!  11. HcPost(attn_out, residual_hc, post, comb) → after_attn_hc
//!  12. mHC pre-FFN (rms_nw + F16Matvec + Sinkhorn + HcWeightedSum) → ffn_cur
//!  13. rms_norm_weighted(ffn_cur, ffn_norm) → ffn_input_norm
//!  14. Router (hash/learned) → selected[6], weights[6]
//!  15. Routed-MoE pipeline → ffn_moe
//!  16. Shared-expert pipeline → ffn_shared
//!  17. ffn_moe += ffn_shared (vec_add)
//!  18. HcPost(ffn_moe, after_attn_hc, post, comb) → residual_hc (next layer)
//!
//! After L=42: head chain (rms_nw + F16Matvec output_hc_fn + HcSigmoidBias
//! + HcWeightedSum) + rms_norm_weighted(output_norm) + Q8_0(output) → logits.

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{Device, DeviceBuffer, Stream};

// Bisect helper: optional reference to the ActivationDump for in-engine
// dump-feed overrides. Set before forward_token via `set_debug_dump`.
static mut DS4_DEBUG_DUMP: Option<*const crate::oracle::ActivationDump> = None;

/// Set a debug ActivationDump pointer for bisect-driven overrides (test only).
/// SAFETY: caller must keep the dump alive across forward_token calls.
pub unsafe fn set_debug_dump(dump: Option<&crate::oracle::ActivationDump>) {
    unsafe { DS4_DEBUG_DUMP = dump.map(|d| d as *const _) };
}

use crate::compressor::{
    CompressorPool, CompressorStateShuffleR4, CompressorStateWrite, F16Roundtrip, Fp8E4m3fnQuantize,
};
use crate::f16::F16Matvec;
use crate::ffn::{Swiglu, SwigluClampWeighted, VecAddInplace};
use crate::head::{HcPost, HcSigmoidBias, HcSinkhorn, HcWeightedSum};
use crate::iq2_xxs::{Iq2XxsPairMatvec, BLOCK_IQ2_XXS_BYTES};
use crate::q2_k::{Q2KAccumulateMatvec, BLOCK_Q2_K_BYTES};
use crate::q8_0::{Q8_0GroupedMatvec, Q8_0Matvec};
use crate::q8_k::Q8KQuantize;
use crate::rms_norm::{RmsNorm, RmsNormNoWeight};
use crate::rope::{RopeParams, RopeTail};
use crate::attention::{AttentionMixed, AttentionSwa};
use crate::weights::{load_to_device, DeviceWeight};

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

/// Sliding-window attention size. Once a layer's raw KV cache reaches this
/// many rows, pushing a new row evicts the oldest (memmove-style slide).
/// Mirrors ds4's `DS4_N_SWA = 128`. Applies to all layers (dense + mixed).
pub const SWA_WINDOW: u32 = 128;

/// Compress ratios per layer (from GGUF metadata
/// `deepseek4.attention.compress_ratios`). Length 43.
pub const COMPRESS_RATIOS: [u32; 43] = [
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
];

/// First three layers use hash-gate router. Rest are learned.
pub const N_HASH_LAYERS: i32 = 3;

// === Weights ===

pub struct CompressorWeights {
    pub wkv: DeviceWeight,    // F16 [n_embd, comp_width]
    pub wgate: DeviceWeight,  // F16 [n_embd, comp_width]
    pub ape: DeviceWeight,    // F16 [comp_width, ratio]
    pub norm: DeviceBuffer<f32>, // F32 [head_dim]
    pub width: u32,
    pub head_dim: u32,
}

/// IndexerWeights (only on ratio==4 layers). Indexer scoring path is
/// skipped for our short-prompt validation (n_comp ≤ 14 < top_k=512), so
/// we don't yet load `indexer.attn_q_b` / `indexer.proj`; the indexer
/// compressor itself IS loaded via [`CompressorWeights`] for state
/// consistency.
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

pub struct LayerWeights {
    pub layer_idx: i32,
    pub ratio: u32,
    pub is_hash_router: bool,

    // mHC
    pub hc_attn_fn: DeviceWeight,
    pub hc_attn_scale: DeviceBuffer<f32>,
    pub hc_attn_base: DeviceBuffer<f32>,
    pub hc_ffn_fn: DeviceWeight,
    pub hc_ffn_scale: DeviceBuffer<f32>,
    pub hc_ffn_base: DeviceBuffer<f32>,

    // Attention
    pub attn_norm: DeviceBuffer<f32>,
    pub attn_q_a: DeviceWeight,
    pub attn_q_b: DeviceWeight,
    pub q_a_norm: DeviceBuffer<f32>,
    pub attn_kv: DeviceWeight,
    pub kv_a_norm: DeviceBuffer<f32>,
    pub attn_sinks: DeviceBuffer<f32>,
    pub attn_output_a: DeviceWeight,
    pub attn_output_b: DeviceWeight,
    pub rope_params: RopeParams,

    // CSA producers
    pub compressor: Option<CompressorWeights>,
    pub indexer_compressor: Option<CompressorWeights>,
    pub indexer: Option<IndexerWeights>,

    // FFN
    pub ffn_norm: DeviceBuffer<f32>,
    pub ffn_gate_inp: DeviceWeight,
    pub tid2eid: Option<Vec<i32>>,
    pub router_bias: Option<Vec<f32>>,
    pub shared: SharedExpertWeights,
    pub routed: RoutedExpertWeights,
}

/// Globally-resident weights (output head + embedding). ~1.6 GB on GPU.
pub struct GlobalWeights {
    pub token_embd: DeviceWeight,
    pub output: DeviceWeight,
    pub output_norm: DeviceBuffer<f32>,
    pub output_hc_fn: DeviceWeight,
    pub output_hc_scale: DeviceBuffer<f32>,
    pub output_hc_base: DeviceBuffer<f32>,
}

/// All-resident model weights: globals + per-layer non-routed weights.
/// Routed-expert weights (~77 GiB) remain host-mmap'd in the GGUF and
/// are streamed per-token by `Engine::forward_layer` for the 6 selected
/// experts.
///
/// Pre-load all layers up front (now feasible after the
/// `amdgpu.no_system_mem_limit=1` unlock — see
/// `project_strix_memory_budget.md`). Total resident: ~9 GiB layer
/// weights + ~1.6 GiB globals.
pub struct ModelWeights {
    pub global: GlobalWeights,
    pub layers: Vec<LayerWeights>,
}

impl ModelWeights {
    pub fn load_all(
        gguf: &MappedGguf,
        device_id: i32,
        rope_params_for_layer: &dyn Fn(i32) -> eyre::Result<RopeParams>,
    ) -> eyre::Result<Self> {
        let global = GlobalWeights::load(gguf, device_id)?;
        let mut layers = Vec::with_capacity(N_LAYER as usize);
        for layer in 0..N_LAYER {
            layers.push(LayerWeights::load(gguf, device_id, layer, rope_params_for_layer)?);
        }
        Ok(Self { global, layers })
    }
}

// === State ===

pub struct CompressorState {
    pub state_kv: DeviceBuffer<f32>,
    pub state_score: DeviceBuffer<f32>,
    pub comp_kv: DeviceBuffer<f32>,
    pub comp_kv_host: Vec<f32>,
    pub n_comp: u32,
    pub width: u32,
    pub head_dim: u32,
}

pub struct LayerState {
    pub kv_cache: DeviceBuffer<f32>,
    pub kv_cache_host: Vec<f32>,
    pub n_raw: u32,
    pub compressor: Option<CompressorState>,
    pub indexer_compressor: Option<CompressorState>,
}

pub struct ModelState {
    pub layers: Vec<LayerState>,
    pub n_kv_max: u32,
}

// === Scratch ===

pub struct Scratch {
    // Cross-layer residual (the layer_input/output_residual signal)
    pub residual: DeviceBuffer<f32>, // [hc_dim]
    pub residual_next: DeviceBuffer<f32>,

    // mHC stage
    pub flat: DeviceBuffer<f32>,
    pub mix: DeviceBuffer<f32>,
    pub split: DeviceBuffer<f32>,
    pub post_attn: DeviceBuffer<f32>,
    pub comb_attn: DeviceBuffer<f32>,
    pub post_ffn: DeviceBuffer<f32>,
    pub comb_ffn: DeviceBuffer<f32>,
    pub attn_cur: DeviceBuffer<f32>,
    pub attn_input_norm: DeviceBuffer<f32>,
    pub after_attn_hc: DeviceBuffer<f32>,
    pub ffn_cur: DeviceBuffer<f32>,
    pub ffn_input_norm: DeviceBuffer<f32>,

    // Attention setup
    pub xq_n_embd: DeviceBuffer<i8>,
    pub xscale_n_embd: DeviceBuffer<f32>,
    pub qr: DeviceBuffer<f32>,
    pub qr_normed: DeviceBuffer<f32>,
    pub qr_xq: DeviceBuffer<i8>,
    pub qr_xscale: DeviceBuffer<f32>,
    pub q: DeviceBuffer<f32>,
    pub q_normed: DeviceBuffer<f32>,
    pub kv_raw: DeviceBuffer<f32>,
    pub kv_normed: DeviceBuffer<f32>,

    // Attention compute
    pub heads: DeviceBuffer<f32>,
    pub low: DeviceBuffer<f32>,
    pub heads_xq: DeviceBuffer<i8>,
    pub heads_xscale: DeviceBuffer<f32>,
    pub low_xq: DeviceBuffer<i8>,
    pub low_xscale: DeviceBuffer<f32>,
    pub attn_out: DeviceBuffer<f32>,

    // Compressor scratch
    pub kv_cur: DeviceBuffer<f32>,
    pub sc_cur: DeviceBuffer<f32>,
    pub kv_cur_idx: DeviceBuffer<f32>,
    pub sc_cur_idx: DeviceBuffer<f32>,
    pub pooled: DeviceBuffer<f32>,
    pub comp_row: DeviceBuffer<f32>,

    // FFN router/MoE
    pub router_logits_host: Vec<f32>,
    pub router_logits: DeviceBuffer<f32>,
    pub d_xq_q8k: DeviceBuffer<u8>,    // [16 blocks]
    pub d_midq_q8k: DeviceBuffer<u8>,  // [8 blocks]
    pub d_gate_e: DeviceBuffer<f32>,   // [N_FF_EXP]
    pub d_up_e: DeviceBuffer<f32>,
    pub d_gate_cat: DeviceBuffer<f32>, // [6 * N_FF_EXP]
    pub d_up_cat: DeviceBuffer<f32>,
    pub d_mid_cat: DeviceBuffer<f32>,
    pub d_mid_e: DeviceBuffer<f32>,
    pub d_ew: DeviceBuffer<f32>,       // [6]
    pub host_gate_cat: Vec<f32>,
    pub host_up_cat: Vec<f32>,
    pub host_mid_cat: Vec<f32>,
    pub ffn_moe: DeviceBuffer<f32>,

    // Per-slot per-layer expert weight buffers (re-used). Sized for max
    // per-expert weight per dtype (computed once at engine init).
    pub d_gate_slot: Vec<DeviceBuffer<u8>>,
    pub d_up_slot: Vec<DeviceBuffer<u8>>,
    pub d_down_slot: Vec<DeviceBuffer<u8>>,

    // Shared expert
    pub gate_sh: DeviceBuffer<f32>,
    pub up_sh: DeviceBuffer<f32>,
    pub mid_sh: DeviceBuffer<f32>,
    pub mid_sh_xq: DeviceBuffer<i8>,
    pub mid_sh_xscale: DeviceBuffer<f32>,
    pub ffn_shared: DeviceBuffer<f32>,

    // Head
    pub head_flat: DeviceBuffer<f32>,
    pub head_pre: DeviceBuffer<f32>,
    pub head_w: DeviceBuffer<f32>,
    pub head_embd: DeviceBuffer<f32>,
    pub head_norm: DeviceBuffer<f32>,
    pub head_xq: DeviceBuffer<i8>,
    pub head_xscale: DeviceBuffer<f32>,
    pub logits: DeviceBuffer<f32>,
}

// === Engine ===

pub struct Engine {
    pub device: Device,
    pub stream: Stream,
    pub rms_w: RmsNorm,
    pub rms_nw: RmsNormNoWeight,
    pub q8: Q8_0Matvec,
    pub q8_grouped: Q8_0GroupedMatvec,
    pub f16: F16Matvec,
    pub rope: RopeTail,
    pub attn_swa: AttentionSwa,
    pub attn_mixed: AttentionMixed,
    pub swiglu: Swiglu,
    pub swiglu_cw: SwigluClampWeighted,
    pub vec_add: VecAddInplace,
    pub q8k: Q8KQuantize,
    pub iq2: Iq2XxsPairMatvec,
    pub q2k: Q2KAccumulateMatvec,
    pub hc_sigmoid: HcSigmoidBias,
    pub hc_weighted: HcWeightedSum,
    pub hc_sinkhorn: HcSinkhorn,
    pub hc_post: HcPost,
    pub compressor_pool: CompressorPool,
    pub compressor_state_write: CompressorStateWrite,
    pub compressor_shuffle: CompressorStateShuffleR4,
    pub fp8: Fp8E4m3fnQuantize,
    pub f16rt: F16Roundtrip,
}

impl Engine {
    pub fn for_arch(device: Device, arch: &str) -> eyre::Result<Self> {
        let stream = Stream::new(device.id)?;
        Ok(Self {
            device,
            stream,
            rms_w: RmsNorm::for_arch(arch)?,
            rms_nw: RmsNormNoWeight::for_arch(arch)?,
            q8: Q8_0Matvec::for_arch(arch)?,
            q8_grouped: Q8_0GroupedMatvec::for_arch(arch)?,
            f16: F16Matvec::for_arch(arch)?,
            rope: RopeTail::for_arch(arch)?,
            attn_swa: AttentionSwa::for_arch(arch)?,
            attn_mixed: AttentionMixed::for_arch(arch)?,
            swiglu: Swiglu::for_arch(arch)?,
            swiglu_cw: SwigluClampWeighted::for_arch(arch)?,
            vec_add: VecAddInplace::for_arch(arch)?,
            q8k: Q8KQuantize::for_arch(arch)?,
            iq2: Iq2XxsPairMatvec::for_arch(arch)?,
            q2k: Q2KAccumulateMatvec::for_arch(arch)?,
            hc_sigmoid: HcSigmoidBias::for_arch(arch)?,
            hc_weighted: HcWeightedSum::for_arch(arch)?,
            hc_sinkhorn: HcSinkhorn::for_arch(arch)?,
            hc_post: HcPost::for_arch(arch)?,
            compressor_pool: CompressorPool::for_arch(arch)?,
            compressor_state_write: CompressorStateWrite::for_arch(arch)?,
            compressor_shuffle: CompressorStateShuffleR4::for_arch(arch)?,
            fp8: Fp8E4m3fnQuantize::for_arch(arch)?,
            f16rt: F16Roundtrip::for_arch(arch)?,
        })
    }
}

impl Scratch {
    pub fn alloc(device_id: i32) -> eyre::Result<Self> {
        let gate_bytes_per_expert =
            (N_FF_EXP as usize) * (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_IQ2_XXS_BYTES;
        let down_bytes_per_expert =
            (N_EMBD as usize) * (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q2_K_BYTES;
        let d_gate_slot: Vec<DeviceBuffer<u8>> = (0..N_EXPERT_USED)
            .map(|_| DeviceBuffer::new(device_id, gate_bytes_per_expert))
            .collect::<Result<_, _>>()?;
        let d_up_slot: Vec<DeviceBuffer<u8>> = (0..N_EXPERT_USED)
            .map(|_| DeviceBuffer::new(device_id, gate_bytes_per_expert))
            .collect::<Result<_, _>>()?;
        let d_down_slot: Vec<DeviceBuffer<u8>> = (0..N_EXPERT_USED)
            .map(|_| DeviceBuffer::new(device_id, down_bytes_per_expert))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            residual: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            residual_next: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            flat: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            mix: DeviceBuffer::new(device_id, HC_MIX_DIM as usize)?,
            split: DeviceBuffer::new(device_id, HC_MIX_DIM as usize)?,
            post_attn: DeviceBuffer::new(device_id, N_HC as usize)?,
            comb_attn: DeviceBuffer::new(device_id, (N_HC * N_HC) as usize)?,
            post_ffn: DeviceBuffer::new(device_id, N_HC as usize)?,
            comb_ffn: DeviceBuffer::new(device_id, (N_HC * N_HC) as usize)?,
            attn_cur: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            attn_input_norm: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            after_attn_hc: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            ffn_cur: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            ffn_input_norm: DeviceBuffer::new(device_id, N_EMBD as usize)?,

            xq_n_embd: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            xscale_n_embd: DeviceBuffer::new(device_id, BLOCKS_N_EMBD as usize)?,
            qr: DeviceBuffer::new(device_id, N_LORA_Q as usize)?,
            qr_normed: DeviceBuffer::new(device_id, N_LORA_Q as usize)?,
            qr_xq: DeviceBuffer::new(device_id, N_LORA_Q as usize)?,
            qr_xscale: DeviceBuffer::new(device_id, BLOCKS_N_LORA_Q as usize)?,
            q: DeviceBuffer::new(device_id, Q_FLAT as usize)?,
            q_normed: DeviceBuffer::new(device_id, Q_FLAT as usize)?,
            kv_raw: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,
            kv_normed: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,

            heads: DeviceBuffer::new(device_id, Q_FLAT as usize)?,
            low: DeviceBuffer::new(device_id, OUT_LOW as usize)?,
            heads_xq: DeviceBuffer::new(device_id, Q_FLAT as usize)?,
            heads_xscale: DeviceBuffer::new(device_id, BLOCKS_GROUPED_OUT as usize)?,
            low_xq: DeviceBuffer::new(device_id, OUT_LOW as usize)?,
            low_xscale: DeviceBuffer::new(device_id, BLOCKS_OUT_LOW as usize)?,
            attn_out: DeviceBuffer::new(device_id, N_EMBD as usize)?,

            // Main compressor (ratio==4: width=1024; ratio==128: width=512).
            // Allocate for the larger case so it covers both.
            kv_cur: DeviceBuffer::new(device_id, (2 * N_HEAD_DIM) as usize)?,
            sc_cur: DeviceBuffer::new(device_id, (2 * N_HEAD_DIM) as usize)?,
            // Indexer compressor (head_dim=128, width=256 for ratio==4).
            kv_cur_idx: DeviceBuffer::new(device_id, (2 * N_INDEXER_HEAD_DIM) as usize)?,
            sc_cur_idx: DeviceBuffer::new(device_id, (2 * N_INDEXER_HEAD_DIM) as usize)?,
            pooled: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,
            comp_row: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,

            router_logits_host: vec![0f32; N_EXPERT as usize],
            router_logits: DeviceBuffer::new(device_id, N_EXPERT as usize)?,
            d_xq_q8k: DeviceBuffer::new(
                device_id,
                (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_Q8K_BYTES,
            )?,
            d_midq_q8k: DeviceBuffer::new(
                device_id,
                (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8K_BYTES,
            )?,
            d_gate_e: DeviceBuffer::new(device_id, N_FF_EXP as usize)?,
            d_up_e: DeviceBuffer::new(device_id, N_FF_EXP as usize)?,
            d_gate_cat: DeviceBuffer::new(device_id, N_EXPERT_USED * (N_FF_EXP as usize))?,
            d_up_cat: DeviceBuffer::new(device_id, N_EXPERT_USED * (N_FF_EXP as usize))?,
            d_mid_cat: DeviceBuffer::new(device_id, N_EXPERT_USED * (N_FF_EXP as usize))?,
            d_mid_e: DeviceBuffer::new(device_id, N_FF_EXP as usize)?,
            d_ew: DeviceBuffer::new(device_id, N_EXPERT_USED)?,
            host_gate_cat: vec![0f32; N_EXPERT_USED * (N_FF_EXP as usize)],
            host_up_cat: vec![0f32; N_EXPERT_USED * (N_FF_EXP as usize)],
            host_mid_cat: vec![0f32; N_EXPERT_USED * (N_FF_EXP as usize)],
            ffn_moe: DeviceBuffer::new(device_id, N_EMBD as usize)?,

            d_gate_slot,
            d_up_slot,
            d_down_slot,

            gate_sh: DeviceBuffer::new(device_id, N_FF_SHARED as usize)?,
            up_sh: DeviceBuffer::new(device_id, N_FF_SHARED as usize)?,
            mid_sh: DeviceBuffer::new(device_id, N_FF_SHARED as usize)?,
            mid_sh_xq: DeviceBuffer::new(device_id, N_FF_SHARED as usize)?,
            mid_sh_xscale: DeviceBuffer::new(device_id, BLOCKS_N_FF_SHARED as usize)?,
            ffn_shared: DeviceBuffer::new(device_id, N_EMBD as usize)?,

            head_flat: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            head_pre: DeviceBuffer::new(device_id, N_HC as usize)?,
            head_w: DeviceBuffer::new(device_id, N_HC as usize)?,
            head_embd: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            head_norm: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            head_xq: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            head_xscale: DeviceBuffer::new(device_id, BLOCKS_N_EMBD as usize)?,
            logits: DeviceBuffer::new(device_id, N_VOCAB as usize)?,
        })
    }
}

impl ModelState {
    pub fn alloc(device_id: i32, n_kv_max: u32) -> eyre::Result<Self> {
        let mut layers = Vec::with_capacity(N_LAYER as usize);
        for layer in 0..N_LAYER {
            let ratio = COMPRESS_RATIOS[layer as usize];
            let compressor = if ratio > 0 {
                Some(CompressorState::alloc(device_id, ratio, N_HEAD_DIM, n_kv_max)?)
            } else {
                None
            };
            let indexer_compressor = if ratio == 4 {
                Some(CompressorState::alloc(
                    device_id,
                    ratio,
                    N_INDEXER_HEAD_DIM,
                    n_kv_max,
                )?)
            } else {
                None
            };
            layers.push(LayerState {
                kv_cache: DeviceBuffer::new(
                    device_id,
                    (n_kv_max as usize) * (N_HEAD_DIM as usize),
                )?,
                kv_cache_host: vec![0f32; (n_kv_max as usize) * (N_HEAD_DIM as usize)],
                n_raw: 0,
                compressor,
                indexer_compressor,
            });
        }
        Ok(Self { layers, n_kv_max })
    }
}

impl CompressorState {
    pub fn alloc(
        device_id: i32,
        ratio: u32,
        head_dim: u32,
        n_kv_max: u32,
    ) -> eyre::Result<Self> {
        let coff = if ratio == 4 { 2 } else { 1 };
        let width = coff * head_dim;
        let state_rows = ratio * coff;
        let n_state = (state_rows as usize) * (width as usize);
        let zeros = vec![0f32; n_state];
        let neg_inf = vec![NEG_INF; n_state];
        let mut state_kv: DeviceBuffer<f32> = DeviceBuffer::new(device_id, n_state)?;
        let mut state_score: DeviceBuffer<f32> = DeviceBuffer::new(device_id, n_state)?;
        state_kv.copy_from_host(&zeros)?;
        state_score.copy_from_host(&neg_inf)?;
        let max_n_comp = (n_kv_max + ratio - 1) / ratio;
        let comp_kv_capacity = (max_n_comp as usize) * (head_dim as usize);
        let comp_kv: DeviceBuffer<f32> = DeviceBuffer::new(device_id, comp_kv_capacity)?;
        Ok(Self {
            state_kv,
            state_score,
            comp_kv,
            comp_kv_host: vec![0f32; comp_kv_capacity],
            n_comp: 0,
            width,
            head_dim,
        })
    }
}

const BLOCK_Q8K_BYTES: usize = crate::q8_k::BLOCK_Q8_K_BYTES;

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

pub fn parse_rope_params(rope_params_blob: &[f32]) -> eyre::Result<RopeParams> {
    let n_ctx_orig = if rope_params_blob[2] != 0.0 {
        ROPE_ORIG_CTX
    } else {
        0
    };
    RopeParams::from_dump_blob(rope_params_blob, n_ctx_orig)
}

// === Weight loading ===

impl GlobalWeights {
    pub fn load(gguf: &MappedGguf, device_id: i32) -> eyre::Result<Self> {
        let token_embd = load_to_device(gguf, "token_embd.weight", device_id)?;
        if token_embd.dtype != GgufType::F16 {
            return Err(eyre!("token_embd dtype {:?} != F16", token_embd.dtype));
        }
        let output = load_to_device(gguf, "output.weight", device_id)?;
        if output.dtype != GgufType::Q8_0 {
            return Err(eyre!("output dtype {:?} != Q8_0", output.dtype));
        }
        let output_norm = load_f32_weight(gguf, "output_norm.weight", device_id, N_EMBD as usize)?;
        let output_hc_fn = load_to_device(gguf, "output_hc_fn.weight", device_id)?;
        let output_hc_scale = load_f32_weight(gguf, "output_hc_scale.weight", device_id, 1)?;
        let output_hc_base = load_f32_weight(gguf, "output_hc_base.weight", device_id, N_HC as usize)?;
        Ok(Self {
            token_embd,
            output,
            output_norm,
            output_hc_fn,
            output_hc_scale,
            output_hc_base,
        })
    }
}

impl LayerWeights {
    /// Load one layer's weights. Designed for streaming: allocate, use,
    /// drop. On Strix Halo with the current UMA carveout, the practical
    /// HIP allocation cap is ~4-5 GB; pre-loading 43 layers (~8 GB) ooms,
    /// so the orchestrator streams a single layer at a time.
    pub fn load(
        gguf: &MappedGguf,
        device_id: i32,
        layer: i32,
        rope_params_for_layer: &dyn Fn(i32) -> eyre::Result<RopeParams>,
    ) -> eyre::Result<Self> {
            let ratio = COMPRESS_RATIOS[layer as usize];
            let is_hash_router = layer < N_HASH_LAYERS;

            // mHC weights.
            let hc_attn_fn = load_to_device(
                gguf,
                &format!("blk.{layer}.hc_attn_fn.weight"),
                device_id,
            )?;
            let hc_attn_scale = load_f32_weight(
                gguf,
                &format!("blk.{layer}.hc_attn_scale.weight"),
                device_id,
                3,
            )?;
            let hc_attn_base = load_f32_weight(
                gguf,
                &format!("blk.{layer}.hc_attn_base.weight"),
                device_id,
                HC_MIX_DIM as usize,
            )?;
            let hc_ffn_fn = load_to_device(
                gguf,
                &format!("blk.{layer}.hc_ffn_fn.weight"),
                device_id,
            )?;
            let hc_ffn_scale = load_f32_weight(
                gguf,
                &format!("blk.{layer}.hc_ffn_scale.weight"),
                device_id,
                3,
            )?;
            let hc_ffn_base = load_f32_weight(
                gguf,
                &format!("blk.{layer}.hc_ffn_base.weight"),
                device_id,
                HC_MIX_DIM as usize,
            )?;

            // Attention.
            let attn_norm = load_f32_weight(
                gguf,
                &format!("blk.{layer}.attn_norm.weight"),
                device_id,
                N_EMBD as usize,
            )?;
            let attn_q_a = load_to_device(
                gguf,
                &format!("blk.{layer}.attn_q_a.weight"),
                device_id,
            )?;
            let attn_q_b = load_to_device(
                gguf,
                &format!("blk.{layer}.attn_q_b.weight"),
                device_id,
            )?;
            let q_a_norm = load_f32_weight(
                gguf,
                &format!("blk.{layer}.attn_q_a_norm.weight"),
                device_id,
                N_LORA_Q as usize,
            )?;
            let attn_kv = load_to_device(
                gguf,
                &format!("blk.{layer}.attn_kv.weight"),
                device_id,
            )?;
            let kv_a_norm = load_f32_weight(
                gguf,
                &format!("blk.{layer}.attn_kv_a_norm.weight"),
                device_id,
                N_HEAD_DIM as usize,
            )?;
            let attn_sinks = load_f32_weight(
                gguf,
                &format!("blk.{layer}.attn_sinks.weight"),
                device_id,
                N_HEAD as usize,
            )?;
            let attn_output_a = load_to_device(
                gguf,
                &format!("blk.{layer}.attn_output_a.weight"),
                device_id,
            )?;
            let attn_output_b = load_to_device(
                gguf,
                &format!("blk.{layer}.attn_output_b.weight"),
                device_id,
            )?;
            // RoPE params: caller provides (from dump in tests, from GGUF
            // metadata in production).
            let rope_params = rope_params_for_layer(layer)?;

            // CSA producers.
            let compressor = if ratio > 0 {
                let comp_width = if ratio == 4 { 1024 } else { 512 };
                Some(CompressorWeights {
                    wkv: load_to_device(
                        gguf,
                        &format!("blk.{layer}.attn_compressor_kv.weight"),
                        device_id,
                    )?,
                    wgate: load_to_device(
                        gguf,
                        &format!("blk.{layer}.attn_compressor_gate.weight"),
                        device_id,
                    )?,
                    ape: load_to_device(
                        gguf,
                        &format!("blk.{layer}.attn_compressor_ape.weight"),
                        device_id,
                    )?,
                    norm: load_f32_weight(
                        gguf,
                        &format!("blk.{layer}.attn_compressor_norm.weight"),
                        device_id,
                        N_HEAD_DIM as usize,
                    )?,
                    width: comp_width,
                    head_dim: N_HEAD_DIM,
                })
            } else {
                None
            };
            let indexer_compressor = if ratio == 4 {
                Some(CompressorWeights {
                    wkv: load_to_device(
                        gguf,
                        &format!("blk.{layer}.indexer_compressor_kv.weight"),
                        device_id,
                    )?,
                    wgate: load_to_device(
                        gguf,
                        &format!("blk.{layer}.indexer_compressor_gate.weight"),
                        device_id,
                    )?,
                    ape: load_to_device(
                        gguf,
                        &format!("blk.{layer}.indexer_compressor_ape.weight"),
                        device_id,
                    )?,
                    norm: load_f32_weight(
                        gguf,
                        &format!("blk.{layer}.indexer_compressor_norm.weight"),
                        device_id,
                        N_INDEXER_HEAD_DIM as usize,
                    )?,
                    width: 256,
                    head_dim: N_INDEXER_HEAD_DIM,
                })
            } else {
                None
            };
            let indexer = if ratio == 4 {
                Some(IndexerWeights {
                    q_b: load_to_device(
                        gguf,
                        &format!("blk.{layer}.indexer.attn_q_b.weight"),
                        device_id,
                    )?,
                    proj: load_to_device(
                        gguf,
                        &format!("blk.{layer}.indexer.proj.weight"),
                        device_id,
                    )?,
                })
            } else {
                None
            };

            // FFN.
            let ffn_norm = load_f32_weight(
                gguf,
                &format!("blk.{layer}.ffn_norm.weight"),
                device_id,
                N_EMBD as usize,
            )?;
            let ffn_gate_inp = load_to_device(
                gguf,
                &format!("blk.{layer}.ffn_gate_inp.weight"),
                device_id,
            )?;
            let tid2eid = if is_hash_router {
                Some(load_i32_tensor(
                    gguf,
                    &format!("blk.{layer}.ffn_gate_tid2eid.weight"),
                )?)
            } else {
                None
            };
            let router_bias = if !is_hash_router {
                let bias_name = format!("blk.{layer}.exp_probs_b.bias");
                if let Some(t) = gguf.gguf().tensor(&bias_name) {
                    if t.dtype != GgufType::F32 {
                        return Err(eyre!("{bias_name} dtype {:?} != F32", t.dtype));
                    }
                    let bytes = gguf.read_tensor(t)?;
                    Some(
                        bytes
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect(),
                    )
                } else {
                    None
                }
            } else {
                None
            };
            let shared = SharedExpertWeights {
                gate: load_to_device(
                    gguf,
                    &format!("blk.{layer}.ffn_gate_shexp.weight"),
                    device_id,
                )?,
                up: load_to_device(
                    gguf,
                    &format!("blk.{layer}.ffn_up_shexp.weight"),
                    device_id,
                )?,
                down: load_to_device(
                    gguf,
                    &format!("blk.{layer}.ffn_down_shexp.weight"),
                    device_id,
                )?,
            };

            // Routed expert weights — fully iGPU-resident (~1.2 GiB/layer).
            let gate = load_to_device(
                gguf,
                &format!("blk.{layer}.ffn_gate_exps.weight"),
                device_id,
            )?;
            let up = load_to_device(
                gguf,
                &format!("blk.{layer}.ffn_up_exps.weight"),
                device_id,
            )?;
            let down = load_to_device(
                gguf,
                &format!("blk.{layer}.ffn_down_exps.weight"),
                device_id,
            )?;
            let gate_bytes_per_expert =
                (N_FF_EXP as usize) * (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_IQ2_XXS_BYTES;
            let up_bytes_per_expert = gate_bytes_per_expert;
            let down_bytes_per_expert =
                (N_EMBD as usize) * (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q2_K_BYTES;
            let routed = RoutedExpertWeights {
                gate,
                up,
                down,
                gate_bytes_per_expert,
                up_bytes_per_expert,
                down_bytes_per_expert,
            };

            Ok(LayerWeights {
                layer_idx: layer,
                ratio,
                is_hash_router,
                hc_attn_fn,
                hc_attn_scale,
                hc_attn_base,
                hc_ffn_fn,
                hc_ffn_scale,
                hc_ffn_base,
                attn_norm,
                attn_q_a,
                attn_q_b,
                q_a_norm,
                attn_kv,
                kv_a_norm,
                attn_sinks,
                attn_output_a,
                attn_output_b,
                rope_params,
                compressor,
                indexer_compressor,
                indexer,
                ffn_norm,
                ffn_gate_inp,
                tid2eid,
                router_bias,
                shared,
                routed,
            })
    }
}

// === Forward pass ===

impl Engine {
    /// Run one full forward pass for token at position `pos`. Input is the
    /// layer-0 residual stream (`layer_input_residual[L=0]`, size `HC_DIM`),
    /// usually computed via `embedding_lookup(token_id) replicated n_hc times`.
    /// On return, `scratch.logits` holds the V4-Flash output logits
    /// (size `N_VOCAB`).
    ///
    /// Updates KV cache + compressor state in `state` across the call.
    pub fn forward_token(
        &self,
        scratch: &mut Scratch,
        state: &mut ModelState,
        weights: &ModelWeights,
        gguf: &MappedGguf,
        input_hc_host: &[f32],
        pos: u32,
        token_id: i32,
    ) -> eyre::Result<()> {
        if input_hc_host.len() != HC_DIM as usize {
            return Err(eyre!(
                "input_hc_host len {} != HC_DIM {}",
                input_hc_host.len(),
                HC_DIM
            ));
        }
        scratch.residual.copy_from_host(input_hc_host)?;

        for layer in 0..N_LAYER as usize {
            let lw = &weights.layers[layer];
            self.forward_layer(scratch, state, lw, gguf, layer, pos, token_id)?;
            std::mem::swap(&mut scratch.residual, &mut scratch.residual_next);
        }

        self.forward_head(scratch, &weights.global)?;
        self.stream.synchronize()?;
        Ok(())
    }

    pub fn forward_layer(
        &self,
        scratch: &mut Scratch,
        state: &mut ModelState,
        lw: &LayerWeights,
        gguf: &MappedGguf,
        layer: usize,
        pos: u32,
        token_id: i32,
    ) -> eyre::Result<()> {
        let _ = layer; // lw.layer_idx is canonical
        let ls = &mut state.layers[layer];
        let ratio = lw.ratio;

        // === (1) mHC pre attn → attn_cur ===
        self.rms_nw
            .launch(&self.stream, &mut scratch.flat, &scratch.residual, 1, HC_DIM, RMS_EPS)?;
        self.f16.matvec(
            &self.stream,
            &mut scratch.mix,
            &lw.hc_attn_fn.buffer,
            &scratch.flat,
            HC_MIX_DIM,
            HC_DIM,
        )?;
        self.hc_sinkhorn.launch(
            &self.stream,
            &mut scratch.split,
            &scratch.mix,
            &lw.hc_attn_scale,
            &lw.hc_attn_base,
            N_HC,
            SINKHORN_ITERS,
            SINKHORN_EPS,
        )?;
        self.hc_weighted.launch(
            &self.stream,
            &mut scratch.attn_cur,
            &scratch.residual,
            &scratch.split,
            N_EMBD,
            N_HC,
        )?;

        // Extract post/comb from split into separate buffers (~80 f32 host-roundtrip).
        // The mhc_chain test does this; not a perf hotspot.
        let mut split_host = vec![0f32; HC_MIX_DIM as usize];
        self.stream.synchronize()?;
        scratch.split.copy_to_host(&mut split_host)?;
        let post_only = &split_host[N_HC as usize..2 * N_HC as usize];
        let comb_only = &split_host
            [2 * N_HC as usize..2 * N_HC as usize + (N_HC * N_HC) as usize];
        scratch.post_attn.copy_from_host(post_only)?;
        scratch.comb_attn.copy_from_host(comb_only)?;

        // === (2) attn_input_norm ===
        self.rms_w.launch_weighted(
            &self.stream,
            &mut scratch.attn_input_norm,
            &scratch.attn_cur,
            &lw.attn_norm,
            N_EMBD,
            RMS_EPS,
        )?;

        // === (3) Q LoRA chain → q_post_rope ===
        self.q8.quantize_input(
            &self.stream,
            &mut scratch.xq_n_embd,
            &mut scratch.xscale_n_embd,
            &scratch.attn_input_norm,
            N_EMBD,
        )?;
        self.q8.matvec(
            &self.stream,
            &mut scratch.qr,
            &lw.attn_q_a.buffer,
            &scratch.xq_n_embd,
            &scratch.xscale_n_embd,
            N_LORA_Q,
            N_EMBD,
        )?;
        self.rms_w.launch_weighted(
            &self.stream,
            &mut scratch.qr_normed,
            &scratch.qr,
            &lw.q_a_norm,
            N_LORA_Q,
            RMS_EPS,
        )?;
        self.q8.quantize_input(
            &self.stream,
            &mut scratch.qr_xq,
            &mut scratch.qr_xscale,
            &scratch.qr_normed,
            N_LORA_Q,
        )?;
        self.q8.matvec(
            &self.stream,
            &mut scratch.q,
            &lw.attn_q_b.buffer,
            &scratch.qr_xq,
            &scratch.qr_xscale,
            Q_FLAT,
            N_LORA_Q,
        )?;
        self.rms_nw.launch(
            &self.stream,
            &mut scratch.q_normed,
            &scratch.q,
            N_HEAD,
            N_HEAD_DIM,
            RMS_EPS,
        )?;
        self.rope.launch_forward(
            &self.stream,
            &mut scratch.q_normed,
            N_HEAD,
            N_HEAD_DIM,
            N_ROT,
            pos,
            &lw.rope_params,
        )?;

        // === (4) KV chain → kv_post_rope ===
        self.q8.matvec(
            &self.stream,
            &mut scratch.kv_raw,
            &lw.attn_kv.buffer,
            &scratch.xq_n_embd,
            &scratch.xscale_n_embd,
            N_HEAD_DIM,
            N_EMBD,
        )?;
        self.rms_w.launch_weighted(
            &self.stream,
            &mut scratch.kv_normed,
            &scratch.kv_raw,
            &lw.kv_a_norm,
            N_HEAD_DIM,
            RMS_EPS,
        )?;
        self.rope.launch_forward(
            &self.stream,
            &mut scratch.kv_normed,
            1,
            N_HEAD_DIM,
            N_ROT,
            pos,
            &lw.rope_params,
        )?;
        // Append to KV cache (host shadow + upload). ds4's
        // `kv_cache_push_raw` (ds4.c:6387) applies BOTH:
        //   1. dsv4_fp8_kv_quantize_row_inplace_cpu on first head_dim-n_rot
        //      = 448 elements (the no-PE portion)
        //   2. f16_to_f32(f32_to_f16(...)) round-trip on all 512 elements
        // Both run before pushing. Without (1) we diverge from the dump's
        // `kv_cached_row` by ~8.6e-2; (2) is a smaller additional quantize.
        self.fp8
            .launch(&self.stream, &mut scratch.kv_normed, N_HEAD_DIM - N_ROT)?;
        self.f16rt
            .launch(&self.stream, &mut scratch.kv_normed, N_HEAD_DIM)?;
        let mut kv_host = vec![0f32; N_HEAD_DIM as usize];
        self.stream.synchronize()?;
        scratch.kv_normed.copy_to_host(&mut kv_host)?;
        // SWA push: if cache has reached SWA_WINDOW (128), evict the oldest
        // row by shifting [1..128) → [0..127) and writing the new row at
        // slot 127. Mirrors `kv_cache_push_raw` (ds4.c:6387) memmove path.
        let stride = N_HEAD_DIM as usize;
        if ls.n_raw < SWA_WINDOW {
            let off = (ls.n_raw as usize) * stride;
            ls.kv_cache_host[off..off + stride].copy_from_slice(&kv_host);
            ls.n_raw += 1;
        } else {
            // Evict-oldest slide.
            let total = (SWA_WINDOW as usize) * stride;
            ls.kv_cache_host.copy_within(stride..total, 0);
            let last_off = ((SWA_WINDOW - 1) as usize) * stride;
            ls.kv_cache_host[last_off..last_off + stride].copy_from_slice(&kv_host);
            // n_raw stays at SWA_WINDOW.
        }
        // Re-upload entire KV cache shadow (simplest; small).
        ls.kv_cache.copy_from_host(&ls.kv_cache_host)?;

        // === (5) Compressor step (ratio>0) ===
        if ratio > 0 {
            let cw = lw
                .compressor
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: missing compressor weights"))?;
            let comp_width = cw.width;
            let pos_mod = pos % ratio;
            let row = if ratio == 4 { 4 + pos_mod } else { pos_mod };

            self.f16.matvec(
                &self.stream,
                &mut scratch.kv_cur,
                &cw.wkv.buffer,
                &scratch.attn_input_norm,
                comp_width,
                N_EMBD,
            )?;
            self.f16.matvec(
                &self.stream,
                &mut scratch.sc_cur,
                &cw.wgate.buffer,
                &scratch.attn_input_norm,
                comp_width,
                N_EMBD,
            )?;
            let cs = ls
                .compressor
                .as_mut()
                .ok_or_else(|| eyre!("L{layer}: missing compressor state"))?;
            self.compressor_state_write.launch(
                &self.stream,
                &mut cs.state_kv,
                &mut cs.state_score,
                &scratch.kv_cur,
                &scratch.sc_cur,
                &cw.ape.buffer,
                comp_width,
                row,
                pos_mod,
            )?;

            // Boundary fires every `ratio` tokens.
            if (pos + 1) % ratio == 0 {
                self.compressor_pool.launch(
                    &self.stream,
                    &mut scratch.pooled,
                    &cs.state_kv,
                    &cs.state_score,
                    N_HEAD_DIM,
                    ratio,
                )?;
                self.rms_w.launch_weighted(
                    &self.stream,
                    &mut scratch.comp_row,
                    &scratch.pooled,
                    &cw.norm,
                    N_HEAD_DIM,
                    RMS_EPS,
                )?;
                let comp_pos = pos + 1 - ratio;
                self.rope.launch_forward(
                    &self.stream,
                    &mut scratch.comp_row,
                    1,
                    N_HEAD_DIM,
                    N_ROT,
                    comp_pos,
                    &lw.rope_params,
                )?;
                self.fp8.launch(
                    &self.stream,
                    &mut scratch.comp_row,
                    N_HEAD_DIM - N_ROT,
                )?;
                self.f16rt
                    .launch(&self.stream, &mut scratch.comp_row, N_HEAD_DIM)?;
                let mut comp_host = vec![0f32; N_HEAD_DIM as usize];
                self.stream.synchronize()?;
                scratch.comp_row.copy_to_host(&mut comp_host)?;
                let coff = (cs.n_comp as usize) * (N_HEAD_DIM as usize);
                cs.comp_kv_host[coff..coff + (N_HEAD_DIM as usize)]
                    .copy_from_slice(&comp_host);
                cs.n_comp += 1;
                cs.comp_kv.copy_from_host(&cs.comp_kv_host)?;
                if ratio == 4 {
                    self.compressor_shuffle.launch(
                        &self.stream,
                        &mut cs.state_kv,
                        &mut cs.state_score,
                        comp_width,
                    )?;
                }
            }
        }

        // === (6) Attention compute ===
        // For our short-prompt validation, n_comp ≤ ~14 < top_k=512 so the
        // indexer is in early-permit mode (mask=None). Indexer pipeline is
        // skipped entirely; indexer compressor state evolution is also
        // deferred (its outputs are unused while early-permit).
        if ratio == 0 {
            self.attn_swa.launch(
                &self.stream,
                &mut scratch.heads,
                &scratch.q_normed,
                &ls.kv_cache,
                &lw.attn_sinks,
                N_HEAD,
                N_HEAD_DIM,
                ls.n_raw,
            )?;
        } else {
            let cs = ls.compressor.as_ref();
            let n_comp = cs.map(|c| c.n_comp).unwrap_or(0);
            let comp_kv_buf = if n_comp > 0 { cs.map(|c| &c.comp_kv) } else { None };
            self.attn_mixed.launch(
                &self.stream,
                &mut scratch.heads,
                &scratch.q_normed,
                &ls.kv_cache,
                comp_kv_buf,
                None,
                &lw.attn_sinks,
                N_HEAD,
                N_HEAD_DIM,
                ls.n_raw,
                n_comp,
            )?;
        }

        // === (7) Output projection (inv_rope + grouped_q8_0 + q8_0) ===
        self.rope.launch_inverse(
            &self.stream,
            &mut scratch.heads,
            N_HEAD,
            N_HEAD_DIM,
            N_ROT,
            pos,
            &lw.rope_params,
        )?;
        self.q8.quantize_input(
            &self.stream,
            &mut scratch.heads_xq,
            &mut scratch.heads_xscale,
            &scratch.heads,
            Q_FLAT,
        )?;
        self.q8_grouped.matvec_grouped(
            &self.stream,
            &mut scratch.low,
            &lw.attn_output_a.buffer,
            &scratch.heads_xq,
            &scratch.heads_xscale,
            GROUP_DIM,
            RANK,
            N_GROUPS,
        )?;
        self.q8.quantize_input(
            &self.stream,
            &mut scratch.low_xq,
            &mut scratch.low_xscale,
            &scratch.low,
            OUT_LOW,
        )?;
        self.q8.matvec(
            &self.stream,
            &mut scratch.attn_out,
            &lw.attn_output_b.buffer,
            &scratch.low_xq,
            &scratch.low_xscale,
            N_EMBD,
            OUT_LOW,
        )?;

        // === (8) mHC post attn ===
        self.hc_post.launch(
            &self.stream,
            &mut scratch.after_attn_hc,
            &scratch.attn_out,
            &scratch.residual,
            &scratch.post_attn,
            &scratch.comb_attn,
            N_EMBD,
            N_HC,
        )?;

        // === (9) mHC pre ffn → ffn_cur ===
        self.rms_nw.launch(
            &self.stream,
            &mut scratch.flat,
            &scratch.after_attn_hc,
            1,
            HC_DIM,
            RMS_EPS,
        )?;
        self.f16.matvec(
            &self.stream,
            &mut scratch.mix,
            &lw.hc_ffn_fn.buffer,
            &scratch.flat,
            HC_MIX_DIM,
            HC_DIM,
        )?;
        self.hc_sinkhorn.launch(
            &self.stream,
            &mut scratch.split,
            &scratch.mix,
            &lw.hc_ffn_scale,
            &lw.hc_ffn_base,
            N_HC,
            SINKHORN_ITERS,
            SINKHORN_EPS,
        )?;
        self.hc_weighted.launch(
            &self.stream,
            &mut scratch.ffn_cur,
            &scratch.after_attn_hc,
            &scratch.split,
            N_EMBD,
            N_HC,
        )?;
        self.stream.synchronize()?;
        scratch.split.copy_to_host(&mut split_host)?;
        let post_only_ffn = &split_host[N_HC as usize..2 * N_HC as usize];
        let comb_only_ffn = &split_host
            [2 * N_HC as usize..2 * N_HC as usize + (N_HC * N_HC) as usize];
        scratch.post_ffn.copy_from_host(post_only_ffn)?;
        scratch.comb_ffn.copy_from_host(comb_only_ffn)?;

        // === (10) ffn_input_norm ===
        self.rms_w.launch_weighted(
            &self.stream,
            &mut scratch.ffn_input_norm,
            &scratch.ffn_cur,
            &lw.ffn_norm,
            N_EMBD,
            RMS_EPS,
        )?;

        // === (11) Routing → selected[6], weights[6] ===
        self.f16.matvec(
            &self.stream,
            &mut scratch.router_logits,
            &lw.ffn_gate_inp.buffer,
            &scratch.ffn_input_norm,
            N_EXPERT,
            N_EMBD,
        )?;
        self.stream.synchronize()?;
        scratch
            .router_logits
            .copy_to_host(&mut scratch.router_logits_host)?;
        let (selected, weights_host) = if lw.is_hash_router {
            let tid2eid = lw
                .tid2eid
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: hash router but no tid2eid"))?;
            hash_router_select(tid2eid, token_id, &scratch.router_logits_host)
        } else {
            // Learned router: probs = sqrt(softplus(logits))
            let mut probs = vec![0f32; N_EXPERT as usize];
            for i in 0..N_EXPERT as usize {
                probs[i] = softplus_stable(scratch.router_logits_host[i]).sqrt();
            }
            let mut selection = probs.clone();
            if let Some(bias) = &lw.router_bias {
                for i in 0..N_EXPERT as usize {
                    selection[i] += bias[i];
                }
            }
            let sel = topk_desc(&selection, N_EXPERT_USED);
            let mut w = [0f32; N_EXPERT_USED];
            let mut sum = 0f32;
            for i in 0..N_EXPERT_USED {
                let p = probs[sel[i] as usize];
                w[i] = p;
                sum += p;
            }
            if sum < 6.103515625e-5 {
                sum = 6.103515625e-5;
            }
            for i in 0..N_EXPERT_USED {
                w[i] = w[i] / sum * EXPERT_WEIGHT_SCALE;
            }
            (sel, w)
        };

        // DEBUG: print selected + weights for bisect crosscheck
        if std::env::var("DS4_DEBUG_SELECTED").is_ok() {
            eprintln!(
                "    L{} selected={:?} weights={:?}",
                lw.layer_idx, selected, weights_host
            );
        }
        // DEBUG: override selected/weights from dump for the given layer (bisect).
        let (selected, weights_host) = if std::env::var("DS4_OVERRIDE_ROUTER_AT")
            .ok()
            .and_then(|s| s.parse::<i32>().ok())
            == Some(lw.layer_idx)
        {
            if let Some(dump) = unsafe { DS4_DEBUG_DUMP } {
                let dump = unsafe { &*dump };
                let pos_i32 = pos as i32;
                let sel_bytes = dump
                    .read_bytes(dump.tensor("expert_selected", lw.layer_idx, pos_i32).unwrap())
                    .unwrap();
                let mut sel = [0i32; 6];
                for i in 0..6 {
                    sel[i] = i32::from_le_bytes([
                        sel_bytes[i * 4],
                        sel_bytes[i * 4 + 1],
                        sel_bytes[i * 4 + 2],
                        sel_bytes[i * 4 + 3],
                    ]);
                }
                let w = dump
                    .read_f32(dump.tensor("expert_weight_out", lw.layer_idx, pos_i32).unwrap())
                    .unwrap();
                let mut wh = [0f32; 6];
                wh.copy_from_slice(&w[..6]);
                eprintln!("    >>> L{} router override from dump", lw.layer_idx);
                (sel, wh)
            } else {
                (selected, weights_host)
            }
        } else {
            (selected, weights_host)
        };

        // === (12) Routed-expert offsets — weights are iGPU-resident ===
        let gbpe = lw.routed.gate_bytes_per_expert;
        let ubpe = lw.routed.up_bytes_per_expert;
        let dbpe = lw.routed.down_bytes_per_expert;
        scratch.d_ew.copy_from_host(&weights_host)?;

        // === (13) Routed MoE pipeline ===
        self.q8k
            .launch(&self.stream, &mut scratch.d_xq_q8k, &scratch.ffn_input_norm, BLOCKS_Q8K_GATE_IN)?;
        let _ = gguf; // no longer needed for expert byte access
        for slot in 0..N_EXPERT_USED {
            let e = selected[slot] as usize;
            self.iq2.launch_with_offsets(
                &self.stream,
                &mut scratch.d_gate_e,
                &mut scratch.d_up_e,
                &lw.routed.gate.buffer,
                e * gbpe,
                &lw.routed.up.buffer,
                e * ubpe,
                &scratch.d_xq_q8k,
                N_FF_EXP,
                BLOCKS_Q8K_GATE_IN,
            )?;
            self.stream.synchronize()?;
            let mut staging = vec![0f32; N_FF_EXP as usize];
            scratch.d_gate_e.copy_to_host(&mut staging)?;
            scratch.host_gate_cat
                [slot * (N_FF_EXP as usize)..(slot + 1) * (N_FF_EXP as usize)]
                .copy_from_slice(&staging);
            scratch.d_up_e.copy_to_host(&mut staging)?;
            scratch.host_up_cat
                [slot * (N_FF_EXP as usize)..(slot + 1) * (N_FF_EXP as usize)]
                .copy_from_slice(&staging);
        }
        scratch.d_gate_cat.copy_from_host(&scratch.host_gate_cat)?;
        scratch.d_up_cat.copy_from_host(&scratch.host_up_cat)?;
        self.swiglu_cw.launch(
            &self.stream,
            &mut scratch.d_mid_cat,
            &scratch.d_gate_cat,
            &scratch.d_up_cat,
            &scratch.d_ew,
            SWIGLU_CLAMP_EXP,
            N_FF_EXP,
            N_EXPERT_USED as u32,
        )?;
        self.stream.synchronize()?;
        scratch.d_mid_cat.copy_to_host(&mut scratch.host_mid_cat)?;
        for slot in 0..N_EXPERT_USED {
            scratch.d_mid_e.copy_from_host(
                &scratch.host_mid_cat[slot * (N_FF_EXP as usize)..(slot + 1) * (N_FF_EXP as usize)],
            )?;
            self.q8k.launch(
                &self.stream,
                &mut scratch.d_midq_q8k,
                &scratch.d_mid_e,
                BLOCKS_Q8K_DOWN_IN,
            )?;
            let e = selected[slot] as usize;
            self.q2k.launch_with_offset(
                &self.stream,
                &mut scratch.ffn_moe,
                &lw.routed.down.buffer,
                e * dbpe,
                &scratch.d_midq_q8k,
                N_EMBD,
                BLOCKS_Q8K_DOWN_IN,
                slot == 0,
            )?;
        }

        // === (14) Shared expert ===
        self.q8
            .quantize_input(&self.stream, &mut scratch.xq_n_embd, &mut scratch.xscale_n_embd, &scratch.ffn_input_norm, N_EMBD)?;
        self.q8.matvec(
            &self.stream,
            &mut scratch.gate_sh,
            &lw.shared.gate.buffer,
            &scratch.xq_n_embd,
            &scratch.xscale_n_embd,
            N_FF_SHARED,
            N_EMBD,
        )?;
        self.q8.matvec(
            &self.stream,
            &mut scratch.up_sh,
            &lw.shared.up.buffer,
            &scratch.xq_n_embd,
            &scratch.xscale_n_embd,
            N_FF_SHARED,
            N_EMBD,
        )?;
        self.swiglu
            .launch(&self.stream, &mut scratch.mid_sh, &scratch.gate_sh, &scratch.up_sh, N_FF_SHARED)?;
        self.q8.quantize_input(
            &self.stream,
            &mut scratch.mid_sh_xq,
            &mut scratch.mid_sh_xscale,
            &scratch.mid_sh,
            N_FF_SHARED,
        )?;
        self.q8.matvec(
            &self.stream,
            &mut scratch.ffn_shared,
            &lw.shared.down.buffer,
            &scratch.mid_sh_xq,
            &scratch.mid_sh_xscale,
            N_EMBD,
            N_FF_SHARED,
        )?;

        // === (15) ffn_moe += ffn_shared ===
        self.vec_add
            .launch(&self.stream, &mut scratch.ffn_moe, &scratch.ffn_shared, N_EMBD)?;

        // === (16) mHC post ffn → residual_next ===
        self.hc_post.launch(
            &self.stream,
            &mut scratch.residual_next,
            &scratch.ffn_moe,
            &scratch.after_attn_hc,
            &scratch.post_ffn,
            &scratch.comb_ffn,
            N_EMBD,
            N_HC,
        )?;

        Ok(())
    }

    pub fn forward_head(
        &self,
        scratch: &mut Scratch,
        weights: &GlobalWeights,
    ) -> eyre::Result<()> {
        // residual (L=42 output) → flat → matvec → sigmoid → weighted_sum
        self.rms_nw.launch(
            &self.stream,
            &mut scratch.head_flat,
            &scratch.residual,
            1,
            HC_DIM,
            RMS_EPS,
        )?;
        self.f16.matvec(
            &self.stream,
            &mut scratch.head_pre,
            &weights.output_hc_fn.buffer,
            &scratch.head_flat,
            N_HC,
            HC_DIM,
        )?;
        self.hc_sigmoid.launch(
            &self.stream,
            &mut scratch.head_w,
            &scratch.head_pre,
            &weights.output_hc_scale,
            &weights.output_hc_base,
            N_HC,
        )?;
        self.hc_weighted.launch(
            &self.stream,
            &mut scratch.head_embd,
            &scratch.residual,
            &scratch.head_w,
            N_EMBD,
            N_HC,
        )?;
        self.rms_w.launch_weighted(
            &self.stream,
            &mut scratch.head_norm,
            &scratch.head_embd,
            &weights.output_norm,
            N_EMBD,
            RMS_EPS,
        )?;
        self.q8.quantize_input(
            &self.stream,
            &mut scratch.head_xq,
            &mut scratch.head_xscale,
            &scratch.head_norm,
            N_EMBD,
        )?;
        self.q8.matvec(
            &self.stream,
            &mut scratch.logits,
            &weights.output.buffer,
            &scratch.head_xq,
            &scratch.head_xscale,
            N_VOCAB,
            N_EMBD,
        )?;
        Ok(())
    }
}

// === Helpers ===

fn softplus_stable(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0f32 + x.exp()).ln()
    }
}

/// Top-K descending insertion sort (mirrors ds4.c:5272).
fn topk_desc(score: &[f32], k: usize) -> [i32; 6] {
    let mut idx = [-1i32; 6];
    for i in 0..score.len() {
        for j in 0..k {
            if idx[j] < 0 || score[i] > score[idx[j] as usize] {
                for m in (j + 1..k).rev() {
                    idx[m] = idx[m - 1];
                }
                idx[j] = i as i32;
                break;
            }
        }
    }
    idx
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


