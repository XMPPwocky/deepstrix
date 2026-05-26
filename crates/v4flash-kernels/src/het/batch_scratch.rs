//! M50: per-batch scratch for prefill.
//!
//! Phase 1 implementation: a shared `DgpuScratch`/`IgpuScratch` used by
//! the existing single-token `forward_layer` in an inner loop over B,
//! with `per_token_residual` ping-ponged in/out. See `forward_prompt_batch`.
//!
//! Phase 2 (this file): adds [`BatchDgpuScratch`] — every per-token field
//! is B-extended to a single contiguous `[B × per_token_size]` buffer.
//! Batched kernels (`*_batched`) read/write with per-batch strides
//! directly, no per-token copies. Used by `forward_prompt_batch_v2`.
//!
//! Memory cost at `B_MAX = 64`:
//! * Phase 1 `BatchScratch`: ~2 MB shared + 64×16 KB residual = ~3 MB.
//! * Phase 2 `BatchDgpuScratch`: ~26 MB of B-extended per-token state.

use color_eyre::eyre;
use v4flash_hip::{Device, DeviceBuffer};

use crate::forward::{
    BLOCKS_GROUPED_OUT, BLOCKS_N_EMBD, BLOCKS_N_FF_SHARED, BLOCKS_N_LORA_Q, BLOCKS_OUT_LOW,
    BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, HC_DIM, HC_MIX_DIM, N_EMBD, N_EXPERT, N_EXPERT_USED,
    N_FF_EXP, N_FF_SHARED, N_HEAD_DIM, N_LORA_Q, OUT_LOW, Q_FLAT,
};
use crate::q8_k::BLOCK_Q8_K_BYTES;

use super::scratch::{DgpuScratch, IgpuScratch};

/// Max prefill batch size. Bumped to 256 for the by-expert MoE
/// experiment — at B=256 expected per-expert reuse is ~6× (vs ~2× at
/// B=64), which is where by-expert grouping should start winning over
/// the L2-amortized by-token kernel. Memory cost at B=256: ~104 MB dGPU
/// scratch + ~24 MB iGPU scratch — trivial vs the 60+ GiB model weights.
pub const B_MAX: usize = 256;

/// Per-batch dGPU + iGPU scratch for prefill.
///
/// Phase 1: Holds ONE shared `DgpuScratch`/`IgpuScratch` used by the
/// engine's existing `forward_layer` (so captured HIP graphs replay
/// consistently — captures bake in scratch buffer pointers, so we
/// can't naively spread them across B independent scratches), plus B
/// small per-token residual buffers swapped in/out around each
/// per-layer call. KV cache + compressor state live in `HetModelState`
/// (per-layer, shared across batch).
///
/// In Phase 2 the shared scratch goes away and per-batch fields are
/// batch-extended (`[B × original_size]`) within a single struct, so
/// batched kernels can index directly without copies. For now this
/// keeps Phase 1 small and lets us validate the layer-major schedule.
pub struct BatchScratch {
    pub shared_dgpu: DgpuScratch,
    pub shared_igpu: IgpuScratch,
    /// Per-token residual buffers ping-ponged into shared scratch
    /// around each `forward_layer` call.
    pub per_token_residual: Vec<v4flash_hip::DeviceBuffer<f32>>,
    /// Per-token residual_next (post-layer-N output buffer).
    pub per_token_residual_next: Vec<v4flash_hip::DeviceBuffer<f32>>,
}

impl BatchScratch {
    pub fn alloc(dgpu_device: Device, igpu_device: Device) -> eyre::Result<Self> {
        use crate::forward::HC_DIM;
        let shared_dgpu = DgpuScratch::alloc(dgpu_device)?;
        let shared_igpu = IgpuScratch::alloc(igpu_device)?;
        dgpu_device.set_current()?;
        let mut per_token_residual = Vec::with_capacity(B_MAX);
        let mut per_token_residual_next = Vec::with_capacity(B_MAX);
        for _ in 0..B_MAX {
            per_token_residual.push(v4flash_hip::DeviceBuffer::new(
                dgpu_device.id,
                HC_DIM as usize,
            )?);
            per_token_residual_next.push(v4flash_hip::DeviceBuffer::new(
                dgpu_device.id,
                HC_DIM as usize,
            )?);
        }
        Ok(Self {
            shared_dgpu,
            shared_igpu,
            per_token_residual,
            per_token_residual_next,
        })
    }

    pub fn b_max(&self) -> usize {
        B_MAX
    }
}

/// M50 Phase 2: per-batch dGPU scratch with B-extended buffers.
///
/// Every field is a single contiguous `DeviceBuffer` sized for the
/// maximum batch (`B_MAX × per_token_size`). Batched kernels read/write
/// with per-batch strides; per-token (stateful) kernels use offset slices.
///
/// Memory at B_MAX=64: ~26 MB total. Trivial vs the 9 GiB model weights.
///
/// Does NOT include head/MTP/stash buffers — `forward_layer_batch_v2`
/// only needs the active per-token state. Head can run in a serial loop
/// over `per_token_residual_next` from the parent `BatchScratch`.
pub struct BatchDgpuScratch {
    // ---- mHC stage ----
    /// `[B, HC_DIM]` — per-token residual (cross-layer flow).
    pub residual: DeviceBuffer<f32>,
    pub residual_next: DeviceBuffer<f32>,
    /// `[B, HC_DIM]` — rms_nw output going into hc_attn_fn.
    pub flat: DeviceBuffer<f32>,
    /// `[B, HC_MIX_DIM]` — f16 narrow matvec output (sinkhorn input).
    pub mix: DeviceBuffer<f32>,
    /// `[B, HC_MIX_DIM]` — sinkhorn output (post + comb embedded).
    pub split: DeviceBuffer<f32>,
    /// `[B, N_EMBD]` — hc_weighted output for attention input.
    pub attn_cur: DeviceBuffer<f32>,
    pub attn_input_norm: DeviceBuffer<f32>,
    pub after_attn_hc: DeviceBuffer<f32>,
    pub ffn_cur: DeviceBuffer<f32>,
    pub ffn_input_norm: DeviceBuffer<f32>,

    // ---- Q chain ----
    pub xq_n_embd: DeviceBuffer<i8>,
    pub xscale_n_embd: DeviceBuffer<f32>,
    pub qr: DeviceBuffer<f32>,
    pub qr_normed: DeviceBuffer<f32>,
    pub qr_xq: DeviceBuffer<i8>,
    pub qr_xscale: DeviceBuffer<f32>,
    pub q: DeviceBuffer<f32>,
    pub q_normed: DeviceBuffer<f32>,

    // ---- KV chain ----
    pub kv_raw: DeviceBuffer<f32>,
    pub kv_normed: DeviceBuffer<f32>,

    // ---- Compressor (used in per-batch serial inner loop) ----
    pub kv_cur: DeviceBuffer<f32>,
    pub sc_cur: DeviceBuffer<f32>,
    pub pooled: DeviceBuffer<f32>,
    pub comp_row: DeviceBuffer<f32>,

    // ---- Per-token attention causality (Phase 4) ----
    /// `[B]` — per-token causal prefix length over raw KV cache.
    pub n_raw_per: DeviceBuffer<i32>,
    /// `[B]` — per-token causal prefix length over comp_kv (0 for ratio=0
    /// layers and for tokens before the first comp boundary).
    pub n_comp_per: DeviceBuffer<i32>,

    // ---- Attention output + output projection ----
    pub heads: DeviceBuffer<f32>,
    pub low: DeviceBuffer<f32>,
    pub heads_xq: DeviceBuffer<i8>,
    pub heads_xscale: DeviceBuffer<f32>,
    pub low_xq: DeviceBuffer<i8>,
    pub low_xscale: DeviceBuffer<f32>,
    pub attn_out: DeviceBuffer<f32>,

    // ---- Router ----
    pub router_logits: DeviceBuffer<f32>,
    pub d_selected: DeviceBuffer<i32>,
    pub d_ew: DeviceBuffer<f32>,
    /// Host readback area for hash router path. `[B, N_EXPERT]`.
    pub router_logits_host: Vec<f32>,

    // ---- Shared expert ----
    pub gate_sh: DeviceBuffer<f32>,
    pub up_sh: DeviceBuffer<f32>,
    pub mid_sh: DeviceBuffer<f32>,
    pub mid_sh_xq: DeviceBuffer<i8>,
    pub mid_sh_xscale: DeviceBuffer<f32>,
    pub ffn_shared: DeviceBuffer<f32>,

    /// `[B, N_EMBD]` — peer-arrival mailbox for iGPU MoE output. Single
    /// batched peer-push from iGPU (Phase 3); then vec_add ffn_shared and
    /// run hc_post.
    pub ffn_moe_recv: DeviceBuffer<f32>,
}

/// M50 Phase 3: per-batch iGPU scratch with B-extended buffers.
///
/// Mirrors [`IgpuScratch`]'s MoE-relevant fields but each sized for
/// `B_MAX × per_token_size`. Used by the batched iGPU MoE in
/// `forward_layer_batch_v2`: one peer-push of `[B, N_EMBD]` ain, one
/// batched iq2 + q2k call chain, one peer-push of `[B, N_EMBD]` ffn_moe
/// back — replaces the per-batch serial loop.
///
/// Memory at B_MAX=64 (rough):
/// * `ffn_input_norm_recv` 64×16KB = 1 MB
/// * `d_xq_q8k` 64×~4.7KB = 300 KB
/// * `d_mid_cat` 64×6×8KB = 3 MB
/// * `d_midq_cat` 64×6×~2.3KB = 880 KB
/// * `ffn_moe` 64×16KB = 1 MB
/// * `d_selected`/`d_ew` 64×6 = 1.5 KB each
/// Total: ~6 MB. Negligible vs the 52 GiB resident expert weights.
pub struct BatchIgpuScratch {
    pub ffn_input_norm_recv: DeviceBuffer<f32>,
    pub d_xq_q8k: DeviceBuffer<u8>,
    pub d_mid_cat: DeviceBuffer<f32>,
    pub d_midq_cat: DeviceBuffer<u8>,
    pub ffn_moe: DeviceBuffer<f32>,
    pub d_selected: DeviceBuffer<i32>,
    pub d_ew: DeviceBuffer<f32>,
    /// Phase 7 by-expert: per-expert pick count built by the group_builder
    /// pre-pass. `[n_expert]` i32. MUST be zeroed before each layer's pre-pass.
    pub group_count: DeviceBuffer<i32>,
    /// Phase 7 by-expert: per-expert (b, slot) member lists, packed as
    /// `(b << 16) | slot`. `[n_expert × max_per_expert]` i32. Only the first
    /// `group_count[e]` entries per expert are valid after the pre-pass.
    /// `max_per_expert = B_MAX` (worst case: every token picks the same expert
    /// in some slot — still fits since each token contributes ≤ n_used picks).
    pub expert_members: DeviceBuffer<i32>,
}

impl BatchIgpuScratch {
    pub fn alloc(igpu_device: Device) -> eyre::Result<Self> {
        igpu_device.set_current()?;
        let id = igpu_device.id;
        let b = B_MAX;
        let xq_bytes_per_batch =
            (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_Q8_K_BYTES;
        let midq_bytes_per_batch =
            (N_EXPERT_USED as usize) * (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES;
        Ok(Self {
            ffn_input_norm_recv: DeviceBuffer::new(id, b * N_EMBD as usize)?,
            d_xq_q8k: DeviceBuffer::new(id, b * xq_bytes_per_batch)?,
            d_mid_cat: DeviceBuffer::new(
                id,
                b * (N_EXPERT_USED as usize) * (N_FF_EXP as usize),
            )?,
            d_midq_cat: DeviceBuffer::new(id, b * midq_bytes_per_batch)?,
            ffn_moe: DeviceBuffer::new(id, b * N_EMBD as usize)?,
            d_selected: DeviceBuffer::new(id, b * N_EXPERT_USED as usize)?,
            d_ew: DeviceBuffer::new(id, b * N_EXPERT_USED as usize)?,
            group_count: DeviceBuffer::new(id, N_EXPERT as usize)?,
            expert_members: DeviceBuffer::new(id, (N_EXPERT as usize) * b)?,
        })
    }

    /// Max per-expert group capacity (= B_MAX). Used by Stage 11's by-expert path.
    pub fn max_per_expert(&self) -> u32 {
        B_MAX as u32
    }
}

impl BatchDgpuScratch {
    pub fn alloc(dgpu_device: Device) -> eyre::Result<Self> {
        dgpu_device.set_current()?;
        let id = dgpu_device.id;
        let b = B_MAX;
        let mk_f32 =
            |n: usize| -> eyre::Result<DeviceBuffer<f32>> { DeviceBuffer::new(id, b * n) };
        let mk_i8 = |n: usize| -> eyre::Result<DeviceBuffer<i8>> { DeviceBuffer::new(id, b * n) };
        let mk_i32 =
            |n: usize| -> eyre::Result<DeviceBuffer<i32>> { DeviceBuffer::new(id, b * n) };
        Ok(Self {
            residual: mk_f32(HC_DIM as usize)?,
            residual_next: mk_f32(HC_DIM as usize)?,
            flat: mk_f32(HC_DIM as usize)?,
            mix: mk_f32(HC_MIX_DIM as usize)?,
            split: mk_f32(HC_MIX_DIM as usize)?,
            attn_cur: mk_f32(N_EMBD as usize)?,
            attn_input_norm: mk_f32(N_EMBD as usize)?,
            after_attn_hc: mk_f32(HC_DIM as usize)?,
            ffn_cur: mk_f32(N_EMBD as usize)?,
            ffn_input_norm: mk_f32(N_EMBD as usize)?,

            xq_n_embd: mk_i8(N_EMBD as usize)?,
            xscale_n_embd: mk_f32(BLOCKS_N_EMBD as usize)?,
            qr: mk_f32(N_LORA_Q as usize)?,
            qr_normed: mk_f32(N_LORA_Q as usize)?,
            qr_xq: mk_i8(N_LORA_Q as usize)?,
            qr_xscale: mk_f32(BLOCKS_N_LORA_Q as usize)?,
            q: mk_f32(Q_FLAT as usize)?,
            q_normed: mk_f32(Q_FLAT as usize)?,

            kv_raw: mk_f32(N_HEAD_DIM as usize)?,
            kv_normed: mk_f32(N_HEAD_DIM as usize)?,

            kv_cur: mk_f32((2 * N_HEAD_DIM) as usize)?,
            sc_cur: mk_f32((2 * N_HEAD_DIM) as usize)?,
            pooled: mk_f32(N_HEAD_DIM as usize)?,
            comp_row: mk_f32(N_HEAD_DIM as usize)?,

            n_raw_per: mk_i32(1)?,
            n_comp_per: mk_i32(1)?,

            heads: mk_f32(Q_FLAT as usize)?,
            low: mk_f32(OUT_LOW as usize)?,
            heads_xq: mk_i8(Q_FLAT as usize)?,
            heads_xscale: mk_f32(BLOCKS_GROUPED_OUT as usize)?,
            low_xq: mk_i8(OUT_LOW as usize)?,
            low_xscale: mk_f32(BLOCKS_OUT_LOW as usize)?,
            attn_out: mk_f32(N_EMBD as usize)?,

            router_logits: mk_f32(N_EXPERT as usize)?,
            d_selected: mk_i32(N_EXPERT_USED)?,
            d_ew: mk_f32(N_EXPERT_USED)?,
            router_logits_host: vec![0.0f32; b * (N_EXPERT as usize)],

            gate_sh: mk_f32(N_FF_SHARED as usize)?,
            up_sh: mk_f32(N_FF_SHARED as usize)?,
            mid_sh: mk_f32(N_FF_SHARED as usize)?,
            mid_sh_xq: mk_i8(N_FF_SHARED as usize)?,
            mid_sh_xscale: mk_f32(BLOCKS_N_FF_SHARED as usize)?,
            ffn_shared: mk_f32(N_EMBD as usize)?,

            ffn_moe_recv: mk_f32(N_EMBD as usize)?,
        })
    }
}
