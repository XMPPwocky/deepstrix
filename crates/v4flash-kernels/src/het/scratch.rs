//! Per-device scratch buffers. Allocated once per inference session.
//!
//! The "_recv" buffers are **peer-arrival mailboxes** — they are written
//! by the *other* device via `hipMemcpyPeerAsync`, and the local device
//! waits on the matching peer-push event before reading them.
//!
//! dGPU holds the residual chain + attention / mHC / shared expert /
//! head / compressor / router buffers. iGPU holds the routed-MoE
//! pipeline buffers + the peer-arrival mailbox for `ffn_input_norm`.

use color_eyre::eyre;
use v4flash_hip::{Device, DeviceBuffer};

use crate::config::{
    BLOCKS_GROUPED_OUT, BLOCKS_N_EMBD, BLOCKS_N_FF_SHARED, BLOCKS_N_LORA_Q, BLOCKS_OUT_LOW,
    BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, HC_DIM, HC_MIX_DIM, INDEXER_TOP_K, N_EMBD, N_EXPERT,
    N_EXPERT_USED, N_FF_EXP, N_FF_SHARED, N_HC, N_HEAD, N_HEAD_DIM, N_INDEXER_HEAD,
    N_INDEXER_HEAD_DIM, N_LORA_Q, N_VOCAB, OUT_LOW, Q_FLAT,
};
use crate::attention::ATTN_MIXED_MAX_KEYS;
use crate::q8_k::BLOCK_Q8_K_BYTES;

pub struct DgpuScratch {
    // Cross-layer residual
    pub residual: DeviceBuffer<f32>,
    pub residual_next: DeviceBuffer<f32>,

    // mHC stage
    pub flat: DeviceBuffer<f32>,
    pub mix: DeviceBuffer<f32>,
    pub split: DeviceBuffer<f32>,
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

    // Split-kernel attention scratch: holds scores out of `attn_score`,
    // then overwritten in place with weights by `attn_softmax_wsum`.
    // Size [N_HEAD, ATTN_MIXED_MAX_KEYS].
    pub attn_scores: DeviceBuffer<f32>,

    // Per-batch counters for the batched-WMMA attention kernels called at
    // B=1 from the decode path. Allocated as 1-element buffers each;
    // overwritten per-layer with [n_raw], [0], [n_comp] before launch.
    // The batched score (f16s, head-tiled WMMA) runs 6× faster than the
    // single-token launch_score at ratio=4 depth-16K — wired only for
    // score for now; the batched smwsum loses at B=1 due to under-fill.
    pub attn_n_raw_per_b1: DeviceBuffer<i32>,
    pub attn_n_raw_offset_per_b1: DeviceBuffer<i32>,
    pub attn_n_comp_per_b1: DeviceBuffer<i32>,

    // Decode K-split smwsum pipeline scratch (per [[decode-long-ctx-analysis]]).
    // partials: [k_split, n_head, head_dim] f32 — written by wsum kernel
    //   pass 2, summed by reduce kernel pass 3. At k_split=16 and the
    //   V4-Flash shape (n_head=64, head_dim=512) this is 2 MiB.
    // inv_per_head: [n_head] f32 — written by softmax_only pass 1,
    //   consumed by reduce pass 3.
    pub attn_partials: DeviceBuffer<f32>,
    pub attn_inv_per_head: DeviceBuffer<f32>,

    // Per-WG sum-sq partials for the multi-WG rms_norm_no_weight kernel.
    // Sized for max n_wgs=64; production uses n_wgs=16.
    pub rms_nw_partials: DeviceBuffer<f32>,
    // 1-element scalar holding inv_rms = 1/sqrt(mean(x²)+eps) for the
    // fused rms_nw + pre-scaled matvec pair. Pair: RmsNormNoWeightMultiWG
    // ::launch_inv_only → F16Matvec::matvec_pre_scaled.
    pub rms_nw_inv_scalar: DeviceBuffer<f32>,

    // Partials for f16_matvec_narrow_ksplit at HC_MIX_DIM=24, n_k_split≤64.
    pub mhc_matvec_partials: DeviceBuffer<f32>,

    // Compressor scratch (lives on dGPU alongside attn_input_norm).
    pub kv_cur: DeviceBuffer<f32>,
    pub sc_cur: DeviceBuffer<f32>,
    pub pooled: DeviceBuffer<f32>,
    pub comp_row: DeviceBuffer<f32>,

    // CSA indexer scratch (used only on ratio==4 layers with
    // n_index_comp > INDEXER_TOP_K; otherwise the entire indexer pipeline
    // short-circuits via ds4's early-permit and these buffers are untouched).
    pub indexer_q: DeviceBuffer<f32>,            // [64 * 128]
    pub indexer_head_weights: DeviceBuffer<f32>, // [64]
    pub indexer_scores: DeviceBuffer<f32>,       // [ATTN_MIXED_MAX_KEYS]
    pub indexer_selected: DeviceBuffer<i32>,     // [INDEXER_TOP_K]
    pub indexer_allowed_bits: DeviceBuffer<u32>, // [ceil(ATTN_MIXED_MAX_KEYS/32)]
    /// Scratch for the bitonic IndexerTopk's per-chunk candidates.
    /// Sized for the worst case: ceil(ATTN_MIXED_MAX_KEYS/4096) chunks
    /// × INDEXER_TOP_K candidates per chunk = up to 6 × 512 = 3072 u32
    /// at ATTN_MIXED_MAX_KEYS=24576.
    pub indexer_topk_scratch: DeviceBuffer<u32>,
    pub active_comp_kv: DeviceBuffer<u16>,       // [INDEXER_TOP_K * N_HEAD_DIM]

    // Shared expert
    pub gate_sh: DeviceBuffer<f32>,
    pub up_sh: DeviceBuffer<f32>,
    pub mid_sh: DeviceBuffer<f32>,
    pub mid_sh_xq: DeviceBuffer<i8>,
    pub mid_sh_xscale: DeviceBuffer<f32>,
    pub ffn_shared: DeviceBuffer<f32>,

    // Mailbox for ffn_moe arriving from iGPU.
    pub ffn_moe_recv: DeviceBuffer<f32>,

    // Router (lives on dGPU). Matvec writes router_logits; topk (or
    // hash router host path) writes d_selected/d_ew. Both are then
    // peer-pushed to iGPU MoE.
    pub router_logits: DeviceBuffer<f32>,
    pub router_logits_host: Vec<f32>,
    pub d_selected: DeviceBuffer<i32>,
    pub d_ew: DeviceBuffer<f32>,

    // Head
    pub head_flat: DeviceBuffer<f32>,
    pub head_pre: DeviceBuffer<f32>,
    pub head_w: DeviceBuffer<f32>,
    pub head_embd: DeviceBuffer<f32>,
    pub head_norm: DeviceBuffer<f32>,
    pub head_xq: DeviceBuffer<i8>,
    pub head_xscale: DeviceBuffer<f32>,
    pub logits: DeviceBuffer<f32>,

    // Sampler scratch (see crate::sampler). partials_max / partials_z
    // hold per-WG reductions consumed by softmax_sample_one. u01 is a
    // 1-element f32 the host writes per decode step. next_token_id is
    // the kernel's only output — a 1-element i32 the host reads back
    // after sampling.
    pub sampler_partials_max: DeviceBuffer<f32>,
    pub sampler_partials_z: DeviceBuffer<f32>,
    pub sampler_u01: DeviceBuffer<f32>,
    pub sampler_next_token_id: DeviceBuffer<i32>,
}

impl DgpuScratch {
    pub fn alloc(dgpu_device: Device) -> eyre::Result<Self> {
        dgpu_device.set_current()?;
        let device_id = dgpu_device.id;
        Ok(Self {
            residual: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            residual_next: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            flat: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            mix: DeviceBuffer::new(device_id, HC_MIX_DIM as usize)?,
            split: DeviceBuffer::new(device_id, HC_MIX_DIM as usize)?,
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

            attn_scores: DeviceBuffer::new(
                device_id,
                (N_HEAD as usize) * (ATTN_MIXED_MAX_KEYS as usize),
            )?,
            attn_n_raw_per_b1: DeviceBuffer::new(device_id, 1)?,
            attn_n_raw_offset_per_b1: {
                let mut b: DeviceBuffer<i32> = DeviceBuffer::new(device_id, 1)?;
                b.copy_from_host(&[0i32])?;
                b
            },
            attn_n_comp_per_b1: DeviceBuffer::new(device_id, 1)?,
            // k_split=16 matches the kernel default; if it ever changes,
            // this allocation needs to grow with it.
            attn_partials: DeviceBuffer::new(
                device_id,
                16 * (N_HEAD as usize) * (N_HEAD_DIM as usize),
            )?,
            attn_inv_per_head: DeviceBuffer::new(device_id, N_HEAD as usize)?,
            rms_nw_partials: DeviceBuffer::new(device_id, 64)?,
            rms_nw_inv_scalar: DeviceBuffer::new(device_id, 1)?,
            // 64 × HC_MIX_DIM=24 = 1536 f32 = 6 KB. n_k_split=32 uses half.
            mhc_matvec_partials: DeviceBuffer::new(device_id, 64 * (HC_MIX_DIM as usize))?,

            kv_cur: DeviceBuffer::new(device_id, (2 * N_HEAD_DIM) as usize)?,
            sc_cur: DeviceBuffer::new(device_id, (2 * N_HEAD_DIM) as usize)?,
            pooled: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,
            comp_row: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,

            indexer_q: DeviceBuffer::new(
                device_id,
                (N_INDEXER_HEAD * N_INDEXER_HEAD_DIM) as usize,
            )?,
            indexer_head_weights: DeviceBuffer::new(device_id, N_INDEXER_HEAD as usize)?,
            indexer_scores: DeviceBuffer::new(device_id, ATTN_MIXED_MAX_KEYS as usize)?,
            indexer_selected: DeviceBuffer::new(device_id, INDEXER_TOP_K as usize)?,
            indexer_allowed_bits: DeviceBuffer::new(
                device_id,
                ((ATTN_MIXED_MAX_KEYS + 31) / 32) as usize,
            )?,
            indexer_topk_scratch: {
                let max_chunks = (ATTN_MIXED_MAX_KEYS + 4095) / 4096;
                DeviceBuffer::new(device_id, (max_chunks * INDEXER_TOP_K) as usize)?
            },
            active_comp_kv: DeviceBuffer::new(
                device_id,
                (INDEXER_TOP_K * N_HEAD_DIM) as usize,
            )?,

            gate_sh: DeviceBuffer::new(device_id, N_FF_SHARED as usize)?,
            up_sh: DeviceBuffer::new(device_id, N_FF_SHARED as usize)?,
            mid_sh: DeviceBuffer::new(device_id, N_FF_SHARED as usize)?,
            mid_sh_xq: DeviceBuffer::new(device_id, N_FF_SHARED as usize)?,
            mid_sh_xscale: DeviceBuffer::new(device_id, BLOCKS_N_FF_SHARED as usize)?,
            ffn_shared: DeviceBuffer::new(device_id, N_EMBD as usize)?,

            ffn_moe_recv: DeviceBuffer::new(device_id, N_EMBD as usize)?,

            // Router (dGPU-resident).
            router_logits: DeviceBuffer::new(device_id, N_EXPERT as usize)?,
            router_logits_host: vec![0f32; N_EXPERT as usize],
            d_selected: DeviceBuffer::new(device_id, N_EXPERT_USED)?,
            d_ew: DeviceBuffer::new(device_id, N_EXPERT_USED)?,

            head_flat: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            head_pre: DeviceBuffer::new(device_id, N_HC as usize)?,
            head_w: DeviceBuffer::new(device_id, N_HC as usize)?,
            head_embd: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            head_norm: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            head_xq: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            head_xscale: DeviceBuffer::new(device_id, BLOCKS_N_EMBD as usize)?,
            logits: DeviceBuffer::new(device_id, N_VOCAB as usize)?,

            sampler_partials_max: DeviceBuffer::new(device_id, crate::sampler::SAMPLER_N_WG as usize)?,
            sampler_partials_z: DeviceBuffer::new(device_id, crate::sampler::SAMPLER_N_WG as usize)?,
            sampler_u01: DeviceBuffer::new(device_id, 1)?,
            sampler_next_token_id: DeviceBuffer::new(device_id, 1)?,
        })
    }
}

pub struct IgpuScratch {
    /// Peer-arrival mailbox for `ffn_input_norm` pushed from dGPU after
    /// mhc_pre_ffn. The routed-MoE pipeline reads from this.
    pub ffn_input_norm_recv: DeviceBuffer<f32>,

    // Routed-MoE pipeline (device-side fused: q8k_xq → iq2_fused →
    // q8k_mid → q2k_down). `d_mid_cat` is the per-slot concatenated
    // mid-quant intermediate; `d_midq_cat` is its q8k-quantized form.
    pub d_xq_q8k: DeviceBuffer<u8>,
    pub d_mid_cat: DeviceBuffer<f32>,
    pub d_midq_cat: DeviceBuffer<u8>,
    pub d_ew: DeviceBuffer<f32>,
    pub d_selected: DeviceBuffer<i32>,
    pub ffn_moe: DeviceBuffer<f32>,
}

impl IgpuScratch {
    pub fn alloc(igpu_device: Device) -> eyre::Result<Self> {
        igpu_device.set_current()?;
        let device_id = igpu_device.id;
        Ok(Self {
            ffn_input_norm_recv: DeviceBuffer::new(device_id, N_EMBD as usize)?,

            d_xq_q8k: DeviceBuffer::new(
                device_id,
                (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_Q8_K_BYTES,
            )?,
            d_mid_cat: DeviceBuffer::new(device_id, N_EXPERT_USED * (N_FF_EXP as usize))?,
            d_midq_cat: DeviceBuffer::new(
                device_id,
                N_EXPERT_USED * (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES,
            )?,
            d_ew: DeviceBuffer::new(device_id, N_EXPERT_USED)?,
            d_selected: DeviceBuffer::new(device_id, N_EXPERT_USED)?,
            ffn_moe: DeviceBuffer::new(device_id, N_EMBD as usize)?,
        })
    }
}
