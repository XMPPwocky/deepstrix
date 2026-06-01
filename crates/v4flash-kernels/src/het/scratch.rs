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
    BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, HC_DIM, HC_MIX_DIM, N_EMBD, N_EXPERT, N_EXPERT_USED,
    N_FF_EXP, N_FF_SHARED, N_HC, N_HEAD, N_HEAD_DIM, N_LORA_Q, N_VOCAB, OUT_LOW, Q_FLAT,
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

    // Compressor scratch (lives on dGPU alongside attn_input_norm).
    pub kv_cur: DeviceBuffer<f32>,
    pub sc_cur: DeviceBuffer<f32>,
    pub pooled: DeviceBuffer<f32>,
    pub comp_row: DeviceBuffer<f32>,

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

            kv_cur: DeviceBuffer::new(device_id, (2 * N_HEAD_DIM) as usize)?,
            sc_cur: DeviceBuffer::new(device_id, (2 * N_HEAD_DIM) as usize)?,
            pooled: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,
            comp_row: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,

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
