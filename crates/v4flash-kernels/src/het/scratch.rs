//! Per-device scratch buffers. Allocated once per inference session.
//!
//! The "_recv" buffers are **peer-arrival mailboxes** — they are written
//! by the *other* device via `hipMemcpyPeerAsync`, and the local device
//! must `wait_event` on the matching peer-push event before reading them
//! (M13.4+; M13.1 uses `.synchronize()` instead).
//!
//! dGPU holds the residual chain + attention/mHC/shared/head buffers.
//! iGPU holds the router + routed-MoE buffers. The router_logits host
//! buffer + sel/weights selection vectors live on the iGPU side too,
//! since the topk computation runs in the iGPU control path (M13.3 will
//! move it onto an iGPU kernel).

use color_eyre::eyre;
use v4flash_hip::{Device, DeviceBuffer};

use crate::forward::{
    BLOCKS_GROUPED_OUT, BLOCKS_N_EMBD, BLOCKS_N_FF_SHARED, BLOCKS_N_LORA_Q, BLOCKS_OUT_LOW,
    BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, HC_DIM, HC_MIX_DIM, N_EMBD, N_EXPERT, N_EXPERT_USED,
    N_FF_EXP, N_FF_SHARED, N_HC, N_HEAD_DIM, N_INDEXER_HEAD_DIM, N_LORA_Q, N_VOCAB, OUT_LOW,
    Q_FLAT,
};
use crate::q8_k::BLOCK_Q8_K_BYTES;

pub struct DgpuScratch {
    // Cross-layer residual
    pub residual: DeviceBuffer<f32>,
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

    /// (Legacy after M14L.) Was the mailbox for `comp_row` peer-pushed
    /// from the iGPU compressor; now compressor runs locally on dGPU so
    /// this buffer is unused. Kept allocated to avoid touching downstream
    /// indexer paths that still reference it.
    pub comp_row_recv: DeviceBuffer<f32>,

    // M14L: compressor scratch — used to live in IgpuScratch.
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

            comp_row_recv: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,
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
    /// Peer-arrival mailbox for `attn_input_norm` pushed from dGPU
    /// after mhc_pre_attn (M13.5). Compressor reads from this.
    pub attn_input_norm_recv: DeviceBuffer<f32>,
    /// Peer-arrival mailbox for `ffn_input_norm` pushed from dGPU
    /// after mhc_pre_ffn (M13.1). Router + routed MoE read from this.
    pub ffn_input_norm_recv: DeviceBuffer<f32>,

    // Compressor scratch — migrated to iGPU in M13.5. The compressor
    // weight matvecs, state-write, pool, RMS, RoPE, FP8 and F16-RT all
    // run on the iGPU compute stream; `comp_row` is peer-pushed back to
    // the dGPU on boundaries.
    pub kv_cur: DeviceBuffer<f32>,
    pub sc_cur: DeviceBuffer<f32>,
    pub kv_cur_idx: DeviceBuffer<f32>,
    pub sc_cur_idx: DeviceBuffer<f32>,
    pub pooled: DeviceBuffer<f32>,
    pub comp_row: DeviceBuffer<f32>,

    pub router_logits: DeviceBuffer<f32>,
    pub router_logits_host: Vec<f32>,

    // Routed-MoE pipeline
    pub d_xq_q8k: DeviceBuffer<u8>,
    pub d_midq_q8k: DeviceBuffer<u8>,
    /// Per-slot concatenated mid-quant buffer:
    /// `[N_USED, BLOCKS_Q8K_DOWN_IN * BLOCK_Q8_K_BYTES]`. Lets the M13.4
    /// MoE pipeline run all 6 q8k quantizations + q2k accumulates
    /// without per-slot copy-to/from-host roundtrips.
    pub d_midq_cat: DeviceBuffer<u8>,
    pub d_gate_e: DeviceBuffer<f32>,
    pub d_up_e: DeviceBuffer<f32>,
    pub d_gate_cat: DeviceBuffer<f32>,
    pub d_up_cat: DeviceBuffer<f32>,
    pub d_mid_cat: DeviceBuffer<f32>,
    pub d_mid_e: DeviceBuffer<f32>,
    pub d_ew: DeviceBuffer<f32>,
    pub host_gate_cat: Vec<f32>,
    pub host_up_cat: Vec<f32>,
    pub host_mid_cat: Vec<f32>,
    pub ffn_moe: DeviceBuffer<f32>,

    /// Device-side `selected[N_EXPERT_USED]` produced by `router_topk` (M13.3).
    /// Host still reads this back to compute per-expert pointer offsets,
    /// but the read is 24 bytes vs the previous 1 KB logits roundtrip.
    pub d_selected: DeviceBuffer<i32>,
    pub host_selected: Vec<i32>,
}

impl IgpuScratch {
    pub fn alloc(igpu_device: Device) -> eyre::Result<Self> {
        igpu_device.set_current()?;
        let device_id = igpu_device.id;
        Ok(Self {
            attn_input_norm_recv: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            ffn_input_norm_recv: DeviceBuffer::new(device_id, N_EMBD as usize)?,

            kv_cur: DeviceBuffer::new(device_id, (2 * N_HEAD_DIM) as usize)?,
            sc_cur: DeviceBuffer::new(device_id, (2 * N_HEAD_DIM) as usize)?,
            kv_cur_idx: DeviceBuffer::new(device_id, (2 * N_INDEXER_HEAD_DIM) as usize)?,
            sc_cur_idx: DeviceBuffer::new(device_id, (2 * N_INDEXER_HEAD_DIM) as usize)?,
            pooled: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,
            comp_row: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,

            router_logits: DeviceBuffer::new(device_id, N_EXPERT as usize)?,
            router_logits_host: vec![0f32; N_EXPERT as usize],

            d_xq_q8k: DeviceBuffer::new(
                device_id,
                (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_Q8_K_BYTES,
            )?,
            d_midq_q8k: DeviceBuffer::new(
                device_id,
                (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES,
            )?,
            d_midq_cat: DeviceBuffer::new(
                device_id,
                N_EXPERT_USED * (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES,
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

            d_selected: DeviceBuffer::new(device_id, N_EXPERT_USED)?,
            host_selected: vec![0i32; N_EXPERT_USED],
        })
    }
}
