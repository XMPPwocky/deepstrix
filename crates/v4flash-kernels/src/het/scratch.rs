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
    N_FF_EXP, N_FF_SHARED, N_HC, N_HEAD, N_HEAD_DIM, N_INDEXER_HEAD_DIM, N_LORA_Q, N_VOCAB,
    OUT_LOW, Q_FLAT,
};
use crate::attention::ATTN_MIXED_MAX_KEYS;
use crate::q8_k::BLOCK_Q8_K_BYTES;

/// M40-P4: per-token in-flight scratch for `forward_pair_interleaved`.
/// Holds every buffer that pre_moe + post_moe write/read inside a single
/// layer pass for ONE token. With per-token streams, t0's and t1's
/// pre_moes run truly in parallel on the dGPU — they cannot share these
/// buffers. Memory cost: ~1 MB per instance × 2 instances = ~2 MB total.
pub struct TokenScratch {
    // Residual flow (input residual + this layer's output ffn_combine result).
    pub residual: DeviceBuffer<f32>,      // HC_DIM
    pub residual_next: DeviceBuffer<f32>, // HC_DIM

    // mhc_pre_attn intermediates + output.
    pub flat: DeviceBuffer<f32>,            // HC_DIM
    pub mix: DeviceBuffer<f32>,             // HC_MIX_DIM
    pub split: DeviceBuffer<f32>,           // HC_MIX_DIM (matches DgpuScratch.split)
    pub attn_cur: DeviceBuffer<f32>,        // N_EMBD
    pub attn_input_norm: DeviceBuffer<f32>, // N_EMBD

    // Q chain.
    pub xq_n_embd: DeviceBuffer<i8>,       // N_EMBD
    pub xscale_n_embd: DeviceBuffer<f32>,  // BLOCKS_N_EMBD
    pub qr: DeviceBuffer<f32>,             // N_LORA_Q
    pub qr_normed: DeviceBuffer<f32>,      // N_LORA_Q
    pub qr_xq: DeviceBuffer<i8>,           // N_LORA_Q
    pub qr_xscale: DeviceBuffer<f32>,      // BLOCKS_N_LORA_Q
    pub q: DeviceBuffer<f32>,              // Q_FLAT
    pub q_normed: DeviceBuffer<f32>,       // Q_FLAT

    // KV chain.
    pub kv_raw: DeviceBuffer<f32>,    // N_HEAD_DIM
    pub kv_normed: DeviceBuffer<f32>, // N_HEAD_DIM

    // Compressor scratch.
    pub kv_cur: DeviceBuffer<f32>,   // 2 * N_HEAD_DIM
    pub sc_cur: DeviceBuffer<f32>,   // 2 * N_HEAD_DIM
    pub pooled: DeviceBuffer<f32>,   // N_HEAD_DIM
    pub comp_row: DeviceBuffer<f32>, // N_HEAD_DIM

    // Attention output + output_proj intermediates.
    pub heads: DeviceBuffer<f32>,         // Q_FLAT
    pub heads_xq: DeviceBuffer<i8>,       // Q_FLAT
    pub heads_xscale: DeviceBuffer<f32>,  // BLOCKS_GROUPED_OUT
    pub low: DeviceBuffer<f32>,           // OUT_LOW
    pub low_xq: DeviceBuffer<i8>,         // OUT_LOW
    pub low_xscale: DeviceBuffer<f32>,    // BLOCKS_OUT_LOW
    pub attn_out: DeviceBuffer<f32>,      // N_EMBD

    // mHC post-attn + pre-ffn.
    pub after_attn_hc: DeviceBuffer<f32>,   // HC_DIM
    pub ffn_cur: DeviceBuffer<f32>,         // N_EMBD
    pub ffn_input_norm: DeviceBuffer<f32>,  // N_EMBD

    // Router.
    pub router_logits: DeviceBuffer<f32>,   // N_EXPERT
    pub router_logits_host: Vec<f32>,
    pub d_selected: DeviceBuffer<i32>,      // N_EXPERT_USED
    pub d_ew: DeviceBuffer<f32>,            // N_EXPERT_USED

    // Shared expert intermediates + output.
    pub gate_sh: DeviceBuffer<f32>,        // N_FF_SHARED
    pub up_sh: DeviceBuffer<f32>,          // N_FF_SHARED
    pub mid_sh: DeviceBuffer<f32>,         // N_FF_SHARED
    pub mid_sh_xq: DeviceBuffer<i8>,       // N_FF_SHARED
    pub mid_sh_xscale: DeviceBuffer<f32>,  // BLOCKS_N_FF_SHARED
    pub ffn_shared: DeviceBuffer<f32>,     // N_EMBD

    // Mailbox: iGPU MoE peer-pushed back to here.
    pub ffn_moe_recv: DeviceBuffer<f32>,   // N_EMBD
}

impl TokenScratch {
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
            kv_cur: DeviceBuffer::new(device_id, (2 * N_HEAD_DIM) as usize)?,
            sc_cur: DeviceBuffer::new(device_id, (2 * N_HEAD_DIM) as usize)?,
            pooled: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,
            comp_row: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,
            heads: DeviceBuffer::new(device_id, Q_FLAT as usize)?,
            heads_xq: DeviceBuffer::new(device_id, Q_FLAT as usize)?,
            heads_xscale: DeviceBuffer::new(device_id, BLOCKS_GROUPED_OUT as usize)?,
            low: DeviceBuffer::new(device_id, OUT_LOW as usize)?,
            low_xq: DeviceBuffer::new(device_id, OUT_LOW as usize)?,
            low_xscale: DeviceBuffer::new(device_id, BLOCKS_OUT_LOW as usize)?,
            attn_out: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            after_attn_hc: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            ffn_cur: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            ffn_input_norm: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            router_logits: DeviceBuffer::new(device_id, N_EXPERT as usize)?,
            router_logits_host: vec![0f32; N_EXPERT as usize],
            d_selected: DeviceBuffer::new(device_id, N_EXPERT_USED)?,
            d_ew: DeviceBuffer::new(device_id, N_EXPERT_USED)?,
            gate_sh: DeviceBuffer::new(device_id, N_FF_SHARED as usize)?,
            up_sh: DeviceBuffer::new(device_id, N_FF_SHARED as usize)?,
            mid_sh: DeviceBuffer::new(device_id, N_FF_SHARED as usize)?,
            mid_sh_xq: DeviceBuffer::new(device_id, N_FF_SHARED as usize)?,
            mid_sh_xscale: DeviceBuffer::new(device_id, BLOCKS_N_FF_SHARED as usize)?,
            ffn_shared: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            ffn_moe_recv: DeviceBuffer::new(device_id, N_EMBD as usize)?,
        })
    }
}

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

    // Split-kernel attention scratch: holds scores out of `attn_score`,
    // then overwritten in place with weights by `attn_softmax_wsum`.
    // Size [N_HEAD, ATTN_MIXED_MAX_KEYS].
    pub attn_scores: DeviceBuffer<f32>,

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

    // M16: router migrated to dGPU. Matvec writes router_logits; topk
    // (or hash router host path) writes d_selected/d_ew. Both are then
    // peer-pushed to iGPU MoE.
    pub router_logits: DeviceBuffer<f32>,
    pub router_logits_host: Vec<f32>,
    pub d_selected: DeviceBuffer<i32>,
    pub d_ew: DeviceBuffer<f32>,
    pub host_selected: Vec<i32>,

    // Head
    pub head_flat: DeviceBuffer<f32>,
    pub head_pre: DeviceBuffer<f32>,
    pub head_w: DeviceBuffer<f32>,
    pub head_embd: DeviceBuffer<f32>,
    pub head_norm: DeviceBuffer<f32>,
    pub head_xq: DeviceBuffer<i8>,
    pub head_xscale: DeviceBuffer<f32>,
    pub logits: DeviceBuffer<f32>,

    /// M40-P1: stash buffers for the layer-major pair-mode forward.
    /// Per layer iteration we run forward_layer for token0 (which
    /// reads `residual` / writes `residual_next`), then we save
    /// `residual_next` into `residual_stash_token0` and load
    /// `residual_stash_token1` (= token1's residual after the previous
    /// layer) into `residual` for token1's forward_layer call. Reuses
    /// the captured graphs since the kernel pointers (`residual` /
    /// `residual_next`) don't change — only their contents do.
    pub residual_stash_token0: DeviceBuffer<f32>,
    pub residual_stash_token1: DeviceBuffer<f32>,
    /// M40-P1: held copy of token0's logits in a pair-forward so they
    /// survive token1's head pass overwriting `logits`.
    pub logits_token0: DeviceBuffer<f32>,

    // ----- M40-P3: per-token stashes for substage-interleaved pair forward -----
    // These persist the pre_moe outputs (after_attn_hc, ffn_input_norm, split)
    // and the post_moe outputs (residual_next) across the interleaved layer
    // flow. token0's pre_moe writes the shared scratch buffers, we stash to
    // _t0; token1's pre_moe overwrites; we stash to _t1; then post_moe restores
    // each token's stash before running shared_expert/ffn_combine for that token.
    // (Same trick as Phase 1's residual_stash but for more buffers.)
    pub after_attn_hc_stash_t0: DeviceBuffer<f32>,
    pub after_attn_hc_stash_t1: DeviceBuffer<f32>,
    pub ffn_input_norm_stash_t0: DeviceBuffer<f32>,
    pub ffn_input_norm_stash_t1: DeviceBuffer<f32>,
    pub split_stash_t0: DeviceBuffer<f32>,
    pub split_stash_t1: DeviceBuffer<f32>,
    pub residual_next_stash_t0: DeviceBuffer<f32>,
    pub residual_next_stash_t1: DeviceBuffer<f32>,
    pub ffn_moe_recv_stash_t0: DeviceBuffer<f32>,
    pub ffn_moe_recv_stash_t1: DeviceBuffer<f32>,
    pub d_selected_stash_t0: DeviceBuffer<i32>,
    pub d_selected_stash_t1: DeviceBuffer<i32>,
    pub d_ew_stash_t0: DeviceBuffer<f32>,
    pub d_ew_stash_t1: DeviceBuffer<f32>,

    /// M40-P4: per-token scratch for forward_pair_interleaved with per-token
    /// streams. Each token's pre_moe + post_moe writes into its own scratch
    /// instance — no aliasing, no stash/restore needed inside a single
    /// layer.
    pub t0: TokenScratch,
    pub t1: TokenScratch,

    // ----- M40-P2: MTP draft scratch (dGPU side) -----
    /// Embedded last token (N_EMBD floats).
    pub mtp_embed: DeviceBuffer<f32>,
    /// `enorm(mtp_embed)` (N_EMBD).
    pub mtp_enorm: DeviceBuffer<f32>,
    /// `e_proj(mtp_enorm)` (N_EMBD).
    pub mtp_eproj: DeviceBuffer<f32>,
    /// `mtp_eproj` repeated across N_HC rows (HC_DIM).
    pub mtp_eproj_hc: DeviceBuffer<f32>,
    /// Per-row RMS norm of prev_hc using `hnorm` (HC_DIM).
    pub mtp_hnorm_hc: DeviceBuffer<f32>,
    /// `h_proj(mtp_hnorm_hc)` per row (HC_DIM).
    pub mtp_hproj_hc: DeviceBuffer<f32>,
    /// `mtp_eproj_hc + mtp_hproj_hc` (HC_DIM) — input residual for the MTP layer.
    pub mtp_input_hc: DeviceBuffer<f32>,
    /// Ping-pong: current MTP HC state, swapping with `mtp_next_hc`
    /// across chained draft iterations.
    pub mtp_state_hc: DeviceBuffer<f32>,
    pub mtp_next_hc: DeviceBuffer<f32>,
    /// MTP head output (N_VOCAB).
    pub mtp_logits: DeviceBuffer<f32>,
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

            attn_scores: DeviceBuffer::new(
                device_id,
                (N_HEAD as usize) * (ATTN_MIXED_MAX_KEYS as usize),
            )?,

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

            // M16: router-on-dGPU scratch.
            router_logits: DeviceBuffer::new(device_id, N_EXPERT as usize)?,
            router_logits_host: vec![0f32; N_EXPERT as usize],
            d_selected: DeviceBuffer::new(device_id, N_EXPERT_USED)?,
            d_ew: DeviceBuffer::new(device_id, N_EXPERT_USED)?,
            host_selected: vec![0i32; N_EXPERT_USED],

            head_flat: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            head_pre: DeviceBuffer::new(device_id, N_HC as usize)?,
            head_w: DeviceBuffer::new(device_id, N_HC as usize)?,
            head_embd: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            head_norm: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            head_xq: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            head_xscale: DeviceBuffer::new(device_id, BLOCKS_N_EMBD as usize)?,
            logits: DeviceBuffer::new(device_id, N_VOCAB as usize)?,

            residual_stash_token0: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            residual_stash_token1: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            logits_token0: DeviceBuffer::new(device_id, N_VOCAB as usize)?,

            after_attn_hc_stash_t0: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            after_attn_hc_stash_t1: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            ffn_input_norm_stash_t0: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            ffn_input_norm_stash_t1: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            split_stash_t0: DeviceBuffer::new(device_id, HC_MIX_DIM as usize)?,
            split_stash_t1: DeviceBuffer::new(device_id, HC_MIX_DIM as usize)?,
            residual_next_stash_t0: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            residual_next_stash_t1: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            ffn_moe_recv_stash_t0: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            ffn_moe_recv_stash_t1: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            d_selected_stash_t0: DeviceBuffer::new(device_id, N_EXPERT_USED)?,
            d_selected_stash_t1: DeviceBuffer::new(device_id, N_EXPERT_USED)?,
            d_ew_stash_t0: DeviceBuffer::new(device_id, N_EXPERT_USED)?,
            d_ew_stash_t1: DeviceBuffer::new(device_id, N_EXPERT_USED)?,

            t0: TokenScratch::alloc(dgpu_device)?,
            t1: TokenScratch::alloc(dgpu_device)?,

            mtp_embed: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            mtp_enorm: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            mtp_eproj: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            mtp_eproj_hc: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            mtp_hnorm_hc: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            mtp_hproj_hc: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            mtp_input_hc: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            mtp_state_hc: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            mtp_next_hc: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            mtp_logits: DeviceBuffer::new(device_id, N_VOCAB as usize)?,
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

    // ----- M40-P3: per-token recv + output buffers for substage pair -----
    // In the substage-interleaved pair forward, we push t0's inputs and
    // immediately start the next dGPU work; iGPU MoE_t0 consumes these
    // buffers asynchronously. If we then push t1 to the SAME recv buffers
    // before MoE_t0 is done reading, MoE_t0 sees t1's data. Per-token
    // recv buffers eliminate this race. Likewise per-token ffn_moe output
    // so the iGPU.xfer push back to dGPU.ffn_moe_recv_t0/t1 doesn't race.
    pub ffn_input_norm_recv_t0: DeviceBuffer<f32>,
    pub ffn_input_norm_recv_t1: DeviceBuffer<f32>,
    pub d_selected_t0: DeviceBuffer<i32>,
    pub d_selected_t1: DeviceBuffer<i32>,
    pub d_ew_t0: DeviceBuffer<f32>,
    pub d_ew_t1: DeviceBuffer<f32>,
    pub ffn_moe_t0: DeviceBuffer<f32>,
    pub ffn_moe_t1: DeviceBuffer<f32>,
    /// M40-P5: per-token router_logits when the router runs on iGPU.
    pub router_logits_t0: DeviceBuffer<f32>,
    pub router_logits_t1: DeviceBuffer<f32>,
    /// Per-token host-readback buffer for hash-router CPU select.
    pub router_logits_host_t0: Vec<f32>,
    pub router_logits_host_t1: Vec<f32>,
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

            ffn_input_norm_recv_t0: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            ffn_input_norm_recv_t1: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            d_selected_t0: DeviceBuffer::new(device_id, N_EXPERT_USED)?,
            d_selected_t1: DeviceBuffer::new(device_id, N_EXPERT_USED)?,
            d_ew_t0: DeviceBuffer::new(device_id, N_EXPERT_USED)?,
            d_ew_t1: DeviceBuffer::new(device_id, N_EXPERT_USED)?,
            ffn_moe_t0: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            ffn_moe_t1: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            router_logits_t0: DeviceBuffer::new(device_id, N_EXPERT as usize)?,
            router_logits_t1: DeviceBuffer::new(device_id, N_EXPERT as usize)?,
            router_logits_host_t0: vec![0f32; N_EXPERT as usize],
            router_logits_host_t1: vec![0f32; N_EXPERT as usize],
        })
    }
}
