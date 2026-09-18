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

use crate::config::{ENGRAM_IN, ENGRAM_OUT, 
    BLOCKS_GROUPED_OUT, BLOCKS_N_EMBD, BLOCKS_N_FF_SHARED, BLOCKS_N_LORA_Q, BLOCKS_OUT_LOW,
    BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, HC_DIM, HC_MIX_DIM, INDEXER_TOP_K, N_EMBD, N_EXPERT,
    N_EXPERT_USED, N_FF_EXP, N_FF_SHARED, N_HC, N_HEAD, N_HEAD_DIM, N_INDEXER_HEAD,
    N_INDEXER_HEAD_DIM, N_LORA_Q, N_VOCAB, OUT_LOW, Q_FLAT, SWA_WINDOW,
};
use crate::attention::ATTN_MIXED_MAX_KEYS;
use crate::q8_k::BLOCK_Q8_K_BYTES;

/// One-hot(copy 0) pre-mix in `split` layout: the initial `pre_mix` of the
/// V4.1 single-pass mHC (ARCH_SPEC §1.1), fed to layer 0's attention collapse.
pub static HC_PRE_ONEHOT: [f32; HC_MIX_DIM as usize] = {
    let mut a = [0.0f32; HC_MIX_DIM as usize];
    a[0] = 1.0;
    a
};

/// `B_MAX` rows of [`HC_PRE_ONEHOT`] for the batched (prefill) layer-0 reset.
pub fn hc_pre_onehot_rows() -> &'static [f32] {
    static ROWS: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
    ROWS.get_or_init(|| HC_PRE_ONEHOT.repeat(super::batch_scratch::B_MAX))
}

/// Rows the batched head handles in one `matvec_bpack` call. Bounds `logits_b`
/// (16 x N_VOCAB x 4 B = 8.3 MB) and mirrors GEMV_BPACK_MAX.
pub const HEAD_BATCH_MAX: usize = 16;

pub struct DgpuScratch {
    // Cross-layer residual
    pub residual: DeviceBuffer<f32>,
    pub residual_next: DeviceBuffer<f32>,

    // mHC stage
    pub flat: DeviceBuffer<f32>,
    pub mix: DeviceBuffer<f32>,
    pub split: DeviceBuffer<f32>,
    /// V4.1 single-pass mHC: the PREVIOUS sub-block's pre-mix, in `split`
    /// layout (only [0..N_HC) meaningful) so `hc_weighted` takes it as-is.
    /// Reset to one-hot(copy 0) at layer 0 of every token; carried across
    /// both sub-blocks of every layer; the head collapses with it (M6).
    pub hc_pre_carry: DeviceBuffer<f32>,
    /// V4.1 Engram (layers 1, 14): one token's dequantised rows `[ENGRAM_IN]` f32
    /// (`stage_engram_rows`), their Q8_0 form, and the wkv output `[ENGRAM_OUT]`.
    /// 32-element stubs when the feature is off.
    pub engram_rows: DeviceBuffer<f32>,
    pub engram_xq: DeviceBuffer<i8>,
    pub engram_xscale: DeviceBuffer<f32>,
    pub engram_kv: DeviceBuffer<f32>,
    pub engram_rows_ready: bool,
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

    // M55: staging for the decode kv window eviction-down copy (the
    // monotonic-append wrap, ~once per 1024 tokens per layer). Same
    // two-hop overlap-safe pattern as prefill's kv_ring_scratch.
    pub kv_wrap_scratch: DeviceBuffer<u16>,

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
    /// V4.1 CSA2 index-K staging (S1a): `wk(latent)` then `k_norm(..)`, both
    /// `N_INDEXER_HEAD_DIM` wide. Separate buffers because the RMSNorm wrapper
    /// requires distinct in/out. Written between the compressor's `rms_w` and its
    /// `rope` — the latent must be read BEFORE the main path rotates it in place.
    pub index_k_row: DeviceBuffer<f32>,
    pub index_k_normed: DeviceBuffer<f32>,

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
    /// ARCH_SPEC 1.5: the candidate blocks layer 20 publishes, consumed by index
    /// sources above it. Sized for the widest store attention can score, since
    /// the block count is `ceil(n_comp / CANDIDATE_BLOCK_SIZE)`.
    pub candidate_block_score: DeviceBuffer<f32>,
    pub candidate_threshold: DeviceBuffer<u32>,
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

    // M56 het-split: dGPU-side MoE scratch for the resident hot experts.
    // Mirrors the iGPU's d_xq_q8k / d_mid_cat / d_midq_cat / ffn_moe.
    pub moe_xq: DeviceBuffer<u8>,
    /// Decode two-box split: request in flight on the remote shard for this
    /// token's layer. Submitted before the local MoE graph, collected at the
    /// combine, so the ~436 us round trip overlaps local compute.
    pub remote_ticket: Option<crate::het::remote_experts::Ticket>,
    /// The `sel` that went with `remote_ticket`, kept so the reply's miss mask
    /// (`proto::RESP_MISS_SHIFT`) can be mapped back to expert ids at the wait
    /// site, where the submit block's `sel_host` is out of scope.
    pub remote_sel: Vec<i32>,
    /// Box 2's weighted partial for this token, `[N_EMBD]` f32.
    pub remote_ffn_moe: Option<DeviceBuffer<f32>>,
    pub remote_ffn_moe_valid: bool,
    pub moe_mid_cat: DeviceBuffer<f32>,
    pub moe_midq_cat: DeviceBuffer<u8>,
    pub ffn_moe_dgpu: DeviceBuffer<f32>,

    // M59: device-resident per-token scalars (written once per token via
    // hipStreamWriteValue32) so rope/append can live inside per-layer
    // graphs: current position, and the monotonic KV append slot
    // (= raw_off + n_raw, identical across layers — counters evolve in
    // lockstep).
    pub pos_dev: DeviceBuffer<u32>,
    pub kv_slot_dev: DeviceBuffer<u32>,

    // Router (lives on dGPU). Matvec writes router_logits; topk (or
    // hash router host path) writes d_selected/d_ew. Both are then
    // peer-pushed to iGPU MoE.
    pub router_logits: DeviceBuffer<f32>,
    pub router_logits_host: Vec<f32>,
    /// M60: backing allocation for d_selected + d_ew (single peer push).
    pub sel_ew_pack: DeviceBuffer<u8>,
    pub d_selected: DeviceBuffer<i32>,
    pub d_ew: DeviceBuffer<f32>,

    // Head
    pub head_flat: DeviceBuffer<f32>,
    pub head_pre: DeviceBuffer<f32>,
    pub head_w: DeviceBuffer<f32>,
    pub head_embd: DeviceBuffer<f32>,
    pub head_norm: DeviceBuffer<f32>,
    pub head_xq: DeviceBuffer<i8>,
    /// Q8_K of head_norm (16×292 B) — Q4_K head via dp4a GEMM@B=1.
    pub head_q8k: DeviceBuffer<u8>,
    /// Q8_K of mid_sh (8×292 B) — Q6_K shexp-down via dp4a GEMM@B=1.
    pub mid_sh_q8k: DeviceBuffer<u8>,
    pub head_xscale: DeviceBuffer<f32>,
    pub logits: DeviceBuffer<f32>,
    /// BATCHED head. Prep (hc_weighted / rms_w / quantize) stays PER ROW so the
    /// numerics are untouched; only the vocab matvec is batched, via
    /// `matvec_bpack`, which reads the ~700 MB tied projection once for all rows
    /// instead of once per row.
    pub head_xq_b: DeviceBuffer<i8>,
    pub head_xscale_b: DeviceBuffer<f32>,
    pub logits_b: DeviceBuffer<f32>,

    // Sampler scratch (see crate::sampler). partials_max / partials_z
    // hold per-WG reductions consumed by softmax_sample_one. u01 is a
    // 1-element f32 the host writes per decode step. next_token_id is
    // the kernel's only output — a 1-element i32 the host reads back
    // after sampling.
    pub sampler_partials_max: DeviceBuffer<f32>,
    pub sampler_partials_z: DeviceBuffer<f32>,
    pub sampler_u01: DeviceBuffer<f32>,
    pub sampler_next_token_id: DeviceBuffer<i32>,
    /// top-p threshold search scratch. `mass` is [N_WG * TOPP_NEDGE] f32
    /// (8.4 KB), `bracket` is the [s_lo, s_hi] log-space bracket and `thr`
    /// the published survivor cutoff. Untouched when top_p >= 1.
    pub sampler_topp_mass: DeviceBuffer<f32>,
    pub sampler_topp_bracket: DeviceBuffer<f32>,
    pub sampler_topp_thr: DeviceBuffer<f32>,
}

impl DgpuScratch {
    pub fn alloc(dgpu_device: Device) -> eyre::Result<Self> {
        dgpu_device.set_current()?;
        let device_id = dgpu_device.id;
        // M60: selected+ew in ONE allocation → single 48-B peer push on the
        // router→iGPU handoff. Layout: [0..24) i32 selected | [24..48) f32 ew.
        let sel_ew_pack = DeviceBuffer::<u8>::new(device_id, 64)?;
        // SAFETY: offsets 0/24 are 4-aligned; producers (router_topk) and
        // consumers agree on the layout; pack lives in the same struct.
        let d_selected = unsafe { sel_ew_pack.view_as::<i32>(0, N_EXPERT_USED) };
        let d_ew = unsafe { sel_ew_pack.view_as::<f32>(24, N_EXPERT_USED) };
        let eg = |n: u32| -> usize { if cfg!(feature = "v41") { n as usize } else { 32 } };
        Ok(Self {
            residual: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            residual_next: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            flat: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            mix: DeviceBuffer::new(device_id, HC_MIX_DIM as usize)?,
            split: DeviceBuffer::new(device_id, HC_MIX_DIM as usize)?,
            engram_rows: DeviceBuffer::new(device_id, eg(ENGRAM_IN))?,
            engram_xq: DeviceBuffer::new(device_id, eg(ENGRAM_IN))?,
            engram_xscale: DeviceBuffer::new(device_id, eg(ENGRAM_IN / 32))?,
            engram_kv: DeviceBuffer::new(device_id, eg(ENGRAM_OUT))?,
            engram_rows_ready: false,
            hc_pre_carry: {
                let mut b = DeviceBuffer::new(device_id, HC_MIX_DIM as usize)?;
                b.copy_from_host(&HC_PRE_ONEHOT)?;
                b
            },
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
            kv_wrap_scratch: DeviceBuffer::new(
                device_id,
                (SWA_WINDOW as usize) * (N_HEAD_DIM as usize),
            )?,
            rms_nw_partials: DeviceBuffer::new(device_id, 64)?,
            rms_nw_inv_scalar: DeviceBuffer::new(device_id, 1)?,
            // 64 × HC_MIX_DIM=24 = 1536 f32 = 6 KB. n_k_split=32 uses half.
            mhc_matvec_partials: DeviceBuffer::new(device_id, 64 * (HC_MIX_DIM as usize))?,

            kv_cur: DeviceBuffer::new(device_id, (2 * N_HEAD_DIM) as usize)?,
            sc_cur: DeviceBuffer::new(device_id, (2 * N_HEAD_DIM) as usize)?,
            pooled: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,
            comp_row: DeviceBuffer::new(device_id, N_HEAD_DIM as usize)?,
            index_k_row: DeviceBuffer::new(device_id, N_INDEXER_HEAD_DIM as usize)?,
            index_k_normed: DeviceBuffer::new(device_id, N_INDEXER_HEAD_DIM as usize)?,

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
            candidate_block_score: DeviceBuffer::new(
                device_id,
                (crate::attention::ATTN_MIXED_MAX_KEYS as usize)
                    .div_ceil(crate::config::CANDIDATE_BLOCK_SIZE as usize),
            )?,
            candidate_threshold: DeviceBuffer::new(device_id, 1)?,
            indexer_topk_scratch: {
                // Two-level bitonic tree merge: L0 = max_chunks*top_k
                // candidates, plus L1 = n_groups*top_k regrouped candidates
                // (group = 4096/top_k chunks). Covers n_comp up to
                // ATTN_MIXED_MAX_KEYS without the old 32768 merge-cap wall.
                let max_chunks = (ATTN_MIXED_MAX_KEYS + 4095) / 4096;
                let group_chunks = 4096 / INDEXER_TOP_K;
                let n_groups = (max_chunks + group_chunks - 1) / group_chunks;
                DeviceBuffer::new(device_id, ((max_chunks + n_groups) * INDEXER_TOP_K) as usize)?
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
            remote_ticket: None,
            remote_sel: Vec::new(),
            remote_ffn_moe: if std::env::var("V41_REMOTE_ADDR").is_ok() {
                Some(DeviceBuffer::new(device_id, N_EMBD as usize)?)
            } else {
                None
            },
            remote_ffn_moe_valid: false,
            moe_xq: DeviceBuffer::new(
                device_id,
                (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_Q8_K_BYTES,
            )?,
            moe_mid_cat: DeviceBuffer::new(
                device_id,
                (N_EXPERT_USED) * (N_FF_EXP as usize),
            )?,
            moe_midq_cat: DeviceBuffer::new(
                device_id,
                (N_EXPERT_USED) * (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES,
            )?,
            ffn_moe_dgpu: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            pos_dev: DeviceBuffer::new(device_id, 1)?,
            kv_slot_dev: DeviceBuffer::new(device_id, 1)?,

            // Router (dGPU-resident).
            router_logits: DeviceBuffer::new(device_id, N_EXPERT as usize)?,
            router_logits_host: vec![0f32; N_EXPERT as usize],
            sel_ew_pack,
            d_selected,
            d_ew,

            head_flat: DeviceBuffer::new(device_id, HC_DIM as usize)?,
            head_pre: DeviceBuffer::new(device_id, N_HC as usize)?,
            head_w: DeviceBuffer::new(device_id, N_HC as usize)?,
            head_xq_b: DeviceBuffer::new(device_id, HEAD_BATCH_MAX * N_EMBD as usize)?,
            head_xscale_b: DeviceBuffer::new(device_id, HEAD_BATCH_MAX * (N_EMBD as usize / 32))?,
            logits_b: DeviceBuffer::new(device_id, HEAD_BATCH_MAX * N_VOCAB as usize)?,
            head_embd: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            head_norm: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            head_xq: DeviceBuffer::new(device_id, N_EMBD as usize)?,
            head_q8k: DeviceBuffer::new(device_id, 16 * 292)?,
            mid_sh_q8k: DeviceBuffer::new(device_id, 8 * 292)?,
            head_xscale: DeviceBuffer::new(device_id, BLOCKS_N_EMBD as usize)?,
            logits: DeviceBuffer::new(device_id, N_VOCAB as usize)?,

            sampler_partials_max: DeviceBuffer::new(device_id, crate::sampler::SAMPLER_N_WG as usize)?,
            sampler_partials_z: DeviceBuffer::new(device_id, crate::sampler::SAMPLER_N_WG as usize)?,
            sampler_u01: DeviceBuffer::new(device_id, 1)?,
            sampler_next_token_id: DeviceBuffer::new(device_id, 1)?,
            sampler_topp_mass: DeviceBuffer::new(
                device_id,
                crate::sampler::sampler_topp_mass_len(),
            )?,
            sampler_topp_bracket: DeviceBuffer::new(device_id, 2)?,
            sampler_topp_thr: DeviceBuffer::new(device_id, 1)?,
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
    /// M60: backing allocation for d_selected + d_ew (single peer push).
    pub sel_ew_pack: DeviceBuffer<u8>,
    pub d_ew: DeviceBuffer<f32>,
    pub d_selected: DeviceBuffer<i32>,
    pub ffn_moe: DeviceBuffer<f32>,
}

impl IgpuScratch {
    pub fn alloc(igpu_device: Device) -> eyre::Result<Self> {
        igpu_device.set_current()?;
        let device_id = igpu_device.id;
        // M60: mirror of DgpuScratch::sel_ew_pack (single 48-B peer push).
        let sel_ew_pack = DeviceBuffer::<u8>::new(device_id, 64)?;
        // SAFETY: see DgpuScratch counterpart.
        let d_selected = unsafe { sel_ew_pack.view_as::<i32>(0, N_EXPERT_USED) };
        let d_ew = unsafe { sel_ew_pack.view_as::<f32>(24, N_EXPERT_USED) };
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
            sel_ew_pack,
            d_ew,
            d_selected,
            ffn_moe: DeviceBuffer::new(device_id, N_EMBD as usize)?,
        })
    }
}
