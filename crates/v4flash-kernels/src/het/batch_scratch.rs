//! Per-batch scratch for prefill.
//!
//! * [`BatchDgpuScratch`] / [`BatchIgpuScratch`] — every per-token field
//!   is B-extended to a single contiguous `[B × per_token_size]` buffer.
//!   Batched kernels (`*_batched`) read/write with per-batch strides
//!   directly, no per-token copies. Used by `forward_prompt_batch_v2` and
//!   the pipelined wrapper.
//! * [`BatchScratch`] — bundle of (shared `DgpuScratch`, shared
//!   `IgpuScratch`, B per-token residual buffers). Test-only convenience
//!   container retained for diagnostic tests; no production caller.
//!
//! Memory cost at `B_MAX = 512`: `BatchDgpuScratch` ~208 MB on dGPU.

use color_eyre::eyre;
use v4flash_hip::{Device, DeviceBuffer};

use crate::attention::{ATTN_MIXED_MAX_KEYS, ATTN_SCORES_STRIDE};
use crate::config::{
    BLOCKS_GROUPED_OUT, BLOCKS_N_EMBD, BLOCKS_N_FF_SHARED, BLOCKS_N_LORA_Q, BLOCKS_OUT_LOW,
    BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, HC_DIM, HC_MIX_DIM, N_EMBD, N_EXPERT, N_EXPERT_USED,
    N_FF_EXP, N_FF_SHARED, N_HEAD, N_HEAD_DIM, N_LORA_Q, OUT_LOW, Q_FLAT,
};
use crate::q8_k::BLOCK_Q8_K_BYTES;

use super::scratch::{DgpuScratch, IgpuScratch};

/// Max prefill batch size. At 512 (vs 256): larger batches grow the hot
/// expert chunks under the skewed (Zipf-y) routing, so the staged iq2
/// path amortizes better over longer member lists — measured +9.4% e2e
/// prefill at depth 1024 (186.6 → 204.2 tok/s, back-to-back). The big
/// scratch growth is `attn_scores` (~+580 MB at f16 / ~290 MB after
/// halving for f16 scores) which is dGPU-resident (16 GiB 9070 XT,
/// ample), NOT on the tight iGPU expert budget; iGPU scratch growth is
/// ~24→48 MB.
pub const B_MAX: usize = 1024;

/// Diagnostic toggle (env `DEEPSTRIX_F32_SCORES=1`): route the batched-
/// prefill attention through the **f32-scores** kernel pair instead of
/// the production f16-scores one. Used to test whether long-ctx
/// accuracy degradation is caused by f16 quantization of pre-softmax
/// logits (`_f16s` writes scores as f16, losing ~3 mantissa digits per
/// logit — bad near softmax ties). When on:
///   * `BatchDgpuScratch::attn_scores` is allocated at *full* f32 size
///     (B_MAX × N_HEAD × ATTN_MIXED_MAX_KEYS f32 elements = ~3.2 GiB at
///     B_MAX=512), instead of the half-sized f16-byte-equivalent layout.
///   * `forward_prefill` dispatches `launch_score_batched_htiled_wmma`
///     + `launch_softmax_wsum_batched_htiled_wmma_ldsv` (both read/write
///     f32) instead of the `_f16s` siblings.
/// Read once at process start; flipping the env var mid-run does
/// nothing.
pub fn use_f32_scores() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("DEEPSTRIX_F32_SCORES")
            .ok()
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}

/// Test-only convenience bundle of (shared `DgpuScratch`,
/// shared `IgpuScratch`, B per-token residual buffers).
///
/// The shared single-token scratches are reused so captured HIP graphs
/// replay consistently — captures bake in scratch buffer pointers, so
/// a fresh scratch per batch element would break replay. KV cache +
/// compressor state live in `HetModelState` (per-layer, shared across
/// the batch).
///
/// Production prefill (`forward_prompt_batch_v2` /
/// `forward_prefill_pipelined`) uses [`BatchDgpuScratch`] instead — no
/// per-token residual ping-pong, no shared single-token scratch.
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
        use crate::config::HC_DIM;
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

/// Per-batch dGPU scratch with B-extended buffers.
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
    /// Scratch ring (≥ `SWA_WINDOW * N_HEAD_DIM`) used by the post-chunk
    /// SWA-eviction pass. After a prefill chunk leaves the cache holding
    /// `n_raw_during_chunk > SWA_WINDOW` rows in the oversized region, we copy
    /// the LAST `SWA_WINDOW` rows here, then copy them back to slots `[0..W)`
    /// of `ls.kv_cache` — two non-overlapping device-to-device copies on the
    /// compute stream. The intermediate buffer makes the shift race-free
    /// even when the source/destination regions in the cache overlap.
    /// `u16` because `ls.kv_cache` is f16-stored.
    pub kv_ring_scratch: DeviceBuffer<u16>,

    // ---- Compressor (used in per-batch serial inner loop) ----
    pub kv_cur: DeviceBuffer<f32>,
    pub sc_cur: DeviceBuffer<f32>,
    pub pooled: DeviceBuffer<f32>,
    pub comp_row: DeviceBuffer<f32>,

    // ---- Per-token attention causality ----
    /// `[B]` — per-token causal prefix length over raw KV cache. Batched
    /// per-position kernels read these arrays (uploaded per chunk/layer).
    /// `pos_per_b[b] = pos0 + b` (constant per chunk).
    /// `row_per_b[b]`, `pos_mod_per_b[b]` depend on ratio (computed per layer).
    pub pos_per_b: DeviceBuffer<i32>,
    pub row_per_b: DeviceBuffer<i32>,
    pub pos_mod_per_b: DeviceBuffer<i32>,
    pub n_raw_per: DeviceBuffer<i32>,
    /// `[B]` — per-token starting slot offset into the (oversized) raw KV
    /// cache. Cache holds `n_raw_before + b` rows during a prefill chunk;
    /// token i's causally-valid window is rows
    /// `[n_raw_offset_per[i] .. n_raw_offset_per[i] + n_raw_per[i])`.
    /// Outside prefill chunks (decode), offsets are 0 and the cache is
    /// the steady-state SWA_WINDOW prefix only.
    pub n_raw_offset_per: DeviceBuffer<i32>,
    /// `[B]` — per-token causal prefix length over comp_kv (0 for ratio=0
    /// layers and for tokens before the first comp boundary).
    pub n_comp_per: DeviceBuffer<i32>,

    // ---- Attention output + output projection ----
    pub heads: DeviceBuffer<f32>,
    /// `[B, n_head, ATTN_MIXED_MAX_KEYS]` — global scores/weights scratch for
    /// the batched split attention kernels (long-context prefill). Replaces
    /// the monolithic kernel's LDS `scores[2304]` (which overflows past ~9K
    /// tokens). ~1.1 GiB at B_MAX=256.
    pub attn_scores: DeviceBuffer<f32>,
    /// `[B, ceil(ATTN_MIXED_MAX_KEYS/32)]` — bitpacked CSA mask passed
    /// to the score kernel. Bit `(c & 31)` of word
    /// `b * max_keys_words + (c >> 5)` is 1 iff comp row c is allowed
    /// for batch token b. Only populated at ratio==4 layers where some
    /// token in the chunk has `n_index_comp > INDEXER_TOP_K`; otherwise
    /// the kernel sees `None` and runs dense.
    pub attn_comp_allowed_bits: DeviceBuffer<u32>,
    // ---- CSA indexer per-token scratch ----
    /// `[B, N_INDEXER_HEAD * N_INDEXER_HEAD_DIM]` — per-token indexer Q
    /// (matvec(attn_q_b) output before RoPE). Allocated full B-batch so
    /// we can write all tokens with one batched matvec; the per-token
    /// IndexerScore + IndexerTopk passes read slices.
    pub indexer_q: DeviceBuffer<f32>,
    /// `[B, N_INDEXER_HEAD]` — per-token head_weights (matvec(proj)
    /// output, post-scale).
    pub indexer_head_weights: DeviceBuffer<f32>,
    /// `[B, ATTN_MIXED_MAX_KEYS]` — batched IndexerScore output. Stride
    /// per token = MAX_KEYS. ~48 MB at B_MAX=512.
    pub indexer_scores: DeviceBuffer<f32>,
    /// `[B, INDEXER_TOP_K]` — batched IndexerTopk output (selected
    /// indices per token, sentinel -1 in unused slots). ~1 MB at B=512.
    pub indexer_selected: DeviceBuffer<i32>,
    /// `[B, max_chunks * INDEXER_TOP_K]` — batched bitonic topk per-chunk
    /// candidates scratch. max_chunks = ceil(MAX_KEYS/4096) = 6. ~6 MB.
    pub indexer_topk_scratch: DeviceBuffer<u32>,
    /// `[B]` — per-token n_index_comp uploaded once per ratio==4 layer.
    pub n_index_comp_per_b: DeviceBuffer<u32>,
    /// `[B, INDEXER_TOP_K, N_HEAD_DIM]` f16 — per-batch gathered comp_kv
    /// rows for the CSA sparse-attention path. Populated by
    /// `IndexerGather::launch_batched` when the indexer fires. Passed to
    /// score+smwsum with `comp_kv_batch_stride = INDEXER_TOP_K` so the
    /// kernels read only the dense top-K rows per batch instead of doing
    /// per-row mask tests on the full sparse n_comp. ~256 MB at B_MAX=512.
    pub attn_active_comp_kv: DeviceBuffer<u16>,
    /// `[n_boundaries_max × coff × ratio × width]` f32 — per-boundary
    /// state_kv snapshots taken at each boundary firing in the prefill
    /// compressor loop. Reused across main + indexer compressors (only
    /// one is active at a time). Sized for the largest case (ratio==4
    /// main: 128 boundaries × 8 × 1024 = 4 MB).
    pub comp_state_kv_snapshots: DeviceBuffer<f32>,
    pub comp_state_score_snapshots: DeviceBuffer<f32>,
    /// `[n_boundaries_max, head_dim]` f32 — pool output per boundary.
    /// Feeds rms_w_batched which writes its result into `comp_rows_batched`.
    pub comp_pooled_batched: DeviceBuffer<f32>,
    /// `[n_boundaries_max, head_dim]` f32 — post-rms_w, then rope/fp8/
    /// f16rt-ed in place. Final values appended to comp_kv via
    /// comp_kv_append_batched. Max 128 × 512 = 256 KB.
    pub comp_rows_batched: DeviceBuffer<f32>,
    /// `[n_boundaries_max]` i32 — per-boundary RoPE positions, uploaded
    /// once per layer×compressor for the batched rope launch.
    pub comp_pos_per_boundary: DeviceBuffer<i32>,
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

    /// `[B, N_EMBD]` — peer-arrival mailbox for iGPU MoE output. Filled
    /// by a single batched peer-push from iGPU; then vec_add ffn_shared
    /// and run hc_post.
    pub ffn_moe_recv: DeviceBuffer<f32>,

    /// M61 prefill het-split: dGPU-side MoE scratch for the hot-expert leg.
    /// `Some` only when `DGPU_HOT_EXPERTS > 0` (~280 MB at B_MAX=1024).
    pub hot: Option<BatchDgpuHotScratch>,
}

/// Static work-item geometry for the dGPU hot-expert prefill leg.
/// Members per hot expert are capped at B_MAX, so each expert needs at
/// most `B_MAX / HOT_CHUNK` chunks. The work-items list is e-major and
/// uploaded ONCE; per-layer launches set grid.y = n_hot × chunks and the
/// matvec kernels' `member_end <= member_start` guard early-exits empty
/// chunks — no per-layer host readback of n_work_items on de.compute.
pub const HOT_MAX_EXPERTS: usize = 256;
pub const HOT_CHUNK: usize = 32;
pub const HOT_CHUNKS_PER_EXPERT: usize = B_MAX / HOT_CHUNK;

/// M61 prefill het-split: dGPU scratch mirroring the iGPU batched MoE
/// pipeline, sized for the hot-expert (resident) slots only. The matvec
/// kernels are the SAME kwide/kwide2 kernels the iGPU runs — only the
/// group builder (dense ids, hits only) and the reduce (own-slots only)
/// differ.
///
/// VRAM: the five big buffers (~280 MB at B_MAX=1024) are non-owning
/// VIEWS into `attn_active_comp_kv` (537 MB), which is idle for the
/// entire MoE phase. ALIASING CONTRACT: comp_kv is written (IndexerGather)
/// and read (attention) strictly BEFORE the router/MoE stages of the same
/// layer, and the hot leg's last read of these views (the ffn_combine
/// vec_add of `ffn_moe_dgpu`) is enqueued before the next layer's
/// attention — all on the single de.compute stream, so reuse is
/// serial-safe. Per-layer comp_kv dirtying can leave NaN bit patterns in
/// the miss slots of `mid_cat`; the post-iq2 q8k quantize reads them but
/// the resulting q8 blocks are never dotted (work items cover hot members
/// only), so that is harmless.
pub struct BatchDgpuHotScratch {
    /// `[B, BLOCKS_Q8K_GATE_IN × 292]` — q8_K-quantized ffn_input_norm.
    /// Bit-identical to the iGPU's d_xq_q8k (same f32 input, same kernel).
    /// View into attn_active_comp_kv.
    pub moe_xq: DeviceBuffer<u8>,
    /// `[B, n_used, N_FF_EXP]` f32 — fused-swiglu mid, hit slots only.
    /// View into attn_active_comp_kv.
    pub mid_cat: DeviceBuffer<f32>,
    /// `[B, n_used, BLOCKS_Q8K_DOWN_IN × 292]` — quantized mid.
    /// View into attn_active_comp_kv.
    pub midq_cat: DeviceBuffer<u8>,
    /// `[B*n_used, N_EMBD]` f32 — q2k by-expert partials, hit slots only.
    /// View into attn_active_comp_kv.
    pub partials: DeviceBuffer<f32>,
    /// `[B, N_EMBD]` f32 — hetsplit-reduced dGPU MoE partial; added to
    /// ffn_moe_recv at ffn_combine. View into attn_active_comp_kv.
    pub ffn_moe_dgpu: DeviceBuffer<f32>,
    /// `[HOT_MAX_EXPERTS]` i32 — DENSE-id group counts. Zeroed per layer
    /// via fill_zero_async on de.compute. Owned.
    pub group_count: DeviceBuffer<i32>,
    /// `[HOT_MAX_EXPERTS × B_MAX]` i32 — dense-id member lists,
    /// `(b<<16)|slot` packed like the iGPU arrays. Owned.
    pub expert_members: DeviceBuffer<i32>,
    /// `[HOT_MAX_EXPERTS × HOT_CHUNKS_PER_EXPERT]` i32 — STATIC e-major
    /// work items `(e<<16)|(c×HOT_CHUNK)`, uploaded once at alloc. Owned.
    pub work_items_static: DeviceBuffer<i32>,
}

impl BatchDgpuHotScratch {
    /// Carve the big buffers out of `comp_kv` (see ALIASING CONTRACT on
    /// the struct). `comp_kv`'s device memory is owned by the parent
    /// `BatchDgpuScratch` and never reallocated, so the views stay valid
    /// for the parent's lifetime.
    fn alloc(id: i32, comp_kv: &DeviceBuffer<u16>) -> eyre::Result<Self> {
        let b = B_MAX;
        let xq_len = b * (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_Q8_K_BYTES;
        let midq_len =
            b * (N_EXPERT_USED as usize) * (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES;
        let mid_len = b * (N_EXPERT_USED as usize) * (N_FF_EXP as usize);
        let partials_len = b * (N_EXPERT_USED as usize) * (N_EMBD as usize);
        let out_len = b * N_EMBD as usize;

        // f32 views first (offset stays 4-aligned), byte views after.
        let mut off = 0usize;
        let mut take = |bytes: usize| {
            let o = off;
            off += (bytes + 255) & !255; // 256-B align each region
            o
        };
        let partials = unsafe { comp_kv.view_as::<f32>(take(partials_len * 4), partials_len) };
        let ffn_moe_dgpu = unsafe { comp_kv.view_as::<f32>(take(out_len * 4), out_len) };
        let mid_cat = unsafe { comp_kv.view_as::<f32>(take(mid_len * 4), mid_len) };
        let moe_xq = unsafe { comp_kv.view_as::<u8>(take(xq_len), xq_len) };
        let midq_cat = unsafe { comp_kv.view_as::<u8>(take(midq_len), midq_len) };

        let mut work_items_static: DeviceBuffer<i32> =
            DeviceBuffer::new(id, HOT_MAX_EXPERTS * HOT_CHUNKS_PER_EXPERT)?;
        let mut wi_host = vec![0i32; HOT_MAX_EXPERTS * HOT_CHUNKS_PER_EXPERT];
        for e in 0..HOT_MAX_EXPERTS {
            for c in 0..HOT_CHUNKS_PER_EXPERT {
                wi_host[e * HOT_CHUNKS_PER_EXPERT + c] =
                    ((e as i32) << 16) | ((c * HOT_CHUNK) as i32);
            }
        }
        work_items_static.copy_from_host(&wi_host)?;
        Ok(Self {
            moe_xq,
            mid_cat,
            midq_cat,
            partials,
            ffn_moe_dgpu,
            group_count: DeviceBuffer::new(id, HOT_MAX_EXPERTS)?,
            expert_members: DeviceBuffer::new(id, HOT_MAX_EXPERTS * b)?,
            work_items_static,
        })
    }
}

/// Per-batch iGPU scratch with B-extended buffers.
///
/// Mirrors [`IgpuScratch`]'s MoE-relevant fields, each sized for
/// `B_MAX × per_token_size`. Used by the batched iGPU MoE in
/// `forward_layer_batch_v2`: one peer-push of `[B, N_EMBD]` ain, one
/// batched iq2 + q2k call chain, one peer-push of `[B, N_EMBD]` ffn_moe
/// back. Total ~6 MB at B_MAX=64 — negligible vs the 52 GiB resident
/// expert weights.
pub struct BatchIgpuScratch {
    pub ffn_input_norm_recv: DeviceBuffer<f32>,
    pub d_xq_q8k: DeviceBuffer<u8>,
    pub d_mid_cat: DeviceBuffer<f32>,
    pub d_midq_cat: DeviceBuffer<u8>,
    pub ffn_moe: DeviceBuffer<f32>,
    pub d_selected: DeviceBuffer<i32>,
    pub d_ew: DeviceBuffer<f32>,
    /// By-expert MoE: per-expert pick count built by the group_builder
    /// pre-pass. `[n_expert]` i32. MUST be zeroed before each layer's pre-pass.
    pub group_count: DeviceBuffer<i32>,
    /// By-expert MoE: per-expert (b, slot) member lists, packed as
    /// `(b << 16) | slot`. `[n_expert × max_per_expert]` i32. Only the first
    /// `group_count[e]` entries per expert are valid after the pre-pass.
    /// `max_per_expert = B_MAX` (worst case: every token picks the same expert
    /// in some slot — still fits since each token contributes ≤ n_used picks).
    pub expert_members: DeviceBuffer<i32>,
    /// Chunked by-expert: flat list of (expert_id<<16 | member_start)
    /// work items built by moe_work_items_builder. Sized for worst case
    /// = `B_MAX * n_used / CHUNK_SIZE + n_expert` (each active expert may
    /// have one extra ceiling chunk).
    pub work_items: DeviceBuffer<i32>,
    /// Chunked by-expert: `[1]` i32. Count of valid entries in
    /// `work_items[]`. MUST be zeroed before each layer's pre-pass.
    /// Read back to host (sync) to set grid.y for the main kernel.
    pub n_work_items: DeviceBuffer<i32>,
    /// Hybrid dispatch: work items for the staged kernel (chunks ≥ threshold).
    /// Same shape as `work_items`. MUST be paired with `n_staged_work_items`.
    pub staged_work_items: DeviceBuffer<i32>,
    /// Hybrid dispatch: work items for the chunked kernel (chunks < threshold).
    /// Same shape as `work_items`. MUST be paired with `n_chunked_work_items`.
    pub chunked_work_items: DeviceBuffer<i32>,
    /// Hybrid dispatch: `[1]` i32 atomic counters, pre-zeroed per layer.
    pub n_staged_work_items: DeviceBuffer<i32>,
    pub n_chunked_work_items: DeviceBuffer<i32>,
    /// Q2K_VARIANT=by_expert: per-(b, slot, row) partial sums written by
    /// `q2_k_matvec_par_by_expert`, then summed across `n_used` by
    /// `q2_k_reduce_partials`. `[B*n_used, N_EMBD]` f32 = 48 MiB at B=512.
    /// Avoids the atomicAdd nondeterminism of an in-place accumulation.
    pub q2k_partials: DeviceBuffer<f32>,
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
            // Worst case work items: every member could be its own chunk
            // (CHUNK_SIZE=1 degenerate), so size for B*n_used. In practice
            // at CHUNK_SIZE=16 we use far less.
            work_items: DeviceBuffer::new(
                id,
                (N_EXPERT as usize) + b * (N_EXPERT_USED as usize),
            )?,
            n_work_items: DeviceBuffer::new(id, 1)?,
            q2k_partials: DeviceBuffer::new(
                id,
                b * (N_EXPERT_USED as usize) * (N_EMBD as usize),
            )?,
            // Hybrid dispatch: two extra work_items arrays, each sized for the
            // worst case (all items land in one bucket).
            staged_work_items: DeviceBuffer::new(
                id,
                (N_EXPERT as usize) + b * (N_EXPERT_USED as usize),
            )?,
            chunked_work_items: DeviceBuffer::new(
                id,
                (N_EXPERT as usize) + b * (N_EXPERT_USED as usize),
            )?,
            n_staged_work_items: DeviceBuffer::new(id, 1)?,
            n_chunked_work_items: DeviceBuffer::new(id, 1)?,
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
        let mk_u16 =
            |n: usize| -> eyre::Result<DeviceBuffer<u16>> { DeviceBuffer::new(id, b * n) };
        let mk_i8 = |n: usize| -> eyre::Result<DeviceBuffer<i8>> { DeviceBuffer::new(id, b * n) };
        let mk_i32 =
            |n: usize| -> eyre::Result<DeviceBuffer<i32>> { DeviceBuffer::new(id, b * n) };
        // Built before the literal so the M61 hot-expert scratch can carve
        // its views out of it (see BatchDgpuHotScratch ALIASING CONTRACT).
        let attn_active_comp_kv = mk_u16(
            (crate::config::INDEXER_TOP_K * crate::config::N_HEAD_DIM) as usize,
        )?;
        let hot = {
            let hot_k: usize = std::env::var("DGPU_HOT_EXPERTS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            if hot_k > 0 {
                Some(BatchDgpuHotScratch::alloc(id, &attn_active_comp_kv)?)
            } else {
                None
            }
        };
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
            // b*N_HEAD_DIM = B_MAX*N_HEAD_DIM ≥ SWA_WINDOW*N_HEAD_DIM.
            kv_ring_scratch: mk_u16(N_HEAD_DIM as usize)?,

            kv_cur: mk_f32((2 * N_HEAD_DIM) as usize)?,
            sc_cur: mk_f32((2 * N_HEAD_DIM) as usize)?,
            pooled: mk_f32(N_HEAD_DIM as usize)?,
            comp_row: mk_f32(N_HEAD_DIM as usize)?,

            pos_per_b: mk_i32(1)?,
            row_per_b: mk_i32(1)?,
            pos_mod_per_b: mk_i32(1)?,
            n_raw_per: mk_i32(1)?,
            n_raw_offset_per: mk_i32(1)?,
            n_comp_per: mk_i32(1)?,

            heads: mk_f32(Q_FLAT as usize)?,
            // Half-sized by default: the f16-scores kernel writes f16
            // into this buffer (see launch_score_batched_htiled_wmma_f16s
            // in attention.rs). N_HEAD * ATTN_MIXED_MAX_KEYS / 2 f32
            // elements ≡ N_HEAD * ATTN_MIXED_MAX_KEYS f16 elements ≡
            // same byte budget as the f32-scores layout would have used.
            //
            // Doubled when DEEPSTRIX_F32_SCORES=1 so the f32-scores
            // kernel pair has the headroom it needs.
            attn_scores: {
                // Sized at ATTN_SCORES_STRIDE (not ATTN_MIXED_MAX_KEYS): the
                // production batched attention runs on the CSA-gathered dense
                // top-K buffer so n_total ≤ ~640 keys. 2048 stride leaves
                // headroom and unblocks B_MAX > 512.
                let per_b = if use_f32_scores() {
                    (N_HEAD * ATTN_SCORES_STRIDE) as usize
                } else {
                    ((N_HEAD * ATTN_SCORES_STRIDE) / 2) as usize
                };
                mk_f32(per_b)?
            },
            // CSA mask scratch. Per-batch bitmap [B, ceil(MAX/32)] u32.
            // For B_MAX=512 and ATTN_MIXED_MAX_KEYS=24576: 512 × 768 = 1.5 MB.
            attn_comp_allowed_bits: {
                let words_per_b = (ATTN_MIXED_MAX_KEYS as usize + 31) / 32;
                DeviceBuffer::new(id, b * words_per_b)?
            },
            // Per-token Indexer Q [N_INDEXER_HEAD * N_INDEXER_HEAD_DIM].
            // mk_f32 multiplies by B internally → 64×128×512 = 16 MB.
            indexer_q: mk_f32((crate::config::N_INDEXER_HEAD
                * crate::config::N_INDEXER_HEAD_DIM) as usize)?,
            // Per-token head_weights [N_INDEXER_HEAD] → 64×512 = 128 KB.
            indexer_head_weights: mk_f32(crate::config::N_INDEXER_HEAD as usize)?,
            // Batched IndexerScore: [B, MAX_KEYS] f32 = ~48 MB at B=512.
            indexer_scores: DeviceBuffer::new(id, b * ATTN_MIXED_MAX_KEYS as usize)?,
            // Batched IndexerTopk selected: [B, top_k] = ~1 MB.
            indexer_selected: DeviceBuffer::new(
                id,
                b * crate::config::INDEXER_TOP_K as usize,
            )?,
            indexer_topk_scratch: {
                let max_chunks = (ATTN_MIXED_MAX_KEYS + 4095) / 4096;
                DeviceBuffer::new(
                    id,
                    b * (max_chunks * crate::config::INDEXER_TOP_K) as usize,
                )?
            },
            n_index_comp_per_b: DeviceBuffer::new(id, b)?,
            attn_active_comp_kv,
            // Per-boundary state snapshots scratch. Sized for the largest
            // compressor (main ratio==4: 8 × 1024 f32 = 32 KB) × max boundaries
            // (B_MAX/4 = 128) = 4 MB per buffer. Reused across compressors.
            comp_state_kv_snapshots: {
                let max_state_per_b = 8 * (2 * crate::config::N_HEAD_DIM) as usize; // ratio*coff * width
                let max_boundaries = B_MAX / 4;
                DeviceBuffer::new(id, max_boundaries * max_state_per_b)?
            },
            comp_state_score_snapshots: {
                let max_state_per_b = 8 * (2 * crate::config::N_HEAD_DIM) as usize;
                let max_boundaries = B_MAX / 4;
                DeviceBuffer::new(id, max_boundaries * max_state_per_b)?
            },
            comp_pooled_batched: {
                let max_boundaries = B_MAX / 4;
                DeviceBuffer::new(id, max_boundaries * crate::config::N_HEAD_DIM as usize)?
            },
            comp_rows_batched: {
                let max_boundaries = B_MAX / 4;
                DeviceBuffer::new(id, max_boundaries * crate::config::N_HEAD_DIM as usize)?
            },
            comp_pos_per_boundary: {
                let max_boundaries = B_MAX / 4;
                DeviceBuffer::new(id, max_boundaries)?
            },
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

            // M61: hot-expert prefill scratch, only when the het-split
            // weights will be loaded (same env gate as weights.rs).
            hot,
        })
    }
}
