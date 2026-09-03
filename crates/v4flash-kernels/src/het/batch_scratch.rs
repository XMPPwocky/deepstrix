//! Per-batch scratch for prefill.
//!
//! * [`BatchDgpuScratch`] / [`BatchIgpuScratch`] — the PER-LANE set: the
//!   few B-extended buffers that are live across the two-lane switch or
//!   are touched by a foreign stream (`de.xfer` / `ie.xfer`). Each lane
//!   owns one of each.
//! * [`BatchDgpuShared`] / [`BatchIgpuShared`] — the SHARED set: every
//!   other B-extended buffer. One instance serves both lanes, because
//!   each of these is first-written and last-read inside a single
//!   lane's `forward_layer_pre_moe_v2` on one in-order stream, and the
//!   host issues the lanes' pre-MoE work sequentially (see the phase
//!   table on [`BatchDgpuShared`]).
//! * [`BatchScratch`] — bundle of (shared `DgpuScratch`, shared
//!   `IgpuScratch`, B per-token residual buffers). Test-only convenience
//!   container retained for diagnostic tests; no production caller.
//!
//! Batched kernels (`*_batched`) read/write with per-batch strides
//! directly, no per-token copies. Used by `forward_prompt_batch_v2` and
//! the pipelined wrapper.
//!
//! Sizing: a scratch is allocated for `rows` tokens (`alloc_rows`);
//! `alloc()` = `alloc_rows(B_MAX)`. The production driver
//! (`forward_prefill_pipelined`) splits every `B_MAX` chunk across two
//! lanes of `ceil(B_MAX/2)` rows each, so the server allocates each lane
//! AND the shared set at `B_MAX.div_ceil(2)` (a shared buffer only ever
//! holds one lane's rows at a time). The single-lane driver
//! (`forward_prefill`) needs `rows >= B_MAX` on all four.
//!
//! Memory at `rows = 512` (B_MAX = 1024, two lanes): per-lane dGPU
//! ~128 MiB (x2), shared dGPU ~629 MiB; per-lane iGPU ~16 MiB (x2),
//! shared iGPU ~82 MiB. Two-lane totals: dGPU ~885 MiB (was 2 x 760),
//! iGPU ~114 MiB (was 2 x 98). See the per-field comments for the
//! within-lane disjoint-lifetime unions (R1/R2/R3 arenas).

use color_eyre::eyre;
use v4flash_hip::{Device, DeviceBuffer};

use crate::attention::{ATTN_MIXED_MAX_KEYS, ATTN_SCORES_STRIDE};
use crate::config::{
    BLOCKS_GROUPED_OUT, BLOCKS_N_EMBD, BLOCKS_N_FF_SHARED, BLOCKS_N_LORA_Q, BLOCKS_OUT_LOW,
    BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, HC_DIM, HC_MIX_DIM, INDEXER_TOP_K, N_EMBD, N_EXPERT,
    N_EXPERT_USED, N_FF_EXP, N_FF_SHARED, N_HEAD, N_HEAD_DIM, N_INDEXER_HEAD,
    N_INDEXER_HEAD_DIM, N_LORA_Q, OUT_LOW, Q_FLAT,
};
use crate::q8_k::BLOCK_Q8_K_BYTES;

use super::scratch::{DgpuScratch, IgpuScratch};

/// Max prefill batch size (tokens per prefill chunk). Larger chunks grow
/// the per-expert member lists under the skewed (Zipf-y) routing, so the
/// kwide MoE kernels amortize better — measured +9.4% e2e prefill going
/// 256 → 512 at depth 1024 (186.6 → 204.2 tok/s, back-to-back), and
/// 1024 has been the production value since 3d9cdd0 (2026-06-05). The
/// chunk is split across two pipeline lanes, so per-lane AND shared
/// scratch are sized at `B_MAX.div_ceil(2)` rows (see
/// [`BatchDgpuScratch::alloc_rows`]). The KV cache is oversized by
/// `B_MAX` rows (`state::KV_CACHE_ROWS`) because both lanes append into
/// the same chunk.
pub const B_MAX: usize = 1024;

/// Diagnostic toggle (env `DEEPSTRIX_F32_SCORES=1`): route the batched-
/// prefill attention through the **f32-scores** kernel pair instead of
/// the production f16-scores one. Used to test whether long-ctx
/// accuracy degradation is caused by f16 quantization of pre-softmax
/// logits (`_f16s` writes scores as f16, losing ~3 mantissa digits per
/// logit — bad near softmax ties). When on:
///   * `BatchDgpuShared::attn_scores` is allocated at *full* f32 size
///     (rows × N_HEAD × ATTN_SCORES_STRIDE f32 elements = 256 MiB at
///     rows=512), instead of the half-sized f16-byte-equivalent layout.
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
/// `forward_prefill_pipelined`) uses [`BatchDgpuScratch`] +
/// [`BatchDgpuShared`] instead — no per-token residual ping-pong, no
/// shared single-token scratch.
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

/// Sequential 256-byte-aligned region carver over one backing
/// allocation. Every `take` hands out a non-owning typed view starting at
/// the current cursor and advances it. The parent buffer's device memory
/// is owned by the enclosing scratch struct and never reallocated, so the
/// views stay valid for the struct's lifetime.
struct Carver<'a, T> {
    base: &'a DeviceBuffer<T>,
    off: usize,
}

impl<'a, T> Carver<'a, T> {
    fn new(base: &'a DeviceBuffer<T>) -> Self {
        Self { base, off: 0 }
    }

    fn take<U>(&mut self, len: usize) -> DeviceBuffer<U> {
        let bytes = len * std::mem::size_of::<U>();
        let o = self.off;
        self.off += align256(bytes);
        // SAFETY: `o` is 256-B aligned (≥ align_of any U we hand out) and
        // `view_as` bounds-checks against the parent's byte length.
        unsafe { self.base.view_as::<U>(o, len) }
    }
}

fn align256(bytes: usize) -> usize {
    (bytes + 255) & !255
}

fn check_rows(who: &str, rows: usize) -> eyre::Result<()> {
    eyre::ensure!(rows > 0 && rows <= B_MAX, "{who} rows={rows} out of (0, B_MAX]");
    Ok(())
}

/// PER-LANE dGPU scratch: the B-extended buffers that outlive one lane's
/// `forward_layer_pre_moe_v2` call or are touched by a stream other than
/// `de.compute`. Everything else lives in [`BatchDgpuShared`].
///
/// Why these cannot be shared between the two pipeline lanes (stream
/// order on `de.compute` per layer is `post_A(L) pre_A(L+1) post_B(L)
/// pre_B(L+1)`, so lane B's pre-MoE runs between lane A's pre-MoE and
/// lane A's post-MoE):
///
/// * `residual` / `residual_next` — the cross-layer HC flow; read in P1
///   and P7, written in P12 (post-MoE), swapped by the driver.
/// * `after_attn_hc` — written P7, read P8 and again in P12 `hc_post`.
/// * `split` — the mHC pre-FFN sinkhorn output (P8), read by P12
///   `hc_post` after the lane switch.
/// * `ffn_shared` — P10 output, read by P12 `vec_add`.
/// * `ffn_moe_recv` — written by `ie.xfer` (peer push), read in P12.
/// * `ffn_input_norm` — read by `de.xfer` (peer push to the iGPU) and by
///   the P11h hot leg; the only fence is `selected_pushed` on `de.xfer`.
/// * `d_selected` / `d_ew` — read by `de.xfer` (same push) and by the
///   P11h hot leg.
/// * `pos_per_b` — uploaded once per chunk, read every layer.
/// * `hot_ffn_moe_dgpu` — the M61 hot-expert reduce output (P11h), read
///   by P12 `vec_add_hot` after the lane switch.
///
/// ~128 MiB at rows=512: residual / residual_next / after_attn_hc 32 MiB
/// each, ffn_input_norm / ffn_shared / ffn_moe_recv / hot_ffn_moe_dgpu
/// 8 MiB each, the rest < 100 KiB.
pub struct BatchDgpuScratch {
    /// Row capacity every B-scaled buffer was sized for. Callers must
    /// never run a batch larger than this through the scratch.
    pub rows: usize,

    /// `[B, HC_DIM]` — per-token residual (cross-layer flow).
    pub residual: DeviceBuffer<f32>,
    pub residual_next: DeviceBuffer<f32>,
    /// `[B, HC_MIX_DIM]` — sinkhorn output (post + comb embedded). Written
    /// in P1 (attn) and again in P8 (ffn); the P8 value is read by P12
    /// `hc_post` AFTER the lane switch, so it is per-lane.
    pub split: DeviceBuffer<f32>,
    /// `[B, HC_DIM]` — mHC post-attention residual (P7 → P8, P12).
    pub after_attn_hc: DeviceBuffer<f32>,
    /// `[B, N_EMBD]` — FFN input (P8). Peer-pushed by `de.xfer` (P11x),
    /// read by the hot leg (P11h).
    pub ffn_input_norm: DeviceBuffer<f32>,

    // ---- Router output (consumed by de.xfer + the hot leg) ----
    pub d_selected: DeviceBuffer<i32>,
    pub d_ew: DeviceBuffer<f32>,

    // ---- Shared expert output ----
    /// `[B, N_EMBD]` — P10 output, read by P12 `vec_add`.
    pub ffn_shared: DeviceBuffer<f32>,

    /// `[B, N_EMBD]` — peer-arrival mailbox for iGPU MoE output. Filled
    /// by a single batched peer-push from iGPU (`ie.xfer`); then vec_add
    /// ffn_shared and run hc_post (P12).
    pub ffn_moe_recv: DeviceBuffer<f32>,

    /// `[B]` — per-token absolute position, `pos_per_b[b] = pos0 + b`.
    /// Uploaded once per chunk, read by every layer's rope launches.
    pub pos_per_b: DeviceBuffer<i32>,

    /// M61 prefill het-split: `[B, N_EMBD]` f32 hetsplit-reduced dGPU MoE
    /// partial (P11h reduce output); added to `ffn_moe_recv` at
    /// ffn_combine (P12, AFTER the lane switch). `Some` only when
    /// `DGPU_HOT_EXPERTS > 0` — must agree with
    /// [`BatchDgpuShared::hot`]. 8 MiB at rows=512.
    pub hot_ffn_moe_dgpu: Option<DeviceBuffer<f32>>,
}

/// SHARED dGPU scratch: one instance serves both pipeline lanes.
///
/// Every field is a single contiguous `DeviceBuffer` (or a non-owning
/// view into one of the arenas below) sized for `rows` tokens
/// (`rows × per_token_size`). Batched kernels read/write with per-batch
/// strides; per-token (stateful) kernels use offset slices.
///
/// ## Why one instance is enough for two lanes
///
/// Both lanes issue ALL of their pre-MoE dGPU work on the single
/// `de.compute` stream, in program order, and the host issues one lane's
/// entire `forward_layer_pre_moe_v2` before starting the other's:
/// `post_A(L) pre_A(L+1) post_B(L) pre_B(L+1)`. Every buffer here is
/// first written and last read INSIDE one lane's pre-MoE call, on
/// `de.compute` (async H2D uploads are also queued on `de.compute`), and
/// nothing here is read by `forward_layer_post_moe_v2` (P12), by
/// `de.xfer`, or by the iGPU. So lane B's writes to a shared buffer are
/// stream-ordered after lane A's last read of it, and the shared
/// instance simply alternates between lanes. Per-field comments name
/// the phase window each buffer relies on.
///
/// Phase order per layer (all de.compute): P1 mhc_pre_attn → P2 q-chain
/// → P3 kv-chain → P4 kv-append+compressor → P5i indexer → P5a
/// score/smwsum → P5e SWA evict → P6 out-proj → P7 mhc_post_attn → P8
/// mhc_pre_ffn → P9 router → P10 shared-expert → P11x peer push
/// (de.xfer) → P11h hot-expert leg → P11i iGPU chain → (lane switch) →
/// P12 post_moe (ffn_combine).
///
/// ## Within-lane lifetime unions (same stream, program ordered)
///
/// * **R1** (`r1_arena`): `flat` (P1, P8) / `q` (P2) / `indexer_scores`
///   (P5i) / `heads` (P5a → P6) all at offset 0, plus the four M61 hot
///   views (`partials`, `mid_cat`, `midq_cat`, `moe_xq`, P11h) at fixed
///   offsets. Live order: flat[P1] < q[P2] < indexer_scores[P5i] <
///   heads[P5a..P6] < flat[P8] < hot[P11h]; nothing in R1 is read after
///   P11h (the hot reduce writes the per-lane `hot_ffn_moe_dgpu`, which
///   is what P12 reads after the lane switch).
/// * **R2** (`q_normed` itself hosts): `low`, `heads_xq`, `heads_xscale`,
///   `low_xq`, `low_xscale`, `attn_out` at distinct offsets. `q_normed`'s
///   last read is smwsum/fused/swa in P5a; the R2 views are first written
///   in P6 (quantize_heads) and `attn_out` is last read in P7; `q_normed`
///   is next written in P2 of the following layer.
/// * **R3** (`r3_arena`): `indexer_q` (written P5i matvec_q, last read by
///   the indexer score kernel) and `indexer_topk_scratch` (written by the
///   topk kernel that runs after the score kernel). Both at offset 0.
///
/// ~629 MiB at rows=512: attn_active_comp_kv 256, attn_scores 128, R1
/// arena 96.5, q_normed/R2 64, attn_cur / attn_input_norm / ffn_cur 8
/// each, R3 16, shared-expert temporaries 13, compressor snapshots 2x4,
/// K-quant activations 5.7, Q-chain 6.8, misc.
///
/// Does NOT include head/MTP/stash buffers — `forward_layer_batch_v2`
/// only needs the active per-token state. Head runs from a separate
/// single-token `DgpuScratch`.
pub struct BatchDgpuShared {
    /// Row capacity every B-scaled buffer was sized for. Must be >= the
    /// rows of every lane that uses this shared set.
    pub rows: usize,

    /// R1 arena backing `flat`, `q`, `indexer_scores`, `heads` and the
    /// hot-expert views (see struct doc). Not used directly.
    pub r1_arena: DeviceBuffer<u8>,
    /// R3 arena backing `indexer_q` and `indexer_topk_scratch`.
    pub r3_arena: DeviceBuffer<u8>,

    // ---- mHC stage ----
    /// `[B, HC_DIM]` — rms_nw output going into hc_attn_fn / hc_ffn_fn.
    /// R1 view @0: live P1 (rms_nw → f16_matvec) and P8 (same pair).
    pub flat: DeviceBuffer<f32>,
    /// `[B, HC_MIX_DIM]` — f16 narrow matvec output (sinkhorn input).
    /// Live P1 and P8 only (written by f16_matvec, read by sinkhorn).
    pub mix: DeviceBuffer<f32>,
    /// `[B, N_EMBD]` — hc_weighted output for attention input (P1).
    pub attn_cur: DeviceBuffer<f32>,
    /// `[B, N_EMBD]` — attention input norm. Written P1; last read by the
    /// indexer proj matvec in P5i (also read P2, P4).
    pub attn_input_norm: DeviceBuffer<f32>,
    /// `[B, N_EMBD]` — hc_weighted output for the FFN input (P8 only).
    pub ffn_cur: DeviceBuffer<f32>,

    // ---- K-quant prefill activations (unsloth UD mix) ----
    /// Q8_K of attn_input_norm `[B, 16*292]` — attn_q_a when Q5_K/Q6_K
    /// (P2 only).
    pub kq_attn_q8k: DeviceBuffer<u8>,
    /// Q8_K of ffn_input_norm `[B, 16*292]` — shexp gate/up when
    /// Q5_K/Q6_K (P10 only).
    pub kq_ffn_q8k: DeviceBuffer<u8>,
    /// Q8_K of mid_sh `[B, 8*292]` — shexp down when Q6_K (P10 only).
    pub kq_mid_q8k: DeviceBuffer<u8>,

    // ---- Q chain (P2; xq/xscale reused in P3 and P10) ----
    pub xq_n_embd: DeviceBuffer<i8>,
    pub xscale_n_embd: DeviceBuffer<f32>,
    pub qr: DeviceBuffer<f32>,
    /// `[B, N_LORA_Q]` — normed q_a output. Written P2; last read by the
    /// indexer matvec_q in P5i.
    pub qr_normed: DeviceBuffer<f32>,
    pub qr_xq: DeviceBuffer<i8>,
    pub qr_xscale: DeviceBuffer<f32>,
    /// `[B, Q_FLAT]` — qb up-projection output. R1 view @0: written by
    /// the qb GEMM, last read by rms_nw_heads (both P2).
    pub q: DeviceBuffer<f32>,
    /// `[B, Q_FLAT]` — normed + roped Q (P2 → P5a). Owned; also the R2
    /// backing allocation for the P6/P7 out-proj temporaries (see struct
    /// doc).
    pub q_normed: DeviceBuffer<f32>,

    // ---- KV chain (P3 → P4 kv_append) ----
    pub kv_raw: DeviceBuffer<f32>,
    pub kv_normed: DeviceBuffer<f32>,
    /// Scratch ring (≥ `SWA_WINDOW * N_HEAD_DIM`) used by the post-chunk
    /// SWA-eviction pass (P5e). After a prefill chunk leaves the cache
    /// holding `n_raw_during_chunk > SWA_WINDOW` rows in the oversized
    /// region, we copy the LAST `SWA_WINDOW` rows here, then copy them
    /// back to slots `[0..W)` of `ls.kv_cache` — two non-overlapping
    /// device-to-device copies on the compute stream. The intermediate
    /// buffer makes the shift race-free even when the source/destination
    /// regions in the cache overlap. `u16` because `ls.kv_cache` is
    /// f16-stored.
    pub kv_ring_scratch: DeviceBuffer<u16>,

    // ---- Compressor (P4; main + indexer compressors back to back) ----
    pub kv_cur: DeviceBuffer<f32>,
    pub sc_cur: DeviceBuffer<f32>,

    // ---- Per-token attention causality (uploaded per layer on de.compute) ----
    /// `[B]` — per-token compressor state row / pos_mod, computed per
    /// layer (P4) from the ratio.
    pub row_per_b: DeviceBuffer<i32>,
    pub pos_mod_per_b: DeviceBuffer<i32>,
    /// `[B]` — per-token causal prefix length over the raw KV cache
    /// (uploaded P5, read P5a).
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
    /// `[B, Q_FLAT]` — attention output (per-head). R1 view @0: written
    /// by swa/smwsum/fused (P5a), rope_inverse in place and last read by
    /// quantize_heads (P6).
    pub heads: DeviceBuffer<f32>,
    /// `[B, n_head, ATTN_SCORES_STRIDE]` — global scores/weights scratch
    /// for the batched split attention kernels (score → smwsum, P5a).
    /// Replaces the monolithic kernel's LDS `scores[2304]` (which
    /// overflows past ~9K tokens). Stored as f16 by the production
    /// `_f16s` pair, so allocated at half the f32 element count (128 MiB
    /// at rows=512; 256 MiB under `DEEPSTRIX_F32_SCORES=1`).
    pub attn_scores: DeviceBuffer<f32>,
    // ---- CSA indexer per-token scratch (P5i) ----
    /// `[B, N_INDEXER_HEAD * N_INDEXER_HEAD_DIM]` — per-token indexer Q
    /// (matvec(attn_q_b) output, then RoPE + QAT in place). R3 view @0:
    /// written by matvec_q, last read by the indexer score kernel (P5i).
    pub indexer_q: DeviceBuffer<f32>,
    /// `[B, N_INDEXER_HEAD]` — per-token head_weights (matvec(proj)
    /// output, post-scale).
    pub indexer_head_weights: DeviceBuffer<f32>,
    /// `[B, ATTN_MIXED_MAX_KEYS]` — batched IndexerScore output. Stride
    /// per token = MAX_KEYS. R1 view @0 (the largest R1 member: 96.5 MiB
    /// at rows=512): written by the score kernel, last read by topk (P5i).
    pub indexer_scores: DeviceBuffer<f32>,
    /// `[B, INDEXER_TOP_K]` — batched IndexerTopk output (selected
    /// indices per token, sentinel -1 in unused slots). Written by topk,
    /// read by the gather (both P5i). 1 MiB at rows=512.
    pub indexer_selected: DeviceBuffer<i32>,
    /// `[B, (max_chunks + n_groups) * INDEXER_TOP_K]` — batched bitonic
    /// topk per-chunk candidates scratch (two-level tree merge). R3 view
    /// @0: written + read only inside the topk launch (P5i), which runs
    /// after the score kernel's last read of `indexer_q`.
    pub indexer_topk_scratch: DeviceBuffer<u32>,
    /// `[B]` — per-token n_index_comp uploaded once per ratio==4 layer
    /// (P5i).
    pub n_index_comp_per_b: DeviceBuffer<u32>,
    /// `[B, INDEXER_TOP_K, N_HEAD_DIM]` f16 — per-batch gathered comp_kv
    /// rows for the CSA sparse-attention path. Populated by
    /// `IndexerGather::launch_batched` when the indexer fires (P5i),
    /// last read by smwsum (P5a). Passed to score+smwsum with
    /// `comp_kv_batch_stride = INDEXER_TOP_K` so the kernels read only
    /// the dense top-K rows per batch instead of doing per-row mask
    /// tests on the full sparse n_comp. 256 MiB at rows=512.
    pub attn_active_comp_kv: DeviceBuffer<u16>,
    /// `[n_boundaries_max × coff × ratio × width]` f32 — per-boundary
    /// state_kv snapshots taken at each boundary firing in the prefill
    /// compressor loop (P4). Reused across main + indexer compressors
    /// (only one is active at a time). Sized for the largest case
    /// (ratio==4 main: `rows/4` boundaries × 8 × 1024 f32 = 4 MiB at
    /// rows=512).
    pub comp_state_kv_snapshots: DeviceBuffer<f32>,
    pub comp_state_score_snapshots: DeviceBuffer<f32>,
    /// `[n_boundaries_max, head_dim]` f32 — pool output per boundary.
    /// Feeds rms_w_batched which writes its result into `comp_rows_batched`.
    pub comp_pooled_batched: DeviceBuffer<f32>,
    /// `[n_boundaries_max, head_dim]` f32 — post-rms_w, then rope/fp8/
    /// f16rt-ed in place. Final values appended to comp_kv via
    /// comp_kv_append_batched. 256 KB at rows=512.
    pub comp_rows_batched: DeviceBuffer<f32>,
    /// `[n_boundaries_max]` i32 — per-boundary RoPE positions, uploaded
    /// once per layer×compressor for the batched rope launch.
    pub comp_pos_per_boundary: DeviceBuffer<i32>,
    /// R2 views into `q_normed` (see struct doc). Written in P6 after
    /// `q_normed`'s last read (P5a); `attn_out` last read by hc_post in
    /// P7; all dead before the next layer's P2 `q_normed` write.
    pub low: DeviceBuffer<f32>,
    pub heads_xq: DeviceBuffer<i8>,
    pub heads_xscale: DeviceBuffer<f32>,
    pub low_xq: DeviceBuffer<i8>,
    pub low_xscale: DeviceBuffer<f32>,
    pub attn_out: DeviceBuffer<f32>,

    // ---- Router (P9) ----
    /// `[B, N_EXPERT]` — gate logits; read by router_topk (P9) which
    /// writes the per-lane `d_selected` / `d_ew`.
    pub router_logits: DeviceBuffer<f32>,
    /// Host readback area for the hash router path (synchronous readback
    /// inside P9). `[B, N_EXPERT]`.
    pub router_logits_host: Vec<f32>,

    // ---- Shared expert temporaries (P10) ----
    pub gate_sh: DeviceBuffer<f32>,
    pub up_sh: DeviceBuffer<f32>,
    pub mid_sh: DeviceBuffer<f32>,
    pub mid_sh_xq: DeviceBuffer<i8>,
    pub mid_sh_xscale: DeviceBuffer<f32>,

    /// M61 prefill het-split: dGPU-side MoE scratch for the hot-expert leg
    /// (P11h). `Some` only when `DGPU_HOT_EXPERTS > 0` (~81 MiB of R1
    /// views + member/work-item lists at rows=512). The reduce OUTPUT is
    /// the per-lane [`BatchDgpuScratch::hot_ffn_moe_dgpu`].
    pub hot: Option<BatchDgpuHotScratch>,
}

/// Static work-item geometry for the dGPU hot-expert prefill leg.
/// Members per hot expert are capped at the scratch's `rows`, so each
/// expert needs at most `ceil(rows / HOT_CHUNK)` chunks
/// ([`hot_chunks_per_expert`]). The work-items list is e-major and
/// uploaded ONCE; per-layer launches set grid.y = n_hot × chunks and the
/// matvec kernels' `member_end <= member_start` guard early-exits empty
/// chunks — no per-layer host readback of n_work_items on de.compute.
pub const HOT_MAX_EXPERTS: usize = 256;
pub const HOT_CHUNK: usize = 32;

/// Chunks per hot expert for a scratch of `rows` tokens.
pub fn hot_chunks_per_expert(rows: usize) -> usize {
    rows.div_ceil(HOT_CHUNK)
}

/// M61 prefill het-split: dGPU scratch mirroring the iGPU batched MoE
/// pipeline, sized for the hot-expert (resident) slots only. The matvec
/// kernels are the SAME kwide/kwide2 kernels the iGPU runs — only the
/// group builder (dense ids, hits only) and the reduce (own-slots only)
/// differ. Lives in [`BatchDgpuShared`]: everything here is written and
/// read inside one lane's P11h on `de.compute`; the reduce output goes to
/// the per-lane `hot_ffn_moe_dgpu`.
///
/// VRAM: the four big intermediates (`partials`, `mid_cat`, `midq_cat`,
/// `moe_xq`; ~81 MiB at rows=512) are non-owning VIEWS into the parent's
/// R1 arena at fixed offsets. LIFETIME CONTRACT: they are written and read
/// only inside the hot leg (P11h: group_builder → q8k → gate/up → q8k →
/// down → reduce), which is enqueued on de.compute after every other R1
/// user of the layer (flat's P8 read is the last), and the reduce's output
/// goes to the OWNED per-lane `hot_ffn_moe_dgpu`, so nothing in R1 is read
/// after P11h. The next R1 access is the next pre-MoE call's P1 `flat`
/// write (the other lane's, or this lane's next layer). Earlier R1 users
/// (heads/flat) leave stale f32 bit patterns (possibly NaN) in the miss
/// slots of `mid_cat`; the post-gate/up q8k quantize reads them but the
/// resulting q8 blocks are never dotted (work items cover hot members
/// only), so that is harmless.
pub struct BatchDgpuHotScratch {
    /// Per-expert member-list capacity (= parent `rows`). Passed to the
    /// hetsplit group builder and the kwide kernels as `max_per_expert`.
    pub max_per_expert: usize,
    /// `ceil(max_per_expert / HOT_CHUNK)` — static work items per expert.
    pub chunks_per_expert: usize,
    /// `[B, BLOCKS_Q8K_GATE_IN × 292]` — q8_K-quantized ffn_input_norm.
    /// Bit-identical to the iGPU's d_xq_q8k (same f32 input, same kernel).
    /// R1 view.
    pub moe_xq: DeviceBuffer<u8>,
    /// `[B, n_used, N_FF_EXP]` f32 — fused-swiglu mid, hit slots only.
    /// R1 view.
    pub mid_cat: DeviceBuffer<f32>,
    /// `[B, n_used, BLOCKS_Q8K_DOWN_IN × 292]` — quantized mid. R1 view.
    pub midq_cat: DeviceBuffer<u8>,
    /// `[B*n_used, N_EMBD]` f32 — q2k by-expert partials, hit slots only.
    /// R1 view. Read by the reduce, which writes the per-lane
    /// `hot_ffn_moe_dgpu`.
    pub partials: DeviceBuffer<f32>,
    /// `[HOT_MAX_EXPERTS]` i32 — DENSE-id group counts. Zeroed per layer
    /// via fill_zero_async on de.compute (stream-ordered, so sharing
    /// between lanes is safe). Owned.
    pub group_count: DeviceBuffer<i32>,
    /// `[HOT_MAX_EXPERTS × max_per_expert]` i32 — dense-id member lists,
    /// `(b<<16)|slot` packed like the iGPU arrays. Owned.
    pub expert_members: DeviceBuffer<i32>,
    /// `[HOT_MAX_EXPERTS × chunks_per_expert]` i32 — STATIC e-major
    /// work items `(e<<16)|(c×HOT_CHUNK)`, uploaded once at alloc. Owned.
    pub work_items_static: DeviceBuffer<i32>,
}

impl BatchDgpuHotScratch {
    fn xq_len(rows: usize) -> usize {
        rows * (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_Q8_K_BYTES
    }
    fn midq_len(rows: usize) -> usize {
        rows * N_EXPERT_USED * (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES
    }
    fn mid_len(rows: usize) -> usize {
        rows * N_EXPERT_USED * (N_FF_EXP as usize)
    }
    fn partials_len(rows: usize) -> usize {
        rows * N_EXPERT_USED * (N_EMBD as usize)
    }

    /// Bytes the four R1 views occupy (each 256-B aligned), in carve
    /// order `partials, mid_cat, moe_xq, midq_cat`.
    pub fn r1_view_bytes(rows: usize) -> usize {
        align256(Self::partials_len(rows) * 4)
            + align256(Self::mid_len(rows) * 4)
            + align256(Self::xq_len(rows))
            + align256(Self::midq_len(rows))
    }

    /// Carve the four intermediates out of the parent's R1 arena (see
    /// LIFETIME CONTRACT on the struct); allocate the rest owned.
    fn alloc(id: i32, rows: usize, r1: &DeviceBuffer<u8>) -> eyre::Result<Self> {
        let chunks_per_expert = hot_chunks_per_expert(rows);

        // f32 views first (offset stays 4-aligned), byte views after.
        // Order must match `r1_view_bytes`.
        let mut carve = Carver::new(r1);
        let partials = carve.take::<f32>(Self::partials_len(rows));
        let mid_cat = carve.take::<f32>(Self::mid_len(rows));
        let moe_xq = carve.take::<u8>(Self::xq_len(rows));
        let midq_cat = carve.take::<u8>(Self::midq_len(rows));
        debug_assert_eq!(carve.off, Self::r1_view_bytes(rows));

        let mut work_items_static: DeviceBuffer<i32> =
            DeviceBuffer::new(id, HOT_MAX_EXPERTS * chunks_per_expert)?;
        let mut wi_host = vec![0i32; HOT_MAX_EXPERTS * chunks_per_expert];
        for e in 0..HOT_MAX_EXPERTS {
            for c in 0..chunks_per_expert {
                wi_host[e * chunks_per_expert + c] =
                    ((e as i32) << 16) | ((c * HOT_CHUNK) as i32);
            }
        }
        work_items_static.copy_from_host(&wi_host)?;
        Ok(Self {
            max_per_expert: rows,
            chunks_per_expert,
            moe_xq,
            mid_cat,
            midq_cat,
            partials,
            group_count: DeviceBuffer::new(id, HOT_MAX_EXPERTS)?,
            expert_members: DeviceBuffer::new(id, HOT_MAX_EXPERTS * rows)?,
            work_items_static,
        })
    }
}

/// Whether the M61 hot-expert prefill scratch should be allocated. Must
/// agree with the residency loader, or the scratch and the weights
/// disagree about whether the split is active.
fn hot_scratch_wanted() -> bool {
    crate::het::weights::dgpu_hot_experts() > 0
}

/// PER-LANE iGPU scratch: the buffers the peer pushes and the group
/// pre-pass touch across stream / lane boundaries.
///
/// * `ffn_input_norm_recv`, `d_selected`, `d_ew` — recv mailboxes written
///   by `de.xfer` (P11x); the iGPU chain waits on `selected_pushed`.
/// * `ffn_moe` — the MoE output, read by `ie.xfer` (peer push back) after
///   `moe_done`; lane B's chain is queued on `ie.compute` while that
///   push may still be pending on `ie.xfer`.
/// * `group_count`, `n_work_items`, `n_staged_work_items`,
///   `n_chunked_work_items` — zeroed with the SYNCHRONOUS `fill_zero`
///   (null-stream hipMemset) at the top of the chain, which is not
///   ordered against the other lane's in-flight `ie.compute` kernels;
///   they stay per-lane so that memset can never race a reader. Tiny.
///
/// ~16 MiB at rows=512 (recv/moe 8 each, the rest < 30 KiB).
pub struct BatchIgpuScratch {
    /// Row capacity every B-scaled buffer was sized for.
    pub rows: usize,
    pub ffn_input_norm_recv: DeviceBuffer<f32>,
    pub ffn_moe: DeviceBuffer<f32>,
    pub d_selected: DeviceBuffer<i32>,
    pub d_ew: DeviceBuffer<f32>,
    /// By-expert MoE: per-expert pick count built by the group_builder
    /// pre-pass. `[n_expert]` i32. MUST be zeroed before each layer's pre-pass.
    pub group_count: DeviceBuffer<i32>,
    /// Chunked by-expert: `[1]` i32. Count of valid entries in
    /// `work_items[]`. MUST be zeroed before each layer's pre-pass.
    /// Read back to host (sync) to set grid.y for the main kernel.
    pub n_work_items: DeviceBuffer<i32>,
    /// Hybrid dispatch: `[1]` i32 atomic counters, pre-zeroed per layer.
    pub n_staged_work_items: DeviceBuffer<i32>,
    pub n_chunked_work_items: DeviceBuffer<i32>,
}

/// SHARED iGPU scratch: one instance serves both pipeline lanes.
///
/// The iGPU chain (P11i) is issued for one lane at a time, entirely on
/// `ie.compute` in program order (q8k → group_builder → work_items →
/// gate/up → q8k → down → reduce). Every buffer here is first written
/// and last read inside that chain: the reduce writes the per-lane
/// `ffn_moe`, which is the only thing `ie.xfer` reads. So lane B's chain
/// is stream-ordered after lane A's last read of every shared buffer,
/// and one instance alternates between lanes. ~82 MiB at rows=512
/// (q2k_partials 48, d_mid_cat 24, d_midq_cat 6.9, d_xq_q8k 2.3,
/// expert_members 0.5, work lists 3 x 13 KiB).
pub struct BatchIgpuShared {
    /// Row capacity every B-scaled buffer was sized for.
    pub rows: usize,
    /// `[B, 16*292]` — q8_K of the recv'd ffn_input_norm (chain head).
    pub d_xq_q8k: DeviceBuffer<u8>,
    /// `[B, n_used, N_FF_EXP]` f32 — fused-swiglu mid (gate/up → q8k).
    pub d_mid_cat: DeviceBuffer<f32>,
    /// `[B, n_used, 8*292]` — q8_K of d_mid_cat (q8k → down).
    pub d_midq_cat: DeviceBuffer<u8>,
    /// By-expert MoE: per-expert (b, slot) member lists, packed as
    /// `(b << 16) | slot`. `[n_expert × max_per_expert]` i32. Only the first
    /// `group_count[e]` entries per expert are valid after the pre-pass.
    /// `max_per_expert = rows` (worst case: every token picks the same expert
    /// in some slot — still fits since each token contributes ≤ n_used picks).
    /// Written by the group builder, last read by the down kernel.
    pub expert_members: DeviceBuffer<i32>,
    /// Chunked by-expert: flat list of (expert_id<<16 | member_start)
    /// work items built by moe_work_items_builder. Sized for worst case
    /// = `rows * n_used / CHUNK_SIZE + n_expert` (each active expert may
    /// have one extra ceiling chunk). Written by the work-items pre-pass,
    /// last read by the down kernel.
    pub work_items: DeviceBuffer<i32>,
    /// Hybrid dispatch: work items for the staged kernel (chunks ≥ threshold).
    /// Same shape as `work_items`. MUST be paired with `n_staged_work_items`.
    pub staged_work_items: DeviceBuffer<i32>,
    /// Hybrid dispatch: work items for the chunked kernel (chunks < threshold).
    /// Same shape as `work_items`. MUST be paired with `n_chunked_work_items`.
    pub chunked_work_items: DeviceBuffer<i32>,
    /// Q2K_VARIANT=by_expert: per-(b, slot, row) partial sums written by
    /// `q2_k_matvec_par_by_expert`, then summed across `n_used` by
    /// `q2_k_reduce_partials` into the per-lane `ffn_moe`.
    /// `[B*n_used, N_EMBD]` f32 = 48 MiB at rows=512.
    /// Avoids the atomicAdd nondeterminism of an in-place accumulation.
    pub q2k_partials: DeviceBuffer<f32>,
}

impl BatchIgpuScratch {
    /// Allocate for the full `B_MAX` chunk (single-lane drivers).
    pub fn alloc(igpu_device: Device) -> eyre::Result<Self> {
        Self::alloc_rows(igpu_device, B_MAX)
    }

    /// Allocate for `rows` tokens. Two-lane callers pass
    /// `B_MAX.div_ceil(2)`; the scratch must never see a batch > `rows`.
    pub fn alloc_rows(igpu_device: Device, rows: usize) -> eyre::Result<Self> {
        check_rows("BatchIgpuScratch", rows)?;
        igpu_device.set_current()?;
        let id = igpu_device.id;
        let b = rows;
        Ok(Self {
            rows,
            ffn_input_norm_recv: DeviceBuffer::new(id, b * N_EMBD as usize)?,
            ffn_moe: DeviceBuffer::new(id, b * N_EMBD as usize)?,
            d_selected: DeviceBuffer::new(id, b * N_EXPERT_USED as usize)?,
            d_ew: DeviceBuffer::new(id, b * N_EXPERT_USED as usize)?,
            group_count: DeviceBuffer::new(id, N_EXPERT as usize)?,
            n_work_items: DeviceBuffer::new(id, 1)?,
            n_staged_work_items: DeviceBuffer::new(id, 1)?,
            n_chunked_work_items: DeviceBuffer::new(id, 1)?,
        })
    }
}

impl BatchIgpuShared {
    /// Allocate for the full `B_MAX` chunk (single-lane drivers).
    pub fn alloc(igpu_device: Device) -> eyre::Result<Self> {
        Self::alloc_rows(igpu_device, B_MAX)
    }

    /// Allocate for `rows` tokens — the max rows of any lane that will
    /// use this shared set (`B_MAX.div_ceil(2)` for the two-lane driver).
    pub fn alloc_rows(igpu_device: Device, rows: usize) -> eyre::Result<Self> {
        check_rows("BatchIgpuShared", rows)?;
        igpu_device.set_current()?;
        let id = igpu_device.id;
        let b = rows;
        let xq_bytes_per_batch =
            (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_Q8_K_BYTES;
        let midq_bytes_per_batch =
            (N_EXPERT_USED as usize) * (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES;
        // Worst case work items: every member could be its own chunk
        // (CHUNK_SIZE=1 degenerate), so size for B*n_used. In practice
        // at CHUNK_SIZE=16 we use far less.
        let work_items_len = (N_EXPERT as usize) + b * (N_EXPERT_USED as usize);
        Ok(Self {
            rows,
            d_xq_q8k: DeviceBuffer::new(id, b * xq_bytes_per_batch)?,
            d_mid_cat: DeviceBuffer::new(
                id,
                b * (N_EXPERT_USED as usize) * (N_FF_EXP as usize),
            )?,
            d_midq_cat: DeviceBuffer::new(id, b * midq_bytes_per_batch)?,
            expert_members: DeviceBuffer::new(id, (N_EXPERT as usize) * b)?,
            work_items: DeviceBuffer::new(id, work_items_len)?,
            // Hybrid dispatch: two extra work_items arrays, each sized for the
            // worst case (all items land in one bucket).
            staged_work_items: DeviceBuffer::new(id, work_items_len)?,
            chunked_work_items: DeviceBuffer::new(id, work_items_len)?,
            q2k_partials: DeviceBuffer::new(
                id,
                b * (N_EXPERT_USED as usize) * (N_EMBD as usize),
            )?,
        })
    }

    /// Max per-expert group capacity (= `rows`). Used by Stage 11's by-expert path.
    pub fn max_per_expert(&self) -> u32 {
        self.rows as u32
    }
}

/// Per-token element counts of the R1 / R3 arena members.
fn flat_len(rows: usize) -> usize {
    rows * HC_DIM as usize
}
fn q_len(rows: usize) -> usize {
    rows * Q_FLAT as usize
}
fn indexer_scores_len(rows: usize) -> usize {
    rows * ATTN_MIXED_MAX_KEYS as usize
}
fn indexer_q_len(rows: usize) -> usize {
    rows * (N_INDEXER_HEAD * N_INDEXER_HEAD_DIM) as usize
}
fn indexer_topk_scratch_len(rows: usize) -> usize {
    // Two-level bitonic tree merge (see scratch.rs): per token
    // L0 = max_chunks*top_k + L1 = n_groups*top_k.
    let max_chunks = (ATTN_MIXED_MAX_KEYS + 4095) / 4096;
    let group_chunks = 4096 / INDEXER_TOP_K;
    let n_groups = (max_chunks + group_chunks - 1) / group_chunks;
    rows * ((max_chunks + n_groups) * INDEXER_TOP_K) as usize
}

/// R1 arena byte size: the largest of its offset-0 members and the hot
/// view pack (see [`BatchDgpuShared`] doc).
pub fn r1_arena_bytes(rows: usize) -> usize {
    [
        flat_len(rows) * 4,
        q_len(rows) * 4,
        indexer_scores_len(rows) * 4,
        q_len(rows) * 4, // heads
        BatchDgpuHotScratch::r1_view_bytes(rows),
    ]
    .into_iter()
    .map(align256)
    .max()
    .unwrap()
}

/// R3 arena byte size: max(indexer_q, indexer_topk_scratch).
pub fn r3_arena_bytes(rows: usize) -> usize {
    align256(indexer_q_len(rows) * 4).max(align256(indexer_topk_scratch_len(rows) * 4))
}

impl BatchDgpuScratch {
    /// Allocate for the full `B_MAX` chunk (single-lane drivers:
    /// `forward_prefill`, `forward_prompt_batch_v2` with b up to B_MAX).
    pub fn alloc(dgpu_device: Device) -> eyre::Result<Self> {
        Self::alloc_rows(dgpu_device, B_MAX)
    }

    /// Allocate for `rows` tokens. The two-lane production driver
    /// (`forward_prefill_pipelined`) never puts more than
    /// `B_MAX.div_ceil(2)` rows in a lane, so the server allocates each
    /// lane at that size (~128 MiB dGPU). Every batched entry point
    /// checks `b <= rows` before touching the scratch.
    pub fn alloc_rows(dgpu_device: Device, rows: usize) -> eyre::Result<Self> {
        check_rows("BatchDgpuScratch", rows)?;
        dgpu_device.set_current()?;
        let id = dgpu_device.id;
        let b = rows;
        let mk_f32 =
            |n: usize| -> eyre::Result<DeviceBuffer<f32>> { DeviceBuffer::new(id, b * n) };
        let mk_i32 =
            |n: usize| -> eyre::Result<DeviceBuffer<i32>> { DeviceBuffer::new(id, b * n) };
        // M61: hot-expert reduce output, only when the het-split weights
        // will be loaded (same env gate as weights.rs and the shared set).
        let hot_ffn_moe_dgpu = if hot_scratch_wanted() {
            Some(mk_f32(N_EMBD as usize)?)
        } else {
            None
        };
        Ok(Self {
            rows,
            residual: mk_f32(HC_DIM as usize)?,
            residual_next: mk_f32(HC_DIM as usize)?,
            split: mk_f32(HC_MIX_DIM as usize)?,
            after_attn_hc: mk_f32(HC_DIM as usize)?,
            ffn_input_norm: mk_f32(N_EMBD as usize)?,
            d_selected: mk_i32(N_EXPERT_USED)?,
            d_ew: mk_f32(N_EXPERT_USED)?,
            ffn_shared: mk_f32(N_EMBD as usize)?,
            ffn_moe_recv: mk_f32(N_EMBD as usize)?,
            pos_per_b: mk_i32(1)?,
            hot_ffn_moe_dgpu,
        })
    }
}

impl BatchDgpuShared {
    /// Allocate for the full `B_MAX` chunk (single-lane drivers).
    pub fn alloc(dgpu_device: Device) -> eyre::Result<Self> {
        Self::alloc_rows(dgpu_device, B_MAX)
    }

    /// Allocate for `rows` tokens — the max rows of any lane that will
    /// use this shared set (`B_MAX.div_ceil(2)` for the two-lane driver,
    /// ~629 MiB dGPU). Every batched entry point checks `b <= rows`.
    pub fn alloc_rows(dgpu_device: Device, rows: usize) -> eyre::Result<Self> {
        check_rows("BatchDgpuShared", rows)?;
        dgpu_device.set_current()?;
        let id = dgpu_device.id;
        let b = rows;
        let mk_f32 =
            |n: usize| -> eyre::Result<DeviceBuffer<f32>> { DeviceBuffer::new(id, b * n) };
        let mk_u16 =
            |n: usize| -> eyre::Result<DeviceBuffer<u16>> { DeviceBuffer::new(id, b * n) };
        let mk_i8 = |n: usize| -> eyre::Result<DeviceBuffer<i8>> { DeviceBuffer::new(id, b * n) };
        let mk_u8 = |n: usize| -> eyre::Result<DeviceBuffer<u8>> { DeviceBuffer::new(id, b * n) };
        let mk_i32 =
            |n: usize| -> eyre::Result<DeviceBuffer<i32>> { DeviceBuffer::new(id, b * n) };
        // Compressor boundaries per chunk: at most one every `ratio >= 4`
        // positions of a lane's `rows` tokens.
        let max_boundaries = rows.div_ceil(4);

        // ---- R1 arena: flat / q / indexer_scores / heads @0, hot views
        // after. Built before the literal so the M61 hot-expert scratch
        // can carve its views out of it (see BatchDgpuHotScratch LIFETIME
        // CONTRACT and the struct doc's lifetime-union table).
        let r1_arena: DeviceBuffer<u8> = DeviceBuffer::new(id, r1_arena_bytes(rows))?;
        let flat = Carver::new(&r1_arena).take::<f32>(flat_len(rows));
        let q = Carver::new(&r1_arena).take::<f32>(q_len(rows));
        let indexer_scores = Carver::new(&r1_arena).take::<f32>(indexer_scores_len(rows));
        let heads = Carver::new(&r1_arena).take::<f32>(q_len(rows));
        let hot = if hot_scratch_wanted() {
            Some(BatchDgpuHotScratch::alloc(id, rows, &r1_arena)?)
        } else {
            None
        };

        // ---- R3 arena: indexer_q / indexer_topk_scratch @0.
        let r3_arena: DeviceBuffer<u8> = DeviceBuffer::new(id, r3_arena_bytes(rows))?;
        let indexer_q = Carver::new(&r3_arena).take::<f32>(indexer_q_len(rows));
        let indexer_topk_scratch =
            Carver::new(&r3_arena).take::<u32>(indexer_topk_scratch_len(rows));

        // ---- R2: out-proj temporaries carved from q_normed (dead from
        // P5a's last read until the next layer's P2 write).
        let q_normed = mk_f32(Q_FLAT as usize)?;
        let mut r2 = Carver::new(&q_normed);
        let low = r2.take::<f32>(b * OUT_LOW as usize);
        let heads_xq = r2.take::<i8>(b * Q_FLAT as usize);
        let heads_xscale = r2.take::<f32>(b * BLOCKS_GROUPED_OUT as usize);
        let low_xq = r2.take::<i8>(b * OUT_LOW as usize);
        let low_xscale = r2.take::<f32>(b * BLOCKS_OUT_LOW as usize);
        let attn_out = r2.take::<f32>(b * N_EMBD as usize);
        debug_assert!(r2.off <= q_normed.byte_len());

        Ok(Self {
            rows,
            r1_arena,
            r3_arena,

            flat,
            mix: mk_f32(HC_MIX_DIM as usize)?,
            attn_cur: mk_f32(N_EMBD as usize)?,
            attn_input_norm: mk_f32(N_EMBD as usize)?,
            ffn_cur: mk_f32(N_EMBD as usize)?,

            kq_attn_q8k: mk_u8((BLOCKS_Q8K_GATE_IN as usize) * 292)?,
            kq_ffn_q8k: mk_u8((BLOCKS_Q8K_GATE_IN as usize) * 292)?,
            kq_mid_q8k: mk_u8((BLOCKS_Q8K_DOWN_IN as usize) * 292)?,
            xq_n_embd: mk_i8(N_EMBD as usize)?,
            xscale_n_embd: mk_f32(BLOCKS_N_EMBD as usize)?,
            qr: mk_f32(N_LORA_Q as usize)?,
            qr_normed: mk_f32(N_LORA_Q as usize)?,
            qr_xq: mk_i8(N_LORA_Q as usize)?,
            qr_xscale: mk_f32(BLOCKS_N_LORA_Q as usize)?,
            q,
            q_normed,

            kv_raw: mk_f32(N_HEAD_DIM as usize)?,
            kv_normed: mk_f32(N_HEAD_DIM as usize)?,
            // rows*N_HEAD_DIM ≥ SWA_WINDOW*N_HEAD_DIM for any rows ≥ 128;
            // the eviction pass copies at most SWA_WINDOW rows through it.
            kv_ring_scratch: DeviceBuffer::new(
                id,
                b.max(crate::config::SWA_WINDOW as usize) * N_HEAD_DIM as usize,
            )?,

            kv_cur: mk_f32((2 * N_HEAD_DIM) as usize)?,
            sc_cur: mk_f32((2 * N_HEAD_DIM) as usize)?,

            row_per_b: mk_i32(1)?,
            pos_mod_per_b: mk_i32(1)?,
            n_raw_per: mk_i32(1)?,
            n_raw_offset_per: mk_i32(1)?,
            n_comp_per: mk_i32(1)?,

            heads,
            // Half-sized by default: the f16-scores kernel writes f16
            // into this buffer (see launch_score_batched_htiled_wmma_f16s
            // in attention.rs). N_HEAD * ATTN_SCORES_STRIDE / 2 f32
            // elements ≡ N_HEAD * ATTN_SCORES_STRIDE f16 elements.
            //
            // Doubled when DEEPSTRIX_F32_SCORES=1 so the f32-scores
            // kernel pair has the headroom it needs.
            attn_scores: {
                // Sized at ATTN_SCORES_STRIDE (not ATTN_MIXED_MAX_KEYS): the
                // production batched attention runs on the CSA-gathered dense
                // top-K buffer so n_total ≤ ~640 keys. 2048 stride leaves
                // headroom.
                let per_b = if use_f32_scores() {
                    (N_HEAD * ATTN_SCORES_STRIDE) as usize
                } else {
                    ((N_HEAD * ATTN_SCORES_STRIDE) / 2) as usize
                };
                mk_f32(per_b)?
            },
            indexer_q,
            // Per-token head_weights [N_INDEXER_HEAD] → 128 KB at rows=512.
            indexer_head_weights: mk_f32(N_INDEXER_HEAD as usize)?,
            indexer_scores,
            // Batched IndexerTopk selected: [B, top_k] = 1 MiB at rows=512.
            indexer_selected: DeviceBuffer::new(id, b * INDEXER_TOP_K as usize)?,
            indexer_topk_scratch,
            n_index_comp_per_b: DeviceBuffer::new(id, b)?,
            attn_active_comp_kv: mk_u16((INDEXER_TOP_K * N_HEAD_DIM) as usize)?,
            // Per-boundary state snapshots scratch. Sized for the largest
            // compressor (main ratio==4: 8 × 1024 f32 = 32 KB) × max
            // boundaries (rows/4) = 4 MiB per buffer at rows=512. Reused
            // across compressors.
            // Floor: one ratio-128 boundary snapshot is coff(1)*128*512 =
            // 65536 elems, larger than rows.div_ceil(4)*8192 once rows < 32.
            // Production lanes (rows >= 512) never hit it; this keeps small
            // `alloc_rows` scratches (tests) from tripping slice_view_mut.
            comp_state_kv_snapshots: {
                let max_state_per_b = 8 * (2 * N_HEAD_DIM) as usize; // ratio*coff * width
                let elems = (max_boundaries * max_state_per_b).max(128 * N_HEAD_DIM as usize);
                DeviceBuffer::new(id, elems)?
            },
            comp_state_score_snapshots: {
                let max_state_per_b = 8 * (2 * N_HEAD_DIM) as usize;
                let elems = (max_boundaries * max_state_per_b).max(128 * N_HEAD_DIM as usize);
                DeviceBuffer::new(id, elems)?
            },
            comp_pooled_batched: DeviceBuffer::new(id, max_boundaries * N_HEAD_DIM as usize)?,
            comp_rows_batched: DeviceBuffer::new(id, max_boundaries * N_HEAD_DIM as usize)?,
            comp_pos_per_boundary: DeviceBuffer::new(id, max_boundaries)?,
            low,
            heads_xq,
            heads_xscale,
            low_xq,
            low_xscale,
            attn_out,

            router_logits: mk_f32(N_EXPERT as usize)?,
            router_logits_host: vec![0.0f32; b * (N_EXPERT as usize)],

            gate_sh: mk_f32(N_FF_SHARED as usize)?,
            up_sh: mk_f32(N_FF_SHARED as usize)?,
            mid_sh: mk_f32(N_FF_SHARED as usize)?,
            mid_sh_xq: mk_i8(N_FF_SHARED as usize)?,
            mid_sh_xscale: mk_f32(BLOCKS_N_FF_SHARED as usize)?,

            // M61: hot-expert prefill scratch, only when the het-split
            // weights will be loaded (same env gate as weights.rs).
            hot,
        })
    }
}
