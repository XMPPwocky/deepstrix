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

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{Device, DeviceBuffer};

use crate::attention::{ATTN_MIXED_MAX_KEYS, ATTN_SCORES_STRIDE};
use crate::config::{ENGRAM_CHUNK, ENGRAM_IN, ENGRAM_OUT, 
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

/// Row pitch (halves) of the f16 activation buffers: +64 keeps every
/// production dim off a power-of-two byte stride.
pub const F16_PAD: u32 = 64;
pub const fn f16_pitch(dim: u32) -> u32 { dim + F16_PAD }

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
/// Rows the DSpark residual capture is sized for. Only speculative verifies
/// draft, and those are bounded by `V41_SMALL_B_OFFLOAD_MAX` (<= 8).
/// Rows of main-model residual the batched path can capture for the DSpark
/// drafter. 128 = `MTP_WINDOW`, the drafter's ring size: prefill seeding wants
/// to replay a FULL ring's worth of prompt positions, not just a verify batch.
/// Costs 3 slots * 128 rows * N_EMBD * 4 B ~= 11 MB.
pub const MTP_CAP_ROWS: usize = 128;

pub struct BatchDgpuScratch {
    /// Row capacity every B-scaled buffer was sized for. Callers must
    /// never run a batch larger than this through the scratch.
    pub rows: usize,

    /// S2 shared selection, PER LANE (V4.1 CSA2 §1.4). The 8 index-source layers run the
    /// indexer; layers between two sources reuse the most recent source's selection.
    ///
    /// This lives in the LANE scratch on purpose. `BatchDgpuShared` is one instance and
    /// the pipelined path runs lane A then lane B at the SAME layer
    /// (`forward_prefill.rs:425-429`), so `sd.indexer_selected` holds one lane's choice
    /// and is immediately overwritten by the other's — the lanes are different TOKENS and
    /// their selections legitimately differ. Keying off the lane's own scratch removes
    /// the aliasing without threading a lane id anywhere.
    ///
    /// `[rows, INDEXER_TOP_K]` i32 (1 MB at rows=512) — the INDICES, not the gathered
    /// rows: re-running the gather at a reuse layer is cheap (~512 rows/token), while
    /// caching `attn_active_comp_kv` would cost 268 MB/lane and compete with the 4.3 GB
    /// attention scratch at 130K. Scoring the whole store is the expensive part, and that
    /// is what gets skipped.
    pub indexer_sel_saved: DeviceBuffer<i32>,
    /// Per-token `min(n_comp, INDEXER_TOP_K)` for the saved selection.
    pub indexer_nsparse_saved: DeviceBuffer<i32>,
    /// Store group the saved selection belongs to
    /// (`kv_source_of(layer).unwrap_or(layer)`), or -1 for none. Guards against a
    /// selection leaking across a kv-source boundary.
    pub indexer_saved_store: i32,

    /// `[B, HC_DIM]` — per-token residual (cross-layer flow).
    pub residual: DeviceBuffer<f32>,
    pub residual_next: DeviceBuffer<f32>,
    /// `[B, HC_MIX_DIM]` — sinkhorn output (post + comb embedded). Written
    /// in P1 (attn) and again in P8 (ffn); the P8 value is read by P12
    /// `hc_post` AFTER the lane switch, so it is per-lane.
    pub split: DeviceBuffer<f32>,
    /// `[B, HC_MIX_DIM]` — V4.1 single-pass mHC (ARCH_SPEC §1.1): the PREVIOUS
    /// sub-block's sinkhorn output per row, which the current sub-block's
    /// collapse reads instead of its own. Reset to one-hot(copy 0) at layer 0;
    /// only the first N_HC entries of each row are read. Unused on V4-Flash.
    pub hc_pre_carry: DeviceBuffer<f32>,
    /// V4.1 Engram prefill: staged rows `[B, ENGRAM_IN]` f32
    /// (`stage_engram_rows_batch`) and per-`ENGRAM_CHUNK` Q8 / wkv-output scratch.
    pub engram_rows: DeviceBuffer<f32>,
    pub engram_xq: DeviceBuffer<i8>,
    pub engram_xscale: DeviceBuffer<f32>,
    pub engram_kv: DeviceBuffer<f32>,
    pub engram_rows_ready: bool,
    /// DSpark: hc-collapsed residual ENTERING each of `MTP_SRC_LAYERS`, for
    /// every row of the batch — `[MTP_CAP_ROWS, 3 * N_EMBD]`.
    ///
    /// A speculative verify has to hand the drafter the residual of whatever
    /// position ends up being the new head, and which row that is is only known
    /// AFTER the batch has run. So capture every row and select afterwards.
    /// Lives on the scratch rather than behind a new parameter because
    /// `forward_layer_pre_moe_v2` already takes `bd` and has five call sites.
    /// Off unless `mtp_capture_rows > 0`.
    pub mtp_src: DeviceBuffer<f32>,
    /// How many rows the last capture wrote, and the ABSOLUTE position of row 0.
    /// Prefill is chunked and two-laned, so the server cannot infer which
    /// positions `mtp_src` holds -- the capture records it here.
    pub mtp_captured: usize,
    pub mtp_captured_pos0: u32,
    pub mtp_capture_rows: usize,
    /// Rows `[0, mtp_lane_cut)` of the last batch were in lane A, the rest in
    /// lane B. The capture is PER LANE and indexed lane-locally, so a caller
    /// that wants global batch row `r` must know where the cut fell. Recorded
    /// on lane A's scratch by the pipelined driver.
    pub mtp_lane_cut: usize,
    /// Uniform `1/N_HC` weights, so `hc_weighted` computes the mean over the
    /// hyper-connection copies — the drafter's `main_hidden`.
    pub mtp_hc_mean: DeviceBuffer<f32>,
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
    /// The remote shard's weighted MoE partial for this lane's chunk,
    /// `[B, N_EMBD]` f32, uploaded from the reply and added at `ffn_combine`
    /// exactly like `hot_ffn_moe_dgpu`.
    ///
    /// PER-LANE, not shared: it is produced in pre-MoE and consumed in post-MoE,
    /// and the two pipeline lanes interleave those phases, so a shared buffer
    /// would let one lane's partial overwrite the other's.
    ///
    /// f32 rather than the protocol's f16 for the first correct version — the
    /// combine's `vec_add` is f32, and taking f16 would mean writing a new
    /// `f16_to_f32_add` kernel inside the same change that first alters
    /// numerics. Costs 2x on the reply (10 KB/token vs 5 KB): ~14.5 ms vs
    /// 7.2 ms at B=1024 on the measured 724 MB/s link, against ~150 ms of local
    /// compute. Switch to f16 once the split is validated.
    pub remote_ffn_moe: Option<DeviceBuffer<f32>>,
    /// Did pre-MoE actually fill `remote_ffn_moe` for the layer now in flight?
    /// The buffer is allocated whenever a remote is attached, but only layers
    /// the remote OWNS produce a partial — and in phase C1 none are consumed at
    /// all. Set on upload, cleared once combined, so a stale partial can never
    /// be added to the wrong layer.
    pub remote_ffn_moe_valid: bool,
    /// Which layer produced the pending partial. Diagnostic only: post-MoE does
    /// not otherwise know its layer index, and "which layers actually combined"
    /// is the question that distinguishes a dead write from a write that lands
    /// somewhere the final logits never read.
    pub remote_ffn_moe_layer: i32,
    /// Request in flight on the remote shard for this lane's layer.
    ///
    /// Submitted in pre-MoE and awaited in POST-MoE, so box 2 computes its half
    /// while this box's iGPU computes the other half. Awaiting it at submit time
    /// (as the first cut did) serialises the ~74 ms round trip at B=1024 in front
    /// of local compute and throws away the whole point of overlapping.
    pub remote_ticket: Option<crate::het::remote_experts::Ticket>,
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
    /// mHC PRE-SCALED path (verify-sized batches only). Decode computes
    /// `mix = (W @ x) * inv_rms` via `rms_nw_mw.launch_inv_only` +
    /// `matvec_pre_scaled`; prefill computed `mix = W @ normalize(x)`, which
    /// rounds 20480 normalised values to f32 BEFORE the dot product instead of
    /// dotting the raw values and scaling once. Mathematically identical,
    /// numerically not -- and `hc_split_sinkhorn`'s 20 iterations amplify the
    /// difference, which is the layer-0 seed of KNOWN_BUGS #0b. One row at a
    /// time, so these are single-row buffers.
    pub mhc_inv_scalar: DeviceBuffer<f32>,
    pub mhc_rms_partials: DeviceBuffer<f32>,
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
    /// f16 activations for the q8_0 f16x WMMA GEMMs (2026-09-08), row pitch
    /// `f16_pitch(dim)` = dim + 64 halves so rows never sit at a power-of-two
    /// stride (out_a's 64 KB rows aliased L2 sets: 18% -> 52% of peak).
    /// x16_n_embd doubles as the shared-expert input (dead by then).
    pub x16_n_embd: DeviceBuffer<u16>,
    /// Q8_K of `ffn_input_norm`, staged for the REMOTE expert shard.
    ///
    /// The dGPU hot-expert path already produces exactly these bytes into
    /// `BatchDgpuHotScratch::moe_xq` ("bit-identical to the iGPU's d_xq_q8k —
    /// same f32 input, same kernel"), but V4.1 runs with `hot_experts = None`
    /// so that scratch is never allocated. Same kernel, own buffer, so the
    /// bytes shipped to box 2 are provably the ones the local iGPU would have
    /// quantised — the remote's arithmetic matches the local path by
    /// construction rather than by agreement.
    ///
    /// Only allocated when a remote shard is attached; `None` costs nothing.
    pub remote_xq: Option<DeviceBuffer<u8>>,

    pub qr16: DeviceBuffer<u16>,
    pub heads16: DeviceBuffer<u16>,
    pub low16: DeviceBuffer<u16>,
    pub mid_sh16: DeviceBuffer<u16>,
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
    /// DECODE's attention scratch, for the verify's per-row replay of the
    /// decode attention chain (`V41_VERIFY_DECODE_ATTN=1`).
    ///
    /// Separate from `attn_scores` because the stride conventions differ:
    /// `launch_score_b1_htiled_wmma` hardcodes `ATTN_MIXED_MAX_KEYS` (82176)
    /// while the batched pair takes `attn_scores_stride` (<= 3072). Slicing the
    /// batched buffer per row would write ~27x past its end. ~21 MB, reused
    /// across rows and lanes because both issue on `de.compute` in order.
    pub verify_scores: DeviceBuffer<f32>,
    pub verify_inv: DeviceBuffer<f32>,
    pub verify_partials: DeviceBuffer<f32>,
    // ---- CSA indexer per-token scratch (P5i) ----
    /// `[B, N_INDEXER_HEAD * N_INDEXER_HEAD_DIM]` — per-token indexer Q
    /// (matvec(attn_q_b) output, then RoPE + QAT in place). R3 view @0:
    /// written by matvec_q, last read by the indexer score kernel (P5i).
    pub indexer_q: DeviceBuffer<f32>,
    /// f16 copy of `indexer_q` for the GEMM-shaped score kernel (2026-09-08),
    /// `[rows, 64*128]` halves = 8 MiB at rows=512.
    pub indexer_q16: DeviceBuffer<u16>,
    /// per-token flag for the threshold top-k fast path (2026-09-08): 1 = selected,
    /// 0 = fall through to the bitonic chain. `[rows]` u32.
    pub indexer_topk_done: DeviceBuffer<u32>,
    /// `[B, N_INDEXER_HEAD]` — per-token head_weights (matvec(proj)
    /// output, post-scale).
    pub indexer_head_weights: DeviceBuffer<f32>,
    /// `[B, ATTN_MIXED_MAX_KEYS]` — batched IndexerScore output. Stride
    /// per token = MAX_KEYS. R1 view @0 (the largest R1 member: 96.5 MiB
    /// at rows=512): written by the score kernel, last read by topk (P5i).
    pub indexer_scores: DeviceBuffer<f32>,
    /// ARCH_SPEC §1.5: the candidate blocks layer 20 publishes, per ROW, consumed
    /// by index sources 24/28/32/36. `[B, ceil(MAX_KEYS / CANDIDATE_BLOCK_SIZE)]`
    /// — 6.3 MiB at rows=512, and only allocated when the pool is on.
    pub candidate_block_score: DeviceBuffer<f32>,
    pub candidate_threshold: DeviceBuffer<u32>,
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
    /// V4.1 CSA2 index-K staging, batched twin of `DgpuScratch::index_k_{row,normed}`
    /// (S1a). `[max_boundaries, N_INDEXER_HEAD_DIM]` each: `wk(latent)` then
    /// `k_norm(..)`. Written between the compressor's batched `rms_w` and its batched
    /// `rope`, because the latter rotates `comp_rows_batched` in place.
    pub index_k_rows_batched: DeviceBuffer<f32>,
    pub index_k_normed_batched: DeviceBuffer<f32>,
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
pub const HOT_MAX_EXPERTS: usize = crate::config::N_EXPERT as usize;
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
    /// WMMA MoE path (IGPU_MOE_WMMA, gfx11): f16 cast of the recv'd
    /// ffn_input_norm `[B, N_EMBD]` (chain head) and f16 fused-swiglu mid
    /// `[B*n_used, N_FF_EXP]` (gate/up → down). 4 + 12 MiB at rows=512.
    pub d_x16: DeviceBuffer<u16>,
    pub d_mid16: DeviceBuffer<u16>,
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

    /// Grow `group_count` to hold `n_groups` group ids.
    ///
    /// The builders' `n_expert` argument is a BUFFER LIMIT, not a guard: a group
    /// id at or above it is dropped with no error. `N_EXPERT` is right only while
    /// the group id IS a raw expert id (the plain builder, and the pager's dense
    /// window where slot == expert id). The speculative verify's SPARSE residency
    /// emits ABSOLUTE pool slots instead, so its bound is the pool -- sizing this
    /// to `N_EXPERT` there was #0b (see `ExpertPager::sparse_group_bound`).
    ///
    /// Grow-only, and a no-op on the prefill path (`N_EXPERT` is already the
    /// allocated size). The wide case is ~10 KB.
    pub fn ensure_group_bound(&mut self, n_groups: u32) -> eyre::Result<()> {
        if self.group_count.len() >= n_groups as usize {
            return Ok(());
        }
        self.group_count = DeviceBuffer::new(self.group_count.device_id(), n_groups as usize)?;
        Ok(())
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
            d_x16: DeviceBuffer::new(id, b * (N_EMBD as usize))?,
            d_mid16: DeviceBuffer::new(id, b * (N_EXPERT_USED as usize) * (N_FF_EXP as usize))?,
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

    /// Grow the by-expert member/work lists to hold `n_groups` group ids with a
    /// stride of `max_per_expert` members, for a chunk of `b` tokens.
    ///
    /// Companion to [`BatchIgpuScratch::ensure_group_bound`] -- the builder's
    /// bound and these three arrays are ONE decision and must be sized together.
    ///
    /// `expert_members` is a dense `[n_groups x max_per_expert]` matrix, so the
    /// wide (sparse-verify) case is only affordable because `max_per_expert` is
    /// the ACTUAL batch there, not `B_MAX`: a group holds at most one entry per
    /// token (a token's top-k picks are distinct experts, and the pager maps
    /// distinct experts to distinct slots), so `b` is exact, and a
    /// 2600-slot pool at b=6 is 62 KB against prefill's 384 x B_MAX.
    ///
    /// Grow-only: a later large-B prefill chunk must still find its own
    /// `N_EXPERT x B_MAX` capacity, so this never shrinks.
    /// Whether [`Self::ensure_group_capacity`] would reallocate. Callers must
    /// drain the stream first when it would: growing FREES the old buffers, and a
    /// previous layer's by-expert kernels may still be queued reading them.
    pub fn group_capacity_ok(&self, n_groups: u32, max_per_expert: u32, b: usize) -> bool {
        self.expert_members.len() >= (n_groups as usize).saturating_mul(max_per_expert as usize)
            && self.work_items.len() >= (n_groups as usize) + b * (N_EXPERT_USED as usize)
    }

    pub fn ensure_group_capacity(
        &mut self,
        n_groups: u32,
        max_per_expert: u32,
        b: usize,
    ) -> eyre::Result<()> {
        let members = (n_groups as usize)
            .checked_mul(max_per_expert as usize)
            .ok_or_else(|| eyre!("group capacity overflow: {n_groups} x {max_per_expert}"))?;
        if self.expert_members.len() < members {
            self.expert_members =
                DeviceBuffer::new(self.expert_members.device_id(), members)?;
        }
        // `moe_work_items_builder` walks the whole group space and emits
        // `ceil(group_count[g] / chunk)` items per active group: at most one
        // header per group plus one per member.
        let wi_len = (n_groups as usize) + b * (N_EXPERT_USED as usize);
        if self.work_items.len() < wi_len {
            let id = self.work_items.device_id();
            self.work_items = DeviceBuffer::new(id, wi_len)?;
            self.staged_work_items = DeviceBuffer::new(id, wi_len)?;
            self.chunked_work_items = DeviceBuffer::new(id, wi_len)?;
        }
        Ok(())
    }
}

/// Per-token element counts of the R1 / R3 arena members.
fn flat_len(rows: usize) -> usize {
    rows * HC_DIM as usize
}
fn q_len(rows: usize) -> usize {
    rows * Q_FLAT as usize
}
/// Keys the per-token indexer scratch must hold. ZERO when this model's
/// indexer can never fire (V4.1): `indexer_scores` is `rows * keys` f32 and
/// dominates the R1 arena (268 MiB at rows=512), and `attn_active_comp_kv`
/// another 268 MiB — all of it dead on a model whose `need_mask` gate
/// (`ratio == 4`) is unreachable.
fn indexer_scratch_keys() -> usize {
    if crate::attention::indexer_scratch_needed() {
        ATTN_MIXED_MAX_KEYS as usize
    } else {
        0
    }
}
fn indexer_scores_len(rows: usize) -> usize {
    rows * indexer_scratch_keys()
}
fn indexer_q_len(rows: usize) -> usize {
    rows * (N_INDEXER_HEAD * N_INDEXER_HEAD_DIM) as usize
}
fn indexer_topk_scratch_len(rows: usize) -> usize {
    if !crate::attention::indexer_scratch_needed() {
        return 0;
    }
    // N-level bitonic tree merge, sized from the SAME ladder the launcher
    // walks (`indexer::topk_merge_levels`). It used to hardcode exactly two
    // levels here and in `scratch.rs` and a third time in the launcher; the
    // launcher then hard-errored past 262,144 compressed positions, mid-layer.
    let per_row: u32 = crate::indexer::topk_merge_levels(ATTN_MIXED_MAX_KEYS, INDEXER_TOP_K)
        .iter()
        .sum();
    rows * per_row as usize
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
            indexer_sel_saved: mk_i32(crate::indexer::INDEXER_TOP_K as usize)?,
            indexer_nsparse_saved: mk_i32(1)?,
            indexer_saved_store: -1,
            residual: mk_f32(HC_DIM as usize)?,
            residual_next: mk_f32(HC_DIM as usize)?,
            split: mk_f32(HC_MIX_DIM as usize)?,
            hc_pre_carry: mk_f32(HC_MIX_DIM as usize)?,
            engram_rows: if cfg!(feature = "v41") { mk_f32(ENGRAM_IN as usize)? } else { DeviceBuffer::new(id, 32)? },
            engram_xq: DeviceBuffer::new(id, if cfg!(feature = "v41") { (ENGRAM_CHUNK * ENGRAM_IN) as usize } else { 32 })?,
            engram_xscale: DeviceBuffer::new(id, if cfg!(feature = "v41") { (ENGRAM_CHUNK * ENGRAM_IN / 32) as usize } else { 32 })?,
            engram_kv: DeviceBuffer::new(id, if cfg!(feature = "v41") { (ENGRAM_CHUNK * ENGRAM_OUT) as usize } else { 32 })?,
            engram_rows_ready: false,
            mtp_src: DeviceBuffer::new(
                id,
                MTP_CAP_ROWS
                    * crate::het::mtp::MTP_SRC_LAYERS.len()
                    * crate::config::N_EMBD as usize,
            )?,
            mtp_capture_rows: 0,
            mtp_lane_cut: 0,
            mtp_captured: 0,
            mtp_captured_pos0: 0,
            mtp_hc_mean: {
                let nh = crate::config::N_HC as usize;
                let mut mb = DeviceBuffer::<f32>::new(id, nh * MTP_CAP_ROWS)?;
                mb.copy_from_host(&vec![1.0f32 / nh as f32; nh * MTP_CAP_ROWS])?;
                mb
            },
            after_attn_hc: mk_f32(HC_DIM as usize)?,
            ffn_input_norm: mk_f32(N_EMBD as usize)?,
            d_selected: mk_i32(N_EXPERT_USED)?,
            d_ew: mk_f32(N_EXPERT_USED)?,
            ffn_shared: mk_f32(N_EMBD as usize)?,
            ffn_moe_recv: mk_f32(N_EMBD as usize)?,
            pos_per_b: mk_i32(1)?,
            hot_ffn_moe_dgpu,
            remote_ffn_moe: if std::env::var("V41_REMOTE_ADDR").is_ok() {
                Some(DeviceBuffer::new(id, b * N_EMBD as usize)?)
            } else {
                None
            },
            remote_ffn_moe_valid: false,
            remote_ffn_moe_layer: -1,
            remote_ticket: None,
        })
    }
}

/// Total score slots (keys) `BatchDgpuShared::attn_scores` must hold for a
/// shared set of `rows` lane rows at `n_kv_max` context.
///
/// Never below the legacy `rows * N_HEAD * ATTN_SCORES_STRIDE`, so V4-Flash
/// and `n_kv_max == 0` callers keep today's exact allocation.
pub fn attn_scores_capacity_keys(rows: usize, n_kv_max: u32) -> usize {
    let legacy = rows * (N_HEAD as usize) * (ATTN_SCORES_STRIDE as usize);
    if n_kv_max == 0 || crate::attention::attn_legacy_stride() {
        return legacy;
    }
    // Every batch shape the two CED phases can present. `b` is the rows of
    // ONE lane, and the replay's segment is at most SWA_WINDOW rows split
    // across two lanes.
    // Widest raw window a row can have: a text row sees SWA_WINDOW, a row
    // inside a vision image block sees IMAGE_RAW_WINDOW_MAX. The allocator
    // does not know whether a tower is loaded, so charge the larger (+25 MiB
    // at rows=512) rather than error mid-prefill on an image chunk.
    let w = crate::het::image_spans::IMAGE_RAW_WINDOW_MAX;
    let replay_rows = (crate::config::SWA_WINDOW as usize).div_ceil(2).min(rows);
    let ced = crate::het::forward_prefill::ced_enabled();
    let mut need = 0usize;
    for (layer, &ratio) in crate::config::COMPRESS_RATIOS.iter().enumerate() {
        if ratio == 0 {
            continue;
        }
        let keys = if crate::attention::scored_keys_are_gathered(ratio) {
            w + crate::config::INDEXER_TOP_K
        } else {
            w + n_kv_max.div_ceil(ratio)
        } as usize;
        // Under CED the decoder layers run ONLY in the bounded replay, at
        // <= SWA_WINDOW/2 rows per lane. Charging them the encoder's `rows`
        // would cost 8x for nothing. With `V41_CED=0` every layer runs over
        // the full chunk, so charge `rows`.
        let b = if ced && layer >= crate::config::CED_DECODER_START {
            replay_rows.max(1)
        } else {
            rows
        };
        need = need.max(b * (N_HEAD as usize) * keys);
    }
    need.max(legacy)
}

impl BatchDgpuShared {
    /// Score slots (keys) `attn_scores` actually holds, in the units the
    /// kernels index it with (f16 slots on the production `_f16s` pair).
    pub fn attn_scores_capacity_keys(&self) -> usize {
        if use_f32_scores() {
            self.attn_scores.len()
        } else {
            self.attn_scores.len() * 2
        }
    }

    /// Per-(row, head) stride for ONE batched score + softmax-wsum pair.
    /// Both kernels must be given this same value; see
    /// [`crate::attention::attn_scores_stride`].
    pub fn attn_scores_stride(&self, batch: u32, n_total_max: u32) -> eyre::Result<u32> {
        crate::attention::attn_scores_stride(
            self.attn_scores_capacity_keys(),
            batch,
            N_HEAD,
            n_total_max,
        )
    }

    /// Allocate for the full `B_MAX` chunk (single-lane drivers).
    pub fn alloc(dgpu_device: Device) -> eyre::Result<Self> {
        Self::alloc_rows(dgpu_device, B_MAX)
    }

    /// Allocate for `rows` tokens — the max rows of any lane that will
    /// use this shared set (`B_MAX.div_ceil(2)` for the two-lane driver,
    /// ~629 MiB dGPU). Every batched entry point checks `b <= rows`.
    ///
    /// Sizes `attn_scores` at the legacy [`ATTN_SCORES_STRIDE`] floor, which
    /// is right for V4-Flash at any context but caps V4.1's CED replay at
    /// ~24K tokens and its ratio-2 encoder at 5888 — use
    /// [`Self::alloc_rows_ctx`] when the caller knows `n_kv_max`.
    pub fn alloc_rows(dgpu_device: Device, rows: usize) -> eyre::Result<Self> {
        Self::alloc_rows_ctx(dgpu_device, rows, 0)
    }

    /// Context-aware twin of [`Self::alloc_rows`]: sizes `attn_scores` so
    /// that every layer of THIS model can score its whole (ungathered)
    /// compressed store at `n_kv_max` tokens.
    ///
    /// `n_kv_max == 0` keeps the legacy floor sizing.
    ///
    /// The buffer must satisfy `b * N_HEAD * (raw + n_comp) <= capacity` for
    /// every call, and the two callers have very different shapes:
    ///
    /// * the CED **encoder** runs `b = rows` (512) over ratio-2 layers, so it
    ///   needs `rows * N_HEAD * (SWA_WINDOW + n_kv/2)`;
    /// * the CED **decoder replay** runs `b <= SWA_WINDOW/2` (64) over
    ///   ratio-1 layers, so it needs `64 * N_HEAD * (SWA_WINDOW + n_kv)`.
    ///
    /// Because the stride is chosen per launch from the capacity
    /// (`attention::attn_scores_stride`), sizing for the max of those two
    /// PRODUCTS — rather than for one worst-case stride at `rows` — halves
    /// the allocation: 3.3 GiB instead of 6.6 GiB at 100K.
    pub fn alloc_rows_ctx(
        dgpu_device: Device,
        rows: usize,
        n_kv_max: u32,
    ) -> eyre::Result<Self> {
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
        // Compressor boundaries per chunk: at most one every `min_ratio` positions of a lane's
        // `rows` tokens.
        //
        // The divisor MUST be the model's minimum non-zero compress ratio, not a literal. It was
        // hardcoded to 4, which holds for V4-Flash (its ratios are 4 and 128) but NOT for V4.1:
        // CSA2 gives layers 20-39 ratio 1, where EVERY position is a boundary. That under-allocated
        // 4x and `comp_pos_per_boundary` overran as soon as a prefill chunk exceeded `rows/4`
        // ("slice_view out of range: offset=0 len=204 parent_len=128"), panicking the worker on any
        // prompt past ~128 tokens. Never exercised before because paged mode took the per-token
        // fallback and the parity harness runs t_n=6.
        let min_ratio = crate::config::COMPRESS_RATIOS
            .iter()
            .copied()
            .filter(|&r| r > 0)
            .min()
            .unwrap_or(4)
            .max(1) as usize;
        let max_boundaries = rows.div_ceil(min_ratio);

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

            mhc_inv_scalar: DeviceBuffer::new(id, 1)?,

            mhc_rms_partials: DeviceBuffer::new(id, 16)?,
            mix: mk_f32(HC_MIX_DIM as usize)?,
            attn_cur: mk_f32(N_EMBD as usize)?,
            attn_input_norm: mk_f32(N_EMBD as usize)?,
            ffn_cur: mk_f32(N_EMBD as usize)?,

            kq_attn_q8k: mk_u8((BLOCKS_Q8K_GATE_IN as usize) * 292)?,
            kq_ffn_q8k: mk_u8((BLOCKS_Q8K_GATE_IN as usize) * 292)?,
            kq_mid_q8k: mk_u8((BLOCKS_Q8K_DOWN_IN as usize) * 292)?,
            xq_n_embd: mk_i8(N_EMBD as usize)?,
            x16_n_embd: DeviceBuffer::new(id, b * f16_pitch(N_EMBD) as usize)?,
            remote_xq: if std::env::var("V41_REMOTE_ADDR").is_ok() {
                Some(DeviceBuffer::new(
                    id,
                    b * (crate::config::BLOCKS_Q8K_GATE_IN as usize) * crate::q8_k::BLOCK_Q8_K_BYTES,
                )?)
            } else {
                None
            },

            qr16: DeviceBuffer::new(id, b * f16_pitch(N_LORA_Q) as usize)?,
            heads16: DeviceBuffer::new(id, b * f16_pitch(Q_FLAT) as usize)?,
            low16: DeviceBuffer::new(id, b * f16_pitch(OUT_LOW) as usize)?,
            mid_sh16: DeviceBuffer::new(id, b * f16_pitch(N_FF_SHARED) as usize)?,
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
            verify_scores: DeviceBuffer::new(
                id,
                N_HEAD as usize * crate::attention::ATTN_MIXED_MAX_KEYS as usize,
            )?,
            verify_inv: DeviceBuffer::new(id, N_HEAD as usize)?,
            verify_partials: DeviceBuffer::new(
                id,
                N_HEAD as usize * 16 * N_HEAD_DIM as usize,
            )?,
            attn_scores: {
                let keys = attn_scores_capacity_keys(rows, n_kv_max);
                // The kernels write f16 unless DEEPSTRIX_F32_SCORES=1, so a
                // f32 element holds two score slots on the production path.
                let f32_elems = if use_f32_scores() { keys } else { keys.div_ceil(2) };
                DeviceBuffer::new(id, f32_elems)?
            },
            indexer_q,
            indexer_q16: DeviceBuffer::new(id, b * (N_INDEXER_HEAD * N_INDEXER_HEAD_DIM) as usize)?,
            indexer_topk_done: DeviceBuffer::new(id, b)?,
            // Per-token head_weights [N_INDEXER_HEAD] → 128 KB at rows=512.
            indexer_head_weights: mk_f32(N_INDEXER_HEAD as usize)?,
            indexer_scores,
            // Batched IndexerTopk selected: [B, top_k] = 1 MiB at rows=512.
            indexer_selected: DeviceBuffer::new(id, b * INDEXER_TOP_K as usize)?,
            indexer_topk_scratch,
            n_index_comp_per_b: DeviceBuffer::new(id, b)?,
            candidate_block_score: if crate::het::forward_layer::candidate_pool_enabled() {
                DeviceBuffer::new(
                    id,
                    b * (crate::attention::ATTN_MIXED_MAX_KEYS as usize)
                        .div_ceil(crate::config::CANDIDATE_BLOCK_SIZE as usize),
                )?
            } else {
                DeviceBuffer::new(id, 1)?
            },
            candidate_threshold: if crate::het::forward_layer::candidate_pool_enabled() {
                DeviceBuffer::new(id, b)?
            } else {
                DeviceBuffer::new(id, 1)?
            },
            // Dense per-token top-K gather target — only ever written by the
            // CSA gather, which cannot run when no layer is gathered.
            attn_active_comp_kv: if crate::attention::indexer_scratch_needed() {
                mk_u16((INDEXER_TOP_K * N_HEAD_DIM) as usize)?
            } else {
                DeviceBuffer::new(id, 1)?
            },
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
            index_k_rows_batched: DeviceBuffer::new(
                id,
                max_boundaries * N_INDEXER_HEAD_DIM as usize,
            )?,
            index_k_normed_batched: DeviceBuffer::new(
                id,
                max_boundaries * N_INDEXER_HEAD_DIM as usize,
            )?,
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
