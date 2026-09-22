//! Layer-major batched prefill.
//!
//! Two production entry points:
//!
//! * `forward_prompt_batch_v2` — single-lane batched prefill. Stateless
//!   matmuls + HC stages run in single B-wide launches against
//!   [`BatchDgpuScratch`] (B-extended per-token buffers); stateful kernels
//!   (rope, KV append, compressor, attention, iGPU MoE) stay in a serial
//!   inner loop using `DeviceBuffer::slice_view` per batch element.
//! * `forward_prefill_pipelined` — two-lane pipelined wrapper around v2;
//!   alternates lane A / lane B so iGPU MoE on chunk N overlaps with
//!   dGPU work on chunk N+1. Used by `deepstrix-chat`.
//!
//! ## Layer-major vs token-major
//!
//! Both produce identical state (per-layer KV cache + compressor are
//! commutative across batch elements only because layer N's per-position
//! state for token b at position pos0+b is only ever written by that
//! one call). Layer-major lets batched kernels amortize per-layer
//! weight reads across the batch.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::DeviceBuffer;

use crate::config::{
    BLOCKS_N_EMBD, BLOCKS_N_FF_SHARED, BLOCKS_N_LORA_Q, BLOCKS_OUT_LOW, EXPERT_WEIGHT_SCALE,
    GROUP_DIM, HC_DIM, HC_MIX_DIM, INDEXER_COMP_WIDTH, INDEXER_TOP_K, N_EMBD, N_EXPERT,
    N_EXPERT_USED, N_FF_SHARED, N_GROUPS, N_HC, N_HEAD, N_HEAD_DIM, N_INDEXER_HEAD,
    N_INDEXER_HEAD_DIM, N_LAYER, N_LORA_Q, N_ROT, N_VOCAB, OUT_LOW, Q_FLAT, RANK, RMS_EPS,
    SINKHORN_EPS, SINKHORN_ITERS, SWA_WINDOW,
};
use crate::attention::{ATTN_MIXED_MAX_KEYS, ATTN_SWA_BATCHED_MAX_KV};
use crate::routing::hash_router_select;

use super::image_spans::{self, ImageSpan};

use super::batch_scratch::{
    BatchDgpuScratch, BatchDgpuShared, BatchIgpuScratch, BatchIgpuShared, B_MAX,
};
use super::engine::HeterogeneousEngine;
use super::prefill_stats::PrefillStats;
use super::scratch::{DgpuScratch, IgpuScratch};
use crate::config::{ENGRAM_CHUNK, ENGRAM_IN, ENGRAM_OUT};
use super::state::{CompKvStore, HetLayerState, HetModelState, KV_CACHE_ROWS};
use super::kv_arena::{store_index_of, KvArena, RowTables, RowTablesDev};
use crate::comp_kv_fp8::FP8_KV_HEAD_ROWS;
use super::sync::{peer_push_f32, peer_push_i32};
use super::weights::{DgpuLayerWeights, HetModelWeights, IgpuLayerWeights};

const ROUTER_WEIGHT_EPS: f32 = 6.103515625e-5;

/// M61 prefill het-split rollback: `DGPU_HOT_PREFILL=0` keeps the decode
/// het-split but routes ALL prefill MoE slots to the iGPU (pre-M61
/// behaviour). Default on when hot experts are loaded.
fn hot_prefill_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("DGPU_HOT_PREFILL").map(|v| v != "0").unwrap_or(true)
    });
    *ON
}

/// Max dGPU-resident slots per token in prefill. Defaults to decode's
/// DGPU_HOT_CAP (default 4) so the prefill slot→device partition matches
/// the decode path exactly — the f32 summation association then matches
/// too, keeping the prefill-vs-sequential oracle drift in the pre-M61
/// kernel-difference class (an uncapped prefill measured 5.4e-2 scaled
/// logit drift vs the 5e-2 oracle bound; matched caps stay ~2.5e-2).
/// Offload cost of the cap is ~nil: with ~8 of 256 experts resident,
/// tokens with >4 resident slots are vanishingly rare.
/// Is the two-box split ACTIVE (phase C2+), i.e. does the local iGPU skip the
/// remote-owned experts and consume the remote's partial at the combine?
/// Default OFF: with it off the local side still computes every expert and the
/// remote reply is discarded (phase C1), so numerics are untouched.
fn remote_split_active() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        matches!(std::env::var("V41_REMOTE_SPLIT").as_deref(),
                 Ok("1") | Ok("on") | Ok("2") | Ok("3") | Ok("4"))
    });
    *ON
}

/// Apply the exclusion remap (local iGPU skips remote-owned picks)?
/// Modes 1 and 3. Mode 2 keeps an all-local remap.
/// Hand the CED replay's DECODER-layer MoE entirely to box 2 (`V41_REPLAY_OFFLOAD=1`).
///
/// Separate from `V41_T2_CATCHALL` on purpose: catch-all is a DECODE-path policy,
/// this is a PREFILL-path one, and they have different capacity preconditions.
/// Box 2's per-layer decoder capacity must be >= the replay union (162 at B=128,
/// measured) or `ensure_layer` cannot make the layer resident for one dispatch.
/// Tying the two together makes a box-2 spec sized for decode silently fail prefill.
fn replay_offload_enabled() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("V41_REPLAY_OFFLOAD").map(|v| v != "0").unwrap_or(false)
    });
    *B
}

/// Mirror of `forward_layer::index_k_enabled` — `V41_INDEX_K=1`, default OFF.
use super::forward_layer::is_index_source_layer;

fn index_k_enabled() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        matches!(std::env::var("V41_INDEX_K").as_deref(), Ok("1") | Ok("on"))
    });
    *B
}

fn remote_exclude() -> bool {
    static E: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        matches!(std::env::var("V41_REMOTE_SPLIT").as_deref(), Ok("1") | Ok("on") | Ok("3"))
    });
    *E
}

/// Add the remote partial at the combine? Mode 1 only.
///
/// The 2/3 split exists to separate the two halves of the change: mode 3
/// EXCLUDES but does not ADD, so its error is purely "contributions missing",
/// while mode 1's extra error over mode 3 is purely "what the partial added".
/// Comparing 1 vs 3 vs 2 says which half is broken without guessing.
fn remote_add_partial() -> bool {
    static A: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        matches!(std::env::var("V41_REMOTE_SPLIT").as_deref(), Ok("1") | Ok("on") | Ok("4"))
    });
    *A
}

/// `V41_REMOTE_SPLIT=2`: take the het-split dispatch with an ALL-LOCAL remap and
/// do NOT add the remote partial. Arithmetically identical to the control, so it
/// isolates "does handing the dispatch a remap change the maths at all?" from
/// "is the exclusion/combine balance right?".
fn remote_split_dryrun() -> bool {
    static D: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        matches!(std::env::var("V41_REMOTE_SPLIT").as_deref(), Ok("2"))
    });
    *D
}

fn hot_prefill_cap() -> u32 {
    static CAP: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        if crate::het::weights::igpu_dedup_hot() {
            // M63: must equal the decode cap (see dgpu_hot_cap) — both are
            // pinned to N_EXPERT_USED, so the partitions still match.
            return crate::het::weights::dgpu_hot_cap();
        }
        std::env::var("DGPU_HOT_CAP_PREFILL")
            .ok()
            .or_else(|| std::env::var("DGPU_HOT_CAP").ok())
            .and_then(|s| s.parse().ok())
            .unwrap_or(4)
    });
    *CAP
}

/// Whether the M61 het-split MoE runs for this prefill layer: hot weights
/// present on BOTH devices, hot scratch allocated (the shared R1 views +
/// member lists AND the lane's reduce output — both are gated by the same
/// `DGPU_HOT_EXPERTS` env, so they agree unless a caller mixed scratches
/// from different processes), and not rolled back.
fn prefill_hot_active(
    dlw: &DgpuLayerWeights,
    ilw: &IgpuLayerWeights,
    bd: &BatchDgpuScratch,
    sd: &BatchDgpuShared,
) -> bool {
    let active = hot_prefill_enabled()
        && bd.hot_ffn_moe_dgpu.is_some()
        && sd.hot.is_some()
        && dlw.hot_experts.is_some()
        && ilw.hot_remap.is_some();
    // M63: a packed iGPU buffer has no non-hetsplit fallback — the plain
    // by-expert builder emits raw expert ids. validate_dedup_preconditions
    // rules every disabling knob out at load, so this can only fire on a
    // future path that forgets to.
    debug_assert!(
        active || !ilw.igpu_packed,
        "L{}: iGPU experts packed but the prefill het-split is inactive",
        ilw.layer_idx
    );
    active
}

/// Every batched entry point runs `b` rows through per-lane scratch
/// that was allocated for `bd.rows` / `bi.rows` tokens and the shared
/// scratch allocated for `sd.rows` / `si.rows` (`alloc_rows` on each).
/// Kernels index by `b`, not by the allocation size, so an oversized
/// batch would silently overrun the B-scaled buffers — refuse it up
/// front.
fn check_scratch_rows(
    who: &str,
    b: usize,
    bd: &BatchDgpuScratch,
    bi: &BatchIgpuScratch,
    sd: &BatchDgpuShared,
    si: &BatchIgpuShared,
) -> eyre::Result<()> {
    if b > bd.rows || b > bi.rows || b > sd.rows || b > si.rows {
        return Err(eyre!(
            "{who}: batch of {b} rows exceeds scratch capacity (dGPU lane rows={}, iGPU lane \
             rows={}, dGPU shared rows={}, iGPU shared rows={})",
            bd.rows,
            bi.rows,
            sd.rows,
            si.rows
        ));
    }
    Ok(())
}

/// Per-row visibility for a batch, or `None` when no span touches it (the
/// all-text fast path, bit-identical to the pre-vision code). Errors if a
/// span straddles the batch — see `image_spans::rows_visibility`.
/// How the rows of a `forward_layer_pre_moe_v2` call relate to the KV state
/// it is handed (docs/v41/MULTISTREAM_DECODE_PLAN.md 3.7).
///
/// `Contiguous`: today's meaning — `b` consecutive positions `pos0..pos0+b` of
/// ONE sequence whose KV is `ls` (window at `ls.raw_off/n_raw`, store at
/// `cs.n_comp`), appended and evicted in place. Every existing caller.
///
/// `Arena`: `b` rows of `b` DIFFERENT streams, one position each (K=1), whose
/// KV lives in a `KvArena` handed over as the same `&mut HetLayerState` (the
/// arena's `state.layers[layer]`, lent through `with_kv_source` like a
/// sequence's). Bases and counts come from the per-row `tables`
/// (`KvArena::tables`) and their device copies `dev` (`RowTablesDev::upload`,
/// on `de.compute`, before the step); the state's scalar counters are ignored;
/// nothing is evicted or advanced here (`KvArena::advance` after the step,
/// `compact_raw` before it). The caller has already uploaded `tables.pos_per`
/// into `bd.pos_per_b[0..b]` (rope), as the contiguous callers do per chunk.
/// Text-only, `CedMode::Exact`, no MTP capture, no image visibility.
pub enum RowLayout<'a> {
    Contiguous,
    /// `next_router`: the NEXT layer's weights, for look-ahead routing (the
    /// router of layer+1 applied to this layer's router input; its box-2-owned
    /// picks are sent as prefetch words with this layer's request).
    /// `next_router2`: layer+2's weights for a second layer of lead
    /// (`V41_LOOKAHEAD_DEPTH=2`); measured 0.68 precision two layers out.
    Arena { tables: &'a RowTables, dev: &'a RowTablesDev, next_router: Option<&'a DgpuLayerWeights>, next_router2: Option<&'a DgpuLayerWeights> },
}

/// M7 CED: mode of one batched layer call under V4.1 Causal Encoder-Decoder
/// prefill (tech report §2.2 / §3.2.2, docs/v41/ENGINE_PORT.md "M7 CED").
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CedMode {
    /// The exact all-stage layer (what the reference forward runs everywhere).
    Exact,
    /// The decoder's Full-mode layer (`CED_DECODER_START`) over encoder-only
    /// prompt rows: mHC pre-mix + attn norm + the global-KV projection into
    /// the shared store, nothing else (no window KV, attention, MoE, and no
    /// carry update — the residual/carry entering the layer are left as-is).
    KvSourceOnly,
    /// The same layer over the bounded-replay segment: the store already holds
    /// these positions (written by `KvSourceOnly`), so the projection is
    /// skipped and the causal comp counts are positional (like a reuse
    /// layer); every other stage runs.
    Replay,
}

/// `V41_CED` (default on under the `v41` feature): encoder-only prefill +
/// Decoder SWA Bounded Replay in `forward_prefill_pipelined` (last-token
/// path). `V41_CED=0` restores the exact all-40-layer prefill.
pub fn ced_enabled() -> bool {
    cfg!(feature = "v41") && std::env::var("V41_CED").map(|v| v != "0").unwrap_or(true)
}

/// `V41_ENGRAM_GEMV=1` puts the Engram `wkv` prefill projection back on the
/// `grid.z = batch` Q8 GEMV (see the call site for why it is 65x over
/// roofline). Rollback knob only.
fn engram_gemv_fallback() -> bool {
    std::env::var("V41_ENGRAM_GEMV").map(|v| v != "0").unwrap_or(false)
}

/// `V41_SWA_MIXED=0` puts the ratio-0 layers (V4.1 layers 0 and 1) back on
/// `attention_swa_batched`. See the call site: that kernel runs ~1150
/// `__syncthreads` per workgroup and measured 16.2 ms per 512-row call on
/// V4-Flash at the identical shape, ~78x its own roofline. The batched WMMA
/// score + softmax-wsum pair computes the same thing with `comp_kv = None`
/// and per-row `n_comp = 0` (`attention_mixed.hip`: "when n_comp == 0 and
/// mask is null, the math reduces exactly to attention_swa").
fn swa_via_mixed() -> bool {
    cfg!(feature = "v41")
        && std::env::var("V41_SWA_MIXED").map(|v| v != "0").unwrap_or(true)
}

/// `V41_MHC_NARROW=1` puts the mHC pre-mix back on `f16_matvec_narrow_batched`.
/// That kernel is grid `(HC_MIX_DIM=24, 1, B)` with one workgroup per
/// (out-row, token) and nothing shared between them: at B=512 / HC_DIM=20480
/// its 12,288 workgroups each read an 80 KB activation row and a 40 KB weight
/// row = 1.47 GB of L2/MALL traffic for 42 MB of unique bytes (V4-Flash
/// measured 559 us per call at the narrower 16384 dim, 2.1 TB/s = the MALL
/// wall). `f16_gemm_wmma_lds_tiled` reads X once. Rollback knob only.
/// Use the NARROW f32 matvec for the mHC pre-mix instead of the WMMA GEMM.
///
/// **DEFAULT ON for b <= 64 since 2026-09-16.** Two independent reasons:
///
/// 1. CORRECTNESS. `ARCH_SPEC` §1.1 requires "mHC math entirely fp32"
///    (:167, and `hc_fn [24, 20480] fp32` at :42). `gemm_batched_wmma` casts the
///    ACTIVATIONS down to f16 before multiplying
///    (`f16_gemm_wmma.hip:104`, `vals[e] = (_Float16)x_row[e]`), so the mix runs
///    in f16. `matvec_narrow_batched` upconverts the weight and multiplies and
///    accumulates in f32 (`f16_matvec_narrow.hip:39`), which is what the spec
///    asks for.
///
/// 2. SPEED at this shape. `HC_MIX_DIM` is 24, and the WMMA grid is
///    `(ceil(24/64), ceil(b/64), 1)` = ONE workgroup for any b <= 64, while the
///    K-loop still reads the whole 983 KB weight. Measured 478 us/call at BOTH
///    b=6 and b=512 -- batch-independent, ~196x off roofline at b=6, with a
///    useful-work fraction near 0.05%. `f16.rs:28` already documents
///    `NARROW_ROWS_THRESHOLD = 64` as "calibrated against the mhc_pre_* calls
///    (n_rows=24) on gfx1201"; the batched path simply bypassed it.
///
/// MEASURED back-to-back, DSpark accept, 648-token prompt:
///
///     WMMA   193 ms/tok  5.19 tok/s   sel_sync 184.1 ms   E 2.314
///     narrow 106 ms/tok  9.48 tok/s   sel_sync 115.9 ms   E 2.574
///
/// The `sel_sync` drop is the mechanism: the mix sits on the stream the
/// per-layer `de.compute.synchronize()` waits on, so removing pointless WMMA
/// work shortens the host's blocking sync. E rising is the correctness half --
/// the drafter is untouched, so better agreement means the VERIFY moved closer
/// to the reference.
///
/// Kept WMMA above 64: `matvec_narrow_batched` launches `n_rows * b` workgroups
/// and hits a MALL wall at prefill batches (`forward_prefill.rs` note on the
/// B=512 case). `V41_MHC_NARROW=0` forces WMMA, `=1` forces narrow.
/// Prefill must reproduce DECODE at verify-sized batches.
///
/// The batched path was optimised with `gemm_batched_wmma`, which casts the
/// activations to f16 (`f16_gemm_wmma.hip`, `vals[e] = (_Float16)x_row[e]`),
/// while decode kept the f32 `de.f16.matvec` at every one of these sites. The
/// two paths therefore computed different numbers BY CONSTRUCTION -- fatal for
/// DSpark, whose verify must reproduce decode -- and each site also violates an
/// explicit fp32 requirement:
///
///   - MoE gate            `ARCH_SPEC:104,162`  "Gate math fp32"
///   - compressor kv/score `ARCH_SPEC:78`       "fp32 kv = wkv(x), score = wgate(x)"
///   - mHC pre-mix         `ARCH_SPEC:167`      "mHC math entirely fp32"
///
/// `matvec_batched` is bit-identical to a per-batch loop of decode's `matvec`
/// (see f16.rs), so below the threshold verify == decode exactly. WMMA is kept
/// for large prefill chunks, where `matvec_batched` re-reads the weight per
/// batch element and goes weight-BW-bound. `V41_PREFILL_F32_MATVEC=0` restores
/// the old all-WMMA behaviour; `=1` forces f32 at every batch size.
/// `V41_PREFILL_PRESUBMIT=1`: issue Stage 10 (the shared expert) AFTER the remote
/// submit instead of before the pager, so its ~173 us of dGPU work lands inside
/// the box-2 RPC window rather than ahead of it.
///
/// Default OFF pending a measured A/B on the verify. Decode's equivalent
/// (`decode_presubmit_reorder` -> `defer_shared`, forward_layer.rs:2210) is
/// default ON and measured +5.0% with byte-identical output, but the verify's
/// balance is different and it has to be scored on E and KLD too, not just wall.
fn prefill_presubmit() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("V41_PREFILL_PRESUBMIT").as_deref() == Ok("1")
    });
    *B
}

fn prefill_f32_matvec(b: u32) -> bool {
    match std::env::var("V41_PREFILL_F32_MATVEC").ok().as_deref() {
        Some("0") => false,
        Some("1") => true,
        _ => b <= 64,
    }
}

/// Use DECODE'S EXACT mHC mix kernel (`launch_inv_only` + `matvec_pre_scaled`,
/// one row at a time) instead of the batched form.
///
/// The verify must reproduce decode; the batched path does not, because it
/// normalises first (`W @ normalize(x)`) where decode scales after
/// (`(W @ x) * inv_rms`). Sinkhorn's 20 iterations amplify the f32 difference.
/// Default ON for b <= 8 (a DSpark verify is B<=6); large-B prefill keeps the
/// batched kernel, where per-row launches would dominate.
/// `V41_MHC_PRE_SCALED=0` disables, `=1` forces at every batch size.
fn mhc_pre_scaled_for(b: u32) -> bool {
    match std::env::var("V41_MHC_PRE_SCALED").ok().as_deref() {
        Some("0") => false,
        Some("1") => true,
        _ => b <= 8,
    }
}

fn mhc_narrow_fallback_for(b: u32) -> bool {
    match std::env::var("V41_MHC_NARROW").ok().as_deref() {
        Some("0") => false,
        Some(_) => true,
        None => b <= 64,
    }
}


/// One prompt row entering the decoder (`CED_DECODER_START`), kept on the host
/// for the bounded replay: token id, residual `[HC_DIM]`, mHC carry
/// `[HC_MIX_DIM]`.
struct ReplayRow {
    tok: i32,
    hc: Vec<f32>,
    carry: Vec<f32>,
}

/// `V41_PREFILL_LOGITS_DUMP=<path>`: append the last-token prefill logits
/// (raw f32 LE, `N_VOCAB` per call) — the CED-vs-exact bit-equality gate.
fn dump_prefill_logits(logits: &[f32]) -> eyre::Result<()> {
    let Ok(path) = std::env::var("V41_PREFILL_LOGITS_DUMP") else { return Ok(()) };
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
    let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    f.write_all(&bytes)?;
    Ok(())
}

fn chunk_visibility(pos0: u32, b: usize, spans: &[ImageSpan]) -> eyre::Result<Option<Vec<(u32, u32)>>> {
    if spans.is_empty() {
        return Ok(None);
    }
    let vis = image_spans::rows_visibility(pos0, b, spans)?;
    if vis.iter().all(|&(l, r)| l == 0 && r == 0) {
        // Spans exist elsewhere in the prompt but none inside this batch.
        // (A 1-token span is impossible — validate_spans requires len >= 2 —
        // so all-(0,0) really means "no image rows here".)
        return Ok(None);
    }
    Ok(Some(vis))
}

/// A prompt prefill that the caller drives one chunk at a time, so a scheduler can
/// interleave batched decode steps with it (docs/v41/MULTISTREAM_DECODE_PLAN.md
/// 5.3: chunks and decode steps are separate forwards). Numerically the same as
/// `forward_prefill_pipelined(last_only = true)`: under CED every chunk runs the
/// encoder range in `KvSourceOnly` and the decoder replays the last `SWA_WINDOW`
/// rows at the end; without CED every chunk runs all layers and the last row's
/// logits are taken. `engram_rows`: one flattened `[T * ENGRAM_IN]` buffer per
/// Engram layer for the whole prompt.
pub struct PrefillJob {
    tokens: Vec<i32>,
    input_hcs: Vec<Vec<f32>>,
    engram_rows: Option<Vec<Vec<f32>>>,
    image_spans: Vec<ImageSpan>,
    pos0: u32,
    chunk_rows: usize,
    chunk_start: usize,
    chunk_idx: usize,
    replay: std::collections::VecDeque<ReplayRow>,
    ced: bool,
    started: std::time::Instant,
    /// Set by `prefill_job_chunk` after the last chunk when CED is off (the head
    /// is taken there); `prefill_job_finish` returns it.
    last_logits: Option<Vec<f32>>,
    /// LAZY inputs: when `input_hcs` is empty the caller supplies the next
    /// chunk's rows through `set_chunk_inputs` right before `prefill_job_chunk`
    /// (embeddings + Engram rows for `next_chunk_range()` only). A 135K-token
    /// prompt's whole-prompt inputs are ~15 GB of host RAM and a 35 s Engram
    /// gather up front; per chunk they are ~60 MB and ~150 ms.
    chunk_inputs: Option<(Vec<Vec<f32>>, Option<Vec<Vec<f32>>>)>,
}

impl PrefillJob {
    pub fn new(
        tokens: Vec<i32>,
        input_hcs: Vec<Vec<f32>>,
        engram_rows: Option<Vec<Vec<f32>>>,
        image_spans: Option<&[ImageSpan]>,
        pos0: u32,
        chunk_rows: usize,
    ) -> eyre::Result<Self> {
        let t = tokens.len();
        if t == 0 {
            return Err(eyre!("PrefillJob: empty prompt"));
        }
        if input_hcs.len() != t && !input_hcs.is_empty() {
            return Err(eyre!("PrefillJob: input_hcs len {} != tokens len {t} (empty = lazy per-chunk inputs)", input_hcs.len()));
        }
        let spans = image_spans.unwrap_or(&[]);
        image_spans::validate_spans(spans, pos0, t)?;
        if let Some(rs) = engram_rows.as_ref() {
            let ein = ENGRAM_IN as usize;
            if rs.iter().any(|r| r.len() != t * ein) {
                return Err(eyre!("PrefillJob: engram rows must be [T * ENGRAM_IN] per Engram layer"));
            }
        }
        Ok(Self {
            tokens,
            input_hcs,
            engram_rows,
            image_spans: spans.to_vec(),
            pos0,
            chunk_rows: chunk_rows.clamp(1, B_MAX),
            chunk_start: 0,
            chunk_idx: 0,
            replay: std::collections::VecDeque::new(),
            ced: ced_enabled(),
            started: std::time::Instant::now(),
            last_logits: None,
            chunk_inputs: None,
        })
    }
    pub fn total(&self) -> usize { self.tokens.len() }
    pub fn done_rows(&self) -> usize { self.chunk_start }
    pub fn chunks_done(&self) -> bool { self.chunk_start >= self.tokens.len() }
    pub fn pos0(&self) -> u32 { self.pos0 }
    pub fn tokens(&self) -> &[i32] { &self.tokens }
    /// True when `new` got no `input_hcs`: every chunk needs `set_chunk_inputs`.
    pub fn lazy_inputs(&self) -> bool { self.input_hcs.is_empty() }
    /// `[start, end)` token indices (into `tokens()`) of the chunk the next
    /// `prefill_job_chunk` will run, given the lane capacities it will see.
    pub fn next_chunk_range(&self, lane_caps: (usize, usize)) -> eyre::Result<(usize, usize)> {
        let t = self.tokens.len();
        let chunk_size = self.chunk_rows.min(lane_caps.0 + lane_caps.1);
        let (end, _) = image_spans::plan_chunk(self.pos0, self.chunk_start, t, chunk_size, Some(lane_caps), &self.image_spans)?;
        Ok((self.chunk_start, end))
    }
    /// Lazy mode: the next chunk's layer-0 HCs (one per token of
    /// `next_chunk_range`) and, if the model has Engram, its rows
    /// (`[n * ENGRAM_IN]` per Engram layer).
    pub fn set_chunk_inputs(&mut self, hcs: Vec<Vec<f32>>, engram: Option<Vec<Vec<f32>>>) {
        self.chunk_inputs = Some((hcs, engram));
    }
}

impl HeterogeneousEngine {
    fn emit_prefill_perfetto(&self) -> eyre::Result<()> {
        if let Some(exp_lock) = &self.perfetto {
            let mut exp = exp_lock.lock().unwrap();
            self.dgpu.events.for_each_pair(|name, s, e| {
                let track = if name.contains(".xfer") || name.contains(".peer_push") { &exp.dgpu_xfer } else { &exp.dgpu_compute };
                exp.emit_slice(track, name, s, e)
            })?;
            self.igpu.events.for_each_pair(|name, s, e| {
                let track = if name.contains(".xfer") || name.contains(".peer_push") { &exp.igpu_xfer } else { &exp.igpu_compute };
                exp.emit_slice(track, name, s, e)
            })?;
            exp.re_anchor(self.dgpu.device, &self.dgpu.compute, &self.dgpu.xfer, self.igpu.device, &self.igpu.compute, &self.igpu.xfer)?;
            self.current_device.store(-1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    /// Run the job's next chunk (at most `chunk_rows` rows, split across the two
    /// lanes as the pipelined prefill does). Returns the rows processed, 0 when
    /// the chunks are already done.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_job_chunk(
        &self,
        job: &mut PrefillJob,
        bd_a: &mut BatchDgpuScratch,
        bi_a: &mut BatchIgpuScratch,
        bd_b: &mut BatchDgpuScratch,
        bi_b: &mut BatchIgpuScratch,
        sd: &mut BatchDgpuShared,
        si: &mut BatchIgpuShared,
        head_scratch: &mut DgpuScratch,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        mut pager: Option<&mut super::expert_pager::ExpertPager>,
    ) -> eyre::Result<usize> {
        let t = job.tokens.len();
        if job.chunk_start >= t {
            return Ok(0);
        }
        self.remote_set_phase_busy_poll(false);
        let lane_caps = (bd_a.rows, bd_b.rows);
        let chunk_size = job.chunk_rows.min(lane_caps.0 + lane_caps.1);
        let split = crate::config::CED_DECODER_START;
        let (chunk_end, b_a) = image_spans::plan_chunk(job.pos0, job.chunk_start, t, chunk_size, Some(lane_caps), &job.image_spans)?;
        let chunk_b = chunk_end - job.chunk_start;
        let is_last_chunk = chunk_end == t;
        let (lazy_hcs, lazy_engram) = match job.chunk_inputs.take() {
            Some((h, e)) => {
                if h.len() != chunk_b || e.as_ref().is_some_and(|rs| rs.iter().any(|r| r.len() != chunk_b * ENGRAM_IN as usize)) {
                    return Err(eyre!("prefill_job_chunk: chunk inputs for {} rows, chunk is {chunk_b}", h.len()));
                }
                (Some(h), e)
            }
            None if job.lazy_inputs() => {
                return Err(eyre!("prefill_job_chunk: lazy job has no inputs for chunk {} (set_chunk_inputs)", job.chunk_idx));
            }
            None => (None, None),
        };
        let chunk_input: &[Vec<f32>] = match lazy_hcs.as_ref() { Some(h) => h.as_slice(), None => &job.input_hcs[job.chunk_start..chunk_end] };
        let chunk_tokens = &job.tokens[job.chunk_start..chunk_end];
        let chunk_pos0 = job.pos0 + job.chunk_start as u32;
        if job.chunk_idx == 0 || job.chunk_idx % 16 == 0 || is_last_chunk {
            let elapsed_s = job.started.elapsed().as_secs_f32();
            tracing::info!(chunk = job.chunk_idx, chunk_pos0, tokens_done = job.chunk_start, tokens_total = t,
                elapsed_s = format!("{elapsed_s:.1}"), "prefill_job_progress");
        }
        self.dgpu.events.reset();
        self.igpu.events.reset();
        let chunk_engram: Option<Vec<Vec<f32>>> = match lazy_hcs.is_some() {
            true => lazy_engram,
            false => job.engram_rows.as_ref().map(|rs| {
                let ein = ENGRAM_IN as usize;
                let (a, z) = (job.chunk_start * ein, chunk_end * ein);
                rs.iter().map(|r| r[a..z].to_vec()).collect()
            }),
        };
        if job.ced {
            let cut = self.forward_prompt_batch_v2_pipelined_range(
                bd_a, bi_a, bd_b, bi_b, sd, si, state, weights, chunk_input, chunk_tokens, chunk_pos0,
                None, Some(&job.image_spans), pager.as_deref_mut(), chunk_engram.as_deref(),
                0..split + 1, CedMode::KvSourceOnly, None,
            )?;
            if cut != b_a {
                return Err(eyre!("PrefillJob: CED lane cut {cut} != planned {b_a}"));
            }
            let take = chunk_b.min(SWA_WINDOW as usize);
            let (m, hc) = (HC_MIX_DIM as usize, HC_DIM as usize);
            for i in chunk_b - take..chunk_b {
                let (src, idx) = if i < b_a { (&*bd_a, i) } else { (&*bd_b, i - b_a) };
                let mut row = ReplayRow { tok: chunk_tokens[i], hc: vec![0f32; hc], carry: vec![0f32; m] };
                src.residual.slice_view(idx * hc, hc).copy_to_host(&mut row.hc)?;
                src.hc_pre_carry.slice_view(idx * m, m).copy_to_host(&mut row.carry)?;
                job.replay.push_back(row);
                if job.replay.len() > SWA_WINDOW as usize {
                    job.replay.pop_front();
                }
            }
        } else {
            self.forward_prompt_batch_v2_pipelined(
                bd_a, bi_a, bd_b, bi_b, sd, si, state, weights, chunk_input, chunk_tokens, chunk_pos0,
                None, Some(&job.image_spans), pager.as_deref_mut(), chunk_engram.as_deref(),
            )?;
            if is_last_chunk {
                let b_a = bd_a.mtp_lane_cut.min(chunk_b);
                let b_b = chunk_b - b_a;
                let (src_bd, last_idx) = if b_b > 0 { (&*bd_b, b_b - 1) } else { (&*bd_a, b_a - 1) };
                job.last_logits = Some(self.head_from_row(head_scratch, src_bd, last_idx, weights)?);
            }
        }
        self.emit_prefill_perfetto()?;
        job.chunk_start = chunk_end;
        job.chunk_idx += 1;
        Ok(chunk_b)
    }

    /// After every chunk: the CED decoder replay over the last window (or,
    /// without CED, the logits the last chunk already produced). Returns the
    /// next-token logits `[N_VOCAB]`; `state` then holds the prompt's KV exactly as
    /// `forward_prefill_pipelined` would have left it.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_job_finish(
        &self,
        job: &mut PrefillJob,
        bd_a: &mut BatchDgpuScratch,
        bi_a: &mut BatchIgpuScratch,
        bd_b: &mut BatchDgpuScratch,
        bi_b: &mut BatchIgpuScratch,
        sd: &mut BatchDgpuShared,
        si: &mut BatchIgpuShared,
        head_scratch: &mut DgpuScratch,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        mut pager: Option<&mut super::expert_pager::ExpertPager>,
    ) -> eyre::Result<Vec<f32>> {
        if !job.chunks_done() {
            return Err(eyre!("PrefillJob: finish called with {} of {} rows done", job.chunk_start, job.tokens.len()));
        }
        if !job.ced {
            return job.last_logits.take().ok_or_else(|| eyre!("PrefillJob: no logits from the last chunk"));
        }
        let t = job.tokens.len();
        let split = crate::config::CED_DECODER_START;
        let b_seg = job.replay.len();
        if b_seg == 0 || b_seg > t {
            return Err(eyre!("PrefillJob: replay segment {b_seg} of {t} rows"));
        }
        let seg_pos0 = job.pos0 + (t - b_seg) as u32;
        // Empty the decoder rings ONLY for a fresh prompt. On a continuation
        // (snapshot restored, `pos0 > 0`) they hold the previous turn's replay
        // rows at positions [pos0 - k, pos0), which is exactly the window the
        // reference decoder carries incrementally; emptying them gave a 1-token
        // suffix a 1-row window for the first SWA_WINDOW generated tokens
        // (KNOWN_BUGS #25). The append below evicts past SWA_WINDOW as usual.
        if job.pos0 == 0 {
            for l in split..N_LAYER as usize {
                state.layers[l].n_raw = 0;
                state.layers[l].raw_off = 0;
            }
        }
        let mut seg_hcs: Vec<Vec<f32>> = Vec::with_capacity(b_seg);
        let mut seg_carry: Vec<Vec<f32>> = Vec::with_capacity(b_seg);
        let mut seg_tokens: Vec<i32> = Vec::with_capacity(b_seg);
        for r in job.replay.drain(..) {
            seg_hcs.push(r.hc);
            seg_carry.push(r.carry);
            seg_tokens.push(r.tok);
        }
        let t0 = std::time::Instant::now();
        self.dgpu.events.reset();
        self.igpu.events.reset();
        let b_a = self.forward_prompt_batch_v2_pipelined_range(
            bd_a, bi_a, bd_b, bi_b, sd, si, state, weights, &seg_hcs, &seg_tokens, seg_pos0,
            None, None, pager.as_deref_mut(), None, split..N_LAYER as usize, CedMode::Replay, Some(&seg_carry),
        )?;
        let (src_bd, last_idx) = if b_seg > b_a { (&*bd_b, b_seg - b_a - 1) } else { (&*bd_a, b_a - 1) };
        let logits = self.head_from_row(head_scratch, src_bd, last_idx, weights)?;
        self.emit_prefill_perfetto()?;
        tracing::info!(replay_tokens = b_seg, seg_pos0, elapsed_s = format!("{:.1}", t0.elapsed().as_secs_f32()),
            total_s = format!("{:.1}", job.started.elapsed().as_secs_f32()), "prefill_job_replay");
        dump_prefill_logits(&logits)?;
        Ok(logits)
    }
}

/// `V41_MS_GRAPHS=0` disables the per-stage HIP graphs of the arena (decode)
/// lane-layer. Default ON: at 1-8 rows the batched driver is host-launch-bound
/// (~40 kernels per lane-layer at 20-40 us each for a few us of GPU work), and
/// a captured stage replays as one launch.
/// `V41_LOOKAHEAD_PREFETCH=0` disables look-ahead routing prefetch (see the
/// router stage of `forward_layer_pre_moe_v2`).
/// `V41_LOOKAHEAD_DEPTH`: layers of routing lead sent as prefetch words
/// (1 = layer+1 only, 2 = also layer+2; default 2).
pub fn lookahead_depth() -> usize {
    static B: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| std::env::var("V41_LOOKAHEAD_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(2).clamp(1, 2));
    *B
}
/// `V41_LOOKAHEAD_TOPK`: how many of each row's look-ahead picks (by rank)
/// become prefetch hints (default 5 of 6).
pub fn lookahead_topk() -> usize {
    static B: std::sync::LazyLock<usize> =
        std::sync::LazyLock::new(|| std::env::var("V41_LOOKAHEAD_TOPK").ok().and_then(|v| v.parse().ok()).unwrap_or(5).clamp(1, 8));
    *B
}
/// `V41_LOOKAHEAD_PREFETCH=1` opts in to speculative look-ahead routing
/// hints. DEFAULT OFF since 2026-09-21: A/B at 5 rows on diverse prompts, same
/// daemon (early paging of the queued request on) -- hints ON: step 227-257 ms,
/// exposed box-2 wait 96-133 ms, box-2 busy 175-212 ms; hints OFF: step
/// 165-237, wait 13-52, busy 93-120. The speculative reads (~40/step, 26%
/// wrong, 26% still in flight at ensure) contend with the demand misses on
/// box 2's two drives and their admits/waits sit on the request path; the
/// daemon's CERTAIN early paging of the queued request gets the overlap
/// without the waste.
pub fn lookahead_prefetch() -> bool {
    static B: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_LOOKAHEAD_PREFETCH").as_deref() == Ok("1"));
    *B
}

pub fn ms_graphs() -> bool {
    static B: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_MS_GRAPHS").as_deref() != Ok("0"));
    *B
}

/// One stage's capture-or-replay handle (see `HeterogeneousEngine::stage_cap`).
/// `skip` = a graph was replayed, run nothing. Dropping an unfinished capture
/// ends it and discards the graph, so an error inside the stage cannot leave
/// the stream in capture mode.
pub struct StageCap<'a> {
    pub skip: bool,
    capturing: bool,
    name: &'static str,
    key: u32,
    de: &'a super::engine::DeviceEngine,
    graphs: &'a super::graph_cache::GraphCache,
}

impl<'a> StageCap<'a> {
    /// Finish: on a fresh capture, instantiate, store and launch it.
    pub fn end(mut self) -> eyre::Result<()> {
        if self.capturing {
            self.capturing = false;
            self.de.events.set_capturing(false);
            let graph = self.de.compute.end_capture()?;
            let exec = std::sync::Arc::new(graph.instantiate()?);
            self.graphs.insert(self.name, self.key, exec.clone());
            exec.launch(&self.de.compute)?;
        }
        Ok(())
    }
}

impl Drop for StageCap<'_> {
    fn drop(&mut self) {
        if self.capturing {
            self.de.events.set_capturing(false);
            let _ = self.de.compute.end_capture();
        }
    }
}

impl HeterogeneousEngine {
    /// Begin a capture-or-replay of a dGPU stage of the arena lane-layer.
    /// `allow` = the caller's static-shape guarantee (arena layout, no dumps
    /// armed). Key = layer | rows << 8 | lane hash << 16, where the lane is
    /// identified by its scratch buffer address (bd_a vs bd_b).
    pub fn stage_cap<'a>(
        &'a self,
        de: &'a super::engine::DeviceEngine,
        name: &'static str,
        layer: usize,
        b: u32,
        lane_ptr: usize,
        allow: bool,
    ) -> eyre::Result<StageCap<'a>> {
        let graphs = &self.dgpu_graphs;
        if !allow || !ms_graphs() {
            return Ok(StageCap { skip: false, capturing: false, name, key: 0, de, graphs });
        }
        let lane = ((lane_ptr >> 8) as u32).wrapping_mul(2654435761) >> 16;
        let key = (layer as u32) | (b << 8) | (lane << 16);
        if let Some(exec) = graphs.get(name, key) {
            exec.launch(&de.compute)?;
            return Ok(StageCap { skip: true, capturing: false, name, key, de, graphs });
        }
        de.compute.begin_capture(v4flash_hip::sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        de.events.set_capturing(true);
        Ok(StageCap { skip: false, capturing: true, name, key, de, graphs })
    }

    /// Route probe hook (`V41_ROUTE_PROBE`): after `forward_layer_pre_moe_v2`
    /// of `layer` on one lane, log the real picks, compute+log the one-layer
    /// look-ahead proxy for layer+1, and the layer-20 activation. Syncs.
    fn route_probe_after_layer(
        &self,
        bd: &BatchDgpuScratch,
        weights: &HetModelWeights,
        layer: usize,
        b: usize,
        lane: usize,
    ) -> eyre::Result<()> {
        if !super::route_probe::enabled() { return Ok(()); }
        let de = &self.dgpu;
        let nu = N_EXPERT_USED;
        let mut look: Option<Vec<i32>> = None;
        if layer + 1 < N_LAYER as usize {
            let nl = &weights.dgpu_layers[layer + 1];
            if !nl.is_hash_router {
                let mut logits = v4flash_hip::DeviceBuffer::<f32>::new(de.device.id, b * N_EXPERT as usize)?;
                let mut sel = v4flash_hip::DeviceBuffer::<i32>::new(de.device.id, b * nu)?;
                let mut ew = v4flash_hip::DeviceBuffer::<f32>::new(de.device.id, b * nu)?;
                de.f16.matvec_batched(&de.compute, &mut logits, &nl.ffn_gate_inp.buffer, &bd.ffn_input_norm, N_EXPERT, N_EMBD, b as u32)?;
                de.router_topk.launch_batched(&de.compute, &mut sel, &mut ew, &logits, nl.router_bias_dev.as_ref(), N_EXPERT, nu as u32, EXPERT_WEIGHT_SCALE, ROUTER_WEIGHT_EPS, b as u32)?;
                de.compute.synchronize()?;
                let mut v = vec![0i32; b * nu];
                sel.copy_to_host(&mut v)?;
                look = Some(v);
            }
        }
        de.compute.synchronize()?;
        let mut picks = vec![0i32; b * nu];
        bd.d_selected.slice_view(0, b * nu).copy_to_host(&mut picks)?;
        let mut act: Option<Vec<f32>> = None;
        if layer == crate::config::CANDIDATE_SOURCE_LAYER as usize {
            let mut a = vec![0f32; b * N_EMBD as usize];
            bd.ffn_input_norm.slice_view(0, b * N_EMBD as usize).copy_to_host(&mut a)?;
            act = Some(a);
        }
        super::route_probe::note(lane, layer, b, &picks, look.as_deref(), act.as_deref())
    }

    /// Layer-major batched prefill using batched kernels.
    ///
    /// Reads `input_hcs[i]` = layer-0 input HC for token `i`, broadcast of
    /// `embed(tokens[i])` to HC_DIM. `tokens[i]` is the token id at
    /// position `pos0 + i` (used by the hash router on bootstrap layers).
    ///
    /// Stateless big matmuls + HC stages run in single B-wide kernel
    /// launches against `batch_dgpu` (B-extended contiguous buffers).
    /// Stateful per-token kernels (rope, kv_append, compressor, attn,
    /// iGPU MoE) loop in a serial inner B loop using `slice_view`.
    ///
    /// After return, `batch_dgpu.residual` contains per-token
    /// post-last-layer HC: every layer writes `residual_next` and is
    /// followed by one swap, so the output is in `residual` for any layer
    /// count (only the *physical* buffer alternates with parity, which
    /// matters for graph capture in decode, not here).
    /// Does NOT compute logits / head — the caller picks which batch
    /// element(s) to feed `forward_head`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prompt_batch_v2(
        &self,
        batch_dgpu: &mut BatchDgpuScratch,
        batch_igpu: &mut BatchIgpuScratch,
        sd: &mut BatchDgpuShared,
        si: &mut BatchIgpuShared,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        input_hcs: &[Vec<f32>],
        tokens: &[i32],
        pos0: u32,
        mut stats: Option<&mut PrefillStats>,
        // Vision-Exp: `(start_pos, len)` of every `[IMAGE_START ..
        // IMAGE_END]` span in the prompt (absolute positions; spans that
        // don't touch this chunk are ignored). A span that straddles the
        // chunk is an error. `None` == text-only (bit-identical to before).
        image_spans: Option<&[ImageSpan]>,
        // M7 expert pager: when Some, this layer's experts are paged out of its
        // pool instead of read from the (placeholder) resident buffers.
        mut pager: Option<&mut super::expert_pager::ExpertPager>,
    ) -> eyre::Result<()> {
        self.remote_set_phase_busy_poll(false);
        let b = tokens.len();
        if b == 0 {
            return Ok(());
        }
        if input_hcs.len() != b {
            return Err(eyre!(
                "forward_prompt_batch_v2: input_hcs len {} != tokens len {b}",
                input_hcs.len()
            ));
        }
        let spans = image_spans.unwrap_or(&[]);
        let vis = chunk_visibility(pos0, b, spans)?;
        for (i, hc) in input_hcs.iter().enumerate() {
            if hc.len() != HC_DIM as usize {
                return Err(eyre!(
                    "forward_prompt_batch_v2: input_hcs[{i}] len {} != HC_DIM {}",
                    hc.len(),
                    HC_DIM
                ));
            }
        }
        check_scratch_rows("forward_prompt_batch_v2", b, batch_dgpu, batch_igpu, sd, si)?;

        self.current_device
            .store(-1, std::sync::atomic::Ordering::Relaxed);
        self.set_current_cached(self.dgpu.device)?;

        // 1. Seed per-token residual buffers in `batch_dgpu.residual`.
        //    `residual` is laid out [B, HC_DIM] contiguous. Each token's
        //    input HC is copied into its slot.
        for i in 0..b {
            let mut slot = batch_dgpu
                .residual
                .slice_view_mut(i * HC_DIM as usize, HC_DIM as usize);
            slot.copy_from_host(&input_hcs[i])?;
        }

        // 1b. Upload pos_per_b = [pos0, pos0+1, ..., pos0+B-1] once per chunk.
        //     Used by batched rope kernels in Stages 2/3/6. Constant across
        //     layers, so uploaded outside the layer loop.
        {
            let pos_host: Vec<i32> = (0..b).map(|i| (pos0 + i as u32) as i32).collect();
            let mut pos_v = batch_dgpu.pos_per_b.slice_view_mut(0, b);
            pos_v.copy_from_host_async(&pos_host, &self.dgpu.compute)?;
        }

        // 2. Layer loop: invoke forward_layer_batch_v2 once per layer.
        //    Each call swaps residual / residual_next internally (we do
        //    the swap here for clarity, mirroring forward_token's per-
        //    layer swap).
        for layer in 0..N_LAYER as usize {
            state.with_kv_source(layer, |ls| self.forward_layer_batch_v2(
                batch_dgpu,
                batch_igpu,
                sd,
                si,
                ls,
                &weights.dgpu_layers[layer],
                &weights.igpu_layers[layer],
                pos0,
                tokens,
                vis.as_deref(),
                stats.as_deref_mut(),
            pager.as_deref_mut(),
))?;
            // Swap residual / residual_next for the next layer: the
            // layer wrote residual_next; next layer reads residual.
            std::mem::swap(&mut batch_dgpu.residual, &mut batch_dgpu.residual_next);
        }

        // Drain any pending async work.
        self.dgpu.compute.synchronize()?;
        Ok(())
    }

    /// Two-lane pipelined version of `forward_prompt_batch_v2`. Splits the
    /// chunk into lane A (first ceil(B/2) tokens) and lane B (the rest)
    /// and interleaves them on the layer loop so:
    ///   per-layer order on de.compute = pre_A stages, pre_B stages,
    ///                                   ffn_combine_A, ffn_combine_B
    ///   per-layer order on ie.compute = q8k+group+wis+iq2+q2k_A,
    ///                                   q8k+group+wis+iq2+q2k_B
    /// The cross-lane dependency that matters is KV writes: lane B's
    /// attention at layer L reads KV slots lane A wrote at the same
    /// layer L. Both lanes share de.compute (FIFO), and pre_A is queued
    /// before pre_B, so lane A's kv_chain/kv_append always sequences
    /// before lane B's attn — no event needed.
    ///
    /// Lane A uses `self.sync_events`, lane B uses `self.sync_events_t1`
    /// (both pre-allocated at engine construction for the decode pair
    /// path; we reuse them here).
    ///
    /// `sd` / `si` are the SHARED scratch sets: one instance serves both
    /// lanes because every buffer in them is first-written and last-read
    /// inside a single lane's pre-MoE call on one in-order stream, and
    /// this driver never interleaves two lanes' pre-MoE work (see
    /// `BatchDgpuShared`). They must be allocated for at least
    /// `max(b_a, b_b)` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prompt_batch_v2_pipelined(
        &self,
        bd_a: &mut BatchDgpuScratch,
        bi_a: &mut BatchIgpuScratch,
        bd_b: &mut BatchDgpuScratch,
        bi_b: &mut BatchIgpuScratch,
        sd: &mut BatchDgpuShared,
        si: &mut BatchIgpuShared,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        input_hcs: &[Vec<f32>],
        tokens: &[i32],
        pos0: u32,
        stats: Option<&mut PrefillStats>,
        image_spans: Option<&[ImageSpan]>,
        pager: Option<&mut super::expert_pager::ExpertPager>,
        engram_rows: Option<&[Vec<f32>]>,
    ) -> eyre::Result<()> {
        self.remote_set_phase_busy_poll(false);
        self.forward_prompt_batch_v2_pipelined_range(
            bd_a, bi_a, bd_b, bi_b, sd, si, state, weights, input_hcs, tokens, pos0, stats,
            image_spans, pager, engram_rows, 0..N_LAYER as usize, CedMode::Exact, None,
        )
        .map(|_| ())
    }

    /// `forward_prompt_batch_v2_pipelined` over the layer range `layers` (M7
    /// CED). `ced` is the mode of `CED_DECODER_START` when it lies in the range
    /// (every other layer runs `Exact`): `KvSourceOnly` requires the range to
    /// END at that layer (no post-MoE / residual swap after it, so on return
    /// `residual` and `hc_pre_carry` are the rows ENTERING it); `Replay`
    /// requires the range to START there. `seed_carry` seeds each row's mHC
    /// carry (`[B, HC_MIX_DIM]`) for a range that does not start at layer 0.
    /// Returns the lane cut `b_a` (rows `[0, b_a)` are in lane A).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prompt_batch_v2_pipelined_range(
        &self,
        bd_a: &mut BatchDgpuScratch,
        bi_a: &mut BatchIgpuScratch,
        bd_b: &mut BatchDgpuScratch,
        bi_b: &mut BatchIgpuScratch,
        sd: &mut BatchDgpuShared,
        si: &mut BatchIgpuShared,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        input_hcs: &[Vec<f32>],
        tokens: &[i32],
        pos0: u32,
        stats: Option<&mut PrefillStats>,
        // See `forward_prompt_batch_v2`. Each LANE is a KV-visible unit
        // (lane A's post-attention eviction runs before lane B appends),
        // so the lane cut is moved off any image span — `lane_split`.
        image_spans: Option<&[ImageSpan]>,
        // M7 expert pager: when Some, this layer's experts are paged out of its
        // pool instead of read from the (placeholder) resident buffers.
        mut pager: Option<&mut super::expert_pager::ExpertPager>,
        // M7/Engram: one flattened [B * ENGRAM_IN] row buffer per Engram layer for
        // THIS call's tokens; batched prefill stages them per Engram layer.
        engram_rows: Option<&[Vec<f32>]>,
        layers: std::ops::Range<usize>,
        ced: CedMode,
        seed_carry: Option<&[Vec<f32>]>,
    ) -> eyre::Result<usize> {
        self.remote_set_phase_busy_poll(false);
        // Same repair as `forward_token_impl`: the steady-state loop below lends
        // each KV-source layer's compressor to its reuse layer and hands it back
        // at the bottom of the iteration, and any `?` in between leaks it.
        state.restore_compressor_lending();
        let b = tokens.len();
        if b == 0 {
            return Ok(0);
        }
        if input_hcs.len() != b {
            return Err(eyre!(
                "forward_prompt_batch_v2_pipelined: input_hcs len {} != tokens len {b}",
                input_hcs.len()
            ));
        }
        let (lo, hi) = (layers.start, layers.end);
        let split = crate::config::CED_DECODER_START;
        if lo >= hi || hi > N_LAYER as usize {
            return Err(eyre!("forward_prompt_batch_v2_pipelined: bad layer range {lo}..{hi}"));
        }
        match ced {
            CedMode::Exact => {}
            CedMode::KvSourceOnly if hi == split + 1 => {}
            CedMode::Replay if lo == split => {}
            _ => {
                return Err(eyre!(
                    "forward_prompt_batch_v2_pipelined: {ced:?} over layers {lo}..{hi} (split {split})"
                ))
            }
        }
        let mode_of = |l: usize| if l == split { ced } else { CedMode::Exact };
        if let Some(c) = seed_carry {
            if lo == 0 {
                return Err(eyre!(
                    "forward_prompt_batch_v2_pipelined: seed_carry with layer 0 (reset to one-hot there)"
                ));
            }
            if c.len() != b || c.iter().any(|r| r.len() != HC_MIX_DIM as usize) {
                return Err(eyre!(
                    "forward_prompt_batch_v2_pipelined: seed_carry must be [{b}, {HC_MIX_DIM}]"
                ));
            }
        }
        // For chunks too small to bother pipelining, fall back to single-lane
        // (exact full-depth only: the single-lane driver has no layer range).
        //
        // NOT when this chunk carries Engram rows: `forward_prompt_batch_v2`
        // has no engram parameter, so the fallback cannot stage them and the
        // first engram layer fails with "Engram rows not staged". Latent in
        // production (prefill chunks are ~128 wide) but it makes a 1-token
        // chunk impossible, which is exactly what a B=1 verify probe is.
        if b < 2
            && ced == CedMode::Exact
            && lo == 0
            && hi == N_LAYER as usize
            && engram_rows.is_none()
        {
            // This path runs the whole batch through lane A, so the cut is `b`.
            // Setting it matters: the ONLY other writer is after this return, so
            // without it a B=1 verify leaves the PREVIOUS call's cut in place and
            // a caller mapping a global row to a lane reads the wrong buffer --
            // the same stale-residual failure as the lane-capture bug, and it
            // bites exactly the one-row verify used to adjudicate divergence.
            bd_a.mtp_lane_cut = b;
            self.forward_prompt_batch_v2(
                bd_a, bi_a, sd, si, state, weights, input_hcs, tokens, pos0, stats, image_spans,
                pager.as_deref_mut(),
            )?;
            return Ok(b);
        }
        let spans = image_spans.unwrap_or(&[]);
        // Two lanes exist to overlap one lane's GPU work with the other's host
        // scheduling. At speculative-verify batch sizes there is no GPU work to
        // hide behind — the whole cost IS the per-layer host scopes — so the
        // split just runs every one of them twice. Below the threshold, put the
        // whole chunk in lane A and skip lane B entirely. Only safe with no
        // image spans, which is the case `lane_split` would otherwise have to
        // cut around.
        // DEFAULT 0 = OFF. It was briefly defaulted to 8 on a wall-time A/B
        // (-16.3/-12.6/-10.1/-5.4% at B=2/4/6/8) — and that was WRONG. Scored
        // against DSpark acceptance instead of the clock, single-lane costs
        // real quality: E[tokens/step] 2.259 two-lane vs 1.735 single-lane on
        // the same prompt. Putting the whole chunk in lane A is not the
        // no-op it looks like; something in the per-lane KV/shared-scratch
        // path is not row-independent.
        //
        // Both "wins" this path produced (this and the small-B expert offload)
        // were faster because they computed something different. On the verify
        // path, wall time alone cannot tell a speedup from a wrong answer —
        // score every change with acceptance.
        let single_lane_max: usize = single_lane_max();
        // ONLY on the non-CED path. The CED branch plans its lane cut in
        // `plan_chunk` and then asserts the range returns the same one
        // ("CED prefill: lane cut N != planned M"), so overriding it here
        // desynchronises the two. `Exact` is the mode the non-CED branch passes;
        // CED passes KvSourceOnly / Replay.
        let b_a = if ced == CedMode::Exact && spans.is_empty() && b <= single_lane_max && b <= bd_a.rows {
            b
        } else {
            image_spans::lane_split(pos0, b, spans, bd_a.rows, bd_b.rows)?
        };
        // Lane A takes rows [0, b_a) and lane B rows [b_a, b); a cut outside the
        // batch would both underflow `b - b_a` and mis-map every global batch
        // row onto a lane below.
        assert!(
            b_a <= b,
            "prefill lane split: lane A cut {b_a} exceeds the batch of {b} rows"
        );
        let b_b = b - b_a;
        // Record where the cut fell. `mtp_src` is captured per lane and indexed
        // lane-locally, so anything selecting a GLOBAL batch row needs this.
        bd_a.mtp_lane_cut = b_a;
        if std::env::var("V41_PREFILL_LANE_DEBUG").as_deref() == Ok("1") {
            tracing::warn!(
                b, b_a, b_b, lo, hi,
                engram_rows = engram_rows.map(|r| r.len()).unwrap_or(0),
                ced = ?ced,
                "prefill.lane_debug"
            );
        }
        // Lane and shared scratches are usually allocated at
        // B_MAX.div_ceil(2) rows (see BatchDgpuScratch::alloc_rows); never
        // exceed what they hold.
        check_scratch_rows("forward_prompt_batch_v2_pipelined lane A", b_a, bd_a, bi_a, sd, si)?;
        check_scratch_rows("forward_prompt_batch_v2_pipelined lane B", b_b, bd_b, bi_b, sd, si)?;
        let tokens_a = &tokens[..b_a];
        let tokens_b = &tokens[b_a..];
        let input_a = &input_hcs[..b_a];
        let input_b = &input_hcs[b_a..];
        let pos0_a = pos0;
        let pos0_b = pos0 + b_a as u32;
        let vis_a = chunk_visibility(pos0_a, b_a, spans)?;
        let vis_b = chunk_visibility(pos0_b, b_b, spans)?;

        self.current_device
            .store(-1, std::sync::atomic::Ordering::Relaxed);
        self.set_current_cached(self.dgpu.device)?;

        for i in 0..b_a {
            let mut slot = bd_a
                .residual
                .slice_view_mut(i * HC_DIM as usize, HC_DIM as usize);
            slot.copy_from_host(&input_a[i])?;
        }
        for i in 0..b_b {
            let mut slot = bd_b
                .residual
                .slice_view_mut(i * HC_DIM as usize, HC_DIM as usize);
            slot.copy_from_host(&input_b[i])?;
        }
        if let Some(c) = seed_carry {
            let m = HC_MIX_DIM as usize;
            for i in 0..b_a {
                bd_a.hc_pre_carry.slice_view_mut(i * m, m).copy_from_host(&c[i])?;
            }
            for i in 0..b_b {
                bd_b.hc_pre_carry.slice_view_mut(i * m, m).copy_from_host(&c[b_a + i])?;
            }
        }
        {
            let pos_a: Vec<i32> = (0..b_a).map(|i| (pos0_a + i as u32) as i32).collect();
            let mut va = bd_a.pos_per_b.slice_view_mut(0, b_a);
            va.copy_from_host_async(&pos_a, &self.dgpu.compute)?;
            let pos_b: Vec<i32> = (0..b_b).map(|i| (pos0_b + i as u32) as i32).collect();
            let mut vb = bd_b.pos_per_b.slice_view_mut(0, b_b);
            vb.copy_from_host_async(&pos_b, &self.dgpu.compute)?;
        }

        // Stats: only lane A collects, to avoid a double mutable borrow of
        // PrefillStats inside the loop. Pipelined mode is for perf benches,
        // not for stats collection — if a caller wants per-batch picks,
        // they should use the single-lane forward_prompt_batch_v2.
        let mut stats_a = stats;

        // Deep pipeline: queue lane A's L+1 pre-MoE immediately after lane A's
        // L post-MoE (NOT after lane B's L post-MoE). On the dgpu.compute FIFO
        // this means lane A's next layer can start as soon as lane A's MoE is
        // back, regardless of how long lane B's MoE still has to run.
        //
        // Stream order in steady state:
        //   ... [wait moe_A(L)] post_A(L) pre_A(L+1) [wait moe_B(L)] post_B(L) pre_B(L+1) ...
        //
        // vs. the shallow version (which we replaced):
        //   ... [wait moe_A(L)] post_A(L) [wait moe_B(L)] post_B(L) pre_A(L+1) pre_B(L+1) ...
        // — the shallow version stalled pre_A(L+1) behind moe_arrived_B(L).

        // Warmup: queue layer 0 pre-MoE for both lanes.
        let layer0 = lo;
        self.stage_engram_lane(bd_a, layer0, engram_rows, 0, b_a)?;
        // Layer 0's pre_moe for both lanes -- the pipeline warm-up. Previously
        // untimed, so one whole layer of pre_moe was missing from every report
        // while the pager/sel_sync counters inside it were not.
        let _t_pre_warm = LayerHostTimer::start(&LH_PRE);
        self.forward_layer_pre_moe_v2(
            bd_a,
            bi_a,
            sd,
            si,
            &mut state.layers[layer0],
            &weights.dgpu_layers[layer0],
            &weights.igpu_layers[layer0],
            pos0_a,
            tokens_a,
            vis_a.as_deref(),
            stats_a.as_deref_mut(),
            &self.sync_events.layers[layer0],
            pager.as_deref_mut(),
            mode_of(layer0),
            RowLayout::Contiguous,
        )?;
        if b_b > 0 {
            self.stage_engram_lane(bd_b, layer0, engram_rows, b_a, b_b)?;
            self.forward_layer_pre_moe_v2(
                bd_b,
                bi_b,
                sd,
                si,
                &mut state.layers[layer0],
                &weights.dgpu_layers[layer0],
                &weights.igpu_layers[layer0],
                pos0_b,
                tokens_b,
                vis_b.as_deref(),
                None,
                &self.sync_events_t1.layers[layer0],
                pager.as_deref_mut(),
                mode_of(layer0),
                RowLayout::Contiguous,
            )?;
        }
        drop(_t_pre_warm);

        // Steady state: for each layer L in 0..N_LAYER-1, queue post_X(L)
        // followed by pre_X(L+1) for the SAME lane, before moving to lane B.
        for layer in lo..(hi - 1) {
            let sev_a_cur = &self.sync_events.layers[layer];
            let sev_b_cur = &self.sync_events_t1.layers[layer];

            let hot_cur = prefill_hot_active(
                &weights.dgpu_layers[layer],
                &weights.igpu_layers[layer],
                bd_a,
                sd,
            );

            // V4.1 reuse layers: lend the KV source's store to layer L+1 for both
            // lanes' pre-MoE halves (post-MoE never touches compressor state).
            let kv_src_next = crate::config::kv_source_of(layer + 1);
            if let Some(src) = kv_src_next {
                let st = state.layers[src].compressor.take();
                state.layers[layer + 1].compressor = st;
            }
            // Lane A: finish layer L, then start layer L+1.
            let _t_post = LayerHostTimer::start(&LH_POST);
            self.forward_layer_post_moe_v2(bd_a, b_a as u32, sev_a_cur, hot_cur)?;
            drop(_t_post);
            std::mem::swap(&mut bd_a.residual, &mut bd_a.residual_next);
            let _t_eng = LayerHostTimer::start(&LH_ENGRAM);
            self.stage_engram_lane(bd_a, layer + 1, engram_rows, 0, b_a)?;
            drop(_t_eng);
            let _t_pre = LayerHostTimer::start(&LH_PRE);
            self.forward_layer_pre_moe_v2(
                bd_a,
                bi_a,
                sd,
                si,
                &mut state.layers[layer + 1],
                &weights.dgpu_layers[layer + 1],
                &weights.igpu_layers[layer + 1],
                pos0_a,
                tokens_a,
                vis_a.as_deref(),
                stats_a.as_deref_mut(),
                &self.sync_events.layers[layer + 1],
                pager.as_deref_mut(),
                mode_of(layer + 1),
                RowLayout::Contiguous,
            )?;
            drop(_t_pre);

            // Lane B: same. TIMED TOO -- these three calls used to be untimed while
            // the counters INSIDE `forward_layer_pre_moe_v2` (LH_PAGER, LH_SEL_SYNC,
            // LH_ENSURE, LH_REMOTE) counted both lanes. That is why a
            // `prefill.layer_host` line could report pager_ms 102.0 > pre_moe_ms
            // 66.2 despite the pager being lexically nested inside pre_moe: the
            // pager total was both lanes and the pre_moe total was lane A only.
            // Every counter is now a BOTH-LANE total, so they are comparable.
            if b_b > 0 {
                let _t_post_b = LayerHostTimer::start(&LH_POST);
                self.forward_layer_post_moe_v2(bd_b, b_b as u32, sev_b_cur, hot_cur)?;
                drop(_t_post_b);
                std::mem::swap(&mut bd_b.residual, &mut bd_b.residual_next);
                let _t_eng_b = LayerHostTimer::start(&LH_ENGRAM);
                self.stage_engram_lane(bd_b, layer + 1, engram_rows, b_a, b_b)?;
                drop(_t_eng_b);
                let _t_pre_b = LayerHostTimer::start(&LH_PRE);
                self.forward_layer_pre_moe_v2(
                    bd_b,
                    bi_b,
                    sd,
                    si,
                    &mut state.layers[layer + 1],
                    &weights.dgpu_layers[layer + 1],
                    &weights.igpu_layers[layer + 1],
                    pos0_b,
                    tokens_b,
                    vis_b.as_deref(),
                    None,
                    &self.sync_events_t1.layers[layer + 1],
                    pager.as_deref_mut(),
                    mode_of(layer + 1),
                    RowLayout::Contiguous,
                )?;
                drop(_t_pre_b);
            }
            if let Some(src) = kv_src_next {
                let st = state.layers[layer + 1].compressor.take();
                state.layers[src].compressor = st;
            }
        }

        // Cooldown: post-MoE for the final layer on both lanes. A source-only
        // last layer has no MoE and leaves `residual` = its input rows.
        let last = hi - 1;
        if mode_of(last) != CedMode::KvSourceOnly {
            let hot_last = prefill_hot_active(
                &weights.dgpu_layers[last],
                &weights.igpu_layers[last],
                bd_a,
                sd,
            );
            // The LAST layer's post_moe, both lanes -- also previously untimed, so
            // one whole layer of post_moe was missing from every report.
            let _t_post_last = LayerHostTimer::start(&LH_POST);
            self.forward_layer_post_moe_v2(bd_a, b_a as u32, &self.sync_events.layers[last], hot_last)?;
            std::mem::swap(&mut bd_a.residual, &mut bd_a.residual_next);
            if b_b > 0 {
                self.forward_layer_post_moe_v2(bd_b, b_b as u32, &self.sync_events_t1.layers[last], hot_last)?;
                std::mem::swap(&mut bd_b.residual, &mut bd_b.residual_next);
            }
            drop(_t_post_last);
        }

        self.dgpu.compute.synchronize()?;
        Ok(b_a)
    }

    /// Chunked prefill driver. Processes `tokens` (length T)
    /// through the v2 batched pipeline in chunks of CHUNK_SIZE=B_MAX. State
    /// carries across chunks via `state.layers[*].{kv_cache,n_raw,compressor}`
    /// (the per-layer fields just keep growing — no special handling).
    ///
    /// Returns logits:
    /// * `last_only=true`: `[N_VOCAB]` for the last token only — typical
    ///   generation start path. Each per-token head is ~16 ms; skipping
    ///   all but the last saves ~T × head_cost wall on the prefill.
    /// * `last_only=false`: `[T × N_VOCAB]` — full per-token logits, for
    ///   prompt-eval / log-prob scoring.
    ///
    /// `head_scratch` is a single-token `DgpuScratch` used for the head
    /// matvec — the head buffers (`head_flat`, `head_pre`, …, `logits`)
    /// aren't B-extended in `BatchDgpuScratch` yet. Per-token head is fast
    /// enough that this isn't on the critical path for `last_only=true`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill(
        &self,
        bd: &mut BatchDgpuScratch,
        bi: &mut BatchIgpuScratch,
        sd: &mut BatchDgpuShared,
        si: &mut BatchIgpuShared,
        head_scratch: &mut DgpuScratch,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        input_hcs: &[Vec<f32>],
        tokens: &[i32],
        pos0: u32,
        last_only: bool,
        mut stats: Option<&mut PrefillStats>,
        // Vision-Exp image spans (absolute `(start_pos, len)`); chunk
        // boundaries are moved so no span straddles a chunk.
        image_spans: Option<&[ImageSpan]>,
        // M7 expert pager: when Some, this layer's experts are paged out of its
        // pool instead of read from the (placeholder) resident buffers.
        mut pager: Option<&mut super::expert_pager::ExpertPager>,
    ) -> eyre::Result<Vec<f32>> {
        self.remote_set_phase_busy_poll(false);
        let t = tokens.len();
        if t == 0 {
            return Ok(Vec::new());
        }
        if input_hcs.len() != t {
            return Err(eyre!(
                "forward_prefill: input_hcs len {} != tokens len {t}",
                input_hcs.len()
            ));
        }
        let spans = image_spans.unwrap_or(&[]);
        image_spans::validate_spans(spans, pos0, t)?;
        let chunk_size = B_MAX;
        // Single-lane driver: every chunk (up to B_MAX rows) runs through
        // ONE lane + ONE shared set, so all four must be allocated at full
        // B_MAX rows (`alloc()`), not the per-lane `alloc_rows(B_MAX.div_ceil(2))`.
        check_scratch_rows("forward_prefill", chunk_size.min(t), bd, bi, sd, si)?;
        let cs_vocab = N_VOCAB as usize;

        let mut out_logits: Vec<f32> = if last_only {
            Vec::with_capacity(cs_vocab)
        } else {
            Vec::with_capacity(t * cs_vocab)
        };

        let mut chunk_start = 0usize;
        while chunk_start < t {
            let (chunk_end, _) =
                image_spans::plan_chunk(pos0, chunk_start, t, chunk_size, None, spans)?;
            let chunk_b = chunk_end - chunk_start;
            let is_last_chunk = chunk_end == t;
            let chunk_input = &input_hcs[chunk_start..chunk_end];
            let chunk_tokens = &tokens[chunk_start..chunk_end];
            let chunk_pos0 = pos0 + chunk_start as u32;

            // Reset event pools at chunk start (mirrors decode's per-token cycle).
            // Fold the PREVIOUS chunk's stages into the request accumulator before
            // clearing. No-op unless DEEPSTRIX_PREFILL_PROFILE=1 (it synchronises,
            // which serialises the two lanes — read per-stage busy time, not wall).
            if super::trace::prefill_profile::enabled() {
                super::trace::prefill_profile::add(
                    "dgpu",
                    &super::trace::rollup_by_name(&self.dgpu.events.harvest()?),
                );
                super::trace::prefill_profile::add(
                    "igpu",
                    &super::trace::rollup_by_name(&self.igpu.events.harvest()?),
                );
            }
            self.dgpu.events.reset();
            self.igpu.events.reset();

            self.forward_prompt_batch_v2(
                bd,
                bi,
                sd,
                si,
                state,
                weights,
                chunk_input,
                chunk_tokens,
                chunk_pos0,
                stats.as_deref_mut(),
                image_spans,
            pager.as_deref_mut(),
)?;

            // After this chunk: if perfetto is attached, emit slices + re-anchor.
            if let Some(exp_lock) = &self.perfetto {
                let mut exp = exp_lock.lock().unwrap();
                self.dgpu.events.for_each_pair(|name, s, e| {
                    let track = if name.contains(".xfer") || name.contains(".peer_push") {
                        &exp.dgpu_xfer
                    } else {
                        &exp.dgpu_compute
                    };
                    exp.emit_slice(track, name, s, e)
                })?;
                self.igpu.events.for_each_pair(|name, s, e| {
                    let track = if name.contains(".xfer") || name.contains(".peer_push") {
                        &exp.igpu_xfer
                    } else {
                        &exp.igpu_compute
                    };
                    exp.emit_slice(track, name, s, e)
                })?;
                exp.re_anchor(
                    self.dgpu.device,
                    &self.dgpu.compute,
                    &self.dgpu.xfer,
                    self.igpu.device,
                    &self.igpu.compute,
                    &self.igpu.xfer,
                )?;
                self.current_device.store(-1, std::sync::atomic::Ordering::Relaxed);
            }

            // residual post-loop holds layer-N output in bd.residual (one
            // swap per layer, so this holds for any N_LAYER parity).
            if last_only {
                if is_last_chunk {
                    out_logits = self.head_from_row(head_scratch, bd, chunk_b - 1, weights)?;
                }
            } else {
                out_logits.extend_from_slice(&self.head_rows(head_scratch, bd, chunk_b, weights)?);
            }

            chunk_start = chunk_end;
        }
        Ok(out_logits)
    }


    /// Stage 10, the batched shared expert, as a callable unit.
    ///
    /// Extracted so it can be issued at EITHER of two points in `pre_moe`:
    /// before the pager/submit (historical order) or after the remote submit
    /// (`V41_PREFILL_PRESUBMIT=1`). It writes `bd.ffn_shared`, which is not read
    /// until `post_moe`'s `vec_add`, so it is free to move within the layer.
    ///
    /// WHY MOVING IT HELPS. `k.shared_expert.down_matvec` is the LAST dGPU kernel
    /// before the measured 1428.8 us x 4265 idle gap, and the remote submit sits
    /// inside that gap; the dGPU then idles again for 1986.6 us x 4320 waiting on
    /// box 2. Issuing the shared expert AFTER the submit puts its ~173 us of dGPU
    /// work inside the RPC window instead of before it. Decode already does this
    /// (`defer_shared`, forward_layer.rs:2210) and measured +5.0% with
    /// byte-identical output.
    #[allow(clippy::too_many_arguments)]
    fn issue_shared_expert_prefill(
        &self,
        sd: &mut BatchDgpuShared,
        bd: &mut BatchDgpuScratch,
        dlw: &DgpuLayerWeights,
        b: u32,
        layer: usize,
        pos0: u32,
        graphs_ok: bool,
    ) -> eyre::Result<()> {
        let de = &self.dgpu;
        // ========================================================
        // Stage 10: Shared expert (BATCHED Q8_0 chains)
        // swiglu + vec_add are pure elementwise → stretch n by B
        // ========================================================
        let _t_shared = de.events.stage("dgpu.shared_expert", &de.compute)?;
        let cap = self.stage_cap(de, "g.shared_expert", layer, b, bd.residual.raw() as usize, graphs_ok)?;
        if !cap.skip {
        {
            let _t = de.events.stage("k.shared_expert.quantize_input", &de.compute)?;
            // Q8_0 gate/up consume the (i8, scale) pair; K-quants (unsloth
            // Q5_K/Q6_K) consume Q8_K — quantize only what's consumed.
            if super::dispatch::any_q8(&[&dlw.shared.gate, &dlw.shared.up]) {
                // Which activation the gate/up GEMM consumes depends on which
                // arm `dense_gemm_prefill` will take, so the predicate lives in
                // one place and is asked here too. dp4a wants (i8, scale) --
                // decode's own `quantize_input_batched` of the same buffer.
                if super::dispatch::small_b_dense_dp4a(b) {
                    de.q8.quantize_input_batched(
                        &de.compute, &mut sd.xq_n_embd, &mut sd.xscale_n_embd,
                        &bd.ffn_input_norm, N_EMBD, b,
                    )?;
                } else {
                    de.q8k.launch_cast_f16_2d(&de.compute, &mut sd.x16_n_embd, &bd.ffn_input_norm,
                        b, N_EMBD, super::batch_scratch::f16_pitch(N_EMBD))?;
                }
            } else {
                de.q8k.launch(
                    &de.compute,
                    &mut sd.kq_ffn_q8k,
                    &bd.ffn_input_norm,
                    crate::config::BLOCKS_Q8K_GATE_IN * b,
                )?;
            }
        }
        {
            let _t = de.events.stage("k.shared_expert.gate_matvec", &de.compute)?;
            super::dispatch::dense_gemm_prefill(
                de, &de.compute, &mut sd.gate_sh, &dlw.shared.gate,
                &sd.xq_n_embd, &sd.xscale_n_embd, &sd.kq_ffn_q8k,
                Some((&sd.x16_n_embd, super::batch_scratch::f16_pitch(N_EMBD))),
                b, N_FF_SHARED, N_EMBD,
            )?;
        }
        {
            let _t = de.events.stage("k.shared_expert.up_matvec", &de.compute)?;
            super::dispatch::dense_gemm_prefill(
                de, &de.compute, &mut sd.up_sh, &dlw.shared.up,
                &sd.xq_n_embd, &sd.xscale_n_embd, &sd.kq_ffn_q8k,
                Some((&sd.x16_n_embd, super::batch_scratch::f16_pitch(N_EMBD))),
                b, N_FF_SHARED, N_EMBD,
            )?;
        }
        {
            let _t = de.events.stage("k.shared_expert.swiglu", &de.compute)?;
            // swiglu — elementwise; stretch n to B * N_FF_SHARED.
            // ds4 5bc1e6d: shared experts use the same swiglu_limit clamp
            // as routed experts (official V4-Flash graph).
            de.swiglu.launch_clamped(
                &de.compute,
                &mut sd.mid_sh,
                &sd.gate_sh,
                &sd.up_sh,
                b * N_FF_SHARED,
                crate::config::SWIGLU_CLAMP_EXP,
            )?;
        }
        {
            let _t = de.events.stage("k.shared_expert.quantize_mid", &de.compute)?;
            if super::dispatch::any_q8(&[&dlw.shared.down]) {
                // Same fork as the gate/up input above, for `down`'s activation.
                if super::dispatch::small_b_dense_dp4a(b) {
                    de.q8.quantize_input_batched(
                        &de.compute, &mut sd.mid_sh_xq, &mut sd.mid_sh_xscale,
                        &sd.mid_sh, N_FF_SHARED, b,
                    )?;
                } else {
                    de.q8k.launch_cast_f16_2d(&de.compute, &mut sd.mid_sh16, &sd.mid_sh,
                        b, N_FF_SHARED, super::batch_scratch::f16_pitch(N_FF_SHARED))?;
                }
            } else {
                de.q8k.launch(
                    &de.compute,
                    &mut sd.kq_mid_q8k,
                    &sd.mid_sh,
                    crate::config::BLOCKS_Q8K_DOWN_IN * b,
                )?;
            }
        }
        {
            let _t = de.events.stage("k.shared_expert.down_matvec", &de.compute)?;
            super::dispatch::dense_gemm_prefill(
                de, &de.compute, &mut bd.ffn_shared, &dlw.shared.down,
                &sd.mid_sh_xq, &sd.mid_sh_xscale, &sd.kq_mid_q8k,
                Some((&sd.mid_sh16, super::batch_scratch::f16_pitch(N_FF_SHARED))),
                b, N_EMBD, N_FF_SHARED,
            )?;
        if super::engine::subtensor_dump_armed(layer as usize) {
            de.compute.synchronize()?;
            super::engine::maybe_dump_subtensor_f32_view(
                layer as usize,
                &format!("pf_ffn_shared_p{pos0}"),
                &bd.ffn_shared.slice_view(0, crate::config::N_EMBD as usize),
            )?;
        }
        }
        }
        cap.end()?;
        drop(_t_shared);
        Ok(())
    }

    /// Logits for rows `0..n` of `bd`, via the batched head when it applies.
    /// Falls back to the per-row chain otherwise. See `forward_head_batch`.
    /// Also the head of a multi-stream step (`forward_step_arena`).
    pub fn head_rows(
        &self,
        head_scratch: &mut DgpuScratch,
        bd: &BatchDgpuScratch,
        n: usize,
        weights: &HetModelWeights,
    ) -> eyre::Result<Vec<f32>> {
        #[cfg(feature = "v41")]
        if self.forward_head_batch(
            head_scratch,
            &bd.residual,
            &bd.hc_pre_carry,
            n as u32,
            &weights.global,
        )? {
            let nv = N_VOCAB as usize;
            let mut out = vec![0f32; n * nv];
            head_scratch.logits_b.slice_view(0, n * nv).copy_to_host(&mut out)?;
            return Ok(out);
        }
        let mut out = Vec::with_capacity(n * N_VOCAB as usize);
        for i in 0..n {
            out.extend_from_slice(&self.head_from_row(head_scratch, bd, i, weights)?);
        }
        Ok(out)
    }

    /// Head over one batched row `idx` of `bd`: residual (+ under V4.1 the mHC
    /// carry the head's collapse reads — decode twin: `forward_head` after
    /// layer N-1 reads `dgpu_scratch.hc_pre_carry`) → logits `[N_VOCAB]`.
    fn head_from_row(
        &self,
        head_scratch: &mut DgpuScratch,
        bd: &BatchDgpuScratch,
        idx: usize,
        weights: &HetModelWeights,
    ) -> eyre::Result<Vec<f32>> {
        let cs_hc = HC_DIM as usize;
        head_scratch
            .residual
            .copy_from_buffer(&bd.residual.slice_view(idx * cs_hc, cs_hc))?;
        if cfg!(feature = "v41") {
            let m = HC_MIX_DIM as usize;
            head_scratch
                .hc_pre_carry
                .copy_from_buffer(&bd.hc_pre_carry.slice_view(idx * m, m))?;
        }
        self.forward_head(head_scratch, &weights.global)?;
        let mut logits = vec![0f32; N_VOCAB as usize];
        head_scratch.logits.copy_to_host(&mut logits)?;
        Ok(logits)
    }

    /// Two-lane pipelined chunked prefill. Same contract as
    /// `forward_prefill` but takes two BatchDgpu/BatchIgpu scratch sets
    /// (one per lane) plus one shared set for both lanes, and calls
    /// `forward_prompt_batch_v2_pipelined`.
    /// For `last_only`, the last token of each chunk lives in lane B if
    /// `chunk_b > 1`, otherwise in lane A.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill_pipelined(
        &self,
        bd_a: &mut BatchDgpuScratch,
        bi_a: &mut BatchIgpuScratch,
        bd_b: &mut BatchDgpuScratch,
        bi_b: &mut BatchIgpuScratch,
        sd: &mut BatchDgpuShared,
        si: &mut BatchIgpuShared,
        head_scratch: &mut DgpuScratch,
        state: &mut HetModelState,
        weights: &HetModelWeights,
        input_hcs: &[Vec<f32>],
        tokens: &[i32],
        pos0: u32,
        last_only: bool,
        mut stats: Option<&mut PrefillStats>,
        cancel: Option<&std::sync::atomic::AtomicBool>,
        // Called after each prefill chunk completes (after the head
        // pass for last_only=true, after the per-token logits copy for
        // last_only=false). Used by the server to pet its forward-
        // progress watchdog — the chunk-grain matters because long-ctx
        // chunks can take many seconds and per-call petting would
        // false-fire the watchdog mid-chunk.
        on_chunk_done: Option<&dyn Fn()>,
        // Vision-Exp image spans (absolute `(start_pos, len)`, sorted,
        // non-overlapping). Chunk ends AND the lane cut inside each chunk
        // are moved off spans (`image_spans::plan_chunk`) so every span is
        // prefilled inside one KV-visible unit; with `None` / empty the
        // chunking is the historical fixed B_MAX / div_ceil(2).
        image_spans: Option<&[ImageSpan]>,
        // M7 expert pager (see forward_layer_pre_moe_v2).
        mut pager: Option<&mut super::expert_pager::ExpertPager>,
        // M7/Engram: one flattened [B * ENGRAM_IN] row buffer per Engram layer for
        // THIS call's tokens; batched prefill stages them per Engram layer.
        engram_rows: Option<&[Vec<f32>]>,
    ) -> eyre::Result<Vec<f32>> {
        self.remote_set_phase_busy_poll(false);
        let t = tokens.len();
        if t == 0 {
            return Ok(Vec::new());
        }
        if input_hcs.len() != t {
            return Err(eyre!(
                "forward_prefill_pipelined: input_hcs len {} != tokens len {t}",
                input_hcs.len()
            ));
        }
        let spans = image_spans.unwrap_or(&[]);
        image_spans::validate_spans(spans, pos0, t)?;
        let lane_caps = (bd_a.rows, bd_b.rows);
        let chunk_size = B_MAX;
        let cs_vocab = N_VOCAB as usize;

        let mut out_logits: Vec<f32> = if last_only {
            Vec::with_capacity(cs_vocab)
        } else {
            Vec::with_capacity(t * cs_vocab)
        };

        // Progress log: long cold prefills can take many minutes;
        // emit a heartbeat every ~16 chunks so the operator can see
        // forward progress instead of guessing whether the engine is
        // stuck. Wall-clock + chunk index lets you extrapolate ETA.
        let prefill_start = std::time::Instant::now();
        let total_chunks = t.div_ceil(chunk_size);
        let mut chunk_idx = 0usize;
        let mut chunk_start = 0usize;

        // M7 CED (tech report §2.2 / §3.2.2): the causal encoder runs over every
        // prompt token and the decoder's global KV is projected from the final
        // encoder hidden state at `CED_DECODER_START`; the decoder itself runs
        // only over the last SWA_WINDOW tokens (Decoder SWA Bounded Replay),
        // after the last chunk. Per-token logits need the decoder over every
        // row, so only the last-token path takes it.
        let ced = ced_enabled() && last_only;
        let split = crate::config::CED_DECODER_START;
        let mut replay: std::collections::VecDeque<ReplayRow> = std::collections::VecDeque::new();
        while chunk_start < t {
            // Caller-driven cancel (typically: HTTP client disconnect).
            // Checked at chunk boundary so latency is bounded by one
            // chunk's wall-clock — ~hundreds of ms at long ctx, cheap
            // at short. Returning an empty Vec is fine for the
            // server's `last_only=true` path; the caller knows to
            // discard the result when the cancel bool is set.
            if let Some(c) = cancel {
                if c.load(std::sync::atomic::Ordering::Relaxed) {
                    tracing::info!(
                        chunk_idx,
                        tokens_done = chunk_start,
                        tokens_total = t,
                        "prefill cancelled by caller"
                    );
                    return Ok(Vec::new());
                }
            }
            // Same planner `forward_prompt_batch_v2_pipelined` re-derives its
            // lane cut from, so `b_a` below matches the lane contents.
            let (chunk_end, b_a) = image_spans::plan_chunk(
                pos0,
                chunk_start,
                t,
                chunk_size,
                Some(lane_caps),
                spans,
            )?;
            let chunk_b = chunk_end - chunk_start;
            let is_last_chunk = chunk_end == t;
            let chunk_input = &input_hcs[chunk_start..chunk_end];
            let chunk_tokens = &tokens[chunk_start..chunk_end];
            let chunk_pos0 = pos0 + chunk_start as u32;

            if chunk_idx == 0 || chunk_idx % 16 == 0 || is_last_chunk {
                let elapsed_s = prefill_start.elapsed().as_secs_f32();
                let toks_per_s = if elapsed_s > 0.001 {
                    chunk_start as f32 / elapsed_s
                } else {
                    0.0
                };
                let eta_s = if toks_per_s > 1.0 {
                    (t - chunk_start) as f32 / toks_per_s
                } else {
                    -1.0
                };
                tracing::warn!(
                    chunk = chunk_idx,
                    total_chunks,
                    chunk_pos0,
                    tokens_done = chunk_start,
                    tokens_total = t,
                    elapsed_s = format!("{elapsed_s:.1}"),
                    tok_per_s = format!("{toks_per_s:.1}"),
                    eta_s = format!("{eta_s:.1}"),
                    "prefill_progress"
                );
            }

            // Fold the PREVIOUS chunk's stages into the request accumulator before
            // clearing. No-op unless DEEPSTRIX_PREFILL_PROFILE=1 (it synchronises,
            // which serialises the two lanes — read per-stage busy time, not wall).
            if super::trace::prefill_profile::enabled() {
                super::trace::prefill_profile::add(
                    "dgpu",
                    &super::trace::rollup_by_name(&self.dgpu.events.harvest()?),
                );
                super::trace::prefill_profile::add(
                    "igpu",
                    &super::trace::rollup_by_name(&self.igpu.events.harvest()?),
                );
            }
            self.dgpu.events.reset();
            self.igpu.events.reset();

            let chunk_engram: Option<Vec<Vec<f32>>> = engram_rows.map(|rs| {
                let ein = ENGRAM_IN as usize;
                let (a, z) = (chunk_start * ein, (chunk_start + chunk_tokens.len()) * ein);
                rs.iter().map(|r| r[a..z].to_vec()).collect()
            });
            if ced {
                // Encoder + the decoder's KV projection; `residual` /
                // `hc_pre_carry` come back holding the rows ENTERING `split`.
                let cut = self.forward_prompt_batch_v2_pipelined_range(
                    bd_a,
                    bi_a,
                    bd_b,
                    bi_b,
                    sd,
                    si,
                    state,
                    weights,
                    chunk_input,
                    chunk_tokens,
                    chunk_pos0,
                    stats.as_deref_mut(),
                    image_spans,
                    pager.as_deref_mut(),
                    chunk_engram.as_deref(),
                    0..split + 1,
                    CedMode::KvSourceOnly,
                    None,
                )?;
                if cut != b_a {
                    return Err(eyre!("CED prefill: lane cut {cut} != planned {b_a}"));
                }
                // Keep the last SWA_WINDOW rows entering the decoder (host ring).
                let take = chunk_b.min(SWA_WINDOW as usize);
                let (m, hc) = (HC_MIX_DIM as usize, HC_DIM as usize);
                for i in chunk_b - take..chunk_b {
                    let (src, idx) = if i < b_a { (&*bd_a, i) } else { (&*bd_b, i - b_a) };
                    let mut row = ReplayRow { tok: chunk_tokens[i], hc: vec![0f32; hc], carry: vec![0f32; m] };
                    src.residual.slice_view(idx * hc, hc).copy_to_host(&mut row.hc)?;
                    src.hc_pre_carry.slice_view(idx * m, m).copy_to_host(&mut row.carry)?;
                    replay.push_back(row);
                    if replay.len() > SWA_WINDOW as usize {
                        replay.pop_front();
                    }
                }
            } else {
                self.forward_prompt_batch_v2_pipelined(
                    bd_a,
                    bi_a,
                    bd_b,
                    bi_b,
                    sd,
                    si,
                    state,
                    weights,
                    chunk_input,
                    chunk_tokens,
                    chunk_pos0,
                    stats.as_deref_mut(),
                    image_spans,
                    pager.as_deref_mut(),
                    // Slice the prompt-wide Engram rows down to this chunk.
                    chunk_engram.as_deref(),
                )?;
            }

            if let Some(exp_lock) = &self.perfetto {
                let mut exp = exp_lock.lock().unwrap();
                self.dgpu.events.for_each_pair(|name, s, e| {
                    let track = if name.contains(".xfer") || name.contains(".peer_push") {
                        &exp.dgpu_xfer
                    } else {
                        &exp.dgpu_compute
                    };
                    exp.emit_slice(track, name, s, e)
                })?;
                self.igpu.events.for_each_pair(|name, s, e| {
                    let track = if name.contains(".xfer") || name.contains(".peer_push") {
                        &exp.igpu_xfer
                    } else {
                        &exp.igpu_compute
                    };
                    exp.emit_slice(track, name, s, e)
                })?;
                exp.re_anchor(
                    self.dgpu.device,
                    &self.dgpu.compute,
                    &self.dgpu.xfer,
                    self.igpu.device,
                    &self.igpu.compute,
                    &self.igpu.xfer,
                )?;
                self.current_device.store(-1, std::sync::atomic::Ordering::Relaxed);
            }

            // THE ACTUAL split the range used, not the one `plan_chunk` planned.
            //
            // `plan_chunk` computes `ceil(chunk_b/2)` for the two-lane driver,
            // but `forward_prompt_batch_v2_pipelined_range` re-decides it and
            // may take the SINGLE-LANE path (`V41_PREFILL_SINGLE_LANE_MAX`),
            // putting every row in lane A. Reading the logits back at the
            // PLANNED cut then takes rows `[cut, chunk_b)` out of lane B's
            // buffer, which that chunk never wrote -- silently, since the
            // buffer holds a previous chunk's rows (or zeros).
            //
            // MEASURED before this fix, single-lane B=6 verify: rows 0-2 agreed
            // with decode to 6.8e-4 nats and rows 3-5 were garbage at ~11 nats
            // (KLD); at B=4 the cliff moved to row 2. Both are exactly
            // `ceil(b/2)`, which is what identified it. That is the "single lane
            // drops DSpark E to 1.74" retraction: the compute was always right,
            // the READBACK was wrong. The CED branch above was already immune --
            // it asserts `cut == b_a` -- which is why this only ever bit the
            // non-CED path.
            let b_a = bd_a.mtp_lane_cut.min(chunk_b);
            let b_b = chunk_b - b_a;

            if last_only {
                if is_last_chunk && !ced {
                    // Last token: lives in lane B if b_b > 0, else lane A.
                    let (src_bd, last_idx) = if b_b > 0 {
                        (&*bd_b, b_b - 1)
                    } else {
                        (&*bd_a, b_a - 1)
                    };
                    let logits = self.head_from_row(head_scratch, src_bd, last_idx, weights)?;
                    dump_prefill_logits(&logits)?;
                    out_logits = logits;
                }
            } else {
                // One weight read per LANE (rows are contiguous within a lane),
                // not one per row.
                out_logits.extend_from_slice(&self.head_rows(head_scratch, bd_a, b_a, weights)?);
                out_logits.extend_from_slice(&self.head_rows(head_scratch, bd_b, b_b, weights)?);
            }

            chunk_start = chunk_end;
            chunk_idx += 1;
            if let Some(f) = on_chunk_done {
                f();
            }
        }

        if ced {
            // Decoder SWA Bounded Replay (§3.2.2): feed the last SWA_WINDOW
            // rows' encoder outputs through the decoder with the decoder rings
            // emptied, so a segment query at index i sees window keys in
            // [max(s, i-W+1), i] and the complete global KV. Approximate by
            // design for N > SWA_WINDOW (identical to the exact path otherwise).
            if let Some(c) = cancel {
                if c.load(std::sync::atomic::Ordering::Relaxed) {
                    return Ok(Vec::new());
                }
            }
            let b_seg = replay.len();
            if b_seg == 0 || b_seg > t {
                return Err(eyre!("CED prefill: replay segment {b_seg} of {t} rows"));
            }
            let seg_pos0 = pos0 + (t - b_seg) as u32;
            // Fresh prompt only (see `prefill_job_finish`, KNOWN_BUGS #25).
            if pos0 == 0 {
                for l in split..N_LAYER as usize {
                    state.layers[l].n_raw = 0;
                    state.layers[l].raw_off = 0;
                }
            }
            let mut seg_hcs: Vec<Vec<f32>> = Vec::with_capacity(b_seg);
            let mut seg_carry: Vec<Vec<f32>> = Vec::with_capacity(b_seg);
            let mut seg_tokens: Vec<i32> = Vec::with_capacity(b_seg);
            for r in replay.drain(..) {
                seg_hcs.push(r.hc);
                seg_carry.push(r.carry);
                seg_tokens.push(r.tok);
            }
            let t0 = std::time::Instant::now();
            // Fold the PREVIOUS chunk's stages into the request accumulator before
            // clearing. No-op unless DEEPSTRIX_PREFILL_PROFILE=1 (it synchronises,
            // which serialises the two lanes — read per-stage busy time, not wall).
            if super::trace::prefill_profile::enabled() {
                super::trace::prefill_profile::add(
                    "dgpu",
                    &super::trace::rollup_by_name(&self.dgpu.events.harvest()?),
                );
                super::trace::prefill_profile::add(
                    "igpu",
                    &super::trace::rollup_by_name(&self.igpu.events.harvest()?),
                );
            }
            self.dgpu.events.reset();
            self.igpu.events.reset();
            let b_a = self.forward_prompt_batch_v2_pipelined_range(
                bd_a,
                bi_a,
                bd_b,
                bi_b,
                sd,
                si,
                state,
                weights,
                &seg_hcs,
                &seg_tokens,
                seg_pos0,
                None,
                // Text-causal replay: `image_spans` is deliberately None here.
                // It drives only (a) raw-window widening and (b) chunk/lane cut
                // planning, both of which exist for V4-Flash's BIDIRECTIONAL
                // in-span window. V4.1 image tokens attend causally (no
                // `get_image_visible` in the reference model.py), so the replay
                // needs neither. Routing is NOT lost: `seg_tokens` carries the
                // synthetic ids >= N_VOCAB, so `image_runs` still applies
                // bias_vl on the replayed decoder layers, and a lane split
                // through an image run stays causally valid. Validated e2e
                // 2026-09-13 with a replay landing inside an image span
                // (seg_pos0=328, span 12..455) — see docs/v41/VISION_PORT.md §5.
                None,
                pager.as_deref_mut(),
                None,
                split..N_LAYER as usize,
                CedMode::Replay,
                Some(&seg_carry),
            )?;
            let (src_bd, last_idx) = if b_seg > b_a { (&*bd_b, b_seg - b_a - 1) } else { (&*bd_a, b_a - 1) };
            let logits = self.head_from_row(head_scratch, src_bd, last_idx, weights)?;
            tracing::info!(
                replay_tokens = b_seg,
                seg_pos0,
                elapsed_s = format!("{:.1}", t0.elapsed().as_secs_f32()),
                total_s = format!("{:.1}", prefill_start.elapsed().as_secs_f32()),
                "ced_replay"
            );
            dump_prefill_logits(&logits)?;
            out_logits = logits;
            if let Some(f) = on_chunk_done {
                f();
            }
        }
        if super::trace::prefill_profile::enabled() {
            // Last chunk / replay is still unharvested at this point.
            super::trace::prefill_profile::add(
                "dgpu",
                &super::trace::rollup_by_name(&self.dgpu.events.harvest()?),
            );
            super::trace::prefill_profile::add(
                "igpu",
                &super::trace::rollup_by_name(&self.igpu.events.harvest()?),
            );
            super::trace::prefill_profile::emit_and_clear(t);
        }
        Ok(out_logits)
    }
}

/// State carried between the four phases of one lane-layer's pre-MoE work
/// (`pre_moe_chain` -> `pre_moe_route` -> `pre_moe_prep` -> `pre_moe_launch`).
/// Owned values only: no phase borrows the pager or the weights across a phase
/// boundary, so the driver can interleave the other lane's phases in between.
#[derive(Default)]
pub struct PreMoeCarry {
    phase: PreMoePhase,
    pub layer: i32,
    pub b: u32,
    cs_n_used: usize,
    cs_n_embd: usize,
    dump_pos: u32,
    cap_ok: bool,
    defer_shared: bool,
    pub remote_split_on: bool,
    split_cap: u32,
    sparse_resid_layer: bool,
    moe_group_bound: u32,
    /// Sequential callers keep the cross-lane iGPU drain in front of `ensure`
    /// (conditional on it being able to evict); the pipelined driver runs both
    /// lanes' preps before either launch and sets this false.
    pub drain_before_ensure: bool,
    /// May `pre_moe_route` emit box-2 look-ahead prefetch hints? Only on the
    /// SEQUENTIAL path: they are read from shared scratch (`sd.look_sel*`) that
    /// the other lane's chain clobbers in the pipelined order.
    pub lookahead_hints_ok: bool,
    /// Promise the daemon (`REQ_FLAG_PARTNER`) that another request for this
    /// same layer follows immediately, so it can merge the two lanes' passes
    /// instead of streaming the layer's experts twice. Set by the pipelined
    /// driver on the lane it routes FIRST; false everywhere else.
    pub partner_follows: bool,
    // route -> prep
    sel_host_remote: Vec<i32>,
    sel_host_audit: Vec<i32>,
    owns_eff: Vec<bool>,
    ids: Vec<u32>,
    replay_offload: bool,
    sparse_resid: bool,
    mc0: u64,
    owns_remote_some: bool,
    // prep -> launch
    n_work_items: u32,
    variant: String,
    wmma_path: bool,
    hot_active: bool,
    max_per_expert: u32,
    chunk_size: u32,
}

#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum PreMoePhase {
    #[default]
    New,
    Chained,
    Routed,
    Prepped,
    Launched,
    Skipped,
}

impl PreMoeCarry {
    /// Phase-order invariant: each phase runs exactly once, in order.
    /// Returns `Ok(false)` when the chain skipped this lane-layer entirely
    /// (`b == 0`, `CedMode::KvSourceOnly`): the later phases are then no-ops.
    fn advance(&mut self, from: PreMoePhase, to: PreMoePhase) -> eyre::Result<bool> {
        if self.phase == PreMoePhase::Skipped {
            return Ok(false);
        }
        if self.phase != from {
            return Err(eyre!("pre-MoE phase order violated on L{}: expected {from:?}, carry is at {:?} (wanted {to:?})", self.layer, self.phase));
        }
        self.phase = to;
        Ok(true)
    }
}

impl HeterogeneousEngine {
    /// One layer of batched prefill, all phases. Reads
    /// `bd.residual` (per-token input HC), writes `bd.residual_next`
    /// (per-token output HC). All other `bd` / `sd` fields are scratch.
    /// Thin wrapper over the split pre-MoE + post-MoE methods;
    /// single-lane callers use this.
    #[allow(clippy::too_many_arguments)]
    /// V4.1 Engram prefill twin of `stage_engram_rows`: `rows` = `b × ENGRAM_IN`
    /// f32, one gathered row set per chunk row, for the next Engram layer.
    /// Stage one lane's Engram rows for `layer`, if it is an Engram layer.
    ///
    /// `rows` holds one flattened `[chunk_b * ENGRAM_IN]` buffer per Engram layer
    /// (in `ENGRAM_LAYERS` order); `off`/`n` select this lane's slice of the chunk.
    /// A no-op for non-Engram layers and when the caller supplied no rows.
    fn stage_engram_lane(
        &self,
        bd: &mut BatchDgpuScratch,
        layer: usize,
        rows: Option<&[Vec<f32>]>,
        off: usize,
        n: usize,
    ) -> eyre::Result<()> {
        let Some(rs) = rows else { return Ok(()) };
        let Some(li) = crate::config::ENGRAM_LAYERS.iter().position(|&l| l as usize == layer)
        else {
            return Ok(());
        };
        let ein = ENGRAM_IN as usize;
        let buf = rs.get(li).ok_or_else(|| {
            eyre!("engram: layer {layer} is Engram index {li} but only {} buffers given", rs.len())
        })?;
        if buf.len() < (off + n) * ein {
            return Err(eyre!(
                "engram: layer {layer} rows hold {} floats, lane needs [{}..{}]",
                buf.len(), off * ein, (off + n) * ein
            ));
        }
        self.stage_engram_rows_batch(bd, &buf[off * ein..(off + n) * ein])
    }

    pub fn stage_engram_rows_batch(&self, bd: &mut BatchDgpuScratch, rows: &[f32]) -> eyre::Result<()> {
        if rows.is_empty() || rows.len() % ENGRAM_IN as usize != 0 || rows.len() > bd.engram_rows.len() {
            return Err(eyre!("stage_engram_rows_batch: {} floats (ENGRAM_IN {}, capacity {})", rows.len(), ENGRAM_IN, bd.engram_rows.len()));
        }
        self.set_current_cached(self.dgpu.device)?;
        bd.engram_rows.slice_view_mut(0, rows.len()).copy_from_host(rows)?;
        bd.engram_rows_ready = true;
        Ok(())
    }

    /// One multi-stream decode step, K=1 (docs/v41/MULTISTREAM_DECODE_PLAN.md
    /// 3.7, M1a step 3): row `i` is `tokens[i]` at the next position of the
    /// stream in `slots[i]`, all `b` rows through the batched layer driver with
    /// `RowLayout::Arena`. `input_hcs[i]` is `embed(tokens[i])` broadcast to
    /// HC_DIM (as for the prompt drivers); `engram_rows` one flattened
    /// `[b * ENGRAM_IN]` buffer per Engram layer, in `ENGRAM_LAYERS` order,
    /// rows in slot order. Compacts any stream whose raw region is full first,
    /// uploads the step's tables, runs the 40 layers, advances every stream.
    /// On return `bd.residual` holds the post-last-layer HC per row and
    /// `bd.hc_pre_carry` the carries: `head_rows(ds, bd, b, weights)` turns
    /// them into `[b * N_VOCAB]` logits.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_step_arena(
        &self,
        bd: &mut BatchDgpuScratch,
        bi: &mut BatchIgpuScratch,
        sd: &mut BatchDgpuShared,
        si: &mut BatchIgpuShared,
        arena: &mut KvArena,
        dev: &mut RowTablesDev,
        slots: &[u32],
        weights: &HetModelWeights,
        input_hcs: &[Vec<f32>],
        tokens: &[i32],
        engram_rows: &mut LazyEngramRows<'_>,
        mut pager: Option<&mut super::expert_pager::ExpertPager>,
    ) -> eyre::Result<RowTables> {
        // Decode-phase link window (the rows are decode rows of live streams).
        self.remote_set_phase_busy_poll(true);
        let b = tokens.len();
        if b == 0 {
            return Ok(RowTables::default());
        }
        if slots.len() != b || input_hcs.len() != b {
            return Err(eyre!(
                "forward_step_arena: {} slots / {} hcs for {b} tokens",
                slots.len(),
                input_hcs.len()
            ));
        }
        for (i, hc) in input_hcs.iter().enumerate() {
            if hc.len() != HC_DIM as usize {
                return Err(eyre!("forward_step_arena: input_hcs[{i}] len {} != HC_DIM", hc.len()));
            }
        }
        check_scratch_rows("forward_step_arena", b, bd, bi, sd, si)?;
        if bd.mtp_capture_rows > 0 {
            return Err(eyre!("forward_step_arena: MTP capture is not supported on arena rows (v1)"));
        }
        self.current_device.store(-1, std::sync::atomic::Ordering::Relaxed);
        self.set_current_cached(self.dgpu.device)?;
        arena.state.restore_compressor_lending();

        // A full raw region moves its window down before the tables are
        // derived (the tables carry the append slot).
        for &slot in slots {
            if arena.needs_compaction(slot) {
                arena.compact_raw(slot, &self.dgpu.compute, &mut sd.kv_ring_scratch)?;
            }
        }
        let tables = arena.tables(slots)?;
        dev.upload(&tables, &self.dgpu.compute)?;

        for (i, hc) in input_hcs.iter().enumerate() {
            let mut slot = bd.residual.slice_view_mut(i * HC_DIM as usize, HC_DIM as usize);
            slot.copy_from_host(hc)?;
        }
        {
            let mut pos_v = bd.pos_per_b.slice_view_mut(0, b);
            pos_v.copy_from_host_async(&tables.pos_per, &self.dgpu.compute)?;
        }
        let ein = ENGRAM_IN as usize;
        for layer in 0..N_LAYER as usize {
            if weights.dgpu_layers[layer].engram.is_some() {
                let li = crate::config::ENGRAM_LAYERS.iter().position(|&l| l as usize == layer);
                let rows = match li { Some(i) => engram_rows.get()?.and_then(|rs| rs.get(i)), None => None };
                match rows {
                    Some(r) if r.len() >= b * ein => self.stage_engram_rows_batch(bd, &r[..b * ein])?,
                    _ => return Err(eyre!("forward_step_arena: layer {layer} needs Engram rows for {b} rows")),
                }
            }
            let dlw = &weights.dgpu_layers[layer];
            let ilw = &weights.igpu_layers[layer];
            let sev = &self.sync_events.layers[layer];
            let hot_active = prefill_hot_active(dlw, ilw, bd, sd);
            arena.state.with_kv_source(layer, |ls| {
                self.forward_layer_pre_moe_v2(
                    bd, bi, sd, si, ls, dlw, ilw, 0, tokens, None, None, sev,
                    pager.as_deref_mut(), CedMode::Exact,
                    RowLayout::Arena { tables: &tables, dev, next_router: weights.dgpu_layers.get(layer + 1), next_router2: weights.dgpu_layers.get(layer + 2) },
                )?;
                self.route_probe_after_layer(bd, weights, layer, b, 0)?;
                self.forward_layer_post_moe_v2(bd, b as u32, sev, hot_active)
            })?;
            std::mem::swap(&mut bd.residual, &mut bd.residual_next);
        }
        self.dgpu.compute.synchronize()?;
        for &slot in slots {
            arena.advance(slot)?;
        }
        Ok(tables)
    }

    /// Two-lane batched decode step over `slots` (K=1 per stream): lane A =
    /// the first half of the slots in `bd_a`/`bi_a`, lane B = the rest in
    /// `bd_b`/`bi_b`, interleaved per layer exactly as the pipelined prefill
    /// driver does — pre(A,L+1) is issued while box 2 still holds lane A's
    /// layer-L tickets, and lane B's whole layer runs under lane A's box-2
    /// wait. With box 2 paging-bound (its disk, 2026-09-20: ~430 ms of a 600 ms
    /// 8-row step), the exposed remote wait of one lane hides the OTHER lane's
    /// dGPU chain, local MoE and combine. Numerics per row are those of
    /// `forward_step_arena` (same kernels, same per-row tables); only the
    /// by-expert batch composition differs (half the rows per launch).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_step_arena_pipelined(
        &self,
        bd_a: &mut BatchDgpuScratch,
        bi_a: &mut BatchIgpuScratch,
        bd_b: &mut BatchDgpuScratch,
        bi_b: &mut BatchIgpuScratch,
        sd: &mut BatchDgpuShared,
        si: &mut BatchIgpuShared,
        arena: &mut KvArena,
        dev_a: &mut RowTablesDev,
        dev_b: &mut RowTablesDev,
        slots: &[u32],
        weights: &HetModelWeights,
        input_hcs: &[Vec<f32>],
        tokens: &[i32],
        engram_rows: &mut LazyEngramRows<'_>,
        mut pager: Option<&mut super::expert_pager::ExpertPager>,
    ) -> eyre::Result<(RowTables, RowTables)> {
        self.remote_set_phase_busy_poll(true);
        let b = tokens.len();
        if b < 2 {
            return Err(eyre!("forward_step_arena_pipelined: needs >= 2 rows (got {b})"));
        }
        if slots.len() != b || input_hcs.len() != b {
            return Err(eyre!("forward_step_arena_pipelined: {} slots / {} hcs for {b} tokens", slots.len(), input_hcs.len()));
        }
        for (i, hc) in input_hcs.iter().enumerate() {
            if hc.len() != HC_DIM as usize {
                return Err(eyre!("forward_step_arena_pipelined: input_hcs[{i}] len {} != HC_DIM", hc.len()));
            }
        }
        let b_a = b.div_ceil(2);
        let b_b = b - b_a;
        check_scratch_rows("forward_step_arena_pipelined", b_a, bd_a, bi_a, sd, si)?;
        check_scratch_rows("forward_step_arena_pipelined", b_b, bd_b, bi_b, sd, si)?;
        if bd_a.mtp_capture_rows > 0 || bd_b.mtp_capture_rows > 0 {
            return Err(eyre!("forward_step_arena_pipelined: MTP capture is not supported on arena rows"));
        }
        self.current_device.store(-1, std::sync::atomic::Ordering::Relaxed);
        self.set_current_cached(self.dgpu.device)?;
        arena.state.restore_compressor_lending();
        for &slot in slots {
            if arena.needs_compaction(slot) {
                arena.compact_raw(slot, &self.dgpu.compute, &mut sd.kv_ring_scratch)?;
            }
        }
        let (slots_a, slots_b) = slots.split_at(b_a);
        let (tokens_a, tokens_b) = tokens.split_at(b_a);
        let tables_a = arena.tables(slots_a)?;
        let tables_b = arena.tables(slots_b)?;
        dev_a.upload(&tables_a, &self.dgpu.compute)?;
        dev_b.upload(&tables_b, &self.dgpu.compute)?;
        for (i, hc) in input_hcs.iter().enumerate() {
            let (bd, k) = if i < b_a { (&mut *bd_a, i) } else { (&mut *bd_b, i - b_a) };
            let mut slot = bd.residual.slice_view_mut(k * HC_DIM as usize, HC_DIM as usize);
            slot.copy_from_host(hc)?;
        }
        {
            let mut va = bd_a.pos_per_b.slice_view_mut(0, b_a);
            va.copy_from_host_async(&tables_a.pos_per, &self.dgpu.compute)?;
            let mut vb = bd_b.pos_per_b.slice_view_mut(0, b_b);
            vb.copy_from_host_async(&tables_b.pos_per, &self.dgpu.compute)?;
        }
        let ein = ENGRAM_IN as usize;
        // Per-lane Engram staging for `layer` (rows are in slot order).
        let mut stage = |this: &Self, bd: &mut BatchDgpuScratch, layer: usize, off: usize, n: usize| -> eyre::Result<()> {
            if weights.dgpu_layers[layer].engram.is_some() {
                let li = crate::config::ENGRAM_LAYERS.iter().position(|&l| l as usize == layer);
                let rows = match li { Some(i) => engram_rows.get()?.and_then(|rs| rs.get(i)), None => None };
                match rows {
                    Some(r) if r.len() >= (off + n) * ein => this.stage_engram_rows_batch(bd, &r[off * ein..(off + n) * ein])?,
                    _ => return Err(eyre!("forward_step_arena_pipelined: layer {layer} needs Engram rows for {n} rows")),
                }
            }
            Ok(())
        };
        let n_layer = N_LAYER as usize;
        // DEPENDENCY-GRAPH ORDER (2026-09-22). Per layer:
        //   combine A, chain A, combine B, chain B, route A, route B,
        //   prep A, launch A, prep B, launch B
        // so that
        //  * both dGPU chains are launched before either lane's router readback,
        //    and the readback is an EVENT wait (`selected_ready`, recorded on the
        //    chain) that is already over -- not a `de.compute` drain;
        //  * both box-2 requests are submitted before either lane's pager or MoE
        //    work, so box 2 receives them within one chain's time of each other
        //    instead of one per lane turnaround.
        //
        // prep and launch stay PAIRED per lane, and that is not a missed
        // opportunity -- it is required by two pieces of shared state:
        //  * `BatchIgpuShared` (`expert_members`, `d_xq_q8k`, `d_x16`, the
        //    work-item arrays): the dispatch writes them then reads them, so B's
        //    writes landing in between make A read B's inputs;
        //  * `ExpertPager::remap_dev[layer]`: ONE buffer per layer, and the two
        //    lanes are on the SAME layer here with DIFFERENT exclusion masks
        //    (`owns_eff` depends on the lane's own picks under the small-B
        //    catch-all), so B's `ensure` would hand A's dispatch B's mask.
        // Both were caught by `multistream_step` G5c: interleaving them made
        // exactly one lane's rows wrong (12 of 24 at S=4).
        //
        // Because launch A is therefore queued before prep B's `ensure`, lane A's
        // MoE for this layer IS in flight while lane B pages, so the conditional
        // cross-lane iGPU drain stays (`drain_before_ensure` left true). Removing
        // it needs the pager to pin the other lane's live slots against eviction
        // (see the note at the drain site) -- that is the proper fix and is not
        // this change.
        macro_rules! rl {
            ($t:expr, $d:expr, $l:expr) => {
                RowLayout::Arena { tables: &$t, dev: &*$d, next_router: weights.dgpu_layers.get($l + 1), next_router2: weights.dgpu_layers.get($l + 2) }
            };
        }
        macro_rules! chain {
            ($bd:expr, $bi:expr, $tables:expr, $dev:expr, $toks:expr, $sev:expr, $off:expr, $n:expr, $l:expr) => {{
                let l: usize = $l;
                stage(self, $bd, l, $off, $n)?;
                arena.state.with_kv_source(l, |ls| {
                    let rl = rl!($tables, $dev, l);
                    self.pre_moe_chain($bd, $bi, sd, si, ls, &weights.dgpu_layers[l], &weights.igpu_layers[l], 0, $toks, None, None,
                        &$sev.layers[l], pager.as_deref_mut(), CedMode::Exact, &rl)
                })?
            }};
        }
        macro_rules! rest_layer {
            ($l:expr, $ca:expr, $cb:expr) => {{
                let l: usize = $l;
                let (ca, cb): (&mut PreMoeCarry, &mut PreMoeCarry) = ($ca, $cb);
                let sev_a = &self.sync_events.layers[l];
                let sev_b = &self.sync_events_t1.layers[l];
                // The look-ahead prefetch hints are read from SHARED scratch
                // (`sd.look_sel*`) that the other lane's chain has already
                // overwritten here, so this path does not emit them (they are
                // default OFF and measured a loss anyway).
                ca.lookahead_hints_ok = false;
                cb.lookahead_hints_ok = false;
                // Lane A's request is followed immediately by lane B's for the
                // same layer: promise it, so an IDLE daemon holds for the pair
                // instead of starting a pass it cannot merge into.
                ca.partner_follows = true;
                cb.partner_follows = false;
                // Both routes first: each waits on its own router EVENT, and both
                // submits are out before any pager work.
                { let rl = rl!(tables_a, dev_a, l); self.pre_moe_route(ca, bd_a, sd, sev_a, pager.as_deref_mut(), &rl)?; }
                { let rl = rl!(tables_b, dev_b, l); self.pre_moe_route(cb, bd_b, sd, sev_b, pager.as_deref_mut(), &rl)?; }
                // Then each lane's prep+launch as ONE unit (see the note above).
                self.pre_moe_prep(ca, bd_a, bi_a, sd, si, &weights.dgpu_layers[l], &weights.igpu_layers[l], sev_a, pager.as_deref_mut())?;
                self.pre_moe_launch(ca, bd_a, bi_a, sd, si, &weights.dgpu_layers[l], &weights.igpu_layers[l], sev_a, pager.as_deref_mut())?;
                self.pre_moe_prep(cb, bd_b, bi_b, sd, si, &weights.dgpu_layers[l], &weights.igpu_layers[l], sev_b, pager.as_deref_mut())?;
                self.pre_moe_launch(cb, bd_b, bi_b, sd, si, &weights.dgpu_layers[l], &weights.igpu_layers[l], sev_b, pager.as_deref_mut())?;
                self.route_probe_after_layer(bd_a, weights, l, b_a, 0)?;
                self.route_probe_after_layer(bd_b, weights, l, b_b, 1)?;
            }};
        }
        let mut ca = chain!(bd_a, bi_a, tables_a, dev_a, tokens_a, self.sync_events, 0, b_a, 0);
        let mut cb = chain!(bd_b, bi_b, tables_b, dev_b, tokens_b, self.sync_events_t1, b_a, b_b, 0);
        rest_layer!(0, &mut ca, &mut cb);
        for layer in 0..n_layer - 1 {
            let hot_a = prefill_hot_active(&weights.dgpu_layers[layer], &weights.igpu_layers[layer], bd_a, sd);
            self.forward_layer_post_moe_v2(bd_a, b_a as u32, &self.sync_events.layers[layer], hot_a)?;
            std::mem::swap(&mut bd_a.residual, &mut bd_a.residual_next);
            let mut ca = chain!(bd_a, bi_a, tables_a, dev_a, tokens_a, self.sync_events, 0, b_a, layer + 1);
            let hot_b = prefill_hot_active(&weights.dgpu_layers[layer], &weights.igpu_layers[layer], bd_b, sd);
            self.forward_layer_post_moe_v2(bd_b, b_b as u32, &self.sync_events_t1.layers[layer], hot_b)?;
            std::mem::swap(&mut bd_b.residual, &mut bd_b.residual_next);
            let mut cb = chain!(bd_b, bi_b, tables_b, dev_b, tokens_b, self.sync_events_t1, b_a, b_b, layer + 1);
            rest_layer!(layer + 1, &mut ca, &mut cb);
        }
        let last = n_layer - 1;
        let hot_a = prefill_hot_active(&weights.dgpu_layers[last], &weights.igpu_layers[last], bd_a, sd);
        self.forward_layer_post_moe_v2(bd_a, b_a as u32, &self.sync_events.layers[last], hot_a)?;
        std::mem::swap(&mut bd_a.residual, &mut bd_a.residual_next);
        let hot_b = prefill_hot_active(&weights.dgpu_layers[last], &weights.igpu_layers[last], bd_b, sd);
        self.forward_layer_post_moe_v2(bd_b, b_b as u32, &self.sync_events_t1.layers[last], hot_b)?;
        std::mem::swap(&mut bd_b.residual, &mut bd_b.residual_next);
        self.dgpu.compute.synchronize()?;
        for &slot in slots {
            arena.advance(slot)?;
        }
        Ok((tables_a, tables_b))
    }

    /// N-lane pipelined arena decode step (2026-09-21): `forward_step_arena_pipelined`
    /// generalised to `lanes.len()` lanes. Rows are split into contiguous
    /// balanced chunks, one per lane, and the layer loop round-robins
    /// post(lane, L) -> pre(lane, L+1) across lanes, so lane i's box-2 wait
    /// overlaps the other lanes' dGPU/iGPU work AND box 2 always has a request
    /// queued (the daemon's early paging of the queued request only pays when
    /// a frame is already there; with two lanes the hub's turnaround equals
    /// box 2's per-request time and the queue never exceeds one).
    /// Lane i uses `self.sync_events_lane(i)`. Returns one `RowTables` per lane.
    pub fn forward_step_arena_lanes(
        &self,
        lanes: &mut [(&mut BatchDgpuScratch, &mut BatchIgpuScratch, &mut RowTablesDev)],
        sd: &mut BatchDgpuShared,
        si: &mut BatchIgpuShared,
        arena: &mut KvArena,
        slots: &[u32],
        weights: &HetModelWeights,
        input_hcs: &[Vec<f32>],
        tokens: &[i32],
        engram_rows: &mut LazyEngramRows<'_>,
        mut pager: Option<&mut super::expert_pager::ExpertPager>,
    ) -> eyre::Result<Vec<RowTables>> {
        self.remote_set_phase_busy_poll(true);
        let b = tokens.len();
        let n = lanes.len();
        if n < 2 || b < n {
            return Err(eyre!("forward_step_arena_lanes: needs >= 2 lanes and >= 1 row per lane (got {n} lanes, {b} rows)"));
        }
        if slots.len() != b || input_hcs.len() != b {
            return Err(eyre!("forward_step_arena_lanes: {} slots / {} hcs for {b} tokens", slots.len(), input_hcs.len()));
        }
        for (i, hc) in input_hcs.iter().enumerate() {
            if hc.len() != HC_DIM as usize {
                return Err(eyre!("forward_step_arena_lanes: input_hcs[{i}] len {} != HC_DIM", hc.len()));
            }
        }
        // Contiguous balanced split: the first (b % n) lanes get one extra row.
        let mut offs: Vec<usize> = Vec::with_capacity(n + 1);
        offs.push(0);
        for i in 0..n {
            let sz = b / n + usize::from(i < b % n);
            offs.push(offs[i] + sz);
        }
        for (i, (bd, bi, _)) in lanes.iter().enumerate() {
            let bl = offs[i + 1] - offs[i];
            check_scratch_rows("forward_step_arena_lanes", bl, bd, bi, sd, si)?;
            if bd.mtp_capture_rows > 0 {
                return Err(eyre!("forward_step_arena_lanes: MTP capture is not supported on arena rows"));
            }
        }
        self.current_device.store(-1, std::sync::atomic::Ordering::Relaxed);
        self.set_current_cached(self.dgpu.device)?;
        arena.state.restore_compressor_lending();
        for &slot in slots {
            if arena.needs_compaction(slot) {
                arena.compact_raw(slot, &self.dgpu.compute, &mut sd.kv_ring_scratch)?;
            }
        }
        let mut tables: Vec<RowTables> = Vec::with_capacity(n);
        for (i, (bd, _, dev)) in lanes.iter_mut().enumerate() {
            let (lo, hi) = (offs[i], offs[i + 1]);
            let t = arena.tables(&slots[lo..hi])?;
            dev.upload(&t, &self.dgpu.compute)?;
            for (k, hc) in input_hcs[lo..hi].iter().enumerate() {
                let mut slot = bd.residual.slice_view_mut(k * HC_DIM as usize, HC_DIM as usize);
                slot.copy_from_host(hc)?;
            }
            let mut v = bd.pos_per_b.slice_view_mut(0, hi - lo);
            v.copy_from_host_async(&t.pos_per, &self.dgpu.compute)?;
            tables.push(t);
        }
        let ein = ENGRAM_IN as usize;
        let mut stage = |this: &Self, bd: &mut BatchDgpuScratch, layer: usize, off: usize, nrows: usize| -> eyre::Result<()> {
            if weights.dgpu_layers[layer].engram.is_some() {
                let li = crate::config::ENGRAM_LAYERS.iter().position(|&l| l as usize == layer);
                let rows = match li { Some(i) => engram_rows.get()?.and_then(|rs| rs.get(i)), None => None };
                match rows {
                    Some(r) if r.len() >= (off + nrows) * ein => this.stage_engram_rows_batch(bd, &r[off * ein..(off + nrows) * ein])?,
                    _ => return Err(eyre!("forward_step_arena_lanes: layer {layer} needs Engram rows for {nrows} rows")),
                }
            }
            Ok(())
        };
        let n_layer = N_LAYER as usize;
        // pre(lane, layer): stage Engram rows, then the pre-MoE half (attention,
        // router, box-2 submit, box-1 MoE) with this lane's tables and events.
        macro_rules! pre {
            ($i:expr, $layer:expr) => {{
                let i: usize = $i;
                let layer: usize = $layer;
                let (lo, hi) = (offs[i], offs[i + 1]);
                let (bd, bi, dev) = &mut lanes[i];
                stage(self, bd, layer, lo, hi - lo)?;
                let t = &tables[i];
                let sev = &self.sync_events_lane(i).layers[layer];
                arena.state.with_kv_source(layer, |ls| {
                    self.forward_layer_pre_moe_v2(bd, bi, sd, si, ls, &weights.dgpu_layers[layer], &weights.igpu_layers[layer], 0, &tokens[lo..hi], None, None,
                        sev, pager.as_deref_mut(), CedMode::Exact,
                        RowLayout::Arena { tables: t, dev, next_router: weights.dgpu_layers.get(layer + 1), next_router2: weights.dgpu_layers.get(layer + 2) })?;
                    self.route_probe_after_layer(bd, weights, layer, hi - lo, i)
                })?;
            }};
        }
        macro_rules! post {
            ($i:expr, $layer:expr) => {{
                let i: usize = $i;
                let layer: usize = $layer;
                let bl = (offs[i + 1] - offs[i]) as u32;
                let (bd, _, _) = &mut lanes[i];
                let hot = prefill_hot_active(&weights.dgpu_layers[layer], &weights.igpu_layers[layer], bd, sd);
                self.forward_layer_post_moe_v2(bd, bl, &self.sync_events_lane(i).layers[layer], hot)?;
                std::mem::swap(&mut bd.residual, &mut bd.residual_next);
            }};
        }
        for i in 0..n {
            pre!(i, 0);
        }
        for layer in 0..n_layer - 1 {
            for i in 0..n {
                post!(i, layer);
                pre!(i, layer + 1);
            }
        }
        let last = n_layer - 1;
        for i in 0..n {
            post!(i, last);
        }
        self.dgpu.compute.synchronize()?;
        for &slot in slots {
            arena.advance(slot)?;
        }
        Ok(tables)
    }

    pub fn forward_layer_batch_v2(
        &self,
        bd: &mut BatchDgpuScratch,
        bi: &mut BatchIgpuScratch,
        sd: &mut BatchDgpuShared,
        si: &mut BatchIgpuShared,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        ilw: &IgpuLayerWeights,
        pos0: u32,
        tokens: &[i32],
        // Per-row `(left, right)` image visibility for THIS batch's rows
        // (`image_spans::rows_visibility`); `None` == all text.
        vis: Option<&[(u32, u32)]>,
        stats: Option<&mut PrefillStats>,
        // M7 expert pager: when Some, this layer's experts are paged out of its
        // pool instead of read from the (placeholder) resident buffers.
        mut pager: Option<&mut super::expert_pager::ExpertPager>,
    ) -> eyre::Result<()> {
        let layer = dlw.layer_idx as usize;
        let b = tokens.len() as u32;
        if b == 0 {
            return Ok(());
        }
        let sev = &self.sync_events.layers[layer];
        let hot_active = prefill_hot_active(dlw, ilw, bd, sd);
        self.forward_layer_pre_moe_v2(bd, bi, sd, si, ls, dlw, ilw, pos0, tokens, vis, stats, sev, pager.as_deref_mut(), CedMode::Exact, RowLayout::Contiguous)?;
        self.forward_layer_post_moe_v2(bd, b, sev, hot_active)?;
        Ok(())
    }

    /// Pre-MoE phase of one prefill layer. Submits dGPU stages 1-10
    /// (attn + router + shared expert), the dGPU→iGPU peer push, the
    /// iGPU MoE chain (iq2 + q2k_down), and the iGPU→dGPU peer push of
    /// the MoE output. Records `sev.moe_arrived` once the MoE result has
    /// landed on dGPU. Does NOT queue `ffn_combine` — the caller drives
    /// that via `forward_layer_post_moe_v2`, which lets two lanes share
    /// the de.compute stream without ffn_combine serializing the second
    /// lane's pre-MoE work behind the first lane's ffn_combine.
    ///
    /// `bd` / `bi` are this lane's per-lane scratch; `sd` / `si` are the
    /// shared sets (one instance for both lanes). Everything this call
    /// leaves for `forward_layer_post_moe_v2`, for `de.xfer` / `ie.xfer`,
    /// or for the other lane to run past lives in `bd` / `bi`; every
    /// `sd` / `si` buffer is dead (last read on de.compute / ie.compute
    /// in program order) by the time this returns.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_layer_pre_moe_v2(
        &self,
        bd: &mut BatchDgpuScratch,
        bi: &mut BatchIgpuScratch,
        sd: &mut BatchDgpuShared,
        si: &mut BatchIgpuShared,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        ilw: &IgpuLayerWeights,
        pos0: u32,
        tokens: &[i32],
        // Vision-Exp per-row `(left, right)` visibility inside an image span
        // (`image_spans::rows_visibility`, already clamped). Drives the raw
        // attention window only; the router-side image flag comes from
        // `tokens` (id >= N_VOCAB). `None` == all text.
        vis: Option<&[(u32, u32)]>,
        stats: Option<&mut PrefillStats>,
        sev: &super::engine::LayerSyncEvents,
        // M7 expert pager: when Some, this layer's experts are paged out of its
        // pool instead of read from the (placeholder) resident buffers.
        mut pager: Option<&mut super::expert_pager::ExpertPager>,
        // M7 CED mode of this call (only `CED_DECODER_START` is ever called
        // with anything but `Exact`).
        ced: CedMode,
        // Multi-stream: which per-row layout the KV in `ls` has (see `RowLayout`).
        rows: RowLayout<'_>,
    ) -> eyre::Result<()> {
        // Sequential composition of the four phases (see `pre_moe_chain`); the
        // pipelined arena driver interleaves them across lanes instead.
        let mut c = self.pre_moe_chain(bd, bi, sd, si, ls, dlw, ilw, pos0, tokens, vis, stats, sev, pager.as_deref_mut(), ced, &rows)?;
        self.pre_moe_route(&mut c, bd, sd, sev, pager.as_deref_mut(), &rows)?;
        self.pre_moe_prep(&mut c, bd, bi, sd, si, dlw, ilw, sev, pager.as_deref_mut())?;
        self.pre_moe_launch(&mut c, bd, bi, sd, si, dlw, ilw, sev, pager)
    }

    pub fn pre_moe_chain(
        &self,
        bd: &mut BatchDgpuScratch,
        bi: &mut BatchIgpuScratch,
        sd: &mut BatchDgpuShared,
        si: &mut BatchIgpuShared,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        ilw: &IgpuLayerWeights,
        pos0: u32,
        tokens: &[i32],
        // Vision-Exp per-row `(left, right)` visibility inside an image span
        // (`image_spans::rows_visibility`, already clamped). Drives the raw
        // attention window only; the router-side image flag comes from
        // `tokens` (id >= N_VOCAB). `None` == all text.
        vis: Option<&[(u32, u32)]>,
        stats: Option<&mut PrefillStats>,
        sev: &super::engine::LayerSyncEvents,
        // M7 expert pager: when Some, this layer's experts are paged out of its
        // pool instead of read from the (placeholder) resident buffers.
        mut pager: Option<&mut super::expert_pager::ExpertPager>,
        // M7 CED mode of this call (only `CED_DECODER_START` is ever called
        // with anything but `Exact`).
        ced: CedMode,
        // Multi-stream: which per-row layout the KV in `ls` has (see `RowLayout`).
        rows: &RowLayout<'_>,
    ) -> eyre::Result<PreMoeCarry> {
        let layer = dlw.layer_idx;
        let arena: Option<(&RowTables, &RowTablesDev)> = match &rows {
            RowLayout::Arena { tables, dev, .. } => Some((*tables, *dev)),
            RowLayout::Contiguous => None,
        };
        // Per-stage HIP graphs (arena decode steps only: static shapes per
        // (layer, rows, lane); a prefill chunk's pos0/visibility vary per call).
        let cap_ok = arena.is_some() && !super::engine::subtensor_dump_armed(layer as usize);
        let lane_ptr = bd.residual.raw() as usize;
        let arena_store = arena.and_then(|_| store_index_of(layer as usize));
        if let Some((t, d)) = arena {
            if t.pos_per.len() != tokens.len() || (d.rows_cap as usize) < tokens.len() {
                return Err(eyre!(
                    "L{layer}: arena tables cover {} rows (device cap {}) but the call has {}",
                    t.pos_per.len(),
                    d.rows_cap,
                    tokens.len()
                ));
            }
            if vis.is_some() || ced != CedMode::Exact || bd.mtp_capture_rows > 0 {
                return Err(eyre!(
                    "L{layer}: RowLayout::Arena is text-only, CedMode::Exact, no MTP capture (v1)"
                ));
            }
            if t.stores.len() != d.stores.len() || (dlw.ratio > 0 && arena_store.map_or(true, |si| si >= t.stores.len())) {
                return Err(eyre!("L{layer}: arena tables have no store for this layer"));
            }
        }
        // Per-row base arrays for this layer's store (None: contiguous, or a
        // dense layer). The `*_rows` launches take them; their `None` form is
        // byte-identical to the old entry points.
        let arena_comp_base: Option<&DeviceBuffer<i32>> =
            arena.zip(arena_store).map(|((_, d), si)| &d.stores[si].comp_base_per);
        let arena_keys_base: Option<&DeviceBuffer<u32>> =
            arena.zip(arena_store).map(|((_, d), si)| &d.stores[si].keys_base_per);
        let arena_state_base: Option<&DeviceBuffer<i32>> =
            arena.zip(arena_store).map(|((_, d), si)| &d.stores[si].state_base_per);
        let arena_fire_state_idx: Option<&DeviceBuffer<i32>> =
            arena.zip(arena_store).map(|((_, d), si)| &d.stores[si].fire_state_idx);
        let arena_fire_dst_row: Option<&DeviceBuffer<i32>> =
            arena.zip(arena_store).map(|((_, d), si)| &d.stores[si].fire_dst_row);
        // Position of row `i`: the sequence's `pos0 + i`, or the row's stream's.
        let pos_at = |i: u32| -> u32 {
            match arena {
                Some((t, _)) => t.pos_per[i as usize] as u32,
                None => pos0 + i,
            }
        };
        // Sub-tensor dumps are tagged with row 0's position (== `pos0` for a
        // contiguous chunk; the stream's position for arena rows).
        let dump_pos = pos_at(0);
        if super::engine::subtensor_dump_armed(layer as usize) {
            DUMP_LAYER_POS.store(((layer as u64) << 32) | dump_pos as u64, std::sync::atomic::Ordering::Relaxed);
        }
        // DSpark: the drafter eats the hc-collapsed residual ENTERING layers
        // 37/38/39. A batched verify does not know until AFTER it runs which
        // row becomes the next head, so capture EVERY row and select later.
        // Stored slot-major, `[3][MTP_CAP_ROWS][N_EMBD]`, because
        // `hc_weighted.launch_batched` writes one contiguous `[b, n_embd]`
        // block per call.
        if bd.mtp_capture_rows > 0 {
            if let Some(slot) = super::mtp::MtpState::src_slot(layer) {
                let ne = crate::config::N_EMBD as usize;
                let rows = bd.mtp_capture_rows.min(super::batch_scratch::MTP_CAP_ROWS);
                let n = tokens.len().min(rows);
                if n > 0 {
                    // `src` below slices `bd.residual` at `skip * nhc * ne` for
                    // `n * nhc * ne` floats, i.e. the WHOLE lane batch has to fit
                    // the lane scratch; `dst` / `wsrc` are sized for
                    // MTP_CAP_ROWS. Either bound broken reads (or writes) past
                    // the buffer into whatever scratch follows it.
                    assert!(
                        tokens.len() <= bd.rows,
                        "mtp capture L{layer}: lane batch of {} rows exceeds lane scratch \
                         capacity {} rows; the `residual` slice would run past the buffer",
                        tokens.len(),
                        bd.rows
                    );
                    assert!(
                        n <= super::batch_scratch::MTP_CAP_ROWS,
                        "mtp capture L{layer}: capturing {n} rows > MTP_CAP_ROWS {}; mtp_src and \
                         mtp_hc_mean are only sized for MTP_CAP_ROWS",
                        super::batch_scratch::MTP_CAP_ROWS
                    );
                    let de = &self.dgpu;
                    self.set_current_cached(de.device)?;
                    // Capture the LAST `n` rows of the batch, not the first.
                    //
                    // For a VERIFY this is a no-op: `mtp_capture_rows == b`, so
                    // skip == 0 and the whole batch is taken either way. It
                    // matters for PREFILL SEEDING, where the chunk can be far
                    // longer than the ring and the rows we want are the MOST
                    // RECENT prompt positions -- the ones the drafter will
                    // actually attend over when generation starts.
                    let skip = tokens.len() - n;
                    let nhc = crate::config::N_HC as usize;
                    let mut dst = bd.mtp_src.slice_view_mut(
                        slot * super::batch_scratch::MTP_CAP_ROWS * ne,
                        n * ne,
                    );
                    // Only `x` moves. The kernel indexes both per batch row
                    // (`x + b*n_hc*n_embd`, `weights + b*w_stride`), but
                    // `mtp_hc_mean` is only `N_HC * MTP_CAP_ROWS` long and is a
                    // CONSTANT 1/n_hc everywhere, so rows [0,n) are numerically
                    // identical to rows [skip, skip+n) -- and skipping it would
                    // run off the end for any chunk longer than MTP_CAP_ROWS.
                    let src = bd.residual.slice_view(skip * nhc * ne, n * nhc * ne);
                    let wsrc = bd.mtp_hc_mean.slice_view(0, n * nhc);
                    de.hc_weighted.launch_batched(
                        &de.compute,
                        &mut dst,
                        &src,
                        &wsrc,
                        crate::config::N_EMBD,
                        crate::config::N_HC,
                        crate::config::N_HC,
                        n as u32,
                    )?;
                    // Which absolute positions these rows are. Both lanes may
                    // capture; the caller takes whichever ran LATER (higher
                    // pos0), since that lane holds the most recent positions.
                    bd.mtp_captured = n;
                    bd.mtp_captured_pos0 = pos0 + skip as u32;
                }
            }
        }
        // Sub-tensor dump, row 0 only, SAME tags as decode's
        // (`engine.rs`, forward_token_impl). Decode has had this since the ds4
        // port; prefill never did, so there was no way to bisect a
        // prefill-vs-decode divergence to a layer or a kernel. Armed by
        // DEEPSTRIX_DUMP_SUBTENSOR_LAYERS + _DIR, no cost otherwise.
        if super::engine::subtensor_dump_armed(layer as usize) {
            self.dgpu.compute.synchronize()?;
            let hc = HC_DIM as usize;
            let ne = crate::config::N_EMBD as usize;
            // Tag with the row's ABSOLUTE position, not just the layer.
            // This runs once per LANE, and lane B's row 0 is position `b_a`, not
            // 0 -- an untagged file is silently whichever lane wrote last, which
            // produced a bogus "diverges at layer 0" reading (relRMSE 1.0 at
            // every layer) when diffed against an oracle dump indexed by
            // position. The position must be in the name for the diff to be
            // well-posed.
            let tag_r = format!("pf_pre_residual_p{dump_pos}");
            let tag_n = format!("pf_attn_input_norm_p{dump_pos}");
            super::engine::maybe_dump_subtensor_f32_view(
                layer as usize, &tag_r, &bd.residual.slice_view(0, hc))?;
            super::engine::maybe_dump_subtensor_f32_view(
                layer as usize, &tag_n, &sd.attn_input_norm.slice_view(0, ne))?;
        }
        if ilw.layer_idx != layer {
            return Err(eyre!(
                "forward_layer_pre_moe_v2: dgpu L{} != igpu L{}",
                layer,
                ilw.layer_idx
            ));
        }
        let ratio = dlw.ratio;
        let b = tokens.len() as u32;
        if b == 0 {
            return Ok(PreMoeCarry { phase: PreMoePhase::Skipped, layer, ..Default::default() });
        }
        check_scratch_rows("forward_layer_pre_moe_v2", b as usize, bd, bi, sd, si)?;

        self.set_current_cached(self.dgpu.device)?;
        let de = &self.dgpu;
        let cs_n_embd = N_EMBD as usize;
        let cs_qflat = Q_FLAT as usize;
        let cs_kvhd = N_HEAD_DIM as usize;
        let cs_n_used = N_EXPERT_USED;

        // ========================================================
        // Stage 1: mhc_pre_attn (BATCHED)
        // rms_nw → f16_narrow → sinkhorn → hc_weighted → rms_w
        // ========================================================
        let _t_mhc_pre = de.events.stage("dgpu.mhc_pre_attn", &de.compute)?;
        if cfg!(feature = "v41") && layer == 0 {
            // Single-pass mHC: every token's layer-0 attention collapses with
            // the initial one-hot(copy 0) pre-mix (ARCH_SPEC §1.1). Rows are
            // independent, so the per-row carry survives the layer-major order.
            let rows = b as usize * HC_MIX_DIM as usize;
            bd.hc_pre_carry
                .slice_view_mut(0, rows)
                .copy_from_host_async(&super::scratch::hc_pre_onehot_rows()[..rows], &de.compute)?;
        }
        if let Some(eg) = dlw.engram.as_ref() {
            // V4.1 Engram on every row of the chunk, ENGRAM_CHUNK rows per pass
            // (decode twin: forward_layer.rs).
            if !bd.engram_rows_ready {
                return Err(eyre!("forward_layer_pre_moe_v2 L{layer}: Engram rows not staged (stage_engram_rows_batch)"));
            }
            let _t = de.events.stage("dgpu.engram", &de.compute)?;
            let (ein, eout, hcd) = (ENGRAM_IN as usize, ENGRAM_OUT as usize, HC_DIM as usize);
            let mut c0 = 0usize;
            while c0 < b as usize {
                let n = (b as usize - c0).min(ENGRAM_CHUNK as usize);
                let rows = bd.engram_rows.slice_view(c0 * ein, n * ein);
                let mut xq = bd.engram_xq.slice_view_mut(0, n * ein);
                let mut xs = bd.engram_xscale.slice_view_mut(0, n * ein / 32);
                let mut kv = bd.engram_kv.slice_view_mut(0, n * eout);
                de.q8.quantize_input_batched(&de.compute, &mut xq, &mut xs, &rows, ENGRAM_IN, n as u32)?;
                // wkv is [25600, 6144] Q8_0 = 167 MB, by far the largest
                // single weight the encoder touches. `matvec_batched`
                // (`q8_0_gemv_batched_warp8`) puts the batch on grid.z with
                // x fastest, so 3200 workgroups separate two visits to the
                // same 52 KB row tile: every z-slice is its own DRAM pass and
                // one 64-row chunk streams 64 x 167 MB = 10.7 GB (16.7 ms at
                // 640 GB/s), i.e. 267 ms per 512-row lane over the two Engram
                // layers. `gemm_lds_tiled` is the LDS-tiled WMMA GEMM that
                // took qb from 8.8 to 1.4 ms on exactly this class of shape:
                // one pass over the weight per (BM=64 x BN=64) tile.
                // Rollback: V41_ENGRAM_GEMV=1.
                // At n <= 8 rows (multi-stream decode steps) the B-packed
                // GEMV reads the 167 MB once for all rows and runs at the
                // dGPU's bandwidth; the LDS-tiled WMMA GEMM at 1-3 rows costs
                // ~1 ms per Engram layer (dgpu.engram 2.15 ms/step, 2 layers).
                if engram_gemv_fallback() || super::dispatch::small_b_dense_dp4a(n as u32) {
                    de.q8.matvec_batched(&de.compute, &mut kv, &eg.wkv.buffer, &xq, &xs, ENGRAM_OUT, ENGRAM_IN, n as u32)?;
                } else {
                    de.q8_wmma.gemm_lds_tiled(&de.compute, &mut kv, &eg.wkv.buffer, &xq, &xs, ENGRAM_OUT, ENGRAM_IN, n as u32)?;
                }
                let mut h = bd.residual.slice_view_mut(c0 * hcd, n * hcd);
                de.engram_gate.launch(&de.compute, &mut h, &kv, &eg.qk, N_HC, N_EMBD, ENGRAM_OUT, N_HC * N_EMBD, RMS_EPS, n as u32)?;
                c0 += n;
            }
            bd.engram_rows_ready = false;
        }
        let cap = self.stage_cap(de, "g.mhc_pre_attn", layer as usize, b, lane_ptr, cap_ok)?;
        if !cap.skip {
        {
            let _t = de.events.stage("k.mhc_pre_attn.rms_nw", &de.compute)?;
            de.rms_nw
                .launch_batched(&de.compute, &mut sd.flat, &bd.residual, 1, HC_DIM, RMS_EPS, b)?;
        }
        {
            let _t = de.events.stage("k.mhc_pre_attn.f16_matvec", &de.compute)?;
            if mhc_pre_scaled_for(b) {
                // DECODE-EXACT path. Decode computes `mix = (W @ x) * inv_rms`
                // (`rms_nw_mw.launch_inv_only` + `matvec_pre_scaled`,
                // forward_layer.rs); the batched path computed
                // `mix = W @ normalize(x)`, rounding 20480 normalised values to
                // f32 BEFORE the dot product instead of dotting raw and scaling
                // once. `hc_split_sinkhorn` then runs 20 doubly-stochastic
                // iterations on the result, amplifying the difference -- measured
                // as the layer-0 seed of the verify-vs-decode divergence
                // (KNOWN_BUGS #0b): 6.5e-03 leaving layer 0, 5.6e-01 by layer 39.
                // One row at a time, so it is bit-identical to decode by
                // construction. Verify-sized batches only; large-B prefill keeps
                // the batched form.
                let hcd = HC_DIM as usize;
                let hmd = HC_MIX_DIM as usize;
                for r in 0..b as usize {
                    let row = bd.residual.slice_view(r * hcd, hcd);
                    de.rms_nw_mw.launch_inv_only(
                        &de.compute,
                        &mut sd.mhc_inv_scalar,
                        &row,
                        &mut sd.mhc_rms_partials,
                        HC_DIM,
                        16,
                        RMS_EPS,
                    )?;
                    let mut mix_row = sd.mix.slice_view_mut(r * hmd, hmd);
                    de.f16.matvec_pre_scaled(
                        &de.compute,
                        &mut mix_row,
                        &dlw.hc_attn_fn.buffer,
                        &row,
                        &sd.mhc_inv_scalar,
                        HC_MIX_DIM,
                        HC_DIM,
                    )?;
                }
            } else if mhc_narrow_fallback_for(b) {
                de.f16.matvec_narrow_batched(
                    &de.compute,
                    &mut sd.mix,
                    &dlw.hc_attn_fn.buffer,
                    &sd.flat,
                    HC_MIX_DIM,
                    HC_DIM,
                    b,
                )?;
            } else {
                de.f16.gemm_batched_wmma(
                    &de.compute,
                    &mut sd.mix,
                    &dlw.hc_attn_fn.buffer,
                    &sd.flat,
                    HC_MIX_DIM,
                    HC_DIM,
                    b,
                )?;
            }
        }
        {
            let _t = de.events.stage("k.mhc_pre_attn.sinkhorn", &de.compute)?;
            de.hc_sinkhorn.launch_batched(
                &de.compute,
                &mut bd.split,
                &sd.mix,
                &dlw.hc_attn_scale,
                &dlw.hc_attn_base,
                N_HC,
                SINKHORN_ITERS,
                SINKHORN_EPS,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.mhc_pre_attn.hc_weighted", &de.compute)?;
            // w_stride: split is [B, HC_MIX_DIM]; pre-sigmoid w is first n_hc.
            // V4.1 collapses with the PREVIOUS sub-block's pre (carry), then
            // carries this sub-block's pre forward (decode twin: forward_layer.rs).
            let w = if cfg!(feature = "v41") { &bd.hc_pre_carry } else { &bd.split };
            de.hc_weighted.launch_batched(&de.compute, &mut sd.attn_cur, &bd.residual, w, N_EMBD, N_HC, HC_MIX_DIM, b)?;
            // Bisect within layer 0 (KNOWN_BUGS #0b): attn_cur is the mHC
            // COLLAPSE output, before attention runs. If it already differs from
            // decode's, the residue is still mHC; if it matches, the divergence
            // is attention onward.
            if super::engine::subtensor_dump_armed(layer as usize) {
                de.compute.synchronize()?;
                super::engine::maybe_dump_subtensor_f32_view(
                    layer as usize,
                    &format!("pf_attn_cur_p{dump_pos}"),
                    &sd.attn_cur.slice_view(0, N_EMBD as usize),
                )?;
            }
            // M7 CED: a source-only call leaves the carry as it entered the
            // layer (the replay re-runs this sub-block and carries it then).
            if cfg!(feature = "v41") && ced != CedMode::KvSourceOnly {
                let rows = b as usize * HC_MIX_DIM as usize;
                let cur = bd.split.slice_view(0, rows);
                bd.hc_pre_carry.slice_view_mut(0, rows).copy_from_buffer_async(&cur, &de.compute)?;
            }
        }
        {
            let _t = de.events.stage("k.mhc_pre_attn.rms_w", &de.compute)?;
            de.rms_w.launch_weighted_batched(
                &de.compute,
                &mut sd.attn_input_norm,
                &sd.attn_cur,
                &dlw.attn_norm,
                N_EMBD,
                RMS_EPS,
                b,
            )?;
        }

        }
        cap.end()?;
        drop(_t_mhc_pre);

        // ========================================================
        // Stage 2: Q chain (BATCHED quantize + matvec + rms + ...)
        // ========================================================
        // M7 CED: a source-only call needs neither Q nor the window KV.
        if ced != CedMode::KvSourceOnly {
        let _t_q = de.events.stage("dgpu.q_chain", &de.compute)?;
        let cap = self.stage_cap(de, "g.q_chain", layer as usize, b, lane_ptr, cap_ok)?;
        if !cap.skip {
        {
            let _t = de.events.stage("k.q_chain.cast_input_f16", &de.compute)?;
            de.q8k.launch_cast_f16_2d(&de.compute, &mut sd.x16_n_embd, &sd.attn_input_norm,
                b, N_EMBD, super::batch_scratch::f16_pitch(N_EMBD))?;
            // `qa_matvec` below takes the dp4a arm at small b, which consumes
            // (xq_n_embd, xscale_n_embd). The only other writer of that pair is
            // the KV chain FURTHER DOWN, so without this the q_a projection
            // would read the PREVIOUS layer's quantisation -- the exact defect
            // the kv_chain comment below records as having cost accept rate.
            // Same source buffer and shape as that one, so the later write is
            // an identical no-op.
            if dlw.attn_q_a.dtype == v4flash_core::gguf::GgufType::Q8_0
                && super::dispatch::small_b_dense_dp4a(b)
            {
                de.q8.quantize_input_batched(
                    &de.compute, &mut sd.xq_n_embd, &mut sd.xscale_n_embd,
                    &sd.attn_input_norm, N_EMBD, b,
                )?;
            }
        }
        {
            let _t = de.events.stage("k.q_chain.qa_matvec", &de.compute)?;
            // Q8_0: LDS-tiled WMMA GEMM (weight shared across BN=64 cols;
            // matvec_batched was weight-BW-bound at B=512). K-quants
            // (unsloth q_a = Q5_K/Q6_K): dp4a register-tiled GEMM on Q8_K
            // activations. The i8 quantize above stays either way —
            // attn_kv (always Q8_0) consumes it.
            if dlw.attn_q_a.dtype != v4flash_core::gguf::GgufType::Q8_0 {
                de.q8k.launch(
                    &de.compute,
                    &mut sd.kq_attn_q8k,
                    &sd.attn_input_norm,
                    crate::config::BLOCKS_Q8K_GATE_IN * b,
                )?;
            }
            super::dispatch::dense_gemm_prefill(
                de,
                &de.compute,
                &mut sd.qr,
                &dlw.attn_q_a,
                &sd.xq_n_embd,
                &sd.xscale_n_embd,
                &sd.kq_attn_q8k,
                Some((&sd.x16_n_embd, super::batch_scratch::f16_pitch(N_EMBD))),
                b,
                N_LORA_Q,
                N_EMBD,
            )?;
        }
        {
            let _t = de.events.stage("k.q_chain.rms_w", &de.compute)?;
            de.rms_w.launch_weighted_batched(
                &de.compute,
                &mut sd.qr_normed,
                &sd.qr,
                &dlw.q_a_norm,
                N_LORA_Q,
                RMS_EPS,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.q_chain.cast_qr_f16", &de.compute)?;
            de.q8k.launch_cast_f16_2d(&de.compute, &mut sd.qr16, &sd.qr_normed,
                b, N_LORA_Q, super::batch_scratch::f16_pitch(N_LORA_Q))?;
        }
        // qb up-projection (M=Q_FLAT=32768, K=N_LORA_Q=1024). Default
        // LDS-tiled WMMA: cooperative-load A+B into LDS per K-outer iter,
        // then WMMA from LDS — kills the s_wait_loadcnt latency throttle
        // that capped both dp4a and the older non-tiled WMMA. Isolated A/B
        // at B=512: dp4a 8.82ms / wmma_old 4.24ms / wmma_lds_tiled 1.38ms
        // → 6.4× over dp4a, 3.1× over the older WMMA. Q_FLAT % 64 == 0 ✓.
        // QB_WMMA=wmma forces the older non-tiled WMMA; QB_WMMA=0 forces dp4a.
        // At verify-sized batches take DECODE'S kernel, so the verify reproduces
        // decode instead of a different kernel family. Re-applies 925fcee/c60a172,
        // whose revert (16ec20a) blamed the alignment for an accept-rate drop
        // 1.788 -> 1.372; that drop was the UNGUARDED KV arm below feeding
        // `matvec_batched` an `xq_n_embd` nothing on the prefill path ever writes,
        // and the KLD it was scored against (2.026 -> 1.631 nats) was measured
        // through #0b. `QB_WMMA` / `Q8_GROUPED_VARIANT` / `Q8_OUT_VARIANT` set
        // explicitly still win.
        let qb_variant = std::env::var("QB_WMMA")
            .unwrap_or_else(|_| if prefill_f32_matvec(b) { "dp4a".into() } else { "f16x".into() });
        if qb_variant != "f16x" {
            // legacy variants consume the Q8_0 quantization of qr
            de.q8.quantize_input_batched(&de.compute, &mut sd.qr_xq, &mut sd.qr_xscale, &sd.qr_normed, N_LORA_Q, b)?;
        }
        match qb_variant.as_str() {
            "f16x" => {
                let _t = de.events.stage("k.q_chain.qb_f16x", &de.compute)?;
                de.q8_wmma.gemm_f16x(&de.compute, &mut sd.q, &dlw.attn_q_b.buffer, &sd.qr16,
                    N_LORA_Q, Q_FLAT, 1, b, super::batch_scratch::f16_pitch(N_LORA_Q))?;
            }
            "0" | "dp4a" => {
                let _t = de.events.stage("k.q_chain.qb_matvec", &de.compute)?;
                de.q8.matvec_batched(
                    &de.compute, &mut sd.q, &dlw.attn_q_b.buffer,
                    &sd.qr_xq, &sd.qr_xscale, Q_FLAT, N_LORA_Q, b,
                )?;
            }
            "wmma" => {
                let _t = de.events.stage("k.q_chain.qb_wmma", &de.compute)?;
                de.q8_wmma.gemm(
                    &de.compute, &mut sd.q, &dlw.attn_q_b.buffer,
                    &sd.qr_xq, &sd.qr_xscale, Q_FLAT, N_LORA_Q, b,
                )?;
            }
            _ => {
                let _t = de.events.stage("k.q_chain.qb_lds", &de.compute)?;
                de.q8_wmma.gemm_lds_tiled(
                    &de.compute, &mut sd.q, &dlw.attn_q_b.buffer,
                    &sd.qr_xq, &sd.qr_xscale, Q_FLAT, N_LORA_Q, b,
                )?;
            }
        }
        {
            let _t = de.events.stage("k.q_chain.rms_nw_heads", &de.compute)?;
            if cfg!(feature = "v41") {
                // V4.1 has no per-head q RMSNorm after wq_b (ARCH_SPEC §1.2);
                // rope reads q_normed, so pass q through (decode twin: forward_layer.rs).
                let n = b as usize * Q_FLAT as usize;
                let src = sd.q.slice_view(0, n);
                sd.q_normed.slice_view_mut(0, n).copy_from_buffer_async(&src, &de.compute)?;
            } else {
                // rms_nw over batch: each batch has [N_HEAD, N_HEAD_DIM] rows.
                // batched API: grid (B, N_HEAD, 1), inner row of N_HEAD_DIM.
                de.rms_nw.launch_batched(&de.compute, &mut sd.q_normed, &sd.q, N_HEAD, N_HEAD_DIM, RMS_EPS, b)?;
            }
        }
        {
            let _t = de.events.stage("k.q_chain.rope", &de.compute)?;
            let pos_v = bd.pos_per_b.slice_view(0, b as usize);
            de.rope.launch_forward_batched(
                &de.compute,
                &mut sd.q_normed,
                &pos_v,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                b,
                &dlw.rope_params,
            )?;
        }
        // KNOWN_BUGS #0b: q_normed is the LAST attention input not yet diffed
        // against decode. Slots, windows and the SWA kernel are all proven
        // identical, so if row 0 of this differs from `dec_q_normed_p<POS>` the
        // bug is in the Q chain (rope position / rms), not in attention.
        }
        cap.end()?;
        if super::engine::subtensor_dump_armed(layer as usize) {
            self.dgpu.compute.synchronize()?;
            let nq = (crate::config::N_HEAD * crate::config::N_HEAD_DIM) as usize;
            super::engine::maybe_dump_subtensor_f32_view(
                layer as usize,
                &format!("pf_q_normed_p{dump_pos}"),
                &sd.q_normed.slice_view(0, nq),
            )?;
        }

        drop(_t_q);

        // ========================================================
        // Stage 3: KV chain (BATCHED matvec + rms; per-token rope/fp8/f16rt)
        // ========================================================
        let _t_kv = de.events.stage("dgpu.kv_chain", &de.compute)?;
        let cap = self.stage_cap(de, "g.kv_chain", layer as usize, b, lane_ptr, cap_ok)?;
        if !cap.skip {
        {
            // Decode uses `de.q8.matvec` here (forward_layer.rs:771), preceded at
            // :759 by quantizing `attn_input_norm` into xq_n_embd/xscale_n_embd.
            // `matvec_batched` is that kernel with a row dimension.
            //
            // THE QUANTIZE IS THE WHOLE POINT. 925fcee replaced this inline with
            // `matvec_batched(&sd.xq_n_embd, &sd.xscale_n_embd, ..)` and no
            // quantize -- and NOTHING on the prefill path writes those buffers
            // (every writer targets `dgpu_scratch`, i.e. decode). It fed the KV
            // projection uninitialised quantisation state, which is what actually
            // cost the accept rate that got the alignment reverted. Every other
            // variant site here guards its own quantize the same way
            // (`if variant != "f16x" { quantize_input_batched(..) }`); this one
            // had no such guard because it was not a variant flip.
            if prefill_f32_matvec(b) {
                de.q8.quantize_input_batched(
                    &de.compute, &mut sd.xq_n_embd, &mut sd.xscale_n_embd,
                    &sd.attn_input_norm, N_EMBD, b,
                )?;
                let _t = de.events.stage("k.kv_chain.matvec", &de.compute)?;
                de.q8.matvec_batched(
                    &de.compute, &mut sd.kv_raw, &dlw.attn_kv.buffer,
                    &sd.xq_n_embd, &sd.xscale_n_embd, N_HEAD_DIM, N_EMBD, b,
                )?;
            } else {
                let _t = de.events.stage("k.kv_chain.gemm_f16x", &de.compute)?;
                de.q8_wmma.gemm_f16x(&de.compute, &mut sd.kv_raw, &dlw.attn_kv.buffer, &sd.x16_n_embd,
                    N_EMBD, N_HEAD_DIM, 1, b, super::batch_scratch::f16_pitch(N_EMBD))?;
            }
        }
        {
            let _t = de.events.stage("k.kv_chain.rms_w", &de.compute)?;
            de.rms_w.launch_weighted_batched(
                &de.compute,
                &mut sd.kv_normed,
                &sd.kv_raw,
                &dlw.kv_a_norm,
                N_HEAD_DIM,
                RMS_EPS,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.kv_chain.rope", &de.compute)?;
            let pos_v = bd.pos_per_b.slice_view(0, b as usize);
            de.rope.launch_forward_batched(
                &de.compute,
                &mut sd.kv_normed,
                &pos_v,
                1,
                N_HEAD_DIM,
                N_ROT,
                b,
                &dlw.rope_params,
            )?;
        }
        {
            let _t = de.events.stage("k.kv_chain.fp8", &de.compute)?;
            if cfg!(feature = "v41") {
                // V4.1 window KV: E4M3 × 2^e per 32 over the whole row (decode twin: forward_layer.rs).
                de.fp4kv.launch_fp8_window(&de.compute, &mut sd.kv_normed, b, N_HEAD_DIM)?;
            } else {
                de.fp8.launch_batched(
                    &de.compute,
                    &mut sd.kv_normed,
                    N_HEAD_DIM - N_ROT,
                    N_HEAD_DIM,
                    b,
                )?;
            }
        }
        {
            // f16rt is pure elementwise — stretch n by B for a single launch.
            let _t = de.events.stage("k.kv_chain.f16rt", &de.compute)?;
            de.f16rt.launch(&de.compute, &mut sd.kv_normed, b * N_HEAD_DIM)?;
        }

        // ========================================================
        // Stage 4: KV cache append + compressor (SERIAL per batch)
        //
        // We capture per-batch n_raw_after / n_comp_after snapshots so
        // Stage 5 attention can use causal prefix lengths instead of the
        // final post-loop values (which would let token i attend to
        // future tokens i+1..B-1).
        // ========================================================
        }
        cap.end()?;
        drop(_t_kv);
        } // ced != KvSourceOnly (stages 2-3)
        let _t_kv_append_comp = de.events.stage("dgpu.kv_append_compressor_serial", &de.compute)?;
        let mut n_raw_after: Vec<u32> = Vec::with_capacity(b as usize);
        let mut n_comp_after: Vec<u32> = Vec::with_capacity(b as usize);

        // Oversized cache + per-token offset. The cache (ls.kv_cache) is
        // sized SWA_WINDOW + B_MAX rows so a prefill chunk can write its full
        // batch into slots [n_raw_before .. n_raw_before + b) WITHOUT evicting
        // any prior content. Attention's `n_raw_offset_per[i]` tells the
        // kernel where each token's causally-valid window begins in the cache:
        //
        //   absolute_position p_i  = chunk_pos0 + i  (lives at cache slot
        //                            n_raw_before + i)
        //   causal window         = [max(0, p_i - W + 1) .. p_i]
        //   in cache slots         = [max(0, n_raw_before+i+1-W) .. n_raw_before+i+1)
        //
        // So n_raw_per[i] = min(n_raw_before+i+1, W) and
        //    n_raw_offset_per[i] = max(0, n_raw_before+i+1 - W).
        //
        // After the chunk's attention runs, an explicit eviction pass copies
        // the last SWA_WINDOW rows back to slots [0..W) and resets ls.n_raw
        // to W (or less, for short prompts) so the steady-state SWA invariant
        // holds for decode and for the next chunk.
        //
        // Vision-Exp: rows inside an `[IMAGE_START .. IMAGE_END]` span use
        // the widened window of `get_window_topk_idxs_visible` —
        //   slots [causal_end - max(W, left+1) .. causal_end + right),
        //   capped at W + 384 keys (`image_spans::raw_window`)
        // — i.e. they also see keys AHEAD of them inside the span. Those
        // keys exist because the whole chunk's K/V is appended (above)
        // before this chunk's attention runs, and `chunk_visibility`
        // guarantees the span lies inside this batch. The compressed /
        // indexer path below stays causal and untouched.
        let n_raw_before = ls.n_raw;
        let mut n_raw_offset_after: Vec<u32> = Vec::with_capacity(b as usize);
        if ced != CedMode::KvSourceOnly {
        match vis {
            None if arena.is_some() => {
                // One position of its own stream per row: the window is the
                // stream's `n_raw` rows plus the row itself, capped at W, ending
                // at the row's append slot (absolute row in the layer buffer).
                let (t, _) = arena.expect("checked");
                let dec = (layer as usize) >= crate::config::CED_DECODER_START;
                let (nrp, slp) = if dec { (&t.n_raw_per_dec, &t.slot_per_dec) } else { (&t.n_raw_per, &t.slot_per) };
                for i in 0..b as usize {
                    let n_per = (nrp[i] as u32 + 1).min(SWA_WINDOW);
                    let offset = (slp[i] as u32 + 1).saturating_sub(n_per);
                    n_raw_after.push(n_per);
                    n_raw_offset_after.push(offset);
                }
            }
            None => {
                for i in 0..b as usize {
                    let causal_end = n_raw_before + i as u32 + 1; // exclusive upper slot
                    let n_per = causal_end.min(SWA_WINDOW);
                    let offset = causal_end.saturating_sub(SWA_WINDOW);
                    // KNOWN_BUGS #0b: does the verify attend the SAME slots as
                    // decode? Arithmetic, so cheap to settle. V41_WINDOW_DBG=1.
                    if i == 0 && std::env::var("V41_WINDOW_DBG").as_deref() == Ok("1") {
                        tracing::info!(
                            layer, pos0, n_raw_before, causal_end, n_per, offset,
                            "window.prefill row0"
                        );
                    }
                    n_raw_after.push(n_per);
                    n_raw_offset_after.push(offset);
                }
            }
            Some(v) => {
                if v.len() != b as usize {
                    return Err(eyre!(
                        "L{layer}: visibility rows {} != batch rows {b}",
                        v.len()
                    ));
                }
                for (i, &(left, right)) in v.iter().enumerate() {
                    let (offset, count) =
                        image_spans::raw_window(n_raw_before, i as u32, left, right);
                    if offset + count > n_raw_before + b {
                        return Err(eyre!(
                            "L{layer}: row {i} raw window [{offset}, {}) reaches past the \
                             chunk's appended K/V ({} rows) — image span not inside chunk",
                            offset + count,
                            n_raw_before + b
                        ));
                    }
                    n_raw_after.push(count);
                    n_raw_offset_after.push(offset);
                }
            }
        }
        // The cache is oversized to SWA_WINDOW + B_MAX rows, so we always
        // use the no-eviction batched append. The post-chunk eviction pass at
        // the end of this layer (after attention) copies the last SWA_WINDOW
        // rows down to slots [0..W) and updates ls.n_raw, restoring the
        // steady-state SWA invariant before the next chunk or decode.
        //
        // The cache bound is tight (no margin): n_raw_before ≤ SWA_WINDOW and
        // b ≤ B_MAX must hold, otherwise launch_batched will OOB-write into
        // the next layer's KV allocation. Guard in debug builds.
        // APPEND AT `raw_off + n_raw_before`, not `n_raw_before`.
        //
        // Readers take their slice from `raw_off * head_dim` and decode appends
        // monotonically at `raw_off + n_raw` (see `HetLayerState::raw_off`), so
        // an append that ignores `raw_off` only agrees with the readers while
        // `raw_off == 0`. It usually is: `normalize_raw_windows` zeroes it
        // before every verify, so LANE A is always safe -- which is exactly why
        // this survived.
        //
        // But a SPECULATIVE append does not evict; it SLIDES `raw_off` instead
        // (see the eviction block), and only once the window is full. So after
        // lane A, `raw_off == b_a`, and LANE B then reads shifted by `b_a`
        // while writing unshifted: its window drops `b_a` rows of real history
        // and picks up `b_a` slots that are not causally its own.
        //
        // MEASURED before this fix, on the rows the verify EMITS from: 4/4
        // catastrophic divergences (KL up to 8.07 nats vs a 0.006 baseline) all
        // at lane B's first row, all at pos >= SWA_WINDOW, minimum pos exactly
        // 128 -- i.e. from the very first slide, and never before it.
        if let Some((t, d)) = arena {
            // Per-row destination slots in the layer's arena buffer
            // (`KvArena::tables`: `region + raw_off + n_raw`, compaction before
            // the step keeps it inside the stream's region).
            let dec = (layer as usize) >= crate::config::CED_DECODER_START;
            let slp = if dec { &t.slot_per_dec } else { &t.slot_per };
            let need = slp.iter().map(|&s| s as usize + 1).max().unwrap_or(0) * N_HEAD_DIM as usize;
            if ls.kv_cache.len() < need {
                return Err(eyre!(
                    "L{layer}: arena raw append slot past the layer buffer ({} f16 < {need})",
                    ls.kv_cache.len()
                ));
            }
            de.kv_append.launch_batched_rows(
                &de.compute,
                &mut ls.kv_cache,
                &sd.kv_normed,
                0,
                N_HEAD_DIM,
                b,
                Some(if dec { &d.slot_per_dec } else { &d.slot_per }),
            )?;
        } else {
        let append_at = ls.raw_off + n_raw_before;
        assert!(
            (append_at + b) as usize <= KV_CACHE_ROWS,
            "kv_append OOB: raw_off={} + n_raw_before={n_raw_before} + b={b} >              KV_CACHE_ROWS={}",
            ls.raw_off,
            KV_CACHE_ROWS,
        );
        de.kv_append.launch_batched(
            &de.compute,
            &mut ls.kv_cache,
            &sd.kv_normed,
            append_at,
            N_HEAD_DIM,
            b,
        )?;
        }
        // Cache now holds n_raw_before + b rows; attention will index with
        // n_raw_offset_per. ls.n_raw is updated to its post-eviction value
        // at the END of this layer (see the eviction-down pass).
        } // ced != KvSourceOnly (window KV append)
        let n_raw_during_chunk = n_raw_before + b;

        // Batched matvec_pair across all B for ratio>0 layers. Produces
        // sd.kv_cur[B, comp_width] + sd.sc_cur[B, comp_width] in one launch.
        // The per-token loop below just READS from those buffers.
        // V4.1 reuse layers (no compressor weights) only read the source's store.
        // M7 CED Replay: the store already holds these positions (written by
        // the source-only pass), so the projection + store write are skipped
        // and the causal counts come from the positional reuse formula below.
        let own_compressor = ratio > 0 && dlw.compressor.is_some() && ced != CedMode::Replay;
        if own_compressor {
            let cw = dlw
                .compressor
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: missing compressor weights"))?;
            let comp_width = cw.width;
            // 2026-09-12: `matvec_pair_batched` launches grid.z = b, so EVERY
            // batch row re-reads the whole weight matrix — at ratio=4
            // (comp_width=1024) that is b × 2 × 1024 × 4096 × 2 B = 8.6 GB per
            // layer per chunk. It only survived because the 16.8 MB weight fits
            // the 64 MB Infinity Cache (~1.5 TB/s), which is still ~5.5 ms and
            // was the true cost of this stage (the serial loop was only ~2.1).
            // The WMMA GEMM tiles batch by BN=64, so the weight is read b/64
            // times instead of b: 8.6 GB -> 134 MB. Roofline ≈ 250 µs.
            //
            // NOT bit-exact: WMMA casts the f32 activation to f16 and
            // accumulates in a different order. `DEEPSTRIX_COMP_GEMM=0` keeps
            // the exact matvec path (and is what the bit-identity oracle for
            // the gather runs against).
            // DEFAULT OFF: measured 2026-09-12, the WMMA GEMM exceeds the
            // project's 5e-2-of-scale oracle bar (argmax still matched). The
            // compressor's gate output feeds a softmax in the pool, so the f16
            // activation cast costs more here than in an ordinary projection.
            // Kept behind the flag as a reference point; the shipped win is the
            // batch-tiled matvec below, which is bit-exact.
            let comp_gemm = std::env::var("DEEPSTRIX_COMP_GEMM")
                .map(|v| v != "0")
                .unwrap_or(false);
            if comp_gemm {
                if prefill_f32_matvec(b) {
                de.f16.matvec_batched(
                        &de.compute,
                        &mut sd.kv_cur,
                        &cw.wkv.buffer,
                        &sd.attn_input_norm,
                        comp_width,
                        N_EMBD,
                        b,
                    )?;
                } else {
                de.f16.gemm_batched_wmma(
                        &de.compute,
                        &mut sd.kv_cur,
                        &cw.wkv.buffer,
                        &sd.attn_input_norm,
                        comp_width,
                        N_EMBD,
                        b,
                    )?;
                }
                if prefill_f32_matvec(b) {
                de.f16.matvec_batched(
                        &de.compute,
                        &mut sd.sc_cur,
                        &cw.wgate.buffer,
                        &sd.attn_input_norm,
                        comp_width,
                        N_EMBD,
                        b,
                    )?;
                } else {
                de.f16.gemm_batched_wmma(
                        &de.compute,
                        &mut sd.sc_cur,
                        &cw.wgate.buffer,
                        &sd.attn_input_norm,
                        comp_width,
                        N_EMBD,
                        b,
                    )?;
                }
            } else if ratio == 1 {
                // V4.1 ratio 1 (layer 20): latent = norm(wkv(x)) — one batched matvec,
                // no gate. `sc_cur` is zeroed so the (identity) 1-row pool sees finite
                // scores; the state/snapshot machinery below is ratio-generic.
                de.f16.matvec_batched(
                    &de.compute,
                    &mut sd.kv_cur,
                    &cw.wkv.buffer,
                    &sd.attn_input_norm,
                    comp_width,
                    N_EMBD,
                    b,
                )?;
                sd.sc_cur.slice_view_mut(0, (b * comp_width) as usize).fill_zero_async(&de.compute)?;
            } else if std::env::var("DEEPSTRIX_COMP_TILED").map(|v| v != "0").unwrap_or(true) {
                de.f16.matvec_pair_batched_tiled(
                    &de.compute,
                    &mut sd.kv_cur,
                    &mut sd.sc_cur,
                    &cw.wkv.buffer,
                    &cw.wgate.buffer,
                    &sd.attn_input_norm,
                    comp_width,
                    N_EMBD,
                    b,
                )?;
            } else {
                de.f16.matvec_pair_batched(
                    &de.compute,
                    &mut sd.kv_cur,
                    &mut sd.sc_cur,
                    &cw.wkv.buffer,
                    &cw.wgate.buffer,
                    &sd.attn_input_norm,
                    comp_width,
                    N_EMBD,
                    b,
                )?;
            }
        }

        // Per-segment batched state_write. Each segment is ≤ `ratio`
        // positions long; within a segment, state_writes go to distinct
        // rows (rows {pos_mod_start..pos_mod_start+seg_len}) so they're
        // safely batched. Segments are bounded by compressor boundaries
        // (where pool+shuffle fire serially) or the chunk end.
        if own_compressor {
            let cw = dlw
                .compressor
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: missing compressor weights"))?;
            let comp_width = cw.width;

            // Precompute per-b (row, pos_mod) and upload once.
            let row_host: Vec<i32> = (0..b)
                .map(|i| {
                    let pos = pos_at(i);
                    let pm = pos % ratio;
                    let row = if ratio == 4 { 4 + pm } else { pm };
                    row as i32
                })
                .collect();
            let pos_mod_host: Vec<i32> =
                (0..b).map(|i| (pos_at(i) % ratio) as i32).collect();
            {
                let mut row_v = sd.row_per_b.slice_view_mut(0, b as usize);
                row_v.copy_from_host_async(&row_host, &de.compute)?;
                let mut pm_v = sd.pos_mod_per_b.slice_view_mut(0, b as usize);
                pm_v.copy_from_host_async(&pos_mod_host, &de.compute)?;
            }

            let cs = ls
                .compressor
                .as_mut()
                .ok_or_else(|| eyre!("L{layer}: missing compressor state"))?;
            // Per-boundary snapshot scratch slot size for THIS compressor.
            let coff_main: u32 = if ratio == 4 { 2 } else { 1 };
            let snap_elems = (coff_main * ratio * comp_width) as usize;
            let n_comp_start = cs.n_comp;
            let mut pos_per_boundary_host: Vec<i32> = Vec::new();

            // ---- FAST PATH (2026-09-12) --------------------------------
            // The serial segment loop below exists because it mirrors ds4's
            // DECODE compressor, which sees one position at a time and keeps
            // a ring buffer + shuffle. In prefill every position of the chunk
            // is already projected into kv_cur/sc_cur, so each boundary's
            // snapshot is a pure gather over those rows and the ring is
            // unnecessary. Measured: the loop was ~384 launches/layer at
            // ratio=4 (≈6.4 ms of the stage's 7.8 ms) against ~30 µs of real
            // work. The gather is bit-for-bit identical — same f32 values,
            // same APE term — so oracles must not move.
            //
            // Preconditions, all satisfied for the production chunk sizes
            // (b=512, ratio ∈ {4,128}); otherwise we fall through to the
            // serial loop unchanged:
            //   - pos0 % ratio == 0  (else carried-in rows for k>0 do not
            //     line up with the previous chunk's state layout)
            //   - at least one boundary fires in this chunk
            let rows_state = coff_main * ratio;
            // DEEPSTRIX_COMP_GATHER=0 forces the serial loop (A/B + rollback).
            let gather_enabled = std::env::var("DEEPSTRIX_COMP_GATHER")
                .map(|v| v != "0")
                .unwrap_or(true);
            // b % ratio == 0 is REQUIRED, not cosmetic: the end-of-chunk state
            // write assumes the chunk ends exactly on a group boundary, so the
            // last complete group is kv_cur[b-ratio .. b) landing in rows
            // 0..ratio-1. With a ragged tail (e.g. b=7, ratio=4) that slice is
            // the wrong positions AND the trailing partial group is lost.
            let fast_ok =
                gather_enabled && arena.is_none() && pos0 % ratio == 0 && b % ratio == 0 && b >= ratio;
            if let Some((t, _)) = arena {
                // ARENA: every row is one position of its own stream, so all b
                // state writes go to distinct accumulator blocks (one launch,
                // `state_base_per`), and the rows whose boundary fires
                // (`fire_rows`, prepared by `KvArena::tables`) are pooled
                // straight out of their blocks below (`fire_state_idx`) — no
                // snapshot, no shuffle (V4.1 ratios 1 and 2 have none; the
                // arena refuses ratio 4). The decode twin is forward_layer.rs
                // `comp_fires_boundary`, one row at a time.
                let si = arena_store.expect("checked at entry");
                let ts = &t.stores[si];
                let row_v = sd.row_per_b.slice_view(0, b as usize);
                let pm_v = sd.pos_mod_per_b.slice_view(0, b as usize);
                de.compressor_state_write.launch_batched_rows(
                    &de.compute,
                    &mut cs.state_kv,
                    &mut cs.state_score,
                    &sd.kv_cur,
                    &sd.sc_cur,
                    &cw.ape.buffer,
                    &row_v,
                    &pm_v,
                    comp_width,
                    b,
                    arena_state_base,
                )?;
                pos_per_boundary_host.extend_from_slice(&ts.fire_comp_pos);
                for k in 0..b {
                    let pos = pos_at(k);
                    let fires = (pos + 1) % ratio == 0;
                    let after = ts.n_comp_per[k as usize] as u32 + u32::from(fires);
                    // The store is 1:1 with boundaries from position 0, so the
                    // stream's counter must agree with the positional formula
                    // (the per-row form of the `V41_COMP_POSITIONAL` audit).
                    if after != (pos + 1) / ratio {
                        return Err(eyre!(
                            "L{layer}: arena row {k} at pos {pos} has n_comp {} (+{}) but the \
                             store should hold {} rows up to it",
                            ts.n_comp_per[k as usize],
                            u32::from(fires),
                            (pos + 1) / ratio
                        ));
                    }
                    n_comp_after.push(after);
                }
                if ts.fire_rows.len() != ts.fire_comp_pos.len()
                    || ts.fire_rows.iter().any(|&r| r < 0 || r as u32 >= b)
                {
                    return Err(eyre!("L{layer}: arena fire table malformed"));
                }
            } else if fast_ok {
                let n_bnd = b / ratio;
                for k in 0..n_bnd {
                    pos_per_boundary_host.push((pos0 + k * ratio) as i32);
                }
                cs.n_comp += n_bnd;
                for k in 0..b {
                    n_comp_after.push(n_comp_start + (k + 1) / ratio);
                }
                {
                    let mut pv = sd
                        .comp_pos_per_boundary
                        .slice_view_mut(0, n_bnd as usize);
                    pv.copy_from_host_async(&pos_per_boundary_host, &de.compute)?;
                }
                de.compressor_state_snapshot.launch_gather(
                    &de.compute,
                    &mut sd.comp_state_kv_snapshots,
                    &mut sd.comp_state_score_snapshots,
                    &sd.kv_cur,
                    &sd.sc_cur,
                    &cs.state_kv,
                    &cs.state_score,
                    &cw.ape.buffer,
                    &sd.comp_pos_per_boundary,
                    comp_width,
                    ratio,
                    rows_state,
                    pos0 as i32,
                    b,
                    n_bnd,
                )?;
                // End-of-chunk state for the NEXT chunk. After the last
                // boundary the serial path leaves rows 0..ratio-1 holding the
                // final complete group (shuffled down for ratio==4, written
                // in place for ratio==128); rows above that are dead until
                // the next group overwrites them. Reproduce with one batched
                // state_write over those `ratio` positions.
                let l_last = b - ratio;               // offset into kv_cur
                let comp_stride = comp_width as usize;
                let kv_seg = sd
                    .kv_cur
                    .slice_view((l_last as usize) * comp_stride, (ratio as usize) * comp_stride);
                let sc_seg = sd
                    .sc_cur
                    .slice_view((l_last as usize) * comp_stride, (ratio as usize) * comp_stride);
                // Destination rows are 0..ratio-1 (post-shuffle slots), NOT
                // row_per_b's 4+pos_mod. With pos0 % ratio == 0 and
                // b % ratio == 0, pos_mod_per_b[0..ratio] == [0..ratio-1],
                // which is exactly both the row list and the APE index list.
                let pm_seg = sd.pos_mod_per_b.slice_view(0, ratio as usize);
                let row_seg = sd.pos_mod_per_b.slice_view(0, ratio as usize);
                de.compressor_state_write.launch_batched(
                    &de.compute,
                    &mut cs.state_kv,
                    &mut cs.state_score,
                    &kv_seg,
                    &sc_seg,
                    &cw.ape.buffer,
                    &row_seg,
                    &pm_seg,
                    comp_width,
                    ratio,
                )?;
            } else {
                let mut i: u32 = 0;
                while i < b {
                    let pos_mod_now = (pos0 + i) % ratio;
                    let seg_len = std::cmp::min(ratio - pos_mod_now, b - i);
                    let seg_end = i + seg_len;

                    // Batched state_write for this segment.
                    let comp_stride = comp_width as usize;
                    let kv_seg = sd.kv_cur.slice_view(
                        (i as usize) * comp_stride,
                        (seg_len as usize) * comp_stride,
                    );
                    let sc_seg = sd.sc_cur.slice_view(
                        (i as usize) * comp_stride,
                        (seg_len as usize) * comp_stride,
                    );
                    let row_seg = sd.row_per_b.slice_view(i as usize, seg_len as usize);
                    let pm_seg = sd.pos_mod_per_b.slice_view(i as usize, seg_len as usize);
                    de.compressor_state_write.launch_batched(
                        &de.compute,
                        &mut cs.state_kv,
                        &mut cs.state_score,
                        &kv_seg,
                        &sc_seg,
                        &cw.ape.buffer,
                        &row_seg,
                        &pm_seg,
                        comp_width,
                        seg_len,
                    )?;

                    // Boundary fire? Snapshot state for batched post-pass at
                    // end-of-chunk. Shuffle still runs immediately so the
                    // NEXT segment's state_write sees correct "old" rows.
                    let comp_fires = (pos0 + seg_end) % ratio == 0;
                    if comp_fires {
                        let k = pos_per_boundary_host.len();
                        let snap_off = k * snap_elems;
                        let mut snap_kv = sd
                            .comp_state_kv_snapshots
                            .slice_view_mut(snap_off, snap_elems);
                        let mut snap_sc = sd
                            .comp_state_score_snapshots
                            .slice_view_mut(snap_off, snap_elems);
                        de.compressor_state_snapshot.launch(
                            &de.compute,
                            &mut snap_kv,
                            &mut snap_sc,
                            &cs.state_kv,
                            &cs.state_score,
                            snap_elems as u32,
                        )?;
                        if ratio == 4 {
                            de.compressor_shuffle.launch(
                                &de.compute,
                                &mut cs.state_kv,
                                &mut cs.state_score,
                                comp_width,
                            )?;
                        }
                        pos_per_boundary_host.push((pos0 + seg_end - ratio) as i32);
                        cs.n_comp += 1;
                    }

                    // n_comp_after semantics: for token at pos = pos0+k, value
                    // reflects cs.n_comp AFTER processing that position. If the
                    // boundary fires at the end of this segment, only the LAST
                    // position sees the post-fire n_comp; earlier positions see pre-fire.
                    let post_fire = cs.n_comp;
                    let pre_fire = if comp_fires { post_fire - 1 } else { post_fire };
                    for k in i..seg_end {
                        let snap = if comp_fires && k == seg_end - 1 {
                            post_fire
                        } else {
                            pre_fire
                        };
                        n_comp_after.push(snap);
                    }
                    i = seg_end;
                }
            }

            // Batched per-boundary stages: pool → rms_w → rope → fp8 →
            // f16rt → comp_kv_append. Replaces what used to be 6 launches
            // per boundary × ~128 boundaries × 21+ layers in the per-token
            // serial loop (~200ms of launch overhead per chunk).
            let n_boundaries = pos_per_boundary_host.len() as u32;
            if n_boundaries > 0 {
                if arena.is_some() {
                    // Pool each firing row's stream block in place (the state
                    // write above already holds its last position).
                    de.compressor_pool.launch_batched_rows(
                        &de.compute,
                        &mut sd.comp_pooled_batched,
                        &cs.state_kv,
                        &cs.state_score,
                        N_HEAD_DIM,
                        ratio,
                        n_boundaries,
                        arena_fire_state_idx,
                    )?;
                } else {
                de.compressor_pool.launch_batched(
                    &de.compute,
                    &mut sd.comp_pooled_batched,
                    &sd.comp_state_kv_snapshots,
                    &sd.comp_state_score_snapshots,
                    N_HEAD_DIM,
                    ratio,
                    n_boundaries,
                )?;
                }
                de.rms_w.launch_weighted_batched(
                    &de.compute,
                    &mut sd.comp_rows_batched,
                    &sd.comp_pooled_batched,
                    &cw.norm,
                    N_HEAD_DIM,
                    RMS_EPS,
                    n_boundaries,
                )?;
                {
                    let mut pv = sd
                        .comp_pos_per_boundary
                        .slice_view_mut(0, n_boundaries as usize);
                    pv.copy_from_host_async(&pos_per_boundary_host, &de.compute)?;
                }
                // V4.1 CSA2 index-K (S1a, PREFILL twin of forward_layer.rs).
                // MUST precede the rope below: that rotates `comp_rows_batched` IN
                // PLACE, and the reference reads the RoPE-free latent. Batched
                // counterparts of the decode chain, same positions.
                // Inert until the sparse gate flips — nothing reads `index_k`.
                if index_k_enabled() {
                    if let Some(iw) = dlw.indexer.as_ref() {
                        if let (Some(wk), Some(knorm), Some(ik)) =
                            (iw.attn_k.as_ref(), iw.k_norm.as_ref(), cs.index_k.as_mut())
                        {
                            let _t = de.events.stage("k.comp_b.index_k", &de.compute)?;
                            if prefill_f32_matvec(n_boundaries) {
                            de.f16.matvec_batched(
                                    &de.compute,
                                    &mut sd.index_k_rows_batched,
                                    &wk.buffer,
                                    &sd.comp_rows_batched,
                                    N_INDEXER_HEAD_DIM,
                                    N_HEAD_DIM,
                                    n_boundaries,
                                )?;
                            } else {
                            de.f16.gemm_batched_wmma(
                                    &de.compute,
                                    &mut sd.index_k_rows_batched,
                                    &wk.buffer,
                                    &sd.comp_rows_batched,
                                    N_INDEXER_HEAD_DIM,
                                    N_HEAD_DIM,
                                    n_boundaries,
                                )?;
                            }
                            de.rms_w.launch_weighted_batched(
                                &de.compute,
                                &mut sd.index_k_normed_batched,
                                &sd.index_k_rows_batched,
                                knorm,
                                N_INDEXER_HEAD_DIM,
                                RMS_EPS,
                                n_boundaries,
                            )?;
                            let pos_v = sd
                                .comp_pos_per_boundary
                                .slice_view(0, n_boundaries as usize);
                            de.rope.launch_forward_batched(
                                &de.compute,
                                &mut sd.index_k_normed_batched,
                                &pos_v,
                                1,
                                N_INDEXER_HEAD_DIM,
                                N_ROT,
                                n_boundaries,
                                &dlw.rope_params,
                            )?;
                            // Packs E2M1 + one E8M0 per 32 and appends — the
                            // reference's `fp4_act_quant(k, 32, True)`.
                            de.index_kv_e2m1.launch_append_batched_rows(
                                &de.compute,
                                ik,
                                &sd.index_k_normed_batched,
                                n_comp_start,
                                n_boundaries,
                                arena_fire_dst_row,
                            )?;
                            if arena.is_none() {
                                cs.n_index_comp = n_comp_start + n_boundaries;
                            }
                        }
                    }
                }
                {
                    let pos_v = sd
                        .comp_pos_per_boundary
                        .slice_view(0, n_boundaries as usize);
                    de.rope.launch_forward_batched(
                        &de.compute,
                        &mut sd.comp_rows_batched,
                        &pos_v,
                        1,
                        N_HEAD_DIM,
                        N_ROT,
                        n_boundaries,
                        &dlw.rope_params,
                    )?;
                }
                match &mut cs.comp_kv {
                    CompKvStore::F16(buf) => {
                        if cfg!(feature = "v41") {
                            // V4.1: E2M1 × E4M3/16 fake quant over the whole row
                            // (decode twin: forward_layer.rs `k.compressor_d.fp4kv`).
                            de.fp4kv.launch(&de.compute, &mut sd.comp_rows_batched, n_boundaries, N_HEAD_DIM)?;
                        } else {
                            de.fp8.launch_batched(
                                &de.compute,
                                &mut sd.comp_rows_batched,
                                N_HEAD_DIM - N_ROT,
                                N_HEAD_DIM,
                                n_boundaries,
                            )?;
                            de.f16rt.launch(
                                &de.compute,
                                &mut sd.comp_rows_batched,
                                n_boundaries * N_HEAD_DIM,
                            )?;
                        }
                        de.comp_kv_append.launch_batched_rows(
                            &de.compute,
                            buf,
                            &sd.comp_rows_batched,
                            n_comp_start,
                            N_HEAD_DIM,
                            n_boundaries,
                            arena_fire_dst_row,
                        )?;
                    }
                    // Packed store: quantise + pack + head-shadow write in
                    // one launch (replaces fp8 -> f16rt -> append).
                    CompKvStore::Fp8 { rows, head } => {
                        if arena.is_some() {
                            return Err(eyre!("L{layer}: arena rows need an f16 main store"));
                        }
                        de.comp_kv_fp8.launch_append_batched(
                            &de.compute,
                            rows,
                            head,
                            &sd.comp_rows_batched,
                            n_comp_start,
                            n_boundaries,
                            FP8_KV_HEAD_ROWS as u32,
                        )?;
                    }
                    CompKvStore::E2m1(_) => {
                        return Err(eyre!("L{layer}: main compressor store cannot be E2M1"));
                    }
                }
            }
        } else if ratio > 0 {
            // V4.1 reuse layer: `ls.compressor` is the source's store, already
            // advanced by the source layer this chunk. Row k sees the rows
            // that existed before the chunk plus the boundaries up to k.
            let cs = ls.compressor.as_ref().ok_or_else(|| eyre!(
                "L{layer}: reuse layer without its source's store (wrap the forward in HetModelState::with_kv_source)"
            ))?;
            // The store is 1:1 with compressor boundaries from position 0, so
            // row k's causal count is positional. It must NOT be derived from
            // `cs.n_comp - boundaries_in_this_call`: in the two-lane driver the
            // source layer has already appended the OTHER lane's rows by the
            // time this lane's reuse layer runs, which let lane A attend to
            // lane B's (future) compressed rows.
            if let Some((t, _)) = arena {
                // Per stream: its store count plus the source layer's boundary
                // for this row, checked against the positional formula.
                let ts = &t.stores[arena_store.expect("checked at entry")];
                for k in 0..b {
                    let pos = pos_at(k);
                    let after = ts.n_comp_per[k as usize] as u32 + u32::from((pos + 1) % ratio == 0);
                    if after != (pos + 1) / ratio {
                        return Err(eyre!(
                            "L{layer}: arena row {k} at pos {pos}: source store holds {after} rows, \
                             positional formula wants {}",
                            (pos + 1) / ratio
                        ));
                    }
                    n_comp_after.push(after);
                }
            } else {
            let need = (pos0 + b) / ratio;
            if cs.n_comp < need {
                return Err(eyre!(
                    "L{layer}: source store n_comp {} < {need} boundaries up to pos {}",
                    cs.n_comp,
                    pos0 + b
                ));
            }
            for k in 0..b {
                n_comp_after.push((pos0 + k + 1) / ratio);
            }
            }
        } else {
            for _ in 0..b {
                n_comp_after.push(0);
            }
        }
        // Clamp to the per-token stride of every comp-indexed scratch
        // buffer (indexer_scores, ...). Production can't exceed it (the
        // server refuses a --ctx whose `attn_max_scored_keys` exceeds
        // ATTN_MIXED_MAX_KEYS), but FAKE_PREFILL_POS benches
        // stamping pos at the cap and decoding past it used to overrun the
        // (since removed) CSA bitmap by one word (the 98304 trap).
        // The OWN-compressor branch derives `n_comp_after` from the live
        // `cs.n_comp` counter with a pre-fire/post-fire snapshot. The REUSE
        // branch computes the same quantity positionally and carries an
        // explicit warning that it must NOT be derived from the live counter,
        // because in the two-lane driver the source layer has already advanced
        // it past this lane's rows. Both stores are 1:1 with boundaries from
        // position 0, so the positional formula is the ground truth for both.
        //
        // `V41_COMP_POSITIONAL=1` logs every disagreement and uses the
        // positional value; `=2` logs without overriding.
        if comp_positional() > 0 && ratio > 0 && n_comp_after.len() == b as usize {
            for k in 0..b as usize {
                let want = (pos_at(k as u32) + 1) / ratio;
                if n_comp_after[k] != want {
                    COMP_POS_MISMATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if std::env::var("V41_COMP_POS_DBG").is_ok() {
                        tracing::warn!(
                            layer, row = k, b, pos0, ratio,
                            got = n_comp_after[k], want,
                            "comp n_comp_after disagrees with the positional formula"
                        );
                    }
                    if comp_positional() == 1 {
                        n_comp_after[k] = want;
                    }
                }
            }
        }
        // HARD ERROR, not a clamp — see the matching note in `forward_layer`.
        // Clamping here truncated the indexer's view of a >131K prompt during
        // PREFILL, silently, on the same stale "production can't reach it"
        // premise that `V41_INDEX_K` invalidated.
        if let Some(&over) = n_comp_after.iter().find(|&&v| v > ATTN_MIXED_MAX_KEYS) {
            return Err(eyre!(
                "L{layer}: prefill n_comp_after {over} exceeds the comp-indexed \
                 scratch stride {ATTN_MIXED_MAX_KEYS} — --ctx admission should \
                 have refused this"
            ));
        }

        // ========================================================
        // CSA indexer compressor — second compressor at head_dim=128,
        // only on ratio==4 layers. Same shape as the main compressor's
        // batched-matvec_pair + per-segment-state_write + on-fire-serial-ops
        // pattern, with:
        //   - head_dim = N_INDEXER_HEAD_DIM (128) vs N_HEAD_DIM (512)
        //   - width    = INDEXER_COMP_WIDTH  (256) vs main's (1024)
        //   - NO FP8 step (only valid at head_dim=512 per ds4.c:6702)
        // Scratch (sd.kv_cur, sd.sc_cur) is
        // reused via slice views — main compressor block has already
        // consumed its writes by this point on the same compute stream.
        // row_per_b / pos_mod_per_b reuse — ratio=4 row formula is the
        // same as main's at ratio=4 (`4 + pm`).
        // ics.n_comp tracks identically to cs.n_comp at ratio==4 layers
        // (both fire on the same boundaries) — we don't need a separate
        // n_index_comp_after vector; downstream (Phase 5 mask) can reuse
        // n_comp_after.
        if ratio == 4 {
            let iw = dlw
                .indexer_compressor
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: missing indexer_compressor weights"))?;
            let ics = ls
                .indexer_compressor
                .as_mut()
                .ok_or_else(|| eyre!("L{layer}: missing indexer_compressor state"))?;
            let icw = INDEXER_COMP_WIDTH; // 256
            let ihd = N_INDEXER_HEAD_DIM; // 128

            // Batched matvec_pair across all B → sd.kv_cur[B,icw] +
            // sd.sc_cur[B,icw] (head of buffers, slice view).
            {
                let mut kv_view =
                    sd.kv_cur.slice_view_mut(0, (b as usize) * (icw as usize));
                let mut sc_view =
                    sd.sc_cur.slice_view_mut(0, (b as usize) * (icw as usize));
                de.f16.matvec_pair_batched(
                    &de.compute,
                    &mut kv_view,
                    &mut sc_view,
                    &iw.wkv.buffer,
                    &iw.wgate.buffer,
                    &sd.attn_input_norm,
                    icw,
                    N_EMBD,
                    b,
                )?;
            }

            // Same snapshot+batched pattern as main compressor above.
            // ratio==4, no fp8 (head_dim=128 != 512).
            let coff_idx: u32 = 2; // ratio==4 ⇒ coff = 2
            let snap_elems_idx = (coff_idx * ratio * icw) as usize;
            let n_idx_comp_start = ics.n_comp;
            let mut pos_per_boundary_idx: Vec<i32> = Vec::new();

            let mut i: u32 = 0;
            while i < b {
                let pos_mod_now = (pos0 + i) % ratio;
                let seg_len = std::cmp::min(ratio - pos_mod_now, b - i);
                let seg_end = i + seg_len;

                let comp_stride = icw as usize;
                let kv_seg = sd.kv_cur.slice_view(
                    (i as usize) * comp_stride,
                    (seg_len as usize) * comp_stride,
                );
                let sc_seg = sd.sc_cur.slice_view(
                    (i as usize) * comp_stride,
                    (seg_len as usize) * comp_stride,
                );
                let row_seg = sd.row_per_b.slice_view(i as usize, seg_len as usize);
                let pm_seg = sd.pos_mod_per_b.slice_view(i as usize, seg_len as usize);
                de.compressor_state_write.launch_batched(
                    &de.compute,
                    &mut ics.state_kv,
                    &mut ics.state_score,
                    &kv_seg,
                    &sc_seg,
                    &iw.ape.buffer,
                    &row_seg,
                    &pm_seg,
                    icw,
                    seg_len,
                )?;

                let comp_fires = (pos0 + seg_end) % ratio == 0;
                if comp_fires {
                    let k = pos_per_boundary_idx.len();
                    let snap_off = k * snap_elems_idx;
                    let mut snap_kv = sd
                        .comp_state_kv_snapshots
                        .slice_view_mut(snap_off, snap_elems_idx);
                    let mut snap_sc = sd
                        .comp_state_score_snapshots
                        .slice_view_mut(snap_off, snap_elems_idx);
                    de.compressor_state_snapshot.launch(
                        &de.compute,
                        &mut snap_kv,
                        &mut snap_sc,
                        &ics.state_kv,
                        &ics.state_score,
                        snap_elems_idx as u32,
                    )?;
                    de.compressor_shuffle.launch(
                        &de.compute,
                        &mut ics.state_kv,
                        &mut ics.state_score,
                        icw,
                    )?;
                    pos_per_boundary_idx.push((pos0 + seg_end - ratio) as i32);
                    ics.n_comp += 1;
                }
                i = seg_end;
            }

            // Batched post-stages for the indexer compressor (no fp8).
            let n_idx_boundaries = pos_per_boundary_idx.len() as u32;
            if n_idx_boundaries > 0 {
                de.compressor_pool.launch_batched(
                    &de.compute,
                    &mut sd.comp_pooled_batched,
                    &sd.comp_state_kv_snapshots,
                    &sd.comp_state_score_snapshots,
                    ihd,
                    ratio,
                    n_idx_boundaries,
                )?;
                de.rms_w.launch_weighted_batched(
                    &de.compute,
                    &mut sd.comp_rows_batched,
                    &sd.comp_pooled_batched,
                    &iw.norm,
                    ihd,
                    RMS_EPS,
                    n_idx_boundaries,
                )?;
                {
                    let mut pv = sd
                        .comp_pos_per_boundary
                        .slice_view_mut(0, n_idx_boundaries as usize);
                    pv.copy_from_host_async(&pos_per_boundary_idx, &de.compute)?;
                }
                {
                    let pos_v = sd
                        .comp_pos_per_boundary
                        .slice_view(0, n_idx_boundaries as usize);
                    de.rope.launch_forward_batched(
                        &de.compute,
                        &mut sd.comp_rows_batched,
                        &pos_v,
                        1,
                        ihd,
                        N_ROT,
                        n_idx_boundaries,
                        &dlw.rope_params,
                    )?;
                }
                // ds4 5bc1e6d: indexer compressor KV rows take the
                // Hadamard128 + FP4 QAT round trip (post-RoPE, pre-append).
                de.indexer_qat.launch(
                    &de.compute,
                    &mut sd.comp_rows_batched,
                    n_idx_boundaries,
                )?;
                match &mut ics.comp_kv {
                    // Packed-E2M1 keys: (code, e) re-derived from the QAT'd
                    // rows, no f16 round trip.
                    CompKvStore::E2m1(rows) => de.index_kv_e2m1.launch_append_batched(
                        &de.compute,
                        rows,
                        &sd.comp_rows_batched,
                        n_idx_comp_start,
                        n_idx_boundaries,
                    )?,
                    other => {
                        de.f16rt.launch(
                            &de.compute,
                            &mut sd.comp_rows_batched,
                            n_idx_boundaries * ihd,
                        )?;
                        de.comp_kv_append.launch_batched(
                            &de.compute,
                            other
                                .f16_mut()
                                .ok_or_else(|| eyre!("L{layer}: indexer compressor store must be f16 or e2m1"))?,
                            &sd.comp_rows_batched,
                            n_idx_comp_start,
                            ihd,
                            n_idx_boundaries,
                        )?;
                    }
                }
            }
        }
        drop(_t_kv_append_comp);
        if ced == CedMode::KvSourceOnly {
            // M7 CED: the decoder's global KV for these rows is in the store;
            // nothing below (window KV, attention, MoE) runs for them.
            return Ok(PreMoeCarry { phase: PreMoePhase::Skipped, layer, ..Default::default() });
        }

        // ========================================================
        // Stage 5: Attention (BATCHED — grid (n_head, B, 1))
        //
        // Causal: each token i attends to KV prefix [0..n_raw_after[i]]
        // and comp_kv prefix [0..n_comp_after[i]]. Per-token prefix
        // lengths live in sd.n_raw_per / sd.n_comp_per device buffers
        // (uploaded fresh per layer from the host snapshots captured in
        // Stage 4).
        // ========================================================
        let n_raw_per_host: Vec<i32> = n_raw_after.iter().map(|&v| v as i32).collect();
        let n_raw_offset_per_host: Vec<i32> =
            n_raw_offset_after.iter().map(|&v| v as i32).collect();
        let n_comp_per_host: Vec<i32> =
            n_comp_after.iter().map(|&v| v as i32).collect();
        let _t_attn = de.events.stage("dgpu.attn_compute", &de.compute)?;
        // Async copies on de.compute so they FIFO with the subsequent
        // attention launch. Avoids the bulk-sync that copy_from_host
        // would impose (~5us each blocks the host AND fences the device).
        {
            let mut nrp_v = sd.n_raw_per.slice_view_mut(0, b as usize);
            nrp_v.copy_from_host_async(&n_raw_per_host, &de.compute)?;
        }
        {
            let mut nrop_v = sd.n_raw_offset_per.slice_view_mut(0, b as usize);
            nrop_v.copy_from_host_async(&n_raw_offset_per_host, &de.compute)?;
        }
        {
            let mut ncp_v = sd.n_comp_per.slice_view_mut(0, b as usize);
            ncp_v.copy_from_host_async(&n_comp_per_host, &de.compute)?;
        }
        let nrp_view = sd.n_raw_per.slice_view(0, b as usize);
        let nrop_view = sd.n_raw_offset_per.slice_view(0, b as usize);
        let ncp_view = sd.n_comp_per.slice_view(0, b as usize);
        if ratio == 0 && swa_via_mixed() {
            // Ratio-0 layers through the batched WMMA mixed pair (see
            // `swa_via_mixed`). `n_comp_per` is already all-zero here (no
            // compressor on a dense layer) and `comp_kv = None`, so the
            // kernels reduce to pure SWA over the raw window.
            let n_total_max = n_raw_after.iter().copied().max().unwrap_or(0);
            if n_total_max > 0 {
                let scores_stride = sd.attn_scores_stride(b, n_total_max)?;
                {
                    let _t = de.events.stage("k.attn.score", &de.compute)?;
                    de.attn_mixed.launch_score_batched_htiled_wmma_f16s(
                        &de.compute,
                        &mut sd.attn_scores,
                        &sd.q_normed,
                        &ls.kv_cache,
                        None,
                        &nrp_view,
                        &nrop_view,
                        &ncp_view,
                        None,
                        N_HEAD,
                        N_HEAD_DIM,
                        n_total_max,
                        b,
                        0,
                        scores_stride,
                    )?;
                }
                {
                    let _t = de.events.stage("k.attn.smwsum", &de.compute)?;
                    de.attn_mixed.launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s(
                        &de.compute,
                        &mut sd.heads,
                        &mut sd.attn_scores,
                        &dlw.attn_sinks,
                        &ls.kv_cache,
                        None,
                        &nrp_view,
                        &nrop_view,
                        &ncp_view,
                        N_HEAD,
                        N_HEAD_DIM,
                        b,
                        0,
                        scores_stride,
                    )?;
                }
            }
        } else if ratio == 0 {
            // Dynamic-LDS stride for attention_swa_batched. Text-only chunks
            // keep SWA_WINDOW here, so the kernel allocates exactly the 1 KiB
            // its old static `scores[128]/weights[128]` arrays used and the
            // occupancy of the text path is unchanged. Image chunks widen it
            // to the chunk's widest window (<= SWA_WINDOW + 384).
            let max_n_kv = if vis.is_some() {
                n_raw_after.iter().copied().max().unwrap_or(0).max(1)
            } else {
                SWA_WINDOW
            };
            if max_n_kv > ATTN_SWA_BATCHED_MAX_KV {
                return Err(eyre!(
                    "L{layer}: raw window {max_n_kv} exceeds attention_swa_batched cap \
                     {ATTN_SWA_BATCHED_MAX_KV}"
                ));
            }
            let _t = de.events.stage("k.attn.swa", &de.compute)?;
            de.attn_swa.launch_batched(
                &de.compute,
                &mut sd.heads,
                &sd.q_normed,
                &ls.kv_cache,
                &dlw.attn_sinks,
                &nrp_view,
                &nrop_view,
                N_HEAD,
                N_HEAD_DIM,
                b,
                max_n_kv,
            )?;
        } else {
            let cs = ls.compressor.as_ref();
            let any_comp = n_comp_after.iter().any(|&v| v > 0);
            let n_total_max = n_raw_after
                .iter()
                .zip(n_comp_after.iter())
                .map(|(&r, &c)| r + c)
                .max()
                .unwrap_or(0);
            // Dense (non-indexer) attention reads rows [0, max n_comp_after)
            // straight from the cache: the f16 buffer, or the FP8 store's
            // f16 head shadow. Only consulted when the indexer does not
            // fire for this chunk (all tokens <= INDEXER_TOP_K comp rows),
            // which is exactly what the shadow covers; `dense_f16` errors
            // otherwise instead of reading past it.
            let comp_kv_buf = if any_comp {
                match cs {
                    Some(c) => {
                        let max_comp = n_comp_after.iter().copied().max().unwrap_or(0);
                        // Mirrors `need_mask` below exactly.
                        let dense_needed = !(ratio == 4
                            && ls.indexer_compressor.is_some()
                            && max_comp > INDEXER_TOP_K);
                        if dense_needed {
                            Some(c.comp_kv.dense_f16(max_comp, &format!("L{layer} prefill"))?)
                        } else {
                            // The indexer fires below; dense buffer unused.
                            None
                        }
                    }
                    None => None,
                }
            } else {
                None
            };
            // Always use the batched split (scores in global, per-row grid,
            // wave-parallel softmax, 16-way ILP wsum). The old monolithic
            // `launch_batched` (LDS scores[2304], used below n_total≤2304) was
            // 12.5× SLOWER at the same B=256 shape — one WG per (head,token)
            // with everything serialized in LDS tanks occupancy — so there is
            // no depth where it wins for batched prefill. It survives only as
            // the correctness oracle in prefill_attention_split_matches_mono.
            // Head-tiled score: one WG per (row, head-group, token) loads the
            // shared MLA latent row once and reuses it across the head group.
            // Score Q·Kᵀ runs as the RDNA4 f16 WMMA GEMM — measured 7.3× over
            // the f32 head-tiled variant on the depth-32k ratio=4 layer.
            // Scores live in DRAM as f16 (the score kernel writes f16,
            // smwsum reads f16). Halves the scores-buffer DRAM round-trip
            // for free; Phase A softmax keeps f32 math. The sd.attn_scores
            // scratch is half-sized accordingly — there is no production
            // f32-scores path. ATTN_FUSED=1 takes the fused-FlashAttention
            // single-kernel path (online softmax, no scores buffer); skips
            // both score and smwsum below.
            // ============================================================
            // CSA indexer mask (per-token, ratio==4 only).
            //
            // For each batch token b at ratio==4 with n_index_comp_per[b] >
            // INDEXER_TOP_K, run matvec(attn_q_b) → RoPE → matvec(proj) →
            // scale → IndexerScore → IndexerTopk → IndexerGather of the
            // selected comp_kv rows into sd.attn_active_comp_kv. The score
            // + smwsum kernels below then read that dense per-token top-K
            // buffer (comp_kv_batch_stride = INDEXER_TOP_K) instead of the
            // full comp_kv. The topk kernel gets `None` for its CSA bitmap
            // (it skips the bitpack) and the score kernel gets `None` for
            // the mask — the gather already handles sparsity.
            //
            // For tokens with n_index_comp_per[b] ≤ INDEXER_TOP_K (early-
            // permit) we set an all-1s mask — leaving the score kernel
            // bit-exact with the pre-mask dense path for that token.
            //
            // For ratio==128 layers and tokens with n_index_comp==0 we
            // pass `None` for the whole-batch mask — the kernel skips
            // the bit test entirely. ratio==128 layers will hit this
            // path because indexer_compressor is None there, so
            // n_index_comp stays 0.
            //
            // The whole pipeline runs serial-per-token on the same
            // compute stream. Cost dominates IndexerScore at long ctx
            // (~70 µs/token at n_index_comp=16K), times B tokens per
            // chunk → maybe ~36 ms added per chunk at B=512, depth 32K.
            // Acceptable per the phase 5 perf budget.
            // V4.1 CSA2 (S1, PREFILL): keys live on the KV-SOURCE's compressor state
            // (`cs.index_k`, packed E2M1), there is no `indexer_compressor`, and the
            // scoring layers are `index_source_layer_ids`, not `ratio == 4`.
            // `n_index_comp_per_b` is filled from `n_comp_after` below, which for V4.1
            // IS the index-K row count (S1a advances them in lockstep) — no change.
            //
            // THIS is the long-context prefill lever: attention is ~30% of prefill at
            // 32K and ~59% at 100K because every query scores the WHOLE compressed
            // store; the indexer makes it 512 rows + a 128 window, flat in context.
            let v41_idx_keys = ls
                .compressor
                .as_ref()
                .and_then(|c| c.index_k.as_ref())
                .filter(|_| index_k_enabled() && is_index_source_layer(layer));
            let v41_force = std::env::var("V41_INDEXER_FORCE").as_deref() == Ok("1");
            // A non-source layer may reuse the last source's selection iff it shares the
            // same compressed store AND this lane actually has one saved.
            let s2_reuse = index_k_enabled()
                && v41_idx_keys.is_none()
                && ls.compressor.is_some()
                && bd.indexer_saved_store >= 0
                && bd.indexer_saved_store
                    == crate::config::index_source_of(layer as usize).unwrap_or(layer as usize) as i32
                && n_comp_after.iter().any(|&v| v > 0);
            let need_mask = if v41_idx_keys.is_some() {
                n_comp_after.iter().any(|&v| v > if v41_force { 0 } else { INDEXER_TOP_K })
            } else {
                ratio == 4
                    && ls.indexer_compressor.is_some()
                    && n_comp_after.iter().any(|&v| v > INDEXER_TOP_K)
            };
            let indexer_fired = if need_mask {
                let _t_ix = de.events.stage("dgpu.prefill_indexer", &de.compute)?;
                let iw = dlw.indexer.as_ref().ok_or_else(|| {
                    eyre!("L{layer}: indexer mask needed but no indexer weights")
                })?;
                // None under v41 — keys come from `v41_idx_keys` instead.
                let ics_opt = ls.indexer_compressor.as_ref();
                let n_words_per_b = ((ATTN_MIXED_MAX_KEYS + 31) / 32) as usize;
                let scale = 1.0f32
                    / ((N_INDEXER_HEAD_DIM as f32) * (N_INDEXER_HEAD as f32)).sqrt();
                let wmma = de.indexer_score_wmma.as_ref().ok_or_else(|| {
                    eyre!("L{layer}: batched prefill indexer requires gfx12 IndexerScoreWmma")
                })?;

                // Batched indexer pipeline. One launch per stage replaces
                // the per-token serial loop (was 8 launches × B × 21L =
                // 86K launches per chunk — the regression root cause).
                // Tokens with n_idx ≤ INDEXER_TOP_K are handled correctly
                // by the batched topk degenerating to all-valid selection.
                let n_idx_max: u32 = n_comp_after.iter().copied().max().unwrap_or(0);
                {
                    let mut v = sd
                        .n_index_comp_per_b
                        .slice_view_mut(0, b as usize);
                    v.copy_from_host_async(&n_comp_after, &de.compute)?;
                }
                // q[B, N_INDEXER_HEAD * N_INDEXER_HEAD_DIM] = attn_q_b @ qr[B].
                // LDS-tiled WMMA GEMM: weight loaded once per (m0,n0) tile
                // and reused across BN=64 output cols. matvec_batched was
                // weight-BW-bound on per-batch rereads at 156 ms/chunk.
                {
                    let _t = de.events.stage("k.indexer.matvec_q", &de.compute)?;
                    if prefill_f32_matvec(b) {
                    de.f16.matvec_batched(
                            &de.compute,
                            &mut sd.indexer_q,
                            &iw.attn_q_b.buffer,
                            &sd.qr_normed,
                            N_INDEXER_HEAD * N_INDEXER_HEAD_DIM,
                            N_LORA_Q,
                            b,
                        )?;
                    } else {
                    de.f16.gemm_batched_wmma(
                            &de.compute,
                            &mut sd.indexer_q,
                            &iw.attn_q_b.buffer,
                            &sd.qr_normed,
                            N_INDEXER_HEAD * N_INDEXER_HEAD_DIM,
                            N_LORA_Q,
                            b,
                        )?;
                    }
                }
                {
                    let _t = de.events.stage("k.indexer.rope", &de.compute)?;
                    let pos_v = bd.pos_per_b.slice_view(0, b as usize);
                    de.rope.launch_forward_batched(
                        &de.compute,
                        &mut sd.indexer_q,
                        &pos_v,
                        N_INDEXER_HEAD,
                        N_INDEXER_HEAD_DIM,
                        N_ROT,
                        b,
                        &dlw.rope_params,
                    )?;
                }
                // ds4 5bc1e6d: Hadamard128 + FP4 QAT round trip on all
                // B × N_INDEXER_HEAD indexer Q rows (post-RoPE, pre-scoring).
                {
                    let _t = de.events.stage("k.indexer.qat", &de.compute)?;
                    // V4.1 does NOT rotate: `fp4_act_quant(q, 32, True)` with no
                    // Hadamard (reference `inference/kernel.py:184`). Using the
                    // rotating kernel yields a plausible but WRONG selection, silently.
                    if v41_idx_keys.is_some() {
                        de.indexer_qat.launch_fp4(&de.compute, &mut sd.indexer_q, b * N_INDEXER_HEAD)?;
                    } else {
                        de.indexer_qat.launch(&de.compute, &mut sd.indexer_q, b * N_INDEXER_HEAD)?;
                    }
                }
                {
                    let _t = de.events.stage("k.indexer.matvec_proj", &de.compute)?;
                    de.f16.matvec_batched(
                        &de.compute,
                        &mut sd.indexer_head_weights,
                        &iw.proj.buffer,
                        &sd.attn_input_norm,
                        N_INDEXER_HEAD,
                        N_EMBD,
                        b,
                    )?;
                }
                {
                    let _t = de.events.stage("k.indexer.scale", &de.compute)?;
                    de.vec_scale.launch(
                        &de.compute,
                        &mut sd.indexer_head_weights,
                        scale,
                        b * N_INDEXER_HEAD,
                    )?;
                }
                {
                    let _t = de.events.stage("k.indexer.score_wmma", &de.compute)?;
                    // Default mw since M52 (2026-06-09): multi-wave Q-staging
                    // kernel, bit-exact vs 1-wave, isolated 32.6→7.45 ms at
                    // the 96K shape, e2e 96K 262→318 tok/s. The 1-wave
                    // kernel stays behind INDEXER_SCORE_VARIANT=sw.
                    static SCORE_MW: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
                        std::env::var("INDEXER_SCORE_VARIANT")
                            .map(|v| v != "sw")
                            .unwrap_or(true)
                    });
                    // 2026-09-08: default "gemm" (8 tokens/WG share each K tile,
                    // 68% of matrix peak vs 15% for mw; bit-exact). mw / sw
                    // stay selectable via INDEXER_SCORE_VARIANT.
                    // Read per call (not LazyLock) so in-process A/B sweeps can flip it.
                    let score_gemm = std::env::var("INDEXER_SCORE_VARIANT").map(|v| v == "gemm").unwrap_or(true);
                    let v41_rows = v41_idx_keys;
                    let ics_e2m1 = ics_opt.and_then(|i| match &i.comp_kv {
                        CompKvStore::E2m1(r) => Some(r),
                        _ => None,
                    });
                    if let Some(rows) = v41_rows.or(ics_e2m1) {
                        // Packed keys: gemm / mw twins expand at their loads;
                        // the 1-wave `sw` kernel has no packed twin.
                        if arena.is_some() {
                            // Per-row key bases: only the mw kernel takes them
                            // (bit-exact with gemm, which has no per-row twin).
                            if v41_rows.is_none() {
                                return Err(eyre!("L{layer}: arena rows need the V4.1 index-K store"));
                            }
                            wmma.launch_batched_mw_e2m1_rows(
                                &de.compute,
                                &mut sd.indexer_scores,
                                &sd.indexer_q,
                                &sd.indexer_head_weights,
                                rows,
                                &sd.n_index_comp_per_b,
                                n_idx_max,
                                ATTN_MIXED_MAX_KEYS,
                                b,
                                arena_keys_base,
                            )?;
                        } else if score_gemm {
                            de.q8k.launch_cast_f16(&de.compute, &mut sd.indexer_q16, &sd.indexer_q,
                                b * N_INDEXER_HEAD * N_INDEXER_HEAD_DIM)?;
                            wmma.launch_batched_gemm_e2m1(
                                &de.compute,
                                &mut sd.indexer_scores,
                                &sd.indexer_q16,
                                &sd.indexer_head_weights,
                                rows,
                                &sd.n_index_comp_per_b,
                                n_idx_max,
                                ATTN_MIXED_MAX_KEYS,
                                b,
                                0,
                            )?;
                        } else if *SCORE_MW {
                            wmma.launch_batched_mw_e2m1(
                                &de.compute,
                                &mut sd.indexer_scores,
                                &sd.indexer_q,
                                &sd.indexer_head_weights,
                                rows,
                                &sd.n_index_comp_per_b,
                                n_idx_max,
                                ATTN_MIXED_MAX_KEYS,
                                b,
                            )?;
                        } else {
                            return Err(eyre!(
                                "L{layer}: INDEXER_SCORE_VARIANT=sw is not available with the packed-E2M1 \
                                 key store (INDEXER_KEYS_E2M1=0 for the f16 store)"
                            ));
                        }
                    } else if arena.is_some() {
                        return Err(eyre!("L{layer}: arena rows need the packed V4.1 index-K store"));
                    } else if score_gemm {
                        de.q8k.launch_cast_f16(&de.compute, &mut sd.indexer_q16, &sd.indexer_q,
                            b * N_INDEXER_HEAD * N_INDEXER_HEAD_DIM)?;
                        wmma.launch_batched_gemm(
                            &de.compute,
                            &mut sd.indexer_scores,
                            &sd.indexer_q16,
                            &sd.indexer_head_weights,
                            ics_opt.expect("f16 indexer key path requires indexer_compressor").comp_kv
                                .f16()
                                .ok_or_else(|| eyre!("L{layer}: indexer compressor store must be f16"))?,
                            &sd.n_index_comp_per_b,
                            n_idx_max,
                            ATTN_MIXED_MAX_KEYS,
                            b,
                            0,
                        )?;
                    } else if *SCORE_MW {
                        wmma.launch_batched_mw(
                            &de.compute,
                            &mut sd.indexer_scores,
                            &sd.indexer_q,
                            &sd.indexer_head_weights,
                            ics_opt.expect("f16 indexer key path requires indexer_compressor").comp_kv
                                .f16()
                                .ok_or_else(|| eyre!("L{layer}: indexer compressor store must be f16"))?,
                            &sd.n_index_comp_per_b,
                            n_idx_max,
                            ATTN_MIXED_MAX_KEYS,
                            b,
                        )?;
                    } else {
                        wmma.launch_batched(
                            &de.compute,
                            &mut sd.indexer_scores,
                            &sd.indexer_q,
                            &sd.indexer_head_weights,
                            ics_opt.expect("f16 indexer key path requires indexer_compressor").comp_kv
                                .f16()
                                .ok_or_else(|| eyre!("L{layer}: indexer compressor store must be f16"))?,
                            &sd.n_index_comp_per_b,
                            n_idx_max,
                            ATTN_MIXED_MAX_KEYS,
                            b,
                        )?;
                    }
                }
                // ARCH_SPEC §1.5 — the same two-level top-k decode runs, PER ROW:
                // layer 20 publishes its best blocks, 24/28/32/36 select only from
                // inside them. Wiring decode alone would make prefill and decode
                // pick different positions on those four layers, which is the
                // prefill/decode divergence class this engine has been bitten by.
                if super::forward_layer::candidate_pool_enabled() {
                    let _t = de.events.stage("k.indexer.candidates", &de.compute)?;
                    let nb_stride = crate::candidate_blocks::CandidateBlocks::n_blocks(
                        ATTN_MIXED_MAX_KEYS,
                    );
                    if layer == crate::config::CANDIDATE_SOURCE_LAYER {
                        de.candidate_blocks.launch_build(
                            &de.compute,
                            &sd.indexer_scores,
                            &mut sd.candidate_block_score,
                            &mut sd.candidate_threshold,
                            sd.n_index_comp_per_b.raw(),
                            ATTN_MIXED_MAX_KEYS,
                            nb_stride,
                            n_idx_max,
                            b,
                        )?;
                    } else if layer > crate::config::CANDIDATE_SOURCE_LAYER {
                        de.candidate_blocks.launch_mask(
                            &de.compute,
                            &mut sd.indexer_scores,
                            &sd.candidate_block_score,
                            &sd.candidate_threshold,
                            sd.n_index_comp_per_b.raw(),
                            ATTN_MIXED_MAX_KEYS,
                            nb_stride,
                            n_idx_max,
                            b,
                        )?;
                    }
                }
                {
                    let _t = de.events.stage("k.indexer.topk_bitonic", &de.compute)?;
                    // INDEXER_TOPK_SELECT=0 disables the threshold fast path.
                    let topk_select = std::env::var("INDEXER_TOPK_SELECT").map(|v| v != "0").unwrap_or(true);
                    de.indexer_topk_bitonic.launch_batched(
                        &de.compute,
                        // S2: write the selection into the LANE's scratch so it survives
                        // to the reuse layers below. `sd` is shared and would alias with
                        // the other lane at the same layer.
                        &mut bd.indexer_sel_saved,
                        None, // no CSA bitmap consumer in batched prefill
                        &mut sd.indexer_topk_scratch,
                        &sd.indexer_scores,
                        &sd.n_index_comp_per_b,
                        n_idx_max,
                        ATTN_MIXED_MAX_KEYS,
                        n_words_per_b as u32,
                        INDEXER_TOP_K,
                        b,
                        if topk_select { Some(&mut sd.indexer_topk_done) } else { None },
                    )?;
                }
                // Gather selected rows from the MAIN compressor's comp_kv
                // (head_dim=512) into a dense per-batch buffer. Attention
                // then reads only top-K rows per token instead of doing
                // dense reads with a -INF mask — 15× DRAM/WMMA savings on
                // both score and smwsum at depth 32K.
                {
                    let _t = de.events.stage("k.indexer.gather", &de.compute)?;
                    let cs_ref = ls
                        .compressor
                        .as_ref()
                        .ok_or_else(|| eyre!("L{layer}: missing compressor state for gather"))?;
                    match &cs_ref.comp_kv {
                        CompKvStore::F16(buf) => de.indexer_gather.launch_batched_rows(
                            &de.compute,
                            &mut sd.attn_active_comp_kv,
                            buf,
                            &bd.indexer_sel_saved,
                            INDEXER_TOP_K,
                            N_HEAD_DIM,
                            b,
                            arena_comp_base,
                        )?,
                        // Packed store: expand FP8 -> f16 on the way into
                        // attn_active_comp_kv; attention is unchanged.
                        CompKvStore::Fp8 { .. } if arena.is_some() => {
                            return Err(eyre!("L{layer}: arena rows need an f16 main store"));
                        }
                        CompKvStore::Fp8 { rows, .. } => de.comp_kv_fp8.launch_gather_batched(
                            &de.compute,
                            &mut sd.attn_active_comp_kv,
                            rows,
                            &bd.indexer_sel_saved,
                            INDEXER_TOP_K,
                            b,
                        )?,
                        CompKvStore::E2m1(_) => {
                            return Err(eyre!("L{layer}: main compressor store cannot be E2M1"));
                        }
                    }
                }
                // Re-upload sparse n_comp_per (= min(actual, INDEXER_TOP_K))
                // so score+smwsum iterate only over the gathered top-K rows.
                {
                    let sparse_n_comp_host: Vec<i32> = n_comp_after
                        .iter()
                        .map(|&v| v.min(INDEXER_TOP_K) as i32)
                        .collect();
                    let mut ncp_v = sd.n_comp_per.slice_view_mut(0, b as usize);
                    ncp_v.copy_from_host_async(&sparse_n_comp_host, &de.compute)?;
                }
                let _ = pos0; // pos consumed via pos_per_b
                // S2: publish which store group this selection belongs to, for the reuse
                // layers that follow in THIS lane.
                if v41_idx_keys.is_some() {
                    // INDEX source, not KV source — see `config::index_source_of`.
                    bd.indexer_saved_store = crate::config::index_source_of(layer as usize)
                        .unwrap_or(layer as usize) as i32;
                }
                _t_ix.end()?;
                true
            } else if s2_reuse {
                // S2 shared selection: this layer is NOT an index source but shares the
                // compressed store with the most recent one, so it reuses that selection.
                // Skip matvec_q / RoPE / fp4 / score / topk entirely — scoring the whole
                // store is the expensive part — and re-run ONLY the gather, which is
                // ~512 rows per token.
                let _t_ix = de.events.stage("dgpu.prefill_indexer_reuse", &de.compute)?;
                // One line per process so a run can be checked for S2 actually engaging
                // without turning on the prefill profile (stage names only appear there).
                {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        tracing::info!(layer, store = bd.indexer_saved_store,
                            "S2 shared selection ACTIVE (reuse layer skipped score+topk)");
                    });
                }
                {
                    let sparse_n_comp_host: Vec<i32> = n_comp_after
                        .iter()
                        .map(|&v| v.min(INDEXER_TOP_K) as i32)
                        .collect();
                    let mut ncp_v = sd.n_comp_per.slice_view_mut(0, b as usize);
                    ncp_v.copy_from_host_async(&sparse_n_comp_host, &de.compute)?;
                }
                let cs_ref = ls.compressor.as_ref().expect("s2_reuse implies a compressor");
                match &cs_ref.comp_kv {
                    CompKvStore::F16(buf) => de.indexer_gather.launch_batched_rows(
                        &de.compute,
                        &mut sd.attn_active_comp_kv,
                        buf,
                        &bd.indexer_sel_saved,
                        INDEXER_TOP_K,
                        N_HEAD_DIM,
                        b,
                        arena_comp_base,
                    )?,
                    _ => return Err(eyre!("L{layer}: S2 reuse needs an f16 main store")),
                }
                _t_ix.end()?;
                true
            } else {
                // An INDEX-SOURCE layer that did NOT fire (its rows are all
                // within INDEXER_TOP_K) must also retract any selection a
                // previous call published for this store: `indexer_saved_store`
                // lives in the lane scratch and outlived the request/chunk/step
                // that wrote it, so the reuse layers below (21-23, 25-27, ...)
                // gathered with a STALE selection whenever a short prompt
                // followed a long one (KNOWN_BUGS #22, 2026-09-20).
                if v41_idx_keys.is_some() {
                    bd.indexer_saved_store = -1;
                }
                false
            };

            // Sparse-attn switch: when indexer fired, attention reads
            // active_comp_kv (per-batch dense top-K) instead of the
            // 8K+-row dense comp_kv. Score+smwsum get a per-batch stride;
            // the bitmask path becomes unused. n_total_max collapses to
            // raw_window + INDEXER_TOP_K.
            let (eff_comp_kv_buf, eff_n_total_max, eff_comp_kv_batch_stride):
                (Option<&DeviceBuffer<u16>>, u32, u32) = if indexer_fired {
                let sparse_max = n_raw_after.iter().zip(n_comp_after.iter())
                    .map(|(&r, &c)| r + c.min(INDEXER_TOP_K))
                    .max().unwrap_or(0);
                (Some(&sd.attn_active_comp_kv), sparse_max, INDEXER_TOP_K)
            } else {
                (comp_kv_buf, n_total_max, 0u32)
            };

            let fused = std::env::var_os("ATTN_FUSED").is_some();
            let f32_scores = super::batch_scratch::use_f32_scores();
            // Per-(row, head) stride of `sd.attn_scores` for THIS launch pair.
            // Derived from the buffer's real capacity at this batch, not from
            // a compile-time constant that assumed compression ratio >= 4:
            // V4.1's ratio-1 decoder layers score `n_raw + n_kv` keys, and the
            // old fixed 3072 stride errored out at 2944 prompt tokens.
            // Score and smwsum MUST be handed the same value.
            let scores_stride = sd.attn_scores_stride(b, eff_n_total_max)?;
            // Dense comp rows straight from the shared store need the row's
            // base; the gathered top-K buffer is already per row.
            let attn_comp_base = if indexer_fired { None } else { arena_comp_base };
            if arena.is_some() && (fused || f32_scores) {
                return Err(eyre!("L{layer}: arena rows need the f16-scores split attention (no ATTN_FUSED / f32 scores)"));
            }
            if !fused {
                let _t = de.events.stage("k.attn.score", &de.compute)?;
                if f32_scores {
                    de.attn_mixed.launch_score_batched_htiled_wmma(
                        &de.compute,
                        &mut sd.attn_scores,
                        &sd.q_normed,
                        &ls.kv_cache,
                        eff_comp_kv_buf,
                        &nrp_view,
                        &nrop_view,
                        &ncp_view,
                        N_HEAD,
                        N_HEAD_DIM,
                        eff_n_total_max,
                        b,
                        scores_stride,
                    )?;
                } else {
                    de.attn_mixed.launch_score_batched_htiled_wmma_f16s_rows(
                        &de.compute,
                        &mut sd.attn_scores,
                        &sd.q_normed,
                        &ls.kv_cache,
                        eff_comp_kv_buf,
                        &nrp_view,
                        &nrop_view,
                        &ncp_view,
                        None, // gather handles sparsity; no -INF mask needed
                        N_HEAD,
                        N_HEAD_DIM,
                        eff_n_total_max,
                        b,
                        eff_comp_kv_batch_stride,
                        scores_stride,
                        attn_comp_base,
                    )?;
                }
            }
            // Head-tiled phase 2: softmax one wave per head + WMMA Phase B
            // W·V via `_ldsv_f16s`. LDS-V staging cooperatively loads each
            // K-tile's 16 V rows once (saves 82.8% of the s_wait_loadcnt
            // stalls the DRAM-V variant ate), and f16 scores halve the
            // Phase A score-DRAM round-trip — together −23% over `_ldsv`
            // at depth 32k, B=256 (13.05 → 10.09 ms p50). The score writer
            // upstream must match the chain (both are f16-only here).
            {
                let _t = de.events.stage(
                    if fused { "k.attn.fused" } else { "k.attn.smwsum" },
                    &de.compute,
                )?;
                if fused {
                    de.attn_mixed.launch_fused_wmma(
                        &de.compute,
                        &mut sd.heads,
                        &sd.q_normed,
                        &dlw.attn_sinks,
                        &ls.kv_cache,
                        eff_comp_kv_buf,
                        &nrp_view,
                        &nrop_view,
                        &ncp_view,
                        N_HEAD,
                        N_HEAD_DIM,
                        eff_n_total_max,
                        b,
                    )?;
                } else if f32_scores {
                    de.attn_mixed.launch_softmax_wsum_batched_htiled_wmma_ldsv(
                        &de.compute,
                        &mut sd.heads,
                        &mut sd.attn_scores,
                        &dlw.attn_sinks,
                        &ls.kv_cache,
                        eff_comp_kv_buf,
                        &nrp_view,
                        &nrop_view,
                        &ncp_view,
                        N_HEAD,
                        N_HEAD_DIM,
                        b,
                        scores_stride,
                    )?;
                } else {
                    de.attn_mixed.launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s_rows(
                        &de.compute,
                        &mut sd.heads,
                        &mut sd.attn_scores,
                        &dlw.attn_sinks,
                        &ls.kv_cache,
                        eff_comp_kv_buf,
                        &nrp_view,
                        &nrop_view,
                        &ncp_view,
                        N_HEAD,
                        N_HEAD_DIM,
                        b,
                        eff_comp_kv_batch_stride,
                        scores_stride,
                        attn_comp_base,
                    )?;
                }
            }
            // `V41_VERIFY_DECODE_ATTN=1`: replay the DECODE attention chain per row,
            // overwriting `sd.heads`.
            //
            // Same reasoning as `V41_VERIFY_DECODE_MOE`: the verify must compute
            // what decode computes, and the two use different kernel families — the
            // batched `*_htiled_wmma*` pair here vs decode's
            // score_b1 -> softmax_only -> wsum_ksplit -> reduce_partials. Decode's
            // are per token, which is fine: looping them keeps the per-LAYER link
            // round trip and pick-readback sync shared across the batch, and that
            // is where speculation's amortisation actually lives.
            //
            // Recomputes rather than replaces so this is one self-contained block,
            // testable with `V41_DSPARK_XCHECK` before anyone pays to delete the
            // superseded work.
            if verify_decode_attn() && arena.is_some() {
                return Err(eyre!("L{layer}: V41_VERIFY_DECODE_ATTN replays one sequence at raw_off 0; not for arena rows"));
            }
            if verify_decode_attn() && (b as usize) <= 16 {
                const K_SPLIT: u32 = 16;
                let qf = crate::config::Q_FLAT as usize;
                let _t_va = de.events.stage("dgpu.verify_decode_attn", &de.compute)?;
                for j in 0..b as usize {
                    let nr = n_raw_after[j];
                    let nc = n_comp_after[j];
                    let q_j = sd.q_normed.slice_view(j * qf, qf);
                    de.attn_mixed.launch_score_b1_htiled_wmma(
                        &de.compute,
                        &mut sd.verify_scores,
                        &q_j,
                        &ls.kv_cache,
                        eff_comp_kv_buf,
                        nr,
                        /*raw_off=*/ 0,
                        nc,
                        N_HEAD,
                        N_HEAD_DIM,
                        nr + nc,
                    )?;
                    de.attn_mixed.launch_softmax_only(
                        &de.compute,
                        &mut sd.verify_scores,
                        &dlw.attn_sinks,
                        &mut sd.verify_inv,
                        N_HEAD,
                        nr,
                        nc,
                    )?;
                    de.attn_mixed.launch_wsum_b1_htiled_ksplit_ldsv(
                        &de.compute,
                        &mut sd.verify_partials,
                        &sd.verify_scores,
                        &ls.kv_cache,
                        eff_comp_kv_buf,
                        N_HEAD,
                        N_HEAD_DIM,
                        nr,
                        nc,
                        K_SPLIT,
                    )?;
                    let mut heads_j = sd.heads.slice_view_mut(j * qf, qf);
                    de.attn_mixed.launch_reduce_partials_apply_inv(
                        &de.compute,
                        &mut heads_j,
                        &sd.verify_partials,
                        &sd.verify_inv,
                        N_HEAD,
                        N_HEAD_DIM,
                        K_SPLIT,
                    )?;
                }
            }
        }
        drop(nrp_view);
        drop(nrop_view);
        drop(ncp_view);
        drop(_t_attn);

        // Post-attention eviction. Cache holds n_raw_during_chunk rows
        // (= n_raw_before + b). Compress back to the SWA invariant:
        //   - if n_raw_during_chunk <= SWA_WINDOW: nothing to do, slots
        //     [0..n_raw_during_chunk) are already the steady state. Update
        //     ls.n_raw = n_raw_during_chunk.
        //   - if n_raw_during_chunk > SWA_WINDOW: copy the LAST SWA_WINDOW
        //     rows down to slots [0..SWA_WINDOW). Set ls.n_raw = SWA_WINDOW.
        //
        // The shift may have source/dest overlap when n_raw_during_chunk is
        // between (SWA_WINDOW, 2*SWA_WINDOW), so route through the kv_ring
        // scratch buffer.
        // SPECULATIVE APPEND (a DSpark verify): the caller took a `KvMark` and
        // will `rollback_kv` after deciding acceptance. The cache is sized
        // SWA_WINDOW + B_MAX precisely so a small batch appends without evicting,
        // so DON'T compact and DON'T reset raw_off here — that physically moved
        // the window and made the mark unaddressable, which capped accept mode at
        // generations shorter than the window. Leave the bytes and raw_off in
        // place; `KvMark::advanced_by` slides the window pointer, and decode's
        // own wrap path compacts later when the append region fills.
        if arena.is_some() {
            // ARENA: the windows and store counters are the KvArena's; the
            // caller advances every row's stream after the step
            // (`KvArena::advance`) and compacts a full region before it.
        } else if speculative_append() {
            // Keep the SWA invariant even before the caller's rollback: the
            // appended speculative rows live past the window in the oversized
            // cache, but `n_raw` is what the NEXT attention reads, and leaving it
            // at n_raw_before + B (e.g. 134) trips
            // "attention_swa: n_kv=134 exceeds kernel cap 128". The caller's
            // `KvMark::advanced_by` recomputes the real window from the mark.
            // SLIDE, don't just clamp. The appended rows sit at
            // [raw_off+n_raw_before, raw_off+n_raw_during_chunk); the live window
            // must be the LAST min(total, SWA_WINDOW) rows ending at the newest
            // one. Clamping n_raw alone left raw_off pointing at the OLDEST 128
            // rows — the newest rows fell outside the window, which is what threw
            // "L8: missing compressor state". Same formula as
            // `KvMark::advanced_by`, so the caller's rollback agrees.
            // DO NOT slide `raw_off` here. This block runs per LANE, and the
            // batched append writes relative to `raw_off` while the whole batch
            // was set up with `raw_off == 0` by `normalize_raw_windows`. Sliding
            // it after lane A meant lane B read shifted by `b_a` -- dropping
            // `b_a` rows of real history and picking up `b_a` slots that are not
            // causally its own. MEASURED: every catastrophic divergence on a row
            // the verify EMITS from (KL up to 8.07 nats against a 0.006
            // baseline) was lane B's first row, at pos >= SWA_WINDOW, minimum
            // exactly 128 -- i.e. from the first slide onward and never before.
            //
            // Sliding it is also unnecessary: the caller's
            // `KvMark::advanced_by` recomputes the real window from the mark at
            // rollback, which is the only place the speculative window has to
            // be right. Carrying the growth in `n_raw` keeps lane B's
            // `causal_end`/`offset` arithmetic consistent with an append that is
            // relative to an unmoved base, and the per-row `n_per` is already
            // `min(causal_end, SWA_WINDOW)` so no row ever asks attention for
            // more keys than the kernel cap.
            ls.n_raw = n_raw_during_chunk;
        } else if n_raw_during_chunk > SWA_WINDOW {
            let src_first_slot = n_raw_during_chunk - SWA_WINDOW;
            let head_dim = N_HEAD_DIM as usize;
            let ring_len = (SWA_WINDOW as usize) * head_dim;
            let src_offset = (src_first_slot as usize) * head_dim;
            // scratch = cache[src_first_slot..src_first_slot+SWA_WINDOW)
            {
                let mut ring_v = sd.kv_ring_scratch.slice_view_mut(0, ring_len);
                let src_v = ls.kv_cache.slice_view(src_offset, ring_len);
                ring_v.copy_from_buffer_async(&src_v, &de.compute)?;
            }
            // cache[0..SWA_WINDOW) = scratch
            {
                let ring_src = sd.kv_ring_scratch.slice_view(0, ring_len);
                let mut dst_v = ls.kv_cache.slice_view_mut(0, ring_len);
                dst_v.copy_from_buffer_async(&ring_src, &de.compute)?;
            }
            ls.n_raw = SWA_WINDOW;
            ls.raw_off = 0;
        } else {
            ls.n_raw = n_raw_during_chunk;
            ls.raw_off = 0;
        }

        // ========================================================
        // Stage 6: Output projection (rope_inv per b, then BATCHED q8)
        // ========================================================
        let _t_out = de.events.stage("dgpu.output_proj", &de.compute)?;
        let cap = self.stage_cap(de, "g.output_proj", layer as usize, b, lane_ptr, cap_ok)?;
        if !cap.skip {
        // SLACK PROBE site `verify_dgpu`: one stall per layer on the dGPU
        // chain. 40 layers x 2 lanes, so the injected total is 80x the tick
        // count -- divide before taking the slope.
        if let Some(ticks) = super::mtp::slack_probe_ticks("verify_dgpu") {
            de.q8.slack_probe_spin(&de.compute, ticks)?;
        }
        {
            let _t = de.events.stage("k.output_proj.rope_inverse", &de.compute)?;
            let pos_v = bd.pos_per_b.slice_view(0, b as usize);
            de.rope.launch_inverse_batched(
                &de.compute,
                &mut sd.heads,
                &pos_v,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                b,
                &dlw.rope_params,
            )?;
        }
        {
            let _t = de.events.stage("k.output_proj.cast_heads_f16", &de.compute)?;
            de.q8k.launch_cast_f16_2d(&de.compute, &mut sd.heads16, &sd.heads,
                b, Q_FLAT, super::batch_scratch::f16_pitch(Q_FLAT))?;
        }
        {
            let _t = de.events.stage("k.output_proj.grouped_matvec", &de.compute)?;
            // LDS-tiled WMMA grouped variant. Per-group shape M=RANK=1024,
            // K=GROUP_DIM=4096 is too small for the legacy per-row GEMV to
            // saturate (1 wave/CU occupancy); the LDS-tiled kernel uses
            // 4 warps × 4 WMMA accs per WG = good occupancy even at small
            // M. Isolated A/B per sub-group at B=512: dp4a 0.42ms vs
            // lds_tiled 0.20ms = 2.1× on each, ~2× on the whole grouped
            // call. Q8_GROUPED_VARIANT=dp4a rolls back.
            // 2026-09-08: default f16x (128x128 f16-activation WMMA GEMM, 52% of
            // matrix peak with the padded heads16 pitch vs 11% for lds_tiled).
            // Dump `heads` BEFORE the variant branch -- inside it the dump only
            // fires on the non-default arm. This is attention's output prior to
            // the output projection, splitting "attention proper" from "the
            // projection kernels" for KNOWN_BUGS #0b.
            if super::engine::subtensor_dump_armed(layer as usize) {
                de.compute.synchronize()?;
                super::engine::maybe_dump_subtensor_f32_view(
                    layer as usize,
                    &format!("pf_heads_p{dump_pos}"),
                    &sd.heads.slice_view(0, Q_FLAT as usize),
                )?;
            }
// Same as `qb_variant`: the "dp4a" arm is
            // `q8_grouped.matvec_grouped_batched`, the batched twin of decode's
            // `matvec_grouped`. Guarded quantize at the branch above writes
            // heads_xq/heads_xscale, so this arm has its input.
            let grp_variant = std::env::var("Q8_GROUPED_VARIANT")
                .unwrap_or_else(|_| if prefill_f32_matvec(b) { "dp4a".into() } else { "f16x".into() });
            if grp_variant != "f16x" {
                de.q8.quantize_input_batched(&de.compute, &mut sd.heads_xq, &mut sd.heads_xscale, &sd.heads, Q_FLAT, b)?;
            }
            if grp_variant == "f16x" {
                de.q8_wmma.gemm_f16x(&de.compute, &mut sd.low, &dlw.attn_output_a.buffer, &sd.heads16,
                    GROUP_DIM, RANK, N_GROUPS, b, super::batch_scratch::f16_pitch(Q_FLAT))?;
            } else if grp_variant == "dp4a" {
                de.q8_grouped.matvec_grouped_batched(
                    &de.compute, &mut sd.low, &dlw.attn_output_a.buffer,
                    &sd.heads_xq, &sd.heads_xscale,
                    GROUP_DIM, RANK, N_GROUPS, b,
                )?;
            } else {
                de.q8_wmma.gemm_lds_tiled_grouped(
                    &de.compute, &mut sd.low, &dlw.attn_output_a.buffer,
                    &sd.heads_xq, &sd.heads_xscale,
                    GROUP_DIM, RANK, N_GROUPS, b,
                )?;
            }
        }
        {
            let _t = de.events.stage("k.output_proj.cast_low_f16", &de.compute)?;
            de.q8k.launch_cast_f16_2d(&de.compute, &mut sd.low16, &sd.low,
                b, OUT_LOW, super::batch_scratch::f16_pitch(OUT_LOW))?;
        }
        {
            let _t = de.events.stage("k.output_proj.matvec_out", &de.compute)?;
            // Same LDS-tiled WMMA the qb path uses. matvec_out shape
            // (M=N_EMBD=4096, K=OUT_LOW=8192) hits the same s_wait_loadcnt
            // throttle on dp4a; LDS-tiled WMMA wins 6.2× at B=512 isolated
            // (8.82 → 1.42 ms). Q8_OUT_VARIANT=dp4a rolls back.
            // Same: the "dp4a" arm is `q8.matvec_batched`, decode's kernel with a
            // row dimension. The guarded quantize below writes low_xq/low_xscale.
            let out_variant = std::env::var("Q8_OUT_VARIANT")
                .unwrap_or_else(|_| if prefill_f32_matvec(b) { "dp4a".into() } else { "f16x".into() });
            if out_variant != "f16x" {
                de.q8.quantize_input_batched(&de.compute, &mut sd.low_xq, &mut sd.low_xscale, &sd.low, OUT_LOW, b)?;
            }
            if out_variant == "f16x" {
                de.q8_wmma.gemm_f16x(&de.compute, &mut sd.attn_out, &dlw.attn_output_b.buffer, &sd.low16,
                    OUT_LOW, N_EMBD, 1, b, super::batch_scratch::f16_pitch(OUT_LOW))?;
            } else if out_variant == "dp4a" {
                de.q8.matvec_batched(
                    &de.compute, &mut sd.attn_out, &dlw.attn_output_b.buffer,
                    &sd.low_xq, &sd.low_xscale, N_EMBD, OUT_LOW, b,
                )?;
            } else {
                de.q8_wmma.gemm_lds_tiled(
                    &de.compute, &mut sd.attn_out, &dlw.attn_output_b.buffer,
                    &sd.low_xq, &sd.low_xscale, N_EMBD, OUT_LOW, b,
                )?;
            }
        }
        }
        cap.end()?;
        drop(_t_out);
        // Bisect within layer 0 (KNOWN_BUGS #0b): attn_out is the attention
        // half's OUTPUT, before hc_post and the whole FFN half. Splits
        // "attention diverges" from "FFN/MoE diverges".
        if super::engine::subtensor_dump_armed(layer as usize) {
            de.compute.synchronize()?;
            super::engine::maybe_dump_subtensor_f32_view(
                layer as usize,
                &format!("pf_attn_out_p{dump_pos}"),
                &sd.attn_out.slice_view(0, N_EMBD as usize),
            )?;
        }

        // ========================================================
        // Stage 7: mhc_post_attn (BATCHED hc_post_from_split)
        // ========================================================
        let _t_mhc_post = de.events.stage("dgpu.mhc_post_attn", &de.compute)?;
        de.hc_post.launch_from_split_batched(
            &de.compute,
            &mut bd.after_attn_hc,
            &sd.attn_out,
            &bd.residual,
            &bd.split,
            N_HC, // n_w (matches single-token path)
            N_EMBD,
            N_HC,
            b,
        )?;
        drop(_t_mhc_post);

        // ========================================================
        // Stage 8: mhc_pre_ffn (BATCHED, same shape as Stage 1)
        // ========================================================
        let _t_mhc_pre_ffn = de.events.stage("dgpu.mhc_pre_ffn", &de.compute)?;
        let cap = self.stage_cap(de, "g.mhc_pre_ffn", layer as usize, b, lane_ptr, cap_ok)?;
        if !cap.skip {
        {
            let _t = de.events.stage("k.mhc_pre_ffn.rms_nw", &de.compute)?;
            de.rms_nw.launch_batched(
                &de.compute,
                &mut sd.flat,
                &bd.after_attn_hc,
                1,
                HC_DIM,
                RMS_EPS,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.mhc_pre_ffn.f16_matvec", &de.compute)?;
            if mhc_narrow_fallback_for(b) {
                de.f16.matvec_narrow_batched(
                    &de.compute,
                    &mut sd.mix,
                    &dlw.hc_ffn_fn.buffer,
                    &sd.flat,
                    HC_MIX_DIM,
                    HC_DIM,
                    b,
                )?;
            } else {
                de.f16.gemm_batched_wmma(
                    &de.compute,
                    &mut sd.mix,
                    &dlw.hc_ffn_fn.buffer,
                    &sd.flat,
                    HC_MIX_DIM,
                    HC_DIM,
                    b,
                )?;
            }
        }
        {
            let _t = de.events.stage("k.mhc_pre_ffn.sinkhorn", &de.compute)?;
            de.hc_sinkhorn.launch_batched(
                &de.compute,
                &mut bd.split,
                &sd.mix,
                &dlw.hc_ffn_scale,
                &dlw.hc_ffn_base,
                N_HC,
                SINKHORN_ITERS,
                SINKHORN_EPS,
                b,
            )?;
        }
        {
            let _t = de.events.stage("k.mhc_pre_ffn.hc_weighted", &de.compute)?;
            let w = if cfg!(feature = "v41") { &bd.hc_pre_carry } else { &bd.split };
            de.hc_weighted.launch_batched(&de.compute, &mut sd.ffn_cur, &bd.after_attn_hc, w, N_EMBD, N_HC, HC_MIX_DIM, b)?;
            if cfg!(feature = "v41") {
                let rows = b as usize * HC_MIX_DIM as usize;
                let cur = bd.split.slice_view(0, rows);
                bd.hc_pre_carry.slice_view_mut(0, rows).copy_from_buffer_async(&cur, &de.compute)?;
            }
        }
        {
            let _t = de.events.stage("k.mhc_pre_ffn.rms_w", &de.compute)?;
            de.rms_w.launch_weighted_batched(
                &de.compute,
                &mut bd.ffn_input_norm,
                &sd.ffn_cur,
                &dlw.ffn_norm,
                N_EMBD,
                RMS_EPS,
                b,
            )?;
        }
        }
        cap.end()?;
        drop(_t_mhc_pre_ffn);

        // ========================================================
        // Stage 9: Router (per-batch wide matvec to match single-token
        // float reduction order; batched matvec_narrow has different
        // accumulation order which makes topk pick different experts
        // when logits are near the threshold. f16.matvec dispatches to
        // wide when n_rows >= 64 (N_EXPERT=256 ≥ 64); a future wide-
        // batched variant could remove the per-token launch overhead.)
        // ========================================================
        let _t_router = de.events.stage("dgpu.router", &de.compute)?;
        let cap = self.stage_cap(de, "g.router_matvec", layer as usize, b, lane_ptr, cap_ok)?;
        if !cap.skip {
        {
            // Gate projection.
            //
            // **The gate MUST be fp32 and MUST match decode bit-for-bit at the
            // batch sizes a DSpark verify uses.** `ARCH_SPEC` says so twice --
            // ":104  s = sqrt(softplus(x_f32 @ W_gate^T)) ... fp32 math" and
            // ":162  Gate math fp32." -- and decode uses `de.f16.matvec`
            // (`forward_layer.rs`), whose wide path `matvec_batched` reproduces
            // bit-identically ("identical single-warp shuffle ... only the launch
            // count drops to 1", f16.rs).
            //
            // `gemm_batched_wmma` does NOT: it casts the activations down to f16
            // (`f16_gemm_wmma.hip`, `vals[e] = (_Float16)x_row[e]`). This gate
            // picks the top 6 of 384 experts, so an f16 rounding near the
            // selection boundary does not perturb a value -- it swaps which
            // experts execute. Prefill/verify therefore selected a DIFFERENT
            // expert set than decode would for the same hidden state, which
            // breaks the property DSpark is built on (the verify must reproduce
            // decode) and is the suspected cause of KNOWN_BUGS #0b, the
            // long-prompt accept degeneracy. The comment that stood here claimed
            // "logits are bit-identical"; that was true of the per-token
            // `matvec` it replaced, not of the WMMA GEMM that replaced it.
            //
            // WMMA is kept only for large prefill chunks, where `matvec_batched`
            // re-reads the weight per batch element and goes weight-BW-bound.
            // `V41_ROUTER_WMMA=1` forces the old path, `=0` forces fp32 always.
            let _t = de.events.stage("k.router.f16_matvec", &de.compute)?;
            let router_wmma = match std::env::var("V41_ROUTER_WMMA").ok().as_deref() {
                Some("1") => true,
                Some("0") => false,
                _ => b > 64,
            };
            if router_wmma {
                de.f16.gemm_batched_wmma(
                    &de.compute,
                    &mut sd.router_logits,
                    &dlw.ffn_gate_inp.buffer,
                    &bd.ffn_input_norm,
                    N_EXPERT,
                    N_EMBD,
                    b,
                )?;
            } else {
                de.f16.matvec_batched(
                    &de.compute,
                    &mut sd.router_logits,
                    &dlw.ffn_gate_inp.buffer,
                    &bd.ffn_input_norm,
                    N_EXPERT,
                    N_EMBD,
                    b,
                )?;
            }
        }
        // Vision-Exp: contiguous runs of image rows (token id >= N_VOCAB,
        // compress-pads included — `Gate.forward`'s `image_mask`). They
        // select experts by top-k(scores + bias_vl) on EVERY layer; text
        // rows keep exp_probs_b / tid2eid, bit-identical to before.
        }
        cap.end()?;
        // Look-ahead routing (arena only, `V41_LOOKAHEAD_PREFETCH=0` off): the
        // NEXT layer's router on THIS layer's router input, read back with the
        // picks below and sent to box 2 as prefetch words. MEASURED 2026-09-22
        // on live agent traffic: 62% (encoder) / 75% (decoder) of next-layer
        // picks predicted; break-even is ~40%.
        let look_next: Option<&DgpuLayerWeights> = match &rows {
            RowLayout::Arena { next_router, .. } if lookahead_prefetch() => next_router.filter(|nl| !nl.is_hash_router),
            _ => None,
        };
        let look_next2: Option<&DgpuLayerWeights> = match &rows {
            RowLayout::Arena { next_router2, .. } if lookahead_prefetch() && lookahead_depth() >= 2 => next_router2.filter(|nl| !nl.is_hash_router),
            _ => None,
        };
        let image_runs = image_spans::image_runs(tokens);
        if !dlw.is_hash_router {
            // Top-k: one block per token in a single launch (B→1 launches).
            // (Image rows are recomputed with bias_vl right below — same
            // stream, FIFO — so this full-batch launch stays as-is.)
            let _t = de.events.stage("k.router.topk", &de.compute)?;
            de.router_topk.launch_batched(
                &de.compute,
                &mut bd.d_selected,
                &mut bd.d_ew,
                &sd.router_logits,
                dlw.router_bias_dev.as_ref(),
                N_EXPERT,
                cs_n_used as u32,
                EXPERT_WEIGHT_SCALE,
                ROUTER_WEIGHT_EPS,
                b,
            )?;
        if let Some(nl) = look_next {
            let _t = de.events.stage("k.router.lookahead", &de.compute)?;
            de.f16.matvec_batched(&de.compute, &mut sd.router_logits, &nl.ffn_gate_inp.buffer, &bd.ffn_input_norm, N_EXPERT, N_EMBD, b)?;
            de.router_topk.launch_batched(&de.compute, &mut sd.look_sel, &mut sd.look_ew, &sd.router_logits, nl.router_bias_dev.as_ref(),
                N_EXPERT, cs_n_used as u32, EXPERT_WEIGHT_SCALE, ROUTER_WEIGHT_EPS, b)?;
        }
        if let Some(nl) = look_next2 {
            let _t = de.events.stage("k.router.lookahead2", &de.compute)?;
            de.f16.matvec_batched(&de.compute, &mut sd.router_logits, &nl.ffn_gate_inp.buffer, &bd.ffn_input_norm, N_EXPERT, N_EMBD, b)?;
            de.router_topk.launch_batched(&de.compute, &mut sd.look_sel2, &mut sd.look_ew2, &sd.router_logits, nl.router_bias_dev.as_ref(),
                N_EXPERT, cs_n_used as u32, EXPERT_WEIGHT_SCALE, ROUTER_WEIGHT_EPS, b)?;
        }
            // KNOWN_BUGS #0b: layer 0's MoE half is where verify diverges from
            // decode while attention is clean. Expert SELECTION is the first
            // thing to rule in or out -- different experts fully explain the
            // observed magnitude. Row 0 only.
            if super::engine::subtensor_dump_armed(layer as usize) {
                de.compute.synchronize()?;
                super::engine::maybe_dump_subtensor_i32(
                    layer as usize,
                    &format!("pf_sel_p{dump_pos}"),
                    &bd.d_selected,
                    cs_n_used,
                )?;
                super::engine::maybe_dump_subtensor_f32_view(
                    layer as usize,
                    &format!("pf_ew_p{dump_pos}"),
                    &bd.d_ew.slice_view(0, cs_n_used),
                )?;
                // The MoE INPUT. Router ids/gates already match, so if this
                // matches too the divergence is in the expert compute itself.
                super::engine::maybe_dump_subtensor_f32_view(
                    layer as usize,
                    &format!("pf_ffn_in_p{dump_pos}"),
                    &bd.ffn_input_norm.slice_view(0, crate::config::N_EMBD as usize),
                )?;
            }
        } else {
            // Hash router: readback all B × N_EXPERT logits, run host
            // select per batch element, upload d_selected + d_ew.
            de.compute.synchronize()?;
            sd.router_logits
                .copy_to_host(&mut sd.router_logits_host)?;
            let tid2eid = dlw
                .tid2eid
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: hash router but no tid2eid"))?;
            let mut all_sel: Vec<i32> = Vec::with_capacity(b as usize * cs_n_used);
            let mut all_ew: Vec<f32> = Vec::with_capacity(b as usize * cs_n_used);
            for i in 0..b as usize {
                if image_spans::is_image_token(tokens[i]) {
                    // Synthetic id: no tid2eid row. Placeholder; overwritten
                    // by the bias_vl top-k launch below.
                    all_sel.extend_from_slice(&[0i32; N_EXPERT_USED]);
                    all_ew.extend_from_slice(&[0f32; N_EXPERT_USED]);
                    continue;
                }
                let logit_slice = &sd.router_logits_host
                    [i * (N_EXPERT as usize)..(i + 1) * (N_EXPERT as usize)];
                let (sel, w) = hash_router_select(tid2eid, tokens[i], logit_slice);
                all_sel.extend_from_slice(&sel);
                all_ew.extend_from_slice(&w);
            }
            // d_selected / d_ew are rows-sized; copy into [0..B*N_USED] view.
            let mut sel_v = bd
                .d_selected
                .slice_view_mut(0, b as usize * cs_n_used);
            sel_v.copy_from_host(&all_sel)?;
            let mut ew_v = bd.d_ew.slice_view_mut(0, b as usize * cs_n_used);
            ew_v.copy_from_host(&all_ew)?;
        }
        if !image_runs.is_empty() {
            let bias_vl = dlw.router_bias_vl_dev.as_ref().ok_or_else(|| {
                eyre!(
                    "L{layer}: image rows in prefill but no bias_vl loaded — put the \
                     sidecar at {} (see het::weights::bias_vl_sidecar_path)",
                    super::weights::BIAS_VL_FILE
                )
            })?;
            let _t = de.events.stage("k.router.topk_vl", &de.compute)?;
            let n_exp = N_EXPERT as usize;
            for &(r0, n) in &image_runs {
                let logits_v = sd.router_logits.slice_view(r0 * n_exp, n * n_exp);
                let mut sel_v = bd.d_selected.slice_view_mut(r0 * cs_n_used, n * cs_n_used);
                let mut ew_v = bd.d_ew.slice_view_mut(r0 * cs_n_used, n * cs_n_used);
                de.router_topk.launch_batched(
                    &de.compute,
                    &mut sel_v,
                    &mut ew_v,
                    &logits_v,
                    Some(bias_vl),
                    N_EXPERT,
                    cs_n_used as u32,
                    EXPERT_WEIGHT_SCALE,
                    ROUTER_WEIGHT_EPS,
                    n as u32,
                )?;
            }
        }
        drop(_t_router);
        let remote_owns_layer = self
            .remote
            .as_ref()
            .and_then(|r| r.lock().ok().map(|c| c.info().owned_count(layer as u32) > 0))
            .unwrap_or(false);
        let remote_split_on = remote_split_active() && remote_owns_layer;
        // DEPENDENCY-GRAPH REORDER (2026-09-22): everything the box-2 request
        // needs from the dGPU is ready right here -- picks, weights, and the
        // activations -- so quantise the activations NOW and mark the router
        // outputs ready. `pre_moe_route` then waits on this EVENT (already past
        // by the time the other lane's chain has been launched) instead of
        // draining the whole compute stream, and the peer push in
        // `pre_moe_prep` waits on the same event.
        if remote_split_on {
            if let Some(xq_dev) = bd.remote_xq_lane.as_mut() {
                self.set_current_cached(self.dgpu.device)?;
                de.q8k.launch(
                    &de.compute,
                    xq_dev,
                    &bd.ffn_input_norm,
                    crate::config::BLOCKS_Q8K_GATE_IN * b,
                )?;
            }
        }
        sev.selected_ready.record(&de.compute)?;

        // M62: accumulate the chunk's picks into the prefill stats bank
        // (one tiny atomicAdd kernel on de.compute, no readback).
        self.record_sel_stats(true, &bd.d_selected, layer as u32, b)?;

        // Stats collection (optional). Copies d_selected to host — sync,
        // fences the device. Don't enable in production prefill.
        if let Some(s) = stats {
            de.compute.synchronize()?;
            let mut sel_host = vec![0i32; (b as usize) * cs_n_used];
            bd.d_selected
                .slice_view(0, (b as usize) * cs_n_used)
                .copy_to_host(&mut sel_host)?;
            s.record_batch(layer as usize, &sel_host, b);
        }

        // Stage 10 (shared expert) is issued HERE by default, or deferred past the
        // remote submit under `V41_PREFILL_PRESUBMIT=1` so its dGPU work lands
        // inside the box-2 RPC window. See `issue_shared_expert_prefill`.
        let defer_shared = prefill_presubmit() && remote_split_active();
        if !defer_shared {
            self.issue_shared_expert_prefill(sd, bd, dlw, b, layer as usize, dump_pos, cap_ok)?;
        }

        // ========================================================
        // Stage 11: iGPU routed MoE (batched).
        //
        // One peer-push of [B × N_EMBD] ffn_input_norm + [B × N_USED]
        // d_selected/d_ew, one batched iGPU MoE call chain (q8k_xq →
        // iq2_fused_swiglu → q8k_mid → q2k_down with by-expert dispatch),
        // one peer-push of [B × N_EMBD] ffn_moe back.
        // ========================================================
        // ---- M7 paged experts -------------------------------------------------
        // Page this layer's experts out of the pool before the MoE reads them.
        //
        // A prefill chunk's routed union at B >> 1 is essentially ALL experts
        // (routing is flat: top-15 is ~28% of picks), so we page the full set
        // rather than read back d_selected. That avoids a host sync on the
        // critical path, and it is a strict SUPERSET of what the router picked,
        // so it cannot under-page (under-paging is what produced garbage before).
        // ---- M7 paged experts -------------------------------------------------
        // Page this layer's FULL expert set into a dense window (slot == expert id).
        // Prefill's group builder sizes group_count/expert_members to N_EXPERT, so a
        // group id must be a raw expert id; the LRU's pool-wide slots would overrun
        // those arrays. With a dense window the pool is a drop-in for the resident
        // buffer and every downstream dispatch is unchanged (no remap, not packed).
        // Two-box split state for this layer, decided once.
        //
        // A single remap cannot encode two splits at the same time: the dGPU
        // hot set wants remote ids NEGATIVE (not its slots) while the iGPU wants
        // them NON-NEGATIVE (skip). V4.1 runs `hot_experts = None` so this never
        // co-occurs today; refuse loudly rather than silently mis-route if it
        // ever does.
        // The het-split builder's cap is a per-token RANK test, not a count of
        // devices. Under the dGPU hot split it deliberately keeps only the top
        // `hot_prefill_cap()` picks for the iGPU. Under the remote split the
        // iGPU must still receive every pick the remote does NOT own, so the cap
        // has to be the full top-k — anything lower silently drops local picks
        // beyond that rank and under-computes the layer with no error.
        // Also pinned whenever the PAGER owns the window: the builder's over-cap
        // branch (`moe_group_builder.hip:116`) falls back to the RAW expert id
        // `g = e` once a pick's rank reaches the cap, which indexes a packed window
        // at the wrong slot. Same shape as the M63 over-cap bug and the
        // hot_prefill_cap()=4 vs N_EXPERT_USED=6 bug: over-cap picks are handed to
        // the other device by raw id, silently.
        let split_cap: u32 = if remote_split_on || pager.is_some() {
            crate::config::N_EXPERT_USED as u32
        } else {
            hot_prefill_cap()
        };
        if remote_split_on && prefill_hot_active(dlw, ilw, bd, sd) {
            return Err(eyre!(
                "L{layer}: dGPU hot split and the two-box remote split are both active; \
                 one remap cannot encode both (see set_remote_exclusion)"
            ));
        }

        // Router picks for this chunk, read back once and shared by the union
        // pager and the remote submit below (both need exactly these ids).
        // Lane picks copied for `V41_GROUP_AUDIT`.
        // Box 2's pick list and weights, masked by the hub's OWN `owns_eff`
        // instead of box 2's advertised HELLO bitmap.
        //
        // `submit`'s mask is the STATIC HELLO bitmap, so a pick the hub
        // reassigned to box 2 without box 2 advertising it is dropped — the
        // comment on `submit_inner`'s mask loop says exactly this. Doing the
        // mask here and submitting unmasked lets the hub choose the split.
        //
        // The empty-slot value is `NO_PICK` (-1), NOT `SENTINEL_EXPERT`
        // (= N_EXPERT = 384): the sentinel is the LOCAL het-split convention,
        // and sending it over the wire gets "expert 384 is not resident here"
        // from the daemon. `ew` must be zeroed in the same slots.
        // Empty = no override, use the old path.
        // Hoisted so the exclusion/audit (which must run AFTER `ensure`) can still
        // see what the masking (now hoisted ABOVE `ensure`) decided.
        // Picks box 1 declined because it does not hold them; box 2 must claim
        // exactly these ON TOP of what it advertises.
        // Hoisted: the ALLOCATOR (below), the EXCLUSION builder and the WEIGHTS
        // VIEW must all agree on the slot space, and the view is chosen ~250
        // lines further down. Deciding this once, here, is what keeps them from
        // diverging -- see the `routed_src` match.
        // KNOWN_BUGS #0b. The sparse view hands the MoE group builder ABSOLUTE
        // pool slots as group ids, and the builder's `n_expert` argument is a
        // BUFFER LIMIT, not a guard: `moe_group_builder.hip:118` drops any id at
        // or above it with no error, and `group_count`/`expert_members` are
        // indexed `g * max_per_expert + pos`. Passing N_EXPERT against a
        // multi-thousand-slot pool therefore dropped EVERY sparse-resident routed
        // expert, while `q2_k_reduce_partials_hetsplit` still claimed its zeroed
        // partial row (it keys on `remap[e] < 0`, which has no bound). Measured
        // cost: verify/decode argmax agreement 43/71 vs 71/71, kld 1.707 vs
        // 0.0066.
        //
        // The fix is to size the group arrays and the bound to the space the
        // remap actually encodes -- `sparse_group_bound()` below, applied at the
        // builder via `ensure_group_bound`/`ensure_group_capacity`. This is a
        // decision that must be made HERE, with the allocator and the weights
        // view: by the time `ensure` has run it has already written absolute
        // slots into the shared remap, and the window view needs the identity
        // map, so there is no safe post-hoc fallback.
        //
        // `sparse_group_ids_in_range()` is the one case widening cannot rescue:
        // work items pack the group id into 16 bits. Unreachable at any pool
        // this box can hold; it falls back to the dense window rather than
        // silently aliasing ids.
        let sparse_resid_layer = (speculative_append() || prefill_unified_pool())
            && !sparse_verify_residency_off()
            && pager
                .as_deref()
                .map(|pg| pg.sparse_group_ids_in_range())
                .unwrap_or(false)
            && !(replay_offload_enabled()
                && remote_split_on
                && (layer as usize) >= crate::config::CED_DECODER_START);
        // Group-id space the MoE by-expert chain must cover for THIS layer.
        // Read here, next to the predicate that chooses the slot space, because
        // the pager is borrowed by `routed_src` by the time the builder runs.
        //   sparse  -> absolute pool slots (`pg.routed`, the whole pool)
        //   window  -> window-relative slots == raw expert ids (`routed_window`)
        let moe_group_bound: u32 = if sparse_resid_layer {
            pager
                .as_deref()
                .map(|pg| pg.sparse_group_bound())
                .unwrap_or(N_EXPERT)
        } else {
            N_EXPERT
        };
        Ok(PreMoeCarry {
            phase: PreMoePhase::Chained,
            layer, b, cs_n_used, cs_n_embd, dump_pos, cap_ok,
            defer_shared, remote_split_on, split_cap, sparse_resid_layer, moe_group_bound,
            drain_before_ensure: true,
            lookahead_hints_ok: true,
            partner_follows: false,
            ..Default::default()
        })
    }

    /// Phase 2 of the pre-MoE lane-layer: wait for the ROUTER (an event, not a
    /// stream drain), read the picks back, split ownership, and SUBMIT the
    /// box-2 request. Touches no iGPU state, so the pipelined driver runs it
    /// for both lanes before either lane's `pre_moe_prep`.
    #[allow(clippy::too_many_arguments)]
    pub fn pre_moe_route(
        &self,
        c: &mut PreMoeCarry,
        bd: &mut BatchDgpuScratch,
        sd: &mut BatchDgpuShared,
        sev: &super::engine::LayerSyncEvents,
        mut pager: Option<&mut super::expert_pager::ExpertPager>,
        rows: &RowLayout<'_>,
    ) -> eyre::Result<()> {
        if !c.advance(PreMoePhase::Chained, PreMoePhase::Routed)? { return Ok(()); }
        let PreMoeCarry { layer, b, cs_n_used, cs_n_embd, remote_split_on, sparse_resid_layer, moe_group_bound, split_cap, lookahead_hints_ok, partner_follows, .. } = *c;
        let _ = (cs_n_embd, split_cap, moe_group_bound);
        let _ = &self.dgpu;
        let look_next: Option<&DgpuLayerWeights> = match &rows {
            RowLayout::Arena { next_router, .. } if lookahead_prefetch() => next_router.filter(|nl| !nl.is_hash_router),
            _ => None,
        };
        let look_next2: Option<&DgpuLayerWeights> = match &rows {
            RowLayout::Arena { next_router2, .. } if lookahead_prefetch() && lookahead_depth() >= 2 => next_router2.filter(|nl| !nl.is_hash_router),
            _ => None,
        };
        let mut sel_host_remote: Vec<i32> = Vec::new();
        let mut sel_host_audit: Vec<i32> = Vec::new();
        let mut sel_for_remote: Vec<i32> = Vec::new();
        let mut owns_eff: Vec<bool> = Vec::new();
        let mut ew_for_remote: Vec<f32> = Vec::new();
        let mut extra_remote = vec![false; N_EXPERT as usize];
        if let Some(pg) = pager.as_deref_mut() {
            // Page the chunk's ACTUAL routed union, not all N_EXPERT. The
            // "union at B >> 1 is essentially everything" argument above holds
            // at large B and is badly false at small B: a B-token chunk touches
            // at most B * N_EXPERT_USED experts, so a 17-token prompt needs 102
            // of 384 and was reading all 384 — ~7.2 GB/layer, ~288 GB over 40
            // layers, to prefill 17 tokens. Costs one de.compute sync per layer
            // to read d_selected back; that is ~1 ms against seconds of reads.
            // At large B the union really is ~everything, and there the dense
            // path's single contiguous H2D per role beats 3*|ids| scattered
            // copies, so keep using it. V41_PAGER_UNION=0 forces dense always.
            if std::env::var("V41_GROUP_AUDIT_VERBOSE").as_deref() == Ok("1") { eprintln!("[pager-stage] L{layer} b={b} union={} unified={} sparse_resid_layer={sparse_resid_layer} bound={moe_group_bound}", super::expert_pager::pager_union_prefill(), prefill_unified_pool()); }
            if super::expert_pager::pager_union_prefill() {
                let _t_pager = LayerHostTimer::start(&LH_PAGER);
                // Per-layer miss histogram. Box 1 pins ENCODER windows only
                // (`prefill_ceiling = CED_DECODER_START + 2`), so a verify's
                // misses should be concentrated at layer >= 20.
                // RACE GUARD (`V41_PAGER_SYNC_IGPU`, default ON).
                //
                // The pager is about to do HOST-side H2D writes that the OTHER
                // lane's still-running iGPU MoE may be reading:
                //   - `remap_dev` is a SINGLE 384-entry buffer, fully rewritten
                //     every layer by `write_window_remap` / `set_remote_exclusion`.
                //     `launch_reduce_partials_hetsplit` reads it to decide which
                //     partial slots this device owns, so a mid-flight change makes
                //     the builder and the reduce disagree.
                //   - the window's expert WEIGHT BYTES, whenever
                //     `window_of(L) == window_of(L+1)` -- with WINDOWS=21 layers
                //     20..39 all share window 20, so 19 consecutive pairs collide.
                //
                // The `de.compute.synchronize()` above transitively drains lane A's
                // previous layer (via `moe_arrived_A`), but NOT lane B's, which sits
                // later in the `ie.compute` FIFO (`post_A(L), pre_A(L+1), post_B(L),
                // pre_B(L+1)`). `copy_from_host` is a blocking `hipMemcpy` that does
                // NOT wait on pending kernels -- this file already records that
                // finding for the repack path.
                //
                // Suspected cause of KNOWN_BUGS #0 (run-to-run nondeterminism): OS
                // page-cache warmth re-times the pager's NVMe reads between server
                // launches, landing the race differently. Set to 0 to measure it.
                // The iGPU drain this guard needs is issued BELOW, right before
                // `ensure` (the first iGPU-side write), not here: the router
                // readback, ownership split and the box-2 submit between here and
                // there need nothing from the iGPU, and issuing the drain first
                // put lane B's whole MoE in front of lane A's submit -- 31-59
                // ms/step of turnaround at 7-8 rows (profile audit 2026-09-21).
                let mc0 = if layer_miss_hist() { pg.counters().prefill_misses } else { 0 };
                let n_sel = (b as usize) * cs_n_used;
                let mut sel_host = vec![0i32; n_sel];
                let _t_sync = LayerHostTimer::start(&LH_SEL_SYNC);
                // Wait on the ROUTER-DONE EVENT, not the stream: the other lane's
                // whole chain may sit behind ours on `de.compute` by now.
                sev.selected_ready.synchronize()?;
                drop(_t_sync);
                let _t_d2h = LayerHostTimer::start(&LH_SEL_D2H);
                bd.d_selected
                    .slice_view(0, n_sel)
                    .copy_to_host(&mut sel_host)?;
                let mut look_host: Vec<i32> = Vec::new();
                if look_next.is_some() {
                    look_host = vec![0i32; n_sel];
                    sd.look_sel.slice_view(0, n_sel).copy_to_host(&mut look_host)?;
                }
                let mut look_host2: Vec<i32> = Vec::new();
                if look_next2.is_some() {
                    look_host2 = vec![0i32; n_sel];
                    sd.look_sel2.slice_view(0, n_sel).copy_to_host(&mut look_host2)?;
                }
                drop(_t_d2h);
                if std::env::var("V41_GROUP_AUDIT_VERBOSE").as_deref() == Ok("1") { eprintln!("[trace] L{layer} A after readback"); }
                if super::expert_pager::pick_trace_on() {
                    for r in 0..b as usize {
                        let row = &sel_host[r * cs_n_used..(r + 1) * cs_n_used];
                        let ids: Vec<String> = row.iter().map(|v| v.to_string()).collect();
                        super::expert_pager::pick_trace(&format!("P {layer} {b} {}", ids.join(" ")));
                    }
                }
                // C3: with the split active, box 2 OWNS half of this layer's
                // experts and computes them itself — so this box must not page
                // them at all. That is the whole point of the split: the working
                // set drops from ~203 experts/layer to ~102, i.e. ~38 GB for all
                // 20 encoder layers, which fits the pool with every layer pinned
                // instead of re-paging each one for every chunk.
                //
                // Correct ONLY because C2 is correct: the local iGPU already
                // skips these picks (exclusion remap) and the remote's partial
                // supplies them at the combine. If either half regresses, this
                // turns a missing expert into silently wrong output rather than
                // an error — so it is gated on the same flag.
                let owns_remote: Option<Vec<bool>> = if remote_split_on
                    && !super::expert_pager::t2_partition()
                {
                    self.remote.as_ref().and_then(|r| {
                        r.lock().ok().map(|c| {
                            (0..N_EXPERT).map(|e| c.owns(layer as u32, e as i32)).collect()
                        })
                    })
                } else {
                    // Under the partition the HELLO bitmap is irrelevant: the pick
                    // loop below assigns every pick by `partition_box2`.
                    None
                };
                // SMALL-B CATCH-ALL: box 1 computes only what it ALREADY HOLDS
                // and hands every MISS to box 2. At B=6 box 1 was otherwise
                // taking ~250 misses per verify and spending ~700 ms reading
                // 4.7 GB off NVMe — 94% of the step — while box 2 sat 83% idle
                // holding its own copy on its own disk.
                // `V41_SMALL_B_CATCHALL_HALF`: 0/unset = all layers,
                // 1 = encoder only, 2 = DECODER only.
                //
                // MEASURED: 93.7% of a B=6 verify's expert misses are in the
                // DECODER half (16-17 encoder vs 235-253 decoder, four verifies).
                // That follows from `prefill_ceiling = CED_DECODER_START + 2` —
                // box 1 pins encoder windows only, so it holds ~nothing on
                // layers 20-39 and misses nearly their whole union.
                //
                // Decoder-only is also where box 2 can take the hand-off: its
                // decoder layers have ~68 slots and a B<=6 per-layer union is
                // <=36. The 170-slot objection that keeps `V41_REPLAY_OFFLOAD`
                // off is a property of the CED replay's 162-wide union at B=128,
                // not of a verify.
                let half = std::env::var("V41_SMALL_B_CATCHALL_HALF")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or(0);
                let in_half = match half {
                    1 => (layer as usize) < crate::config::CED_DECODER_START,
                    2 => (layer as usize) >= crate::config::CED_DECODER_START,
                    _ => true,
                };
                let small_b_catchall = remote_split_on
                    && (b as usize) <= small_b_catchall_max()
                    && super::expert_pager::ExpertPager::t2_catchall()
                    && in_half;
                let mut seen = vec![false; N_EXPERT as usize];
                let mut ids: Vec<u32> = Vec::with_capacity(N_EXPERT as usize);
                let mut skipped_remote = 0usize;
                let note_hot = super::expert_pager::hot_set::enabled() && (b as usize) <= small_b_catchall_max().max(8);
                for &sv in &sel_host {
                    if note_hot && (0..N_EXPERT as i32).contains(&sv) {
                        // Every pick (not just the first per expert): the mass
                        // is what the hot set is ranked by.
                        super::expert_pager::hot_set::note_pick(layer as usize, sv as u32);
                    }
                    if (0..N_EXPERT as i32).contains(&sv) && !seen[sv as usize] {
                        seen[sv as usize] = true;
                        if let Some(o) = owns_remote.as_ref() {
                            if o[sv as usize] {
                                skipped_remote += 1;
                                continue;
                            }
                        }
                        // DETERMINISTIC variant (`V41_SMALL_B_CATCHALL_DET=1`):
                        // hand box 2 EVERY pick, not just the ones box 1 happens
                        // to be missing. The residency-based rule makes the
                        // box1/box2 partition a function of request history, and
                        // f32 addition is not associative, so the verify's output
                        // then depends on what the server served before — the same
                        // reasoning that put decode's catch-all on mode 2
                        // (`t2_catchall_deterministic`). Box 2's per-layer capacity
                        // is ~68 decoder slots against a B<=6 union of <=18, so it
                        // can take the whole batch; the 170-slot objection that
                        // keeps `V41_REPLAY_OFFLOAD` off is a property of the CED
                        // replay's 162-wide union at B=128, not of a verify.
                        if remote_split_on && super::expert_pager::t2_partition() {
                            // Box 2's share: always hers.
                            if super::expert_pager::partition_box2(layer as i32, sv as u32) {
                                extra_remote[sv as usize] = true;
                                continue;
                            }
                            // PARTITIONED PAGING (`V41_B1_PAGE_MISSES=1`): box 1's
                            // hash-half is paged from box 1's OWN disk (concurrent
                            // misses, `V41_PAGER_MISS_PAR`) and computed here, so both
                            // NVMes carry miss traffic (MEASURED 2026-09-21: box 1
                            // 4.2 GB/s + box 2 3.4). Overrides the prefetch hand-off
                            // below and the small-B catch-all for this half; only
                            // sensible with the two-lane step, which hides the read
                            // under box 2's leg.
                            if super::expert_pager::b1_page_misses() {
                                ids.push(sv as u32);
                                continue;
                            }
                            // Box 1's share but NOT resident. `ensure` would read it
                            // from this box's dm-crypt disk SYNCHRONOUSLY, blocking
                            // the layer: measured 285 ms of a 375 ms verify step (42
                            // misses x ~6.8 ms), against 14 ms of box-2 round trip.
                            // Hand it to box 2 for THIS step (she is `--paged`, so she
                            // can serve anything, and her read overlaps our compute)
                            // and queue the async fill, so the share converges onto
                            // box 1 without ever blocking a step.
                            //
                            // MUST live here as well as in the decode path: the DSpark
                            // verify runs through THIS driver, and wiring only the
                            // decode side left the prefetcher firing 64 times a
                            // request instead of ~1300, with box-1 misses unchanged.
                            if super::expert_pager::b1_prefetch()
                                && !pg.is_resident(layer as i32, sv as u32)
                            {
                                pg.prefetch_hint(layer as i32, sv as u32);
                                extra_remote[sv as usize] = true;
                                continue;
                            }
                        }
                        if small_b_catchall
                            && (small_b_catchall_det() || !pg.is_resident(layer as i32, sv as u32))
                        {
                            extra_remote[sv as usize] = true;
                            continue;
                        }
                        ids.push(sv as u32);
                    }
                }
                // `sd.look_sel`/`look_sel2` are SHARED dGPU scratch, so in the
                // pipelined order the other lane's chain has overwritten them by
                // the time we read them (same hazard as `remote_xq`, but these
                // only steer box-2 PREFETCH hints, so the pipelined driver does
                // not emit them; default OFF anyway and measured a loss).
                if lookahead_hints_ok && remote_split_on && (!look_host.is_empty() || !look_host2.is_empty()) {
                    // Rank cut (`V41_LOOKAHEAD_TOPK`): the look-ahead picks come
                    // back in descending selection order and their precision
                    // falls with rank -- decoder 0.97/0.92/0.83/0.72/0.57/0.43
                    // (probe, 2026-09-21). The 5th/6th are mostly wasted reads
                    // that contend with the demand misses on box 2's drives.
                    // L+1 words first (nearest deadline), then L+2.
                    let topk = lookahead_topk().min(cs_n_used);
                    let mut words: Vec<u32> = Vec::with_capacity(look_host.len() + look_host2.len());
                    for (lh, nl) in [(&look_host, layer as i32 + 1), (&look_host2, layer as i32 + 2)] {
                        if lh.is_empty() { continue; }
                        let mut seen_l = vec![false; N_EXPERT as usize];
                        for row in lh.chunks(cs_n_used) {
                            for &sv in &row[..topk] {
                                if !(0..N_EXPERT as i32).contains(&sv) || seen_l[sv as usize] { continue; }
                                seen_l[sv as usize] = true;
                                let box2 = owns_remote.as_ref().is_some_and(|o| o[sv as usize])
                                    || (super::expert_pager::t2_partition() && super::expert_pager::partition_box2(nl, sv as u32));
                                if box2 {
                                    words.push(((nl as u32) << 16) | (sv as u32));
                                }
                            }
                        }
                    }
                    super::remote_experts::push_prefetch_words(&words);
                }
                if skipped_remote > 0 && std::env::var("V41_REMOTE_DBG").is_ok() {
                    eprintln!(
                        "[c3-dbg] L{layer} paging {} experts, skipped {skipped_remote} owned by box 2",
                        ids.len()
                    );
                }
                // T2 CATCH-ALL on the CED REPLAY (`V41_T2_CATCHALL=1`).
                //
                // The replay runs a 128-token window through the 20 DECODER layers,
                // and box 1 has `dense_windows=1` — so it streams a whole layer union
                // (~162 experts x 18.8 MB) into its single window, 20 times over:
                // ~50-60 GB of reads per replay, measured at 11.8 s = 37% of a 6k
                // prefill. Box 2 already holds these layers, so hand it the ENTIRE
                // decoder-layer MoE and page nothing here. Box 2's per-layer capacity
                // must be >= the replay union (162 at B=128) or `ensure_layer` cannot
                // make them all resident at once for the dispatch.
                if std::env::var("V41_GROUP_AUDIT_VERBOSE").as_deref() == Ok("1") { eprintln!("[trace] L{layer} B after pick loop"); }
                let replay_offload = replay_offload_enabled()
                    && remote_split_on
                    && (layer as usize) >= crate::config::CED_DECODER_START;
                // Paired with the exclusion builder below: the SPARSE LRU and
                // `set_remote_exclusion` are incompatible (see there), so the two
                // decisions must be made from ONE predicate.
                let sparse_resid = sparse_resid_layer;
                debug_assert_eq!(sparse_resid, !replay_offload
                    && (speculative_append() || prefill_unified_pool())
                    && !sparse_verify_residency_off()
                    && pg.sparse_group_ids_in_range());
                // MOVED ABOVE `ensure` so box 2 starts while box 1 pages.
                // Pure reorder, not a policy change: the split DECISION
                // (`extra_remote` vs `ids`) is taken in the pick loop further up,
                // before any paging, so nothing here reads residency that `ensure`
                // is about to change. Only the exclusion/audit below genuinely
                // need post-`ensure` state, and they stay there.
                sel_host_remote = sel_host;
                if std::env::var("V41_GROUP_AUDIT_VERBOSE").as_deref() == Ok("1") { eprintln!("[trace] L{layer} C before remote submit: remote_split_on={remote_split_on} remote={} replay_offload={} ids={}", self.remote.is_some(), replay_offload, ids.len()); }
                if group_audit() {
                    sel_host_audit = sel_host_remote.clone();
                }
                // Two-box split: tell the iGPU to skip the experts box 2 owns.
                // Built here, while we still hold `&mut pg`; consumed below via
                // `pg.remap_dev` under the shared borrow.
                if remote_split_on {
                    if let Some(remote) = self.remote.as_ref() {
                        let _t_owns = LayerHostTimer::start(&LH_OWNS);
                        let owns: Vec<bool> = if super::expert_pager::t2_partition() {
                            // Partition: box 2 owns exactly what `extra_remote` says.
                            vec![false; N_EXPERT as usize]
                        } else {
                            let c = remote
                                .lock()
                                .map_err(|_| eyre!("remote expert client mutex poisoned"))?;
                            (0..N_EXPERT).map(|e| c.owns(layer as u32, e as i32)).collect()
                        };
                        drop(_t_owns);
                        let dry = !remote_exclude();
                        if std::env::var("V41_REMOTE_DBG").is_ok() {
                            let n_owned = owns.iter().filter(|&&o| o).count();
                            let picks_remote = sel_host_remote
                                .iter()
                                .filter(|&&e| (0..N_EXPERT as i32).contains(&e) && owns[e as usize])
                                .count();
                            eprintln!(
                                "[excl-dbg] L{layer} owned={n_owned}/{N_EXPERT} \
                                 picks_remote={picks_remote}/{} dry={dry}",
                                sel_host_remote.len(),
                            );
                        }
                        // Under replay offload EVERY expert on this layer is box 2's.
                        // Box 2 must claim the picks box 1 declined for lack of
                        // residency, or they are computed by NOBODY — silently,
                        // since the check below validates this vector, not box
                        // 2's advertised table.
                        owns_eff = (0..N_EXPERT as usize)
                            .map(|e| replay_offload || extra_remote[e] || (!dry && owns[e]))
                            .collect();
                        // Only when the hub actually reassigned something. With
                        // nothing reassigned `owns_eff` IS box 2's advertised
                        // bitmap, so this would be a no-op — but leaving the old
                        // path untouched keeps the default byte-identical.
                        if !dry && extra_remote.iter().any(|&x| x) {
                            sel_for_remote = sel_host_remote
                                .iter()
                                .map(|&e| {
                                    if (0..N_EXPERT as i32).contains(&e) && owns_eff[e as usize] {
                                        e
                                    } else {
                                        super::remote_experts::NO_PICK
                                    }
                                })
                                .collect();
                        }
                }
                // SUBMIT BEFORE PAGING. Box 2 needs only the router's picks and the
                // activations, both ready above; it does NOT need box 1 to have
                // finished `ensure`. Issued here, box 2's ~33 ms of round trip
                // overlaps box 1's paging AND its whole dGPU chain, instead of
                // starting ~160 lines later and being waited on in post_moe.
                if let (true, Some(remote), Some(xq_dev)) =
                    (remote_split_on, self.remote.as_ref(), bd.remote_xq_lane.as_mut())
                {
                    {
                        let _t_remote = LayerHostTimer::start(&LH_REMOTE);
                        let n_sel = (b as usize) * cs_n_used;
                        let xq_bytes = (b as usize)
                            * (crate::config::BLOCKS_Q8K_GATE_IN as usize)
                            * crate::q8_k::BLOCK_Q8_K_BYTES;
                        // The pager just left the iGPU current (it binds its own device
                        // to allocate slots). `de.q8k` is a dGPU module and HIP resolves
                        // module handles against the CURRENT device, so launching
                        // without rebinding fails with hipErrorInvalidHandle.
                        //
                        // This MUST be the uncached bind. `set_current_cached` skips the
                        // real `hipSetDevice` when its cached id already matches, and the
                        // pager switched devices via `Device::set_current` directly —
                        // the engine's cache never saw it and is stale, so the cached
                        // setter is a silent no-op here. Re-sync the cache after, or the
                        // next cached call inherits the same staleness.
                        self.dgpu.device.set_current()?;
                        self.current_device.store(
                            self.dgpu.device.id,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        // Same kernel the iGPU would use, so the bytes match by
                        // construction rather than by agreement.
                        // xq was quantised on the chain right after the router and is
                        // covered by `selected_ready`, already waited on above: D2H only.
                        let _t_rsync = LayerHostTimer::start(&LH_REMOTE_SYNC);
                        let mut xq_host = vec![0u8; xq_bytes];
                        xq_dev.slice_view(0, xq_bytes).copy_to_host(&mut xq_host)?;
                        let mut ew_host = vec![0f32; n_sel];
                        bd.d_ew.slice_view(0, n_sel).copy_to_host(&mut ew_host)?;
                        drop(_t_rsync);
                        // Hash what box 1 SENDS. If xq repeats across layers, the stale
                        // value is box 1's own `ffn_input_norm`, not anything remote.
                        if std::env::var("V41_REMOTE_DBG").is_ok() {
                            let mut hx: u64 = 0xcbf29ce484222325;
                            for &v in xq_host.iter().step_by(37) {
                                hx ^= v as u64;
                                hx = hx.wrapping_mul(0x100000001b3);
                            }
                            let mut hs: u64 = 0xcbf29ce484222325;
                            for &v in sel_host_remote.iter() {
                                hs ^= v as u64;
                                hs = hs.wrapping_mul(0x100000001b3);
                            }
                            eprintln!("[submit-src] L{layer} b={b} xq_hash={hx:016x} sel_hash={hs:016x}");
                        }
                        // `sel_host` above is this chunk's router picks; reuse it.
                        let t_sub = super::perfetto::now_ns();
                        let ticket = remote
                            .lock()
                            .map_err(|_| eyre!("remote expert client mutex poisoned"))?
                            // Last arg is `resp_f32`, NOT "is the split on". This used to
                            // read `remote_split_on` — an unrelated boolean that happens to
                            // be true whenever we get here, so it worked by coincidence and
                            // would have panicked the moment the two diverged. Now explicit.
                            //
                            // f32 is LOAD-BEARING: the consumer below calls
                            // `RemotePartial::f32()`, which asserts `is_f32`. It is not a
                            // free choice, and asking for f16 panics with "partial is f16"
                            // (measured 2026-09-14) until an f16 remote-add path exists.
                            //
                            // Worth building: at B=512 an f32 partial is 512*5120*4 =
                            // 10.49 MB/request, 3760 requests = 39.4 GB over a 724 MB/s
                            // link = ~54 s of a 160 s prefill. f16 halves it. See
                            // docs/v41/PREFILL_100K_PROFILE.md.
                            // MASKED vs UNMASKED. `submit` filters the picks down to what
                            // box 2 ADVERTISED it owns, leaving the rest for box 1. Under
                            // the small-B offload box 1 computes nothing, so a masked
                            // submit leaves every unadvertised pick computed by NOBODY —
                            // silently, because `verify_routing_exactly_once` validates
                            // the hub's own `owns_eff`, not box 2's advertised table. That
                            // halved the drafter's acceptance (E 2.12 -> 1.10) before it
                            // was caught. `submit_unmasked` is what the hub already does
                            // under T2 catch-all: hand box 2 every pick and let it page.
                            .submit_dispatch(
                                !sel_for_remote.is_empty(),
                                layer as u32,
                                b as usize,
                                &xq_host,
                                if sel_for_remote.is_empty() { &sel_host_remote } else { &sel_for_remote },
                                if sel_for_remote.is_empty() {
                                    &ew_host
                                } else {
                                    // Zero the weights of the slots we masked out.
                                    ew_for_remote = ew_host
                                        .iter()
                                        .zip(&sel_for_remote)
                                        .map(|(&w, &e)| if e == super::remote_experts::NO_PICK { 0.0 } else { w })
                                        .collect();
                                    &ew_for_remote
                                },
                                true,
                                // Another request for this SAME layer follows
                                // immediately (the other lane, routed next by the
                                // pipelined driver), so the daemon may hold for it
                                // and run both as one MoE pass. False on the
                                // sequential path, where nothing follows.
                                partner_follows,
                            )?;
                        let t_sub_end = super::perfetto::now_ns();
                        // Stash, don't wait: the local iGPU MoE for this layer is issued
                        // right after this block, and post-MoE collects the reply. The
                        // gap between the `submit` and `wait` slices on the
                        // `remote.expert (host)` track IS the overlap we bought.
                        bd.remote_ticket = ticket;
                        bd.remote_ffn_moe_layer = layer as i32;
                        if let Some(pf) = self.perfetto.as_ref() {
                            if let Ok(pf) = pf.lock() {
                                let _ = pf.emit_host_slice(
                                    pf.remote_uuid,
                                    &format!("submit L{layer} b={b}"),
                                    t_sub, t_sub_end,
                                );
                            }
                        }
                    }
                }
                } // if remote_split_on -- box-2 submit only. Local paging below runs with or
                  // without a remote (it did NOT after f1fbe3f: the whole tail of this block,
                  // ensure included, sat inside the remote branch; KNOWN_BUGS #20/#21).
                c.ids = ids;
                c.replay_offload = replay_offload;
                c.sparse_resid = sparse_resid;
                c.mc0 = mc0;
            }
        }
        c.sel_host_remote = sel_host_remote;
        c.sel_host_audit = sel_host_audit;
        c.owns_eff = owns_eff;
        Ok(())
    }

    /// Phase 3: box-1 residency (`ensure`), exclusion, the deferred shared
    /// expert, and the peer push of this lane's picks/activations to the iGPU.
    /// Stops BEFORE the iGPU MoE dispatch, which reads shared `si` scratch and
    /// therefore must not be interleaved across lanes.
    /// The pipelined driver runs this for both lanes AFTER both `pre_moe_route`s
    /// and BEFORE either `pre_moe_launch`, so no MoE is in flight on the iGPU
    /// while `ensure` evicts (the driver checks that with `moe_done` queries)
    /// and it sets `drain_before_ensure = false`; sequential callers keep the
    /// (conditional) cross-lane drain.
    #[allow(clippy::too_many_arguments)]
    pub fn pre_moe_prep(
        &self,
        c: &mut PreMoeCarry,
        bd: &mut BatchDgpuScratch,
        bi: &mut BatchIgpuScratch,
        sd: &mut BatchDgpuShared,
        si: &mut BatchIgpuShared,
        dlw: &DgpuLayerWeights,
        ilw: &IgpuLayerWeights,
        sev: &super::engine::LayerSyncEvents,
        mut pager: Option<&mut super::expert_pager::ExpertPager>,
    ) -> eyre::Result<()> {
        if !c.advance(PreMoePhase::Routed, PreMoePhase::Prepped)? { return Ok(()); }
        let PreMoeCarry {
            layer, b, cs_n_used, cs_n_embd, dump_pos, cap_ok, defer_shared, remote_split_on, split_cap,
            sparse_resid_layer, moe_group_bound, replay_offload, sparse_resid, mc0, drain_before_ensure, owns_remote_some,
            ref mut ids, ref owns_eff, ref sel_host_remote, ref sel_host_audit, ..
        } = *c;
        let de = &self.dgpu;
        self.set_current_cached(self.dgpu.device)?;
        if let Some(pg) = pager.as_deref_mut() {
            if super::expert_pager::pager_union_prefill() {
                let _t_pager = LayerHostTimer::start(&LH_PAGER);
                if std::env::var("V41_GROUP_AUDIT_VERBOSE").as_deref() == Ok("1") { eprintln!("[trace] L{layer} D at ensure site"); }
                // RACE GUARD (see the note at the top of this block), now paid
                // ONLY when this `ensure` can actually write (2026-09-22).
                //
                // The two hazards were (a) `remap_dev` being one shared 384-entry
                // buffer and (b) evicting a slot whose bytes the OTHER lane's
                // in-flight MoE is reading. (a) IS GONE: `ExpertPager::remap_dev`
                // is `Vec<DeviceBuffer<i32>>`, one per layer (its own doc-comment:
                // "Per LAYER rather than a ring because each layer's MoE is
                // captured as its own HIP graph"), and the two lanes are never on
                // the same layer inside one step. (b) needs an EVICTION, which
                // only happens when some id is not already resident.
                //
                // At 8 rows `pager.misses_per_step` is 0.25-0.95 for the whole
                // step -- i.e. ~99% of the 80 lane-layers page nothing, and each
                // was paying a full cross-lane iGPU drain (31-59 ms/step) for a
                // write that never came. The dense/union paths always rewrite the
                // window, so they keep the drain unconditionally.
                //
                // *** THIS IS THE SECOND-BEST FIX. *** It makes the common case
                // free but leaves the drain on exactly the lane-layers that page,
                // which are the expensive ones, and it still couples the lanes
                // whenever it fires. The PROPER fix is to make eviction incapable
                // of touching the other lane's live slots, so no drain is ever
                // needed: give the pager a `pinned` set -- the other lane's
                // in-flight picks for the layer its MoE is still running -- and
                // have the victim search skip it, which is exactly what box 2's
                // `ExpertShard::pinned` already does for its own queued requests.
                // With that in place delete this block and `V41_PAGER_SYNC_IGPU`
                // outright. Related: the router readback below (`LH_SEL_SYNC`,
                // 25-49 ms/step) is still a full dGPU drain issued the instant the
                // chain is launched; it wants software pipelining -- launch this
                // lane's chain, go service the OTHER lane's reply and chain, then
                // come back for these picks with the wait already over.
                let ensure_may_evict = !replay_offload
                    && (!sparse_resid || ids.iter().any(|&e| !pg.is_resident(layer as i32, e)));
                if drain_before_ensure && ensure_may_evict && std::env::var("V41_PAGER_SYNC_IGPU").as_deref() != Ok("0") {
                    let _t_isync = LayerHostTimer::start(&LH_PAGER_SYNC_IGPU);
                    self.igpu.compute.synchronize()?;
                }
                let _t_ensure = LayerHostTimer::start(&LH_ENSURE);
                let audit_v = std::env::var("V41_GROUP_AUDIT_VERBOSE").as_deref() == Ok("1");
                if audit_v {
                    eprintln!("[ensure-audit] L{layer} b={b} replay_offload={replay_offload} sparse_resid={sparse_resid} ids={} first={:?} owns_remote={} remote_split_on={remote_split_on}",
                        ids.len(), ids.first(), owns_remote_some);
                }
                if replay_offload {
                    ids.clear();
                } else if sparse_resid {
                    // SPECULATIVE VERIFY: use the SPARSE decode-LRU residency, not
                    // prefill's dense windows.
                    //
                    // Prefill needs `slot == expert id` inside a contiguous
                    // N_EXPERT window, so ONE LAYER occupies one window and only
                    // `dense_windows` layers are resident at once. A B=6 verify
                    // picks ~18 DISTINCT experts per layer — it does not need 384
                    // slots — but inheriting the dense pager made all 40 layers
                    // thrash the windows (prefill_misses ~4924/run) even while box
                    // 1's decode LRU sat at 2201 slots with ~0 misses that the
                    // verify could not touch. That is why the T2 catch-all (hand
                    // EVERYTHING to box 2) won, leaving box 1's iGPU idle through
                    // every verify.
                    //
                    // `ensure` maps arbitrary experts to arbitrary pool slots via
                    // `slot_of` and fills the same `remap_dev` the het-split builder
                    // already indexes through, so the MoE kernels are unchanged.
                    // #0b ROOT-CAUSE TEST (V41_SPARSE_REMAP_SYNC=1).
                    //
                    // `remap_dev` is ONE buffer shared by all 40 layers, and
                    // `ensure` re-uploads it per layer. The WINDOW path never
                    // writes remap (ensure_layer_dense/_union leave the identity
                    // self-map -(e)-1, the same for every layer), so a re-upload
                    // is a no-op there. The SPARSE path writes PER-LAYER LRU
                    // slots -- so if the host reaches layer L+1's ensure while
                    // layer L's MoE is still queued, layer L's kernel reads
                    // L+1's remap: right expert ids, wrong weights, no error.
                    // That asymmetry is exactly why only the sparse view breaks.
                    if std::env::var("V41_SPARSE_REMAP_SYNC").as_deref() == Ok("1") {
                        self.igpu.compute.synchronize()?;
                        self.dgpu.compute.synchronize()?;
                    }
                    // A real prefill chunk is a SCAN: confine its misses so it
                    // cannot evict decode's warm set. The verify is not a scan
                    // (it is this conversation's next few tokens), so it keeps
                    // the whole pool.
                    // Real prefill: count it as prefill (it runs through
                    // `ensure`, which otherwise books it as decode), and bound
                    // where its misses may land only if asked to.
                    if !speculative_append() {
                        pg.set_count_as_prefill(true);
                        let scan = prefill_scan_slots();
                        if scan < pg.slots() as usize {
                            let n = pg.slots() as usize;
                            pg.set_scan_window(Some((n - scan, n)));
                        }
                    }
                    let r = pg.ensure(layer as i32, &ids).map(|_| ());
                    pg.set_scan_window(None);
                    pg.set_count_as_prefill(false);
                    r?;
                } else if ids.len() * 10 >= N_EXPERT as usize * 9 {
                    pg.ensure_layer_dense(layer as i32)?;
                } else {
                    pg.ensure_layer_union(layer as i32, &ids)?;
                }
                drop(_t_ensure);
                if audit_v {
                    let probe: Vec<(u32, Option<u32>, i32)> = ids.iter().take(3).map(|&e| (e, pg.resident_slot(layer as i32, e), pg.remap()[e as usize])).collect();
                    eprintln!("[ensure-audit] L{layer} after ensure: (id, slot_of, host remap) = {probe:?}");
                }
                // DIAGNOSTIC (multi-stream harness): `V41_PAGER_SYNC_AFTER_ENSURE=1`
                // drains BOTH devices after paging, before the MoE dispatch reads
                // the pool — tests whether missed experts can be read before they land.
                if std::env::var("V41_PAGER_SYNC_AFTER_ENSURE").is_ok() {
                    self.igpu.device.set_current()?;
                    self.igpu.device.synchronize()?;
                    self.dgpu.device.set_current()?;
                    self.dgpu.device.synchronize()?;
                    self.current_device.store(self.dgpu.device.id, std::sync::atomic::Ordering::Relaxed);
                }
                if layer_miss_hist() {
                    let d = pg.counters().prefill_misses.saturating_sub(mc0);
                    LAYER_MISS[layer as usize].fetch_add(d, std::sync::atomic::Ordering::Relaxed);
                }
                // Prefill READ-AHEAD (`V41_PREFILL_READAHEAD=1`, default off).
                // This layer's `ensure` is done and its MoE is about to be
                // issued, so a hint fired here has that whole layer's compute as
                // lead time -- the most this loop can give without predicting
                // anything. Only NON-RESIDENT owned experts are hinted, so the
                // volume is the miss set, not the layer.
                //
                // It is a page-cache hint, never a pool admission: layer L's MoE
                // may still be queued against the very slots a device-side
                // prefetch of L+1 would evict, which is silent wrong output (the
                // hazard `V41_SPARSE_REMAP_SYNC` exists for). See
                // `ExpertPager::readahead_layer`.
                if super::expert_pager::prefill_readahead() {
                    let ahead = layer as i32 + pg.readahead_depth();
                    // The SAME ownership predicate the target layer's dispatch
                    // will use, or the hint warms bytes box 2 reads.
                    let split = remote_split_on && super::expert_pager::t2_partition();
                    pg.readahead_layer(ahead, |e| {
                        split && super::expert_pager::partition_box2(ahead, e)
                    });
                }
                if remote_split_on && !owns_eff.is_empty() {
                        // The exclusion builder MUST match the allocator above.
                        // `set_remote_exclusion` rebuilds the whole remap from
                        // `window_of(layer)`, which is only valid for the DENSE
                        // window where slot == expert id. Under the sparse LRU an
                        // expert's slot is anywhere in the pool, so that rebuild
                        // stamps 0 ("not ours") over every LRU assignment outside
                        // the layer's nominal window — and since box 2 does not own
                        // those either, the picks are computed by NOBODY. That is
                        // exactly what `verify_routing_exactly_once` caught:
                        //   L18 expert 251: computed by 0 devices, remap[251]=0.
                        // `mark_remote_after_ensure` is the decode-path twin: it
                        // only TOUCHES the remote entries and leaves the LRU slots
                        // that `ensure` just assigned intact.
                        let _t_excl = LayerHostTimer::start(&LH_EXCL);
                        if sparse_resid {
                            pg.mark_remote_after_ensure(|e| owns_eff[e as usize])?;
                        } else {
                            pg.set_remote_exclusion(layer as i32, |e| owns_eff[e as usize])?;
                        }
                        drop(_t_excl);
                        let _t_audit = LayerHostTimer::start(&LH_AUDIT);
                        // Every routed pick must be computed by EXACTLY ONE device.
                        // Decode has had this check since the catch-all landed
                        // (`forward_layer.rs`); PREFILL has had none, and it is the
                        // path about to gain packed windows — where a mis-encoded
                        // remap entry is silent (wrong slot => another expert's
                        // weights, no error). O(picks) via a bitmap, so the B*6
                        // prefill batch is fine.
                        super::forward_layer::verify_routing_exactly_once(
                            layer as i32,
                            &sel_host_remote,
                            pg.remap(),
                            Some(&owns_eff[..]),
                        )?;
                }
            } else {
                if std::env::var("V41_GROUP_AUDIT_VERBOSE").as_deref() == Ok("1") { eprintln!("[pager-stage] L{layer} b={b} -> ensure_layer_dense (union off)"); }
                pg.ensure_layer_dense(layer as i32)?;
            }
        }

        // ---- Two-box split, phase C1: ship this layer's activations to the
        // remote shard. See docs/v41/REMOTE_EXPERTS.md §6.
        //
        // Issued BEFORE the local iGPU chain so the ~74 ms round trip at
        // B=1024 overlaps the ~150 ms of local compute instead of adding to
        // it. Whether that overlap actually happens is exactly what the
        // `remote.submit` / `remote.wait` host tracks exist to show: a thin
        // `remote.wait` beside a fat `igpu.routed_moe` means it is hidden.
        //
        // C1 discards the partial — the local iGPU still computes every
        // expert, so numerics are unchanged and this is purely additive. The
        // exclusion remap and the combine operand (which DO change numerics)
        // come next, and only then may `ensure_layer_union` stop paging the
        // remote-owned half.
        // Gated on `remote_split_on`, not merely on a remote being attached.
        // Phase C1 submitted unconditionally and discarded the reply, which was
        // right while the split was being validated but costs a real ~5.7 MB
        // round trip per layer once it is off — it showed up as a 48.6 -> 42.5
        // tok/s regression in the split-OFF arm of every A/B.

        // DEFERRED Stage 10. With `V41_PREFILL_PRESUBMIT=1` the shared expert is
        // issued HERE, after the remote submit, so its ~173 us of dGPU work lands
        // inside the box-2 RPC window instead of ahead of it. `bd.ffn_shared` is
        // not read until `post_moe`'s `vec_add`, so this is the last legal point.
        //
        // NOT OPTIONAL once `defer_shared` is set: the call site above SKIPS the
        // shared expert entirely in that case, so without this the layer would add
        // a stale `ffn_shared` and be silently wrong.
        if defer_shared {
            self.issue_shared_expert_prefill(sd, bd, dlw, b, layer as usize, dump_pos, cap_ok)?;
        }
        let pager_window;
        let (routed_src, moe_remap, moe_packed): (
            &crate::model_weights::RoutedExpertWeights,
            Option<&v4flash_hip::DeviceBuffer<i32>>,
            bool,
        ) = match pager.as_deref() {
            // A VIEW of this layer's dense window, not the whole pool: the window's
            // slot i holds expert i, so the kernel indexes it exactly like a resident
            // buffer while other layers stay resident in other windows.
            Some(pg) => {
                // THIRD half of the allocator/exclusion pairing: the WEIGHTS VIEW.
                //
                // `ensure` (sparse decode LRU) writes ABSOLUTE pool slots, which
                // are only meaningful against the whole pool -- that is what the
                // decode path hands the dispatch (`pg.routed`, forward_layer.rs).
                // `ensure_layer_union`/`_dense` write WINDOW-RELATIVE slots, valid
                // only against `routed_window(layer)`.
                //
                // Handing an absolute slot to the window view reads
                // `window_base(w) + slot`, and with V41_PAGER_WINDOWS=4 /
                // STRIDE=128 the base is 128/256/384 for most layers while the LRU
                // slots start at `dense_slots()` = 768. That is off the end of a
                // 384-wide view: another expert's weights, no error, plausible
                // output. It is correct only when window_base == 0 for every layer
                // (dense_windows == 1), which is NOT the default.
                let routed_view: &crate::model_weights::RoutedExpertWeights =
                    if sparse_resid_layer {
                        &pg.routed
                    } else {
                        pager_window = pg.routed_window(layer as i32);
                        &pager_window
                    };
                // ALWAYS hand the dispatch the remap when the pager owns the
                // window, not just under the remote split.
                //
                // With a dense window `slot == raw expert id`, so a `None` remap
                // used to be harmless — the PLAIN group builder's raw ids happened
                // to be correct slots. That equivalence is exactly what packed
                // windows break (`moe_group_builder.hip:116` mode 0 takes the group
                // id FROM the remap: `g = (dense >= 0) ? e : (-dense - 1)`), so a
                // layer that fell through to `None` would index a packed window by
                // raw id and read another expert's weights — silently, with no
                // error and plausible-looking output. Passing it unconditionally is
                // a no-op today (mode 0 with an all-local remap is arithmetically
                // identical to the plain builder) and the precondition for packing.
                (routed_view, Some(pg.remap_dev(layer as i32)), false)
            }
            None => (&ilw.routed, ilw.hot_remap.as_ref(), ilw.igpu_packed),
        };
        let gbpe = routed_src.gate_bytes_per_expert as u32;
        let ubpe = routed_src.up_bytes_per_expert as u32;
        let dbpe = routed_src.down_bytes_per_expert as u32;
        let mid_blocks_bytes = (crate::config::BLOCKS_Q8K_DOWN_IN as usize)
            * crate::q8_k::BLOCK_Q8_K_BYTES;
        // Stage 9 router_topk + Stage 10 shared expert wrote bd.d_selected,
        // bd.d_ew, bd.ffn_input_norm on de.compute. We're about to read
        // them from de.xfer. Use the LayerSyncEvents event chain
        // (selected_ready → wait → push → selected_pushed → igpu waits)
        // instead of a host sync, so de.compute is free to keep queuing
        // the next lane's pre-MoE work while xfer/igpu drain this lane.
        self.set_current_cached(self.dgpu.device)?;
        // `selected_ready` was recorded on the chain right after the router.
        de.xfer.wait_event(&sev.selected_ready)?;
        // Single batched peer-push of all B activations + routing.
        let ain_v = bd
            .ffn_input_norm
            .slice_view(0, (b as usize) * cs_n_embd);
        let dsel_v = bd.d_selected.slice_view(0, (b as usize) * cs_n_used);
        let dew_v = bd.d_ew.slice_view(0, (b as usize) * cs_n_used);
        let mut bi_ain = bi
            .ffn_input_norm_recv
            .slice_view_mut(0, (b as usize) * cs_n_embd);
        let mut bi_sel = bi
            .d_selected
            .slice_view_mut(0, (b as usize) * cs_n_used);
        let mut bi_ew = bi.d_ew.slice_view_mut(0, (b as usize) * cs_n_used);
        {
            let _t_peer_ain = de.events.stage("dgpu.peer_push_ffn_input_norm", &de.xfer)?;
            {
                let _t = de.events.stage("k.peer_push.ain", &de.xfer)?;
                peer_push_f32(&ain_v, &mut bi_ain, &de.xfer)?;
            }
            {
                let _t = de.events.stage("k.peer_push.d_selected", &de.xfer)?;
                peer_push_i32(&dsel_v, &mut bi_sel, &de.xfer)?;
            }
            {
                let _t = de.events.stage("k.peer_push.d_ew", &de.xfer)?;
                peer_push_f32(&dew_v, &mut bi_ew, &de.xfer)?;
            }
            drop(_t_peer_ain);
        }
        sev.selected_pushed.record(&de.xfer)?;
        drop(bi_ain);
        drop(bi_sel);
        drop(bi_ew);

        // ========================================================
        // M61 prefill het-split: the dGPU computes its RESIDENT (hot)
        // experts' (b, slot) members itself, on de.compute, fully async
        // with the iGPU MoE below (which skips those slots). Same
        // group-builder machinery as the iGPU path but in DENSE id space,
        // and with a STATIC work-items list — grid.y covers the worst
        // case (n_hot × ceil(rows/HOT_CHUNK)) and the matvec kernels'
        // `member_end <= member_start` guard early-exits empty chunks, so
        // nothing here blocks on a host readback of n_work_items (a
        // de.compute sync would stall the lane pipeline). The partial is
        // added to ffn_moe_recv at ffn_combine (post_moe).
        // ========================================================
        let hot_active = prefill_hot_active(dlw, ilw, bd, sd);
        if hot_active {
            use super::batch_scratch::HOT_CHUNK;
            let hot = dlw.hot_experts.as_ref().unwrap();
            // Intermediates live in the SHARED R1 arena (this is the last
            // R1 user of the lane's pre-MoE call; see BatchDgpuHotScratch);
            // the reduce output is the PER-LANE buffer post_moe reads.
            let hd = sd.hot.as_mut().unwrap();
            let ffn_moe_dgpu = bd.hot_ffn_moe_dgpu.as_mut().unwrap();
            let cap = hot_prefill_cap();
            // Member-list stride / chunk count follow the shared scratch
            // rows (hd.max_per_expert == sd.rows >= b), not B_MAX.
            let max_per_expert = hd.max_per_expert as u32;
            let n_wi = hot.n_hot * hd.chunks_per_expert as u32;
            let _t_hot = de.events.stage("dgpu.hot_moe_prefill", &de.compute)?;
            hd.group_count.fill_zero_async(&de.compute)?;
            de.moe_group_builder.launch_hetsplit(
                &de.compute,
                &mut hd.group_count,
                &mut hd.expert_members,
                &bd.d_selected,
                &hot.remap,
                /*mode=*/ 1,
                cap,
                b,
                cs_n_used as u32,
                hot.n_hot,
                max_per_expert,
            )?;
            de.q8k.launch(
                &de.compute,
                &mut hd.moe_xq,
                &bd.ffn_input_norm,
                crate::config::BLOCKS_Q8K_GATE_IN * b,
            )?;
            // Single-prefill-kernel formats (IQ2_S/IQ2_XS/IQ3_XXS/IQ3_S) go
            // through the dispatcher; IQ2_XXS falls through to its kwide kernel.
            if !super::dispatch::moe_gate_up_chunked(
                de, routed_src.gate.dtype, &de.compute, &mut hd.mid_cat, &hot.gate, &hot.up,
                &hd.moe_xq, &bd.d_ew, &hd.group_count, &hd.expert_members,
                &hd.work_items_static, n_wi, gbpe, ubpe, cs_n_used as u32,
                max_per_expert, HOT_CHUNK as u32, crate::config::SWIGLU_CLAMP_EXP,
                crate::config::N_FF_EXP, crate::config::BLOCKS_Q8K_GATE_IN,
            )? {
                de.iq2.launch_fused_swiglu_kwide(
                    &de.compute,
                    &mut hd.mid_cat,
                    &hot.gate,
                    &hot.up,
                    &hd.moe_xq,
                    &bd.d_ew,
                    &hd.group_count,
                    &hd.expert_members,
                    &hd.work_items_static,
                    gbpe,
                    ubpe,
                    cs_n_used as u32,
                    max_per_expert,
                    HOT_CHUNK as u32,
                    crate::config::SWIGLU_CLAMP_EXP,
                    crate::config::N_FF_EXP,
                    crate::config::BLOCKS_Q8K_GATE_IN,
                    n_wi,
                )?;
            }
            de.q8k.launch(
                &de.compute,
                &mut hd.midq_cat,
                &hd.mid_cat,
                crate::config::BLOCKS_Q8K_DOWN_IN * (cs_n_used as u32) * b,
            )?;
            match routed_src.down.dtype {
                v4flash_core::gguf::GgufType::IQ3_XXS => de.iq3.launch_by_expert_kwide2(
                    &de.compute, &mut hd.partials, &hot.down, &hd.midq_cat,
                    &hd.group_count, &hd.expert_members, &hd.work_items_static, n_wi,
                    dbpe, mid_blocks_bytes as u32, cs_n_used as u32,
                    max_per_expert, HOT_CHUNK as u32, N_EMBD,
                    crate::config::BLOCKS_Q8K_DOWN_IN,
                )?,
                v4flash_core::gguf::GgufType::MXFP4 => de.mxfp4.launch_by_expert_kwide2(
                    &de.compute, &mut hd.partials, &hot.down, &hd.midq_cat,
                    &hd.group_count, &hd.expert_members, &hd.work_items_static, n_wi,
                    dbpe, mid_blocks_bytes as u32, cs_n_used as u32,
                    max_per_expert, HOT_CHUNK as u32, N_EMBD,
                    crate::config::BLOCKS_Q8K_DOWN_IN,
                )?,
                _ => de.q2k.launch_by_expert_kwide2(
                    &de.compute,
                    &mut hd.partials,
                    &hot.down,
                    &hd.midq_cat,
                    &hd.group_count,
                    &hd.expert_members,
                    &hd.work_items_static,
                    dbpe,
                    mid_blocks_bytes as u32,
                    cs_n_used as u32,
                    max_per_expert,
                    HOT_CHUNK as u32,
                    N_EMBD,
                    crate::config::BLOCKS_Q8K_DOWN_IN,
                    n_wi,
                )?,
            }
            de.q2k.launch_reduce_partials_hetsplit(
                &de.compute,
                ffn_moe_dgpu,
                &hd.partials,
                &bd.d_selected,
                &hot.remap,
                /*mode=*/ 1,
                cap,
                cs_n_used as u32,
                N_EMBD,
                b,
            )?;
        }

        // Single batched iGPU MoE call chain. iq2 uses by-expert
        // dispatch (group_builder + work_items pre-pass), q2_k stays
        // by-token (could also be by-expert but smaller perf lever).
        c.hot_active = hot_active;
        Ok(())
    }

    /// Phase 4: the whole iGPU MoE dispatch, group builder through `moe_done` /
    /// `moe_arrived`. Reads SHARED `BatchIgpuShared` scratch (`expert_members`,
    /// `d_xq_q8k`, `d_x16`, the work-item arrays), so one lane's phase 4 runs to
    /// completion before the other lane's starts; `pre_moe_prep` is the
    /// interleavable part -- cutting INSIDE this region had lane A's dispatch
    /// read lane B's inputs (caught by multistream_step G5c, 2026-09-22).
    #[allow(clippy::too_many_arguments)]
    pub fn pre_moe_launch(
        &self,
        c: &mut PreMoeCarry,
        bd: &mut BatchDgpuScratch,
        bi: &mut BatchIgpuScratch,
        sd: &mut BatchDgpuShared,
        si: &mut BatchIgpuShared,
        dlw: &DgpuLayerWeights,
        ilw: &IgpuLayerWeights,
        sev: &super::engine::LayerSyncEvents,
        pager: Option<&mut super::expert_pager::ExpertPager>,
    ) -> eyre::Result<()> {
        if !c.advance(PreMoePhase::Prepped, PreMoePhase::Launched)? { return Ok(()); }
        let PreMoeCarry {
            layer, b, cs_n_used, cs_n_embd, moe_group_bound, split_cap, sparse_resid_layer, hot_active, ref sel_host_audit, ..
        } = *c;
        let _ = (sd, dlw, ilw, hot_active, cs_n_embd);
        let pager_window;
        let (routed_src, moe_remap, moe_packed): (
            &crate::model_weights::RoutedExpertWeights,
            Option<&v4flash_hip::DeviceBuffer<i32>>,
            bool,
        ) = match pager.as_deref() {
            // A VIEW of this layer's dense window, not the whole pool: the window's
            // slot i holds expert i, so the kernel indexes it exactly like a resident
            // buffer while other layers stay resident in other windows.
            Some(pg) => {
                // THIRD half of the allocator/exclusion pairing: the WEIGHTS VIEW.
                //
                // `ensure` (sparse decode LRU) writes ABSOLUTE pool slots, which
                // are only meaningful against the whole pool -- that is what the
                // decode path hands the dispatch (`pg.routed`, forward_layer.rs).
                // `ensure_layer_union`/`_dense` write WINDOW-RELATIVE slots, valid
                // only against `routed_window(layer)`.
                //
                // Handing an absolute slot to the window view reads
                // `window_base(w) + slot`, and with V41_PAGER_WINDOWS=4 /
                // STRIDE=128 the base is 128/256/384 for most layers while the LRU
                // slots start at `dense_slots()` = 768. That is off the end of a
                // 384-wide view: another expert's weights, no error, plausible
                // output. It is correct only when window_base == 0 for every layer
                // (dense_windows == 1), which is NOT the default.
                let routed_view: &crate::model_weights::RoutedExpertWeights =
                    if sparse_resid_layer {
                        &pg.routed
                    } else {
                        pager_window = pg.routed_window(layer as i32);
                        &pager_window
                    };
                // ALWAYS hand the dispatch the remap when the pager owns the
                // window, not just under the remote split.
                //
                // With a dense window `slot == raw expert id`, so a `None` remap
                // used to be harmless — the PLAIN group builder's raw ids happened
                // to be correct slots. That equivalence is exactly what packed
                // windows break (`moe_group_builder.hip:116` mode 0 takes the group
                // id FROM the remap: `g = (dense >= 0) ? e : (-dense - 1)`), so a
                // layer that fell through to `None` would index a packed window by
                // raw id and read another expert's weights — silently, with no
                // error and plausible-looking output. Passing it unconditionally is
                // a no-op today (mode 0 with an all-local remap is arithmetically
                // identical to the plain builder) and the precondition for packing.
                (routed_view, Some(pg.remap_dev(layer as i32)), false)
            }
            None => (&ilw.routed, ilw.hot_remap.as_ref(), ilw.igpu_packed),
        };
        let gbpe = routed_src.gate_bytes_per_expert as u32;
        let ubpe = routed_src.up_bytes_per_expert as u32;
        let dbpe = routed_src.down_bytes_per_expert as u32;
        let mid_blocks_bytes = (crate::config::BLOCKS_Q8K_DOWN_IN as usize)
            * crate::q8_k::BLOCK_Q8_K_BYTES;
        self.set_current_cached(self.igpu.device)?;
        let ie = &self.igpu;
        // Wait for the dGPU→iGPU peer-push to land before any iGPU compute
        // reads the recv buffers. Replaces the old de.xfer.synchronize().
        ie.compute.wait_event(&sev.selected_pushed)?;
        // f16 WMMA MoE path (2026-09-08): f16 activations end to end, no
        // Q8_K quantize on either side of the gate/up. See dispatch.rs.
        let variant_peek = std::env::var("IQ2_VARIANT").unwrap_or_else(|_| "kwide".into());
        let wmma_path = super::dispatch::igpu_moe_wmma_selected(
            routed_src.gate.dtype,
            routed_src.down.dtype,
            ie.is_gfx11,
            &variant_peek,
            super::dispatch::igpu_moe_wmma_env_enabled(),
        );
        if wmma_path {
            let _t_cast = ie.events.stage("igpu.cast_f16_pre_moe", &ie.compute)?;
            ie.q8k.launch_cast_f16(&ie.compute, &mut si.d_x16, &bi.ffn_input_norm_recv, N_EMBD * b)?;
        } else {
            // q8k quantize ain[B*N_EMBD] → d_xq_q8k[B*blocks].
            let _t_q8k_pre = ie.events.stage("igpu.q8k_quantize_pre_iq2", &ie.compute)?;
            ie.q8k.launch(
                &ie.compute,
                &mut si.d_xq_q8k,
                &bi.ffn_input_norm_recv,
                crate::config::BLOCKS_Q8K_GATE_IN * b,
            )?;
        }
        // Chunked by-expert iq2. Three pre-passes then main kernel:
        //   1. moe_group_builder: invert d_selected → group_count + expert_members.
        //   2. moe_work_items_builder: chunk popular groups → work_items + n_work_items.
        //   3. host sync + readback n_work_items to set main kernel grid.y.
        //   4. iq2 chunked main kernel.
        // Chunk size = how many members per WG the iq2/q2k kernels handle.
        // tile8_row32 caps at 8 (ejpir-style block8 unpack); others use 32.
        // Default kwide since M51 (2026-06-09): k-widened lanes, −35% kernel
        // vs staged, +30% e2e prefill. staged/staged_v2/chunked/tile8/hybrid
        // remain opt-in.
        #[allow(non_snake_case)]
        let CHUNK_SIZE: u32 = if variant_peek == "tile8" { 8 } else { 32 };
        // Group arrays and the builder's bound are ONE decision -- the bound is a
        // buffer limit, so a mismatch drops picks silently (#0b). `moe_group_bound`
        // is the pool when the sparse verify residency is in play, `N_EXPERT`
        // otherwise.
        //
        // `expert_members` is dense `[bound x max_per_expert]`, so the wide case
        // pays for itself only by dropping the stride from `B_MAX` to the actual
        // chunk: a group can hold at most one entry per token, so `b` is exact
        // (and the sparse path only ever runs a B<=16 verify). Prefill keeps the
        // `B_MAX` stride it already has allocated.
        let max_per_expert = if moe_group_bound > N_EXPERT {
            b
        } else {
            si.max_per_expert()
        };
        if bi.group_count.len() < moe_group_bound as usize
            || !si.group_capacity_ok(moe_group_bound, max_per_expert, b as usize)
        {
            // Growing FREES the old buffers, and the previous layer's by-expert
            // kernels can still be queued on them (the chain's only sync is the
            // work-item readback, ahead of the main kernel). Grow-only, so this
            // drains at most once per process.
            ie.compute.synchronize()?;
        }
        bi.ensure_group_bound(moe_group_bound)?;
        si.ensure_group_capacity(moe_group_bound, max_per_expert, b as usize)?;
        if moe_group_bound > N_EXPERT {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                eprintln!(
                    "moe group space: WIDE {moe_group_bound} ids x {max_per_expert} members \
                     (sparse verify residency; N_EXPERT={N_EXPERT})"
                );
            });
        }
        bi.group_count.fill_zero()?;
        {
            let _t_grp = ie.events.stage("igpu.moe_group_builder", &ie.compute)?;
            let BatchIgpuScratch {
                group_count,
                d_selected,
                ..
            } = bi;
            let BatchIgpuShared { expert_members, .. } = si;
            // Take the het-split builder whenever a remap is in play: either the
            // M56 dGPU hot split (hot_active), or the M7 pager, whose remap sends
            // EVERY pick to a pool slot. The plain builder emits raw expert ids,
            // which cannot index a packed/pooled buffer.
            if let Some(rm) = moe_remap {
                // M61: build groups for the MISS slots only — the dGPU
                // owns the resident slots (up to the per-token cap).
                ie.moe_group_builder.launch_hetsplit(
                    &ie.compute,
                    group_count,
                    expert_members,
                    d_selected,
                    rm,
                    /*mode=*/ 0,
                    split_cap,
                    b,
                    cs_n_used as u32,
                    moe_group_bound,
                    max_per_expert,
                )?;
                // `V41_GROUP_AUDIT=1`: every pick the remap marks OURS must end
                // up in exactly one expert's member list. A short count means the
                // builder silently dropped local picks and the layer is
                // under-computed — the failure mode this file warns about twice
                // (the M63 over-cap bug, hot_prefill_cap()=4 vs N_EXPERT_USED=6)
                // and the one that matches the small-B catch-all corrupting only
                // when a LANE HOLDS >= 2 ROWS.
                if group_audit() {
                    ie.compute.synchronize()?;
                    let mut gc = vec![0i32; moe_group_bound as usize];
                    group_count
                        .slice_view(0, moe_group_bound as usize)
                        .copy_to_host(&mut gc)?;
                    let enqueued: i64 = gc.iter().map(|&v| v as i64).sum();
                    // Expected: picks whose remap entry is NEGATIVE ("ours at slot").
                    let mut remap_host = vec![0i32; N_EXPERT as usize];
                    rm.slice_view(0, N_EXPERT as usize).copy_to_host(&mut remap_host)?;
                    let expected: i64 = sel_host_audit
                        .iter()
                        .filter(|&&e| {
                            (0..N_EXPERT as i32).contains(&e) && remap_host[e as usize] < 0
                        })
                        .count() as i64;
                    if enqueued != expected {
                        tracing::warn!(
                            layer, b, enqueued, expected,
                            deficit = expected - enqueued,
                            "GROUP_AUDIT: het-split builder dropped local picks"
                        );
                    }
                    // `V41_GROUP_AUDIT_VERBOSE=1` (multi-stream harness, KNOWN_BUGS #20):
                    // print, on stderr (no tracing subscriber in tests), the groups the
                    // builder filled and, for row 0's picks, id / remap / pager slot /
                    // a checksum of the slot's gate bytes — the per-history diff of the
                    // batched MoE's bookkeeping.
                    if std::env::var("V41_GROUP_AUDIT_VERBOSE").as_deref() == Ok("1") {
                        let groups: Vec<(usize, i32)> = gc.iter().enumerate().filter(|(_, &c)| c != 0).map(|(g, &c)| (g, c)).collect();
                        eprintln!("[group-audit] L{layer} b={b} enqueued={enqueued} expected={expected} bound={moe_group_bound} max_per_expert={max_per_expert} groups(slot:count)={groups:?}");
                        if let Some(pg) = pager.as_deref() {
                            let gbpe = pg.routed.gate_bytes_per_expert;
                            let probe = gbpe.min(65536);
                            let mut bytes = vec![0u8; probe];
                            for (k, &e) in sel_host_audit.iter().take(cs_n_used).enumerate() {
                                if !(0..N_EXPERT as i32).contains(&e) { continue; }
                                let slot = pg.resident_slot(layer as i32, e as u32);
                                let sum = match slot {
                                    Some(sl) if (sl as usize + 1) * gbpe <= pg.routed.gate.buffer.len() => {
                                        pg.routed.gate.buffer.slice_view(sl as usize * gbpe, probe).copy_to_host(&mut bytes)?;
                                        bytes.iter().fold(0u64, |a, &x| a.wrapping_mul(1099511628211).wrapping_add(x as u64))
                                    }
                                    _ => 0,
                                };
                                eprintln!("[group-audit]   pick{k}: id={e} remap={} slot_of={slot:?} gate_fnv={sum:#x}", remap_host[e as usize]);
                            }
                        }
                    }
                }
            } else {
                // Emits RAW expert ids as group ids — incompatible with a
                // de-duplicated iGPU buffer (see prefill_hot_active).
                if moe_packed {
                    return Err(eyre!(
                        "L{layer}: iGPU experts are packed (IGPU_DEDUP_HOT) but the prefill \
                         het-split is inactive — the plain group builder would index the \
                         wrong experts"
                    ));
                }
                // Raw expert ids, so this branch's group space IS N_EXPERT. It is
                // only reachable with no remap at all, i.e. no pager -- and the
                // sparse view needs one -- so a wide bound here would mean the
                // slot space and the builder had diverged.
                debug_assert_eq!(moe_group_bound, N_EXPERT);
                ie.moe_group_builder.launch(
                    &ie.compute,
                    group_count,
                    expert_members,
                    d_selected,
                    b,
                    cs_n_used as u32,
                    N_EXPERT,
                    max_per_expert,
                )?;
            }
        }
        // IQ2_VARIANT env: "staged" (default), "chunked", "hybrid", or "tile8".
        // - hybrid: split work items by chunk size; staged for large, chunked for small.
        // - tile8:  ejpir-port block8 + tile8_row32 (chunk_size auto-set to 8 above).
        // IQ2_HYBRID_THRESHOLD env: chunk-size cutoff (default 8).
        let variant = variant_peek.clone();
        // Carried out to the q2k_down dispatch below; staged/chunked path
        // assigns it inside its else-branch, hybrid path leaves it 0 (which
        // is fine — Q2K_VARIANT=by_expert is forbidden with hybrid below).
        let mut n_work_items: u32 = 0;
        let threshold: u32 = std::env::var("IQ2_HYBRID_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        if variant == "hybrid" && routed_src.gate.dtype != v4flash_core::gguf::GgufType::IQ2_XXS {
            return Err(eyre!(
                "IQ2_VARIANT=hybrid unsupported on a layer with {:?} gate/up \
                 (unsloth blk.26); unset IQ2_VARIANT",
                routed_src.gate.dtype
            ));
        }
        if variant == "hybrid" {
            // launch_work_items_split atomicAdds into these counters.
            // Their doc-comment promises "pre-zeroed per layer" — honour
            // that here. Without this, the readback at copy_to_host below
            // returns prev_counter+actual_count, the downstream iq2 grid
            // is overstated, and the staged/chunked kernels read past the
            // real work_items[] tail into uninit slots.
            bi.n_staged_work_items.fill_zero()?;
            bi.n_chunked_work_items.fill_zero()?;
            {
                let BatchIgpuScratch {
                    n_staged_work_items,
                    n_chunked_work_items,
                    group_count,
                    ..
                } = bi;
                let BatchIgpuShared {
                    staged_work_items,
                    chunked_work_items,
                    ..
                } = si;
                let _t_wis = ie.events.stage("igpu.moe_work_items_split", &ie.compute)?;
                let max_items = staged_work_items.len() as u32;
                ie.moe_group_builder.launch_work_items_split(
                    &ie.compute,
                    staged_work_items,
                    chunked_work_items,
                    n_staged_work_items,
                    n_chunked_work_items,
                    group_count,
                    moe_group_bound,
                    CHUNK_SIZE,
                    threshold,
                    max_items,
                )?;
            }
            let _t_wi = LayerHostTimer::start(&LH_WORK_ITEMS_SYNC);
            ie.compute.synchronize()?;
            drop(_t_wi);
            let mut counts = [0i32; 1];
            bi.n_staged_work_items.copy_to_host(&mut counts)?;
            let n_staged = counts[0] as u32;
            bi.n_chunked_work_items.copy_to_host(&mut counts)?;
            let n_chunked = counts[0] as u32;
            {
                let BatchIgpuScratch {
                    d_ew,
                    group_count,
                    ..
                } = bi;
                let BatchIgpuShared {
                    d_mid_cat,
                    d_xq_q8k,
                    expert_members,
                    staged_work_items,
                    chunked_work_items,
                    ..
                } = si;
                if n_staged > 0 {
                    let _t_st = ie.events.stage("igpu.iq2_staged", &ie.compute)?;
                    ie.iq2.launch_fused_swiglu_chunked_staged(
                        &ie.compute, d_mid_cat,
                        &routed_src.gate.buffer, &routed_src.up.buffer,
                        d_xq_q8k, d_ew,
                        group_count, expert_members, staged_work_items,
                        gbpe, ubpe, cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                        crate::config::SWIGLU_CLAMP_EXP,
                        crate::config::N_FF_EXP,
                        crate::config::BLOCKS_Q8K_GATE_IN,
                        n_staged,
                    )?;
                }
                if n_chunked > 0 {
                    let _t_ch = ie.events.stage("igpu.iq2_chunked", &ie.compute)?;
                    ie.iq2.launch_fused_swiglu_chunked(
                        &ie.compute, d_mid_cat,
                        &routed_src.gate.buffer, &routed_src.up.buffer,
                        d_xq_q8k, d_ew,
                        group_count, expert_members, chunked_work_items,
                        gbpe, ubpe, cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                        crate::config::SWIGLU_CLAMP_EXP,
                        crate::config::N_FF_EXP,
                        crate::config::BLOCKS_Q8K_GATE_IN,
                        n_chunked,
                    )?;
                }
            }
        } else {
            bi.n_work_items.fill_zero()?;
            {
                let BatchIgpuScratch {
                    n_work_items,
                    group_count,
                    ..
                } = bi;
                let BatchIgpuShared { work_items, .. } = si;
                let _t_wi = ie.events.stage("igpu.moe_work_items", &ie.compute)?;
                ie.moe_group_builder.launch_work_items(
                    &ie.compute,
                    work_items,
                    n_work_items,
                    group_count,
                    moe_group_bound,
                    CHUNK_SIZE,
                    work_items.len() as u32,
                )?;
            }
            // Host readback of the work-item COUNT to size the kwide launch: a
            // full iGPU drain per lane-layer that also waits for the OTHER lane's
            // MoE queued ahead of it (profile audit 2026-09-21; untimed until now).
            let _t_wic = LayerHostTimer::start(&LH_WORK_ITEMS_COUNT);
            ie.compute.synchronize()?;
            let mut n_wi_host = [0i32; 1];
            bi.n_work_items.copy_to_host(&mut n_wi_host)?;
            drop(_t_wic);
            n_work_items = n_wi_host[0] as u32;
            {
                let BatchIgpuScratch {
                    d_ew,
                    group_count,
                    ..
                } = bi;
                let BatchIgpuShared {
                    d_mid_cat,
                    d_xq_q8k,
                    d_x16,
                    d_mid16,
                    expert_members,
                    work_items,
                    ..
                } = si;
                // Formats with a single prefill kernel (IQ2_S, IQ2_XS,
                // IQ3_XXS-as-gate/up, IQ3_S): chunked by-expert, IQ2_VARIANT
                // does not apply. IQ2_XXS returns false and takes the zoo below.
                let handled = if wmma_path {
                    let _t_wm = ie.events.stage("igpu.gateup_wmma", &ie.compute)?;
                    match routed_src.gate.dtype {
                        v4flash_core::gguf::GgufType::IQ2_S => ie.iq2s.launch_fused_swiglu_wmma_f16out(
                            &ie.compute, d_mid16,
                            &routed_src.gate.buffer, &routed_src.up.buffer,
                            d_x16, d_ew,
                            group_count, expert_members, work_items,
                            n_work_items,
                            gbpe, ubpe, cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                            crate::config::SWIGLU_CLAMP_EXP,
                            crate::config::N_FF_EXP,
                            crate::config::BLOCKS_Q8K_GATE_IN,
                        )?,
                        _ => ie.iq2xs.launch_fused_swiglu_wmma_f16out(
                            &ie.compute, d_mid16,
                            &routed_src.gate.buffer, &routed_src.up.buffer,
                            d_x16, d_ew,
                            group_count, expert_members, work_items,
                            n_work_items,
                            gbpe, ubpe, cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                            crate::config::SWIGLU_CLAMP_EXP,
                            crate::config::N_FF_EXP,
                            crate::config::BLOCKS_Q8K_GATE_IN,
                        )?,
                    }
                    true
                } else {
                    // Label reflects the kernel that actually runs (kwide vs
                    // the chunked rollback) — a trace is the only way to
                    // confirm the kwide path is live, and IQ2_S gate/up is
                    // essentially the whole iGPU MoE prefill in the
                    // Vision-Exp mix.
                    let _t_ch = ie.events.stage(
                        super::dispatch::pair_prefill_stage(routed_src.gate.dtype),
                        &ie.compute,
                    )?;
                    super::dispatch::moe_gate_up_chunked(
                        ie, routed_src.gate.dtype, &ie.compute, d_mid_cat,
                        &routed_src.gate.buffer, &routed_src.up.buffer,
                        d_xq_q8k, d_ew,
                        group_count, expert_members, work_items,
                        n_work_items,
                        gbpe, ubpe, cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                        crate::config::SWIGLU_CLAMP_EXP,
                        crate::config::N_FF_EXP,
                        crate::config::BLOCKS_Q8K_GATE_IN,
                    )?
                };
                if handled {
                } else if variant == "tile8" {
                    let _t_t8 = ie.events.stage("igpu.iq2_tile8", &ie.compute)?;
                    ie.iq2.launch_fused_swiglu_tile8_row32(
                        &ie.compute, d_mid_cat,
                        &routed_src.gate.buffer, &routed_src.up.buffer,
                        d_xq_q8k, d_ew,
                        group_count, expert_members, work_items,
                        gbpe, ubpe, cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                        crate::config::SWIGLU_CLAMP_EXP,
                        crate::config::N_FF_EXP,
                        crate::config::BLOCKS_Q8K_GATE_IN,
                        n_work_items,
                    )?;
                } else if variant == "kwide" {
                    let _t_kw = ie.events.stage("igpu.iq2_kwide", &ie.compute)?;
                    ie.iq2.launch_fused_swiglu_kwide(
                        &ie.compute, d_mid_cat,
                        &routed_src.gate.buffer, &routed_src.up.buffer,
                        d_xq_q8k, d_ew,
                        group_count, expert_members, work_items,
                        gbpe, ubpe, cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                        crate::config::SWIGLU_CLAMP_EXP,
                        crate::config::N_FF_EXP,
                        crate::config::BLOCKS_Q8K_GATE_IN,
                        n_work_items,
                    )?;
                } else if variant == "staged_v2" {
                    let _t_s2 = ie.events.stage("igpu.iq2_staged_v2", &ie.compute)?;
                    ie.iq2.launch_fused_swiglu_chunked_staged_v2(
                        &ie.compute, d_mid_cat,
                        &routed_src.gate.buffer, &routed_src.up.buffer,
                        d_xq_q8k, d_ew,
                        group_count, expert_members, work_items,
                        gbpe, ubpe, cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                        crate::config::SWIGLU_CLAMP_EXP,
                        crate::config::N_FF_EXP,
                        crate::config::BLOCKS_Q8K_GATE_IN,
                        n_work_items,
                    )?;
                } else if variant != "chunked" {
                    let _t_st = ie.events.stage("igpu.iq2_staged", &ie.compute)?;
                    ie.iq2.launch_fused_swiglu_chunked_staged(
                        &ie.compute, d_mid_cat,
                        &routed_src.gate.buffer, &routed_src.up.buffer,
                        d_xq_q8k, d_ew,
                        group_count, expert_members, work_items,
                        gbpe, ubpe, cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                        crate::config::SWIGLU_CLAMP_EXP,
                        crate::config::N_FF_EXP,
                        crate::config::BLOCKS_Q8K_GATE_IN,
                        n_work_items,
                    )?;
                } else {
                    let _t_ch = ie.events.stage("igpu.iq2_chunked", &ie.compute)?;
                    ie.iq2.launch_fused_swiglu_chunked(
                        &ie.compute, d_mid_cat,
                        &routed_src.gate.buffer, &routed_src.up.buffer,
                        d_xq_q8k, d_ew,
                        group_count, expert_members, work_items,
                        gbpe, ubpe, cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                        crate::config::SWIGLU_CLAMP_EXP,
                        crate::config::N_FF_EXP,
                        crate::config::BLOCKS_Q8K_GATE_IN,
                        n_work_items,
                    )?;
                }
            }
        }
        if !wmma_path {
            let _t_q8k_post = ie.events.stage("igpu.q8k_quantize_post_iq2", &ie.compute)?;
            ie.q8k.launch(
                &ie.compute,
                &mut si.d_midq_cat,
                &si.d_mid_cat,
                crate::config::BLOCKS_Q8K_DOWN_IN * (cs_n_used as u32) * b,
            )?;
        }
        {
            let _t_q2k = ie.events.stage("igpu.q2k_down", &ie.compute)?;
            // Default `by_expert`: invert (B, expert) iteration so each
            // expert's row-tile is read once instead of B*n_used times.
            // Was: 94% DRAM-BW-bound on redundant weight reads (PMC L2 hit
            // 4%, MemUnitBusy 99.98%). Reuses the iq2 group arrays
            // (group_count / expert_members / work_items) built once per
            // layer by moe_group_builder — no extra pre-pass. Writes per-
            // (b, slot) partials, then a tiny reduce kernel sums across
            // slots to produce out — fully deterministic (no atomicAdd).
            // E2E: +14-23% prefill throughput at B_MAX=512 across all
            // depths (4K..64K); ~flat at B≤64; small-batch regression for
            // pathological B<32 prefills.
            // Q2K_VARIANT=bxn rolls back to the original kernel.
            // Hybrid IQ2 is incompatible (splits work_items into two
            // buckets); error out if both are requested simultaneously.
            // Default kwide2 since M53 (2026-06-09): row-pair activation
            // reuse on top of kwide's unpack-once loop; bit-exact vs
            // by_expert. kwide/by_expert/bxn stay opt-in.
            let down_dt = routed_src.down.dtype;
            let q2k_variant = if down_dt == v4flash_core::gguf::GgufType::Q2_K {
                std::env::var("Q2K_VARIANT").unwrap_or_else(|_| "kwide2".into())
            } else {
                // IQ3_XXS / MXFP4 implement only the kwide2 shape; the
                // env variants are Q2_K-only.
                "kwide2".into()
            };
            let use_kwide2 = q2k_variant == "kwide2";
            let use_kwide = q2k_variant == "kwide";
            let use_by_expert = use_kwide || use_kwide2 || q2k_variant == "by_expert";
            if use_by_expert && variant == "hybrid" {
                return Err(eyre!(
                    "Q2K_VARIANT=by_expert/kwide not supported with IQ2_VARIANT=hybrid \
                     (would need to combine staged_+chunked_work_items). \
                     Set Q2K_VARIANT=bxn to opt out."
                ));
            }
            if hot_active && !use_by_expert {
                // bxn iterates d_selected directly: it would read mid for
                // the dGPU-resident slots the iGPU never computed (garbage)
                // and double-count them after the combine adds the dGPU
                // partial.
                return Err(eyre!(
                    "Q2K_VARIANT=bxn incompatible with prefill het-split \
                     (dGPU owns the resident slots). Set DGPU_HOT_PREFILL=0 \
                     to roll back the split instead."
                ));
            }
            if use_by_expert {
                // No partials zero-fill (M53): router topk yields 8 DISTINCT
                // experts per token, so group_count[e] ≤ B = max_per_expert —
                // the builder's overflow guard can't fire and every (b, slot)
                // pair is written by exactly one work item. (The 128 MB/layer
                // fill was ~44 ms/chunk of pure overhead.) If routing ever
                // allows duplicate experts per token, restore the fill.
                if wmma_path {
                    let _t_dw = ie.events.stage("igpu.down_wmma", &ie.compute)?;
                    ie.iq3.launch_by_expert_wmma(
                        &ie.compute, &mut si.q2k_partials,
                        &routed_src.down.buffer, &si.d_mid16,
                        &bi.group_count, &si.expert_members, &si.work_items,
                        n_work_items, dbpe, crate::config::N_FF_EXP,
                        cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                        N_EMBD, crate::config::BLOCKS_Q8K_DOWN_IN,
                    )?;
                } else if use_kwide2 && down_dt == v4flash_core::gguf::GgufType::IQ3_XXS {
                    ie.iq3.launch_by_expert_kwide2(
                        &ie.compute, &mut si.q2k_partials,
                        &routed_src.down.buffer, &si.d_midq_cat,
                        &bi.group_count, &si.expert_members, &si.work_items,
                        n_work_items, dbpe, mid_blocks_bytes as u32,
                        cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                        N_EMBD, crate::config::BLOCKS_Q8K_DOWN_IN,
                    )?;
                } else if use_kwide2 && down_dt == v4flash_core::gguf::GgufType::MXFP4 {
                    ie.mxfp4.launch_by_expert_kwide2(
                        &ie.compute, &mut si.q2k_partials,
                        &routed_src.down.buffer, &si.d_midq_cat,
                        &bi.group_count, &si.expert_members, &si.work_items,
                        n_work_items, dbpe, mid_blocks_bytes as u32,
                        cs_n_used as u32, max_per_expert, CHUNK_SIZE,
                        N_EMBD, crate::config::BLOCKS_Q8K_DOWN_IN,
                    )?;
                } else if use_kwide2 {
                    ie.q2k.launch_by_expert_kwide2(
                        &ie.compute,
                        &mut si.q2k_partials,
                        &routed_src.down.buffer,
                        &si.d_midq_cat,
                        &bi.group_count,
                        &si.expert_members,
                        &si.work_items,
                        dbpe,
                        mid_blocks_bytes as u32,
                        cs_n_used as u32,
                        max_per_expert,
                        CHUNK_SIZE,
                        N_EMBD,
                        crate::config::BLOCKS_Q8K_DOWN_IN,
                        n_work_items,
                    )?;
                } else if use_kwide {
                    ie.q2k.launch_by_expert_kwide(
                        &ie.compute,
                        &mut si.q2k_partials,
                        &routed_src.down.buffer,
                        &si.d_midq_cat,
                        &bi.group_count,
                        &si.expert_members,
                        &si.work_items,
                        dbpe,
                        mid_blocks_bytes as u32,
                        cs_n_used as u32,
                        max_per_expert,
                        CHUNK_SIZE,
                        N_EMBD,
                        crate::config::BLOCKS_Q8K_DOWN_IN,
                        n_work_items,
                    )?;
                } else {
                    ie.q2k.launch_by_expert(
                        &ie.compute,
                        &mut si.q2k_partials,
                        &routed_src.down.buffer,
                        &si.d_midq_cat,
                        &bi.group_count,
                        &si.expert_members,
                        &si.work_items,
                        dbpe,
                        mid_blocks_bytes as u32,
                        cs_n_used as u32,
                        max_per_expert,
                        CHUNK_SIZE,
                        N_EMBD,
                        crate::config::BLOCKS_Q8K_DOWN_IN,
                        n_work_items,
                    )?;
                }
                // MUST match the group builder's selector above, which takes the
                // het-split path on `moe_remap.is_some()`. Keying the reduce off
                // `hot_active` instead let the two disagree: with the two-box
                // remap the builder skipped the remote-owned experts, then the
                // PLAIN reduce summed all `cs_n_used` partial slots anyway —
                // including the ones nothing had written this layer, i.e. stale
                // rows from a previous layer. Symptom was a moderate, coherent
                // perturbation (output kept counting but lost a clause), not
                // garbage, which is exactly what summing a few stale slots looks
                // like.
                if moe_remap.is_some() {
                    // M61: sum ONLY the miss slots this device computed —
                    // resident slots' partials are stale here (the dGPU
                    // holds their contribution).
                    ie.q2k.launch_reduce_partials_hetsplit(
                        &ie.compute,
                        &mut bi.ffn_moe,
                        &si.q2k_partials,
                        &bi.d_selected,
                        moe_remap.unwrap(),
                        /*mode=*/ 0,
                        split_cap,
                        cs_n_used as u32,
                        N_EMBD,
                        b,
                    )?;
                } else {
                    ie.q2k.launch_reduce_partials(
                        &ie.compute,
                        &mut bi.ffn_moe,
                        &si.q2k_partials,
                        cs_n_used as u32,
                        N_EMBD,
                        b,
                    )?;
                }
            } else {
                ie.q2k.launch_batched_bxn(
                    &ie.compute,
                    &mut bi.ffn_moe,
                    &routed_src.down.buffer,
                    &si.d_midq_cat,
                    &bi.d_selected,
                    dbpe,
                    mid_blocks_bytes as u32,
                    cs_n_used as u32,
                    N_EMBD,
                    crate::config::BLOCKS_Q8K_DOWN_IN,
                    b,
                )?;
            }
        }
        // `V41_VERIFY_DECODE_MOE=1`: recompute this layer's local MoE with the
        // DECODE path's kernels, overwriting what the by-expert chain above
        // produced.
        //
        // WHY. The verify inherits the PREFILL MoE — `moe_gate_up_chunked` plus
        // the by-expert kwide/`q2k_down` chain, iterating by EXPERT and
        // accumulating across the batch, and on the WMMA branch never
        // quantising activations to Q8_K at all. Decode uses
        // `moe_*_hetsplit`, per token, over Q8_K activations. Two deliberate
        // implementations that were never required to agree, because prefill
        // only consumes the LAST row's logits — a speculative verify is the
        // first consumer of all of them. MEASURED divergence at B=6: argmax
        // agreement 0.49-0.61 with decode, cos 0.75-0.79 on the logit vectors
        // (identical compute would be ~0.9999).
        //
        // This recomputes rather than replaces so the change is ONE
        // self-contained block: it proves or refutes the diagnosis before
        // anyone pays for the surgery to skip the wasted chain. Bounded by b
        // because it costs b x decode's MoE.
        if verify_decode_moe() && (b as usize) <= 16 {
            if let Some(remap) = moe_remap {
                let nu = cs_n_used;
                let ne = crate::config::N_EMBD as usize;
                let ffe = crate::config::N_FF_EXP as usize;
                let xqb = super::remote_experts::XQ_BYTES_PER_TOKEN;
                let mqb = super::remote_experts::MIDQ_BYTES_PER_SLOT;
                let _t_vm = ie.events.stage("igpu.verify_decode_moe", &ie.compute)?;
                // Decode quantises the activation to Q8_K; the WMMA branch
                // above may have cast to f16 instead and left this untouched.
                ie.q8k.launch(
                    &ie.compute,
                    &mut si.d_xq_q8k,
                    &bi.ffn_input_norm_recv,
                    crate::config::BLOCKS_Q8K_GATE_IN * b,
                )?;
                for j in 0..b as usize {
                    {
                        let xq_j = si.d_xq_q8k.slice_view(j * xqb, xqb);
                        let ew_j = bi.d_ew.slice_view(j * nu, nu);
                        let sel_j = bi.d_selected.slice_view(j * nu, nu);
                        let mut mid_j = si.d_mid_cat.slice_view_mut(j * nu * ffe, nu * ffe);
                        super::dispatch::moe_gate_up_batch_hetsplit(
                            ie, routed_src.gate.dtype, &ie.compute, &mut mid_j,
                            &routed_src.gate.buffer, &routed_src.up.buffer,
                            &xq_j, &ew_j, &sel_j, remap, 0, nu as u32, gbpe, ubpe,
                            nu as u32, crate::config::SWIGLU_CLAMP_EXP,
                            crate::config::N_FF_EXP, crate::config::BLOCKS_Q8K_GATE_IN,
                        )?;
                    }
                    {
                        let mid_j = si.d_mid_cat.slice_view(j * nu * ffe, nu * ffe);
                        let mut midq_j = si.d_midq_cat.slice_view_mut(j * nu * mqb, nu * mqb);
                        ie.q8k.launch(
                            &ie.compute, &mut midq_j, &mid_j,
                            crate::config::BLOCKS_Q8K_DOWN_IN * nu as u32,
                        )?;
                    }
                    {
                        let midq_j = si.d_midq_cat.slice_view(j * nu * mqb, nu * mqb);
                        let sel_j = bi.d_selected.slice_view(j * nu, nu);
                        let mut out_j = bi.ffn_moe.slice_view_mut(j * ne, ne);
                        super::dispatch::moe_down_batched_hetsplit(
                            ie, routed_src.down.dtype, &ie.compute, &mut out_j,
                            &routed_src.down.buffer, &midq_j, &sel_j, remap, 0,
                            nu as u32, dbpe, mqb as u32, nu as u32,
                            crate::config::N_EMBD, crate::config::BLOCKS_Q8K_DOWN_IN,
                        )?;
                    }
                }
            }
        }
        // Record MoE-done so the iGPU xfer can wait without a host sync.
        sev.moe_done.record(&ie.compute)?;
        ie.xfer.wait_event(&sev.moe_done)?;

        // Single batched peer-push back of bi.ffn_moe[B*N_EMBD].
        let bi_ffn_view = bi.ffn_moe.slice_view(0, (b as usize) * cs_n_embd);
        let mut bd_moe_dst = bd
            .ffn_moe_recv
            .slice_view_mut(0, (b as usize) * cs_n_embd);
        {
            let _t_peer_moe = ie.events.stage("igpu.peer_push_ffn_moe", &ie.xfer)?;
            peer_push_f32(&bi_ffn_view, &mut bd_moe_dst, &ie.xfer)?;
        }
        // moe_arrived fires once ffn_moe has landed on dGPU; ffn_combine
        // below waits on it instead of the old ie.xfer.synchronize().
        sev.moe_arrived.record(&ie.xfer)?;
        drop(bi_ffn_view);
        drop(bd_moe_dst);
        self.set_current_cached(self.dgpu.device)?;
        // moe_arrived is the post-MoE handoff event; forward_layer_post_moe_v2
        // queues the wait + ffn_combine. Pre-MoE returns here without
        // touching de.compute again, so the caller can interleave another
        // lane's pre-MoE before queueing this lane's ffn_combine.
        Ok(())
    }

    /// Stage 12: dGPU ffn_combine. Separated from the pre-MoE body so the
    /// pipelined caller can queue both lanes' pre-MoE work on de.compute
    /// before any ffn_combine reads its post-MoE inputs. `sev.moe_arrived`
    /// must be the same LayerSyncEvents passed to the matching pre-MoE call.
    fn forward_layer_post_moe_v2(
        &self,
        bd: &mut BatchDgpuScratch,
        b: u32,
        sev: &super::engine::LayerSyncEvents,
        hot_active: bool,
    ) -> eyre::Result<()> {
        self.set_current_cached(self.dgpu.device)?;
        let de = &self.dgpu;
        de.compute.wait_event(&sev.moe_arrived)?;
        // KNOWN_BUGS-style routed-only dump (twin of decode's `dec_ffn_routed`):
        // `ffn_moe_recv` row 0 BEFORE the shared/remote/hot adds below.
        {
            let lp = DUMP_LAYER_POS.load(std::sync::atomic::Ordering::Relaxed);
            let (dl, dp) = ((lp >> 32) as usize, (lp & 0xffff_ffff) as u32);
            if lp != u64::MAX && super::engine::subtensor_dump_armed(dl) {
                de.compute.synchronize()?;
                super::engine::maybe_dump_subtensor_f32_view(
                    dl,
                    &format!("pf_ffn_routed_p{dp}"),
                    &bd.ffn_moe_recv.slice_view(0, N_EMBD as usize),
                )?;
                DUMP_LAYER_POS.store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
            }
        }
        // NOT one bracket around the whole function. The blocking box-2 `wait()`
        // sits between the two vec_adds below, and a stage bracket spanning it
        // records the HOST STALL as dGPU device time: on a perfetto trace it
        // draws one enormous `dgpu.ffn_combine` slice covering the entire RPC,
        // so the lane reads ~6 ms busy while the GPU is idle. Every "GPU busy %"
        // taken from that slice was inflated by the RPC wait. Bracket the two
        // halves separately instead, so the idle between them shows as idle.
        {
            let _t_combine = de.events.stage("dgpu.ffn_combine.local", &de.compute)?;
            let _t = de.events.stage("k.ffn_combine.vec_add", &de.compute)?;
            de.vec_add.launch(
                &de.compute,
                &mut bd.ffn_moe_recv,
                &bd.ffn_shared,
                b * N_EMBD,
            )?;
        }

        // Two-box split: collect the remote's reply. Awaited HERE, not at
        // submit time, so box 2 computed its half of this layer while this box's
        // iGPU computed the other half — the local MoE was issued between the
        // two. The gap between the `submit` and `wait` slices on the
        // `remote.expert (host)` perfetto track is exactly that overlap.
        if let Some(t) = bd.remote_ticket.take() {
            let layer = bd.remote_ffn_moe_layer;
            let t_wait = super::perfetto::now_ns();
            let remote = self
                .remote
                .as_ref()
                .ok_or_else(|| eyre!("remote ticket pending but no client"))?;
            let partial = remote
                .lock()
                .map_err(|_| eyre!("remote expert client mutex poisoned"))?
                .wait(t)?;
            // SLACK PROBE site `remote`: hold the partial back by a known
            // amount, i.e. pretend box 2 (or the link) was slower. Regressing
            // the step against it gives the box-2 leg's share of the critical
            // path -- the thing an rtt counter cannot tell you, because exposed
            // wait collapses to ~0 whenever box 1 is the slower side.
            if let Some(ticks) = super::mtp::slack_probe_ticks("remote") {
                // wall_clock64 ticks are 100 MHz, so ticks/100 = microseconds.
                std::thread::sleep(std::time::Duration::from_micros(ticks / 100));
            }
            let t_wait_end = super::perfetto::now_ns();
            if let Some(pf) = self.perfetto.as_ref() {
                if let Ok(pf) = pf.lock() {
                    let _ = pf.emit_host_slice(
                        pf.remote_uuid,
                        &format!(
                            "wait L{layer} rtt={}us link={}us remote={}us page={}us/{}",
                            partial.rtt_us, partial.link_us(), partial.t_remote_compute_us,
                            partial.t_remote_page_us, partial.n_remote_miss,
                        ),
                        t_wait, t_wait_end,
                    );
                    emit_remote_page_slice(&pf, layer as u32, &partial, t_wait_end);
                }
            }
            super::trace::phase::add(
                &super::trace::phase::REMOTE_WAIT_NS,
                (t_wait_end - t_wait) as u64,
            );
            super::trace::phase::add(&super::trace::phase::REMOTE_RTT_NS, (partial.rtt_us as u64) * 1000);
            super::trace::phase::add(&super::trace::phase::REMOTE_SRV_NS, (partial.t_remote_server_us as u64) * 1000);
            super::trace::phase::add(&super::trace::phase::REMOTE_PAGE_NS, partial.t_remote_page_us as u64 * 1000);
            super::trace::phase::add(&super::trace::phase::REMOTE_COMPUTE_NS, partial.t_remote_compute_us as u64 * 1000);
            super::trace::phase::add(&super::trace::phase::REMOTE_MISSES, partial.n_remote_miss as u64);
            if layer_host_timing() {
                LH_REMOTE_WAIT.fetch_add((t_wait_end - t_wait) as u64 / 1000, std::sync::atomic::Ordering::Relaxed);
            }
            if remote_add_partial() {
                let rows = (b as usize) * N_EMBD as usize;
                let src = partial.f32();
                // Hash box 2's HOST-SIDE response before it is copied anywhere.
                // The combine-dbg print reads the DEVICE buffer after syncing
                // de.compute, so a repeated value there could be a readback
                // artifact. This distinguishes "box 2 sent the same bytes for
                // two layers" (a routing bug) from "box 1 read stale device
                // memory" (a stream-ordering artifact).
                if std::env::var("V41_REMOTE_DBG").is_ok() {
                    let mut h: u64 = 0xcbf29ce484222325;
                    for &v in src.iter().step_by(97) {
                        h ^= v.to_bits() as u64;
                        h = h.wrapping_mul(0x100000001b3);
                    }
                    let l2: f64 =
                        src.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                    let nz = src.iter().filter(|&&x| x != 0.0).count();
                    eprintln!(
                        "[partial-src] L{layer} b={b} seq_layer={} hash={h:016x} l2={l2:.4} \
                         nonzero={nz}/{} first={:?}",
                        partial.layer, src.len(), &src[..4.min(src.len())]
                    );
                }
                if src.len() != rows {
                    return Err(eyre!(
                        "L{layer}: remote partial has {} f32 rows, expected {rows}",
                        src.len()
                    ));
                }
                bd.remote_ffn_moe
                    .as_mut()
                    .ok_or_else(|| eyre!("remote partial pending but buffer unallocated"))?
                    .slice_view_mut(0, rows)
                    .copy_from_host(src)?;
                bd.remote_ffn_moe_valid = true;
            }
            remote
                .lock()
                .map_err(|_| eyre!("remote expert client mutex poisoned"))?
                .recycle(partial);
        }

        // Two-box split: + the remote shard's MoE partial.
        //
        // MUST live here in POST-MoE, not pre-MoE. `ffn_moe_recv` is the dGPU's
        // landing buffer for the iGPU's `ffn_moe`, delivered by peer push and
        // gated by `sev.moe_arrived` (waited above). Adding to it before the
        // push OVERWRITES the addition — which is exactly what happened: the
        // buffer measurably changed (34.71 -> 35.76) and the logits came out
        // BIT-IDENTICAL to not adding at all.
        if std::env::var("V41_REMOTE_DBG").is_ok() && bd.remote_ffn_moe.is_some() {
            // Does the exclusion actually remove mass from the local leg, and is
            // the remote's partial the right size to replace it? If the local
            // norm here matches a no-split run, the iGPU never skipped anything
            // and we are double-counting; if it dropped but the sum is still
            // wrong, the two sets are not complementary.
            let n = (b as usize) * N_EMBD as usize;
            let mut local = vec![0f32; n];
            let mut rem = vec![0f32; n];
            de.compute.synchronize()?;
            bd.ffn_moe_recv.slice_view(0, n).copy_to_host(&mut local)?;
            if bd.remote_ffn_moe_valid {
                bd.remote_ffn_moe.as_ref().unwrap().slice_view(0, n).copy_to_host(&mut rem)?;
            }
            let l2 = |v: &[f32]| v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
            eprintln!(
                "[combine-dbg] L{} valid={} b={b} local_l2={:.4} remote_l2={:.4} ratio={:.3}",
                bd.remote_ffn_moe_layer, bd.remote_ffn_moe_valid,
                l2(&local), l2(&rem), l2(&rem) / l2(&local).max(1e-9),
            );
        }
        // Second half of the combine, AFTER the blocking wait above. Separate
        // bracket from `dgpu.ffn_combine.local` on purpose -- see the note there.
        let _t_combine_remote = de.events.stage("dgpu.ffn_combine.remote", &de.compute)?;
        if bd.remote_ffn_moe_valid {
            // Two-box split: the iGPU skipped every expert box 2 owns (its
            // remap entry was non-negative), so this partial is the rest of the
            // sum, not a duplicate. Cleared immediately: the buffer outlives the
            // layer and adding it twice would double-count.
            let _t = de.events.stage("k.ffn_combine.vec_add_remote", &de.compute)?;
            let remote = bd
                .remote_ffn_moe
                .as_ref()
                .expect("remote_ffn_moe_valid implies the buffer exists");
            de.vec_add.launch(
                &de.compute,
                &mut bd.ffn_moe_recv,
                remote,
                b * N_EMBD,
            )?;
            if std::env::var("V41_REMOTE_DBG").is_ok() {
                // Did the add actually change the buffer hc_post reads? If
                // after == before, the vec_add is dead and everything upstream
                // of it is irrelevant.
                let n = (b as usize) * N_EMBD as usize;
                let mut after = vec![0f32; n];
                de.compute.synchronize()?;
                bd.ffn_moe_recv.slice_view(0, n).copy_to_host(&mut after)?;
                let l2: f64 = after.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
                eprintln!("[combine-dbg] AFTER add: ffn_moe_recv l2={l2:.4}");
            }
            bd.remote_ffn_moe_valid = false;
        }
        if hot_active {
            // M61: + the dGPU's resident-expert MoE partial. Queued on
            // de.compute in pre_moe, so stream order already guarantees
            // it's complete here.
            let _t = de.events.stage("k.ffn_combine.vec_add_hot", &de.compute)?;
            let ffn_moe_dgpu = bd
                .hot_ffn_moe_dgpu
                .as_ref()
                .expect("hot_active implies bd.hot_ffn_moe_dgpu");
            de.vec_add.launch(
                &de.compute,
                &mut bd.ffn_moe_recv,
                ffn_moe_dgpu,
                b * N_EMBD,
            )?;
        }
        {
            let _t = de.events.stage("k.ffn_combine.hc_post", &de.compute)?;
            de.hc_post.launch_from_split_batched(
                &de.compute,
                &mut bd.residual_next,
                &bd.ffn_moe_recv,
                &bd.after_attn_hc,
                &bd.split,
                N_HC,
                N_EMBD,
                N_HC,
                b,
            )?;
        }
        Ok(())
    }
}

/// Host wall inside the per-layer verify/prefill body, split by phase.
///
/// The perfetto gap analysis put 20.8 ms/layer between
/// `k.shared_expert.down_matvec` and `k.ffn_combine.vec_add` with every DEVICE
/// track idle and the real work at ~5.6 ms/layer — i.e. ~18 ms/layer running
/// nowhere and covered by no `events.stage()` scope. These attribute it to the
/// three host calls the loop actually makes. `V41_LAYER_HOST_TIMING=1`.
pub static LH_POST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LH_PRE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LH_ENGRAM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LH_PAGER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LH_SEL_SYNC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LH_REMOTE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Sub-timers INSIDE the pager block. The perfetto trace puts ~15 ms/layer of
/// host gap there while the iGPU's whole MoE chain is ~150 us and box 2 answers
/// in ~150 us, so the block's own breakdown is what names the owner.
pub static LH_ENSURE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LH_OWNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LH_EXCL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LH_AUDIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LH_REMAP_H2D: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// iGPU work-item readback sync (one per layer, MoE dispatch) and the exposed
/// box-2 wait, so a caller reading the counters sees the whole per-layer host
/// serialization.
pub static LH_WORK_ITEMS_SYNC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LH_REMOTE_WAIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Inside `lh.pager_block`, previously untimed (profile audit 2026-09-21):
/// the `V41_PAGER_SYNC_IGPU` cross-lane iGPU drain, the D2H copies after
/// `sel_sync`, and the dGPU drain + D2H inside `lh.remote_submit`.
pub static LH_PAGER_SYNC_IGPU: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Exposed part of the Engram SSD gather: time the first Engram layer waited
/// for the helper thread (`LazyEngramRows::get`).
pub static LH_ENGRAM_JOIN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// kwide path: iGPU drain + D2H of the work-item count before the MoE launch.
pub static LH_WORK_ITEMS_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LH_SEL_D2H: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LH_REMOTE_SYNC: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Programmatic switch (the multistream profile turns it on): OR-ed with the env.
pub static LH_FORCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn layer_host_timing() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    LH_FORCE.load(std::sync::atomic::Ordering::Relaxed)
        || *V.get_or_init(|| std::env::var("V41_LAYER_HOST_TIMING").as_deref() == Ok("1"))
}

/// Read-and-clear every LH_* counter (microseconds), for a caller that folds
/// them into its own per-step profile (`multistream` "ms.stage").
pub fn take_layer_host_timing() -> Vec<(&'static str, u64)> {
    use std::sync::atomic::Ordering::Relaxed;
    vec![
        ("lh.pre_moe", LH_PRE.swap(0, Relaxed)),
        ("lh.post_moe", LH_POST.swap(0, Relaxed)),
        ("lh.engram", LH_ENGRAM.swap(0, Relaxed)),
        ("lh.pager_block", LH_PAGER.swap(0, Relaxed)),
        ("lh.sel_sync", LH_SEL_SYNC.swap(0, Relaxed)),
        ("lh.remote_submit", LH_REMOTE.swap(0, Relaxed)),
        ("lh.ensure", LH_ENSURE.swap(0, Relaxed)),
        ("lh.owns", LH_OWNS.swap(0, Relaxed)),
        ("lh.excl", LH_EXCL.swap(0, Relaxed)),
        ("lh.audit", LH_AUDIT.swap(0, Relaxed)),
        ("lh.remap_h2d", LH_REMAP_H2D.swap(0, Relaxed)),
        ("lh.work_items_sync", LH_WORK_ITEMS_SYNC.swap(0, Relaxed)),
        ("lh.remote_wait", LH_REMOTE_WAIT.swap(0, Relaxed)),
        ("lh.pager_sync_igpu", LH_PAGER_SYNC_IGPU.swap(0, Relaxed)),
        ("lh.engram_join", LH_ENGRAM_JOIN.swap(0, Relaxed)),
        ("lh.work_items_count", LH_WORK_ITEMS_COUNT.swap(0, Relaxed)),
        ("lh.sel_d2h", LH_SEL_D2H.swap(0, Relaxed)),
        ("lh.remote_sync", LH_REMOTE_SYNC.swap(0, Relaxed)),
    ]
}

pub struct LayerHostTimer {
    t0: std::time::Instant,
    acc: &'static std::sync::atomic::AtomicU64,
}

impl LayerHostTimer {
    pub fn start(acc: &'static std::sync::atomic::AtomicU64) -> Option<Self> {
        layer_host_timing().then(|| Self { t0: std::time::Instant::now(), acc })
    }
}

impl Drop for LayerHostTimer {
    fn drop(&mut self) {
        self.acc.fetch_add(
            self.t0.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

/// Emit and clear; call once per verify/prefill.
///
/// UNITS: every counter here is a BOTH-LANE total for the whole call, and the
/// `*_per_layer_us` fields divide by `layers`, so they are "per layer, both
/// lanes". They were NOT comparable before 2026-09-16: LH_POST/LH_PRE/LH_ENGRAM
/// wrapped lane A only (and skipped the warm-up and cool-down layers entirely)
/// while LH_PAGER/LH_SEL_SYNC/LH_ENSURE/LH_REMOTE live inside
/// `forward_layer_pre_moe_v2` and always counted both lanes -- which is how a
/// report could show `pager_ms 102.0 > pre_moe_ms 66.2` with the pager nested
/// INSIDE pre_moe. If you are comparing against a number recorded before that
/// fix, the pre/post figures there are roughly half of the real total.
///
/// Note `pager_ms` ENCLOSES `sel_sync_ms`; they are not additive. And `sel_sync`
/// is not overhead -- it is the host waiting for the dGPU to execute the layer's
/// attention/router/shared-expert chain, and measures about the same as the
/// dGPU's busy time per lane-layer.
pub fn emit_layer_host_timing(tag: &str, layers: usize) {
    if !layer_host_timing() {
        return;
    }
    use std::sync::atomic::Ordering::Relaxed;
    let (post, pre, eng) = (
        LH_POST.swap(0, Relaxed),
        LH_PRE.swap(0, Relaxed),
        LH_ENGRAM.swap(0, Relaxed),
    );
    let (pager, sel_sync, remote) = (
        LH_PAGER.swap(0, Relaxed),
        LH_SEL_SYNC.swap(0, Relaxed),
        LH_REMOTE.swap(0, Relaxed),
    );
    tracing::info!(
        tag,
        layers,
        post_moe_ms = format!("{:.1}", post as f64 / 1000.0),
        pre_moe_ms = format!("{:.1}", pre as f64 / 1000.0),
        engram_ms = format!("{:.1}", eng as f64 / 1000.0),
        post_per_layer_us = post / layers.max(1) as u64,
        pre_per_layer_us = pre / layers.max(1) as u64,
        pager_ms = format!("{:.1}", pager as f64 / 1000.0),
        sel_sync_ms = format!("{:.1}", sel_sync as f64 / 1000.0),
        remote_ms = format!("{:.1}", remote as f64 / 1000.0),
        ensure_ms = format!("{:.1}", LH_ENSURE.swap(0, Relaxed) as f64 / 1000.0),
        owns_ms = format!("{:.1}", LH_OWNS.swap(0, Relaxed) as f64 / 1000.0),
        excl_ms = format!("{:.1}", LH_EXCL.swap(0, Relaxed) as f64 / 1000.0),
        audit_ms = format!("{:.1}", LH_AUDIT.swap(0, Relaxed) as f64 / 1000.0),
        remap_h2d_ms = format!("{:.1}", LH_REMAP_H2D.swap(0, Relaxed) as f64 / 1000.0),
        "prefill.layer_host"
    );
}


/// `V41_SMALL_B_CATCHALL_MAX=N`: for chunks of N rows or fewer, box 1 computes
/// only the picks it ALREADY HOLDS and hands every miss to box 2, which pages it
/// from its own disk — decode's T2 catch-all rule, applied to the speculative
/// verify. Default 0 (off) so it A/Bs in one binary.
///
/// This only works because the hub masks box 2's pick list itself (see
/// `sel_for_remote`): `submit`'s own mask is the static HELLO bitmap and would
/// silently drop every reassigned pick.
/// Live value of the small-B catch-all threshold.
///
/// RUNTIME-SETTABLE (not a `OnceLock`) so both arms can be interleaved inside
/// ONE process against ONE weight load. Under the shadow probe that is a true
/// control: DECODE drives the generated text, so flipping this per step cannot
/// change WHAT is generated, only whether the batched verify reproduces it.
/// Every earlier A/B of this flag compared separate servers and was confounded
/// both by cold start and by its own effect on the text it was scored against.
/// `usize::MAX` means "not yet seeded from the environment".
static SMALL_B_CATCHALL: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

pub fn small_b_catchall_max() -> usize {
    use std::sync::atomic::Ordering::Relaxed;
    let v = SMALL_B_CATCHALL.load(Relaxed);
    if v != usize::MAX {
        return v;
    }
    let seed = std::env::var("V41_SMALL_B_CATCHALL_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    SMALL_B_CATCHALL.store(seed, Relaxed);
    seed
}

/// Set the small-B catch-all threshold at runtime. See `small_b_catchall_max`.
pub fn set_small_b_catchall_max(v: usize) {
    SMALL_B_CATCHALL.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Live value of `V41_PREFILL_SINGLE_LANE_MAX`. Runtime-settable for the same
/// reason as `small_b_catchall_max`: so the arms interleave in ONE process.
static SINGLE_LANE_MAX: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

pub fn single_lane_max() -> usize {
    use std::sync::atomic::Ordering::Relaxed;
    let v = SINGLE_LANE_MAX.load(Relaxed);
    if v != usize::MAX {
        return v;
    }
    let seed = std::env::var("V41_PREFILL_SINGLE_LANE_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    SINGLE_LANE_MAX.store(seed, Relaxed);
    seed
}

/// Set the single-lane threshold at runtime. See `single_lane_max`.
pub fn set_single_lane_max(v: usize) {
    SINGLE_LANE_MAX.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Set for the duration of a DSpark verify so the post-attention pass leaves the
/// raw window exactly where the caller's `KvMark` addresses it (no compaction,
/// no raw_off reset). See the eviction block and `KvMark::advanced_by`.
/// (layer << 32 | pos) of the batched layer whose sub-tensor dump is armed, set
/// by `forward_layer_pre_moe_v2`, consumed by `forward_layer_post_moe_v2` for the
/// routed-only dump. Dump-only plumbing; `u64::MAX` = nothing pending.
static DUMP_LAYER_POS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

static SPECULATIVE_APPEND: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn speculative_append() -> bool {
    SPECULATIVE_APPEND.load(std::sync::atomic::Ordering::Relaxed)
}

/// Scope guard: sets the speculative-append flag, clears it on drop.
pub struct SpeculativeAppend;
impl SpeculativeAppend {
    pub fn begin() -> Self {
        SPECULATIVE_APPEND.store(true, std::sync::atomic::Ordering::Relaxed);
        Self
    }
}
impl Drop for SpeculativeAppend {
    fn drop(&mut self) {
        SPECULATIVE_APPEND.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

/// `V41_SPARSE_VERIFY_RESIDENCY=0` reverts the verify to prefill's dense-window
/// pager (see the call site).
/// `V41_PREFILL_UNIFIED_POOL=1`: run PREFILL through the same sparse/LRU
/// residency the verify uses, instead of the dense per-layer windows.
///
/// Windows reserve `windows * stride` slots that sit idle through decode (the
/// two phases never overlap), and a window can only see its OWN contents -- so
/// an expert decode already holds is invisible to prefill and gets re-read from
/// NVMe at 6-8 ms. MEASURED on a real agent turn: 5,132-7,152 prefill misses and
/// 22.7-34.7 s of blocking box-1 reads per request. The unified pool makes every
/// resident expert a free hit; `V41_PREFILL_SCAN_SLOTS` stops the scan evicting
/// decode's warm set by confining prefill's MISSES to a small region.
///
/// Needs the per-layer `remap_dev`: one shared buffer made the sparse path race
/// under the two-lane driver, which is why this could not exist before.
pub fn prefill_unified_pool() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("V41_PREFILL_UNIFIED_POOL").as_deref() == Ok("1")
    });
    *B
}

/// Slots a prefill scan may allocate from. DEFAULT UNBOUNDED (the whole pool).
///
/// The scan-resistance argument assumes prefill is a FOREIGN scan whose
/// contents the reuse workload does not want. That is false here: turn N+1's
/// prefill is the suffix of the same conversation decode just generated from,
/// so the two want largely the SAME experts, and LRU recency already keeps the
/// right ones (the last chunks prefilled are exactly what decode needs next).
/// Bounding the scan therefore costs thrash without buying protection —
/// MEASURED at 768 slots: ~6,840 distinct experts per chunk cycling through
/// 768 slots, 18,004 misses on a 17.5K-token prefill, ~126 s of the 156 s wall.
///
/// Kept as a knob for a genuinely foreign scan (e.g. a cold unrelated prompt
/// served between turns of a live conversation).
pub fn prefill_scan_slots() -> usize {
    static N: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("V41_PREFILL_SCAN_SLOTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(usize::MAX)
    });
    *N
}

pub fn sparse_verify_residency_off() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("V41_SPARSE_VERIFY_RESIDENCY").as_deref() == Ok("0")
    });
    *B
}

/// `V41_GROUP_AUDIT=1`: verify the het-split builder enqueued every local pick.
pub fn group_audit() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("V41_GROUP_AUDIT").as_deref() == Ok("1"))
}

/// `V41_SMALL_B_CATCHALL_DET=1`: the small-B catch-all hands box 2 every pick
/// instead of only the ones box 1 is missing, making the split history-
/// independent. See the call site.
pub fn small_b_catchall_det() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("V41_SMALL_B_CATCHALL_DET").as_deref() == Ok("1"))
}

/// `V41_COMP_POSITIONAL`: 1 = log and override `n_comp_after` with the
/// positional formula, 2 = log only, 0/unset = off.
pub fn comp_positional() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("V41_COMP_POSITIONAL").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
    })
}

/// Count of rows whose `n_comp_after` disagreed with the positional formula.
pub static COMP_POS_MISMATCH: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `V41_LAYER_MISS_HIST=1`: per-layer expert-miss histogram for one verify.
pub static LAYER_MISS: [std::sync::atomic::AtomicU64; 40] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 40];

pub fn layer_miss_hist() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("V41_LAYER_MISS_HIST").as_deref() == Ok("1"))
}

/// Emit and clear, split at `CED_DECODER_START` — box 1 pins encoder windows
/// only, so a verify's misses should be concentrated in the decoder half.
pub fn emit_layer_miss_hist(tag: &str) {
    if !layer_miss_hist() {
        return;
    }
    use std::sync::atomic::Ordering::Relaxed;
    let v: Vec<u64> = LAYER_MISS.iter().map(|a| a.swap(0, Relaxed)).collect();
    let split = crate::config::CED_DECODER_START;
    let enc: u64 = v[..split].iter().sum();
    let dec: u64 = v[split..].iter().sum();
    let tot = enc + dec;
    if tot == 0 {
        return;
    }
    tracing::info!(
        tag,
        encoder_misses = enc,
        decoder_misses = dec,
        decoder_pct = format!("{:.1}", 100.0 * dec as f64 / tot as f64),
        "prefill.layer_miss_hist"
    );
}

/// `V41_VERIFY_DECODE_MOE=1`: recompute the batched path's local MoE with the
/// DECODE kernels, so a speculative verify produces the logits decode would.
pub fn verify_decode_moe() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("V41_VERIFY_DECODE_MOE").as_deref() == Ok("1"))
}

/// `V41_VERIFY_DECODE_ATTN=1`: replay DECODE's attention chain per row in the
/// batched driver, so a speculative verify produces the attention decode would.
pub fn verify_decode_attn() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("V41_VERIFY_DECODE_ATTN").as_deref() == Ok("1"))
}

/// Draw box 2's own paging for one request on the `remote.pager` lane.
///
/// Box 2 reports only a DURATION (proto v2 `t_page_us`), not timestamps, so the
/// position is reconstructed: the request lands on box 2 about half a link-time
/// after it was sent, and `ensure_layer*` runs BEFORE the MoE compute, so the
/// paging sits at the front of box 2's service window. The DURATION, the LAYER
/// and the MISS COUNT are exact; treat the sub-request placement as indicative.
pub(crate) fn emit_remote_page_slice(
    pf: &super::perfetto::DeviceTimingExporter,
    layer: u32,
    partial: &super::remote_experts::RemotePartial,
    t_wait_end: u64,
) {
    if partial.t_remote_page_us == 0 {
        return;
    }
    let rtt_ns = partial.rtt_us as u64 * 1_000;
    let half_link_ns = partial.link_us() as u64 * 500;
    let start = t_wait_end.saturating_sub(rtt_ns).saturating_add(half_link_ns);
    let end = start + partial.t_remote_page_us as u64 * 1_000;
    let _ = pf.emit_host_slice(
        pf.remote_pager_uuid,
        &format!(
            "page L{layer} {}us / {} miss",
            partial.t_remote_page_us, partial.n_remote_miss
        ),
        start,
        end,
    );
}


/// Engram rows for one decode step: gathered on a helper thread while the step
/// runs layer 0, joined at the first Engram layer. The synchronous gather was
/// 13.8 ms/step at 8 rows (5-26), serial on the hub thread before the forward
/// (profile audit 2026-09-21); layers 1 and 14 are the only consumers, so the
/// SSD reads can hide under layer 0. `get()` joins at most once and counts the
/// exposed wait into `lh.engram_join`.
pub struct LazyEngramRows<'scope> {
    ready: Option<Vec<Vec<f32>>>,
    pending: Option<std::thread::ScopedJoinHandle<'scope, eyre::Result<Vec<Vec<f32>>>>>,
}

impl<'scope> LazyEngramRows<'scope> {
    /// Rows already in hand (or `None` when the model has no Engram layers).
    pub fn ready(rows: Option<Vec<Vec<f32>>>) -> Self {
        Self { ready: rows, pending: None }
    }
    /// Rows still being gathered on a scoped thread.
    pub fn pending(h: std::thread::ScopedJoinHandle<'scope, eyre::Result<Vec<Vec<f32>>>>) -> Self {
        Self { ready: None, pending: Some(h) }
    }
    pub fn get(&mut self) -> eyre::Result<Option<&[Vec<f32>]>> {
        if let Some(h) = self.pending.take() {
            let t = std::time::Instant::now();
            let rows = h.join().map_err(|_| eyre!("engram gather thread panicked"))??;
            if layer_host_timing() {
                LH_ENGRAM_JOIN.fetch_add(t.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            }
            self.ready = Some(rows);
        }
        Ok(self.ready.as_deref())
    }
}
