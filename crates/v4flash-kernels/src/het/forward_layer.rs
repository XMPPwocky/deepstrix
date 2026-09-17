//! Per-layer dispatch for the het orchestrator.
//!
//! Pipeline (event-driven overlap mode — `ExecMode::HetParallel`):
//!
//!  1. **dGPU**: mHC pre-attn → attn_cur → attn_input_norm
//!  2. **dGPU**: Q chain + KV chain
//!  3. **dGPU**: KV cache push (FP8 + F16-roundtrip + SWA slide)
//!  4. **dGPU** (ratio>0): compressor step + boundary fire → comp_kv
//!  5. **dGPU**: attention (swa or mixed) → heads → attn_out
//!  6. **dGPU**: mHC post-attn → after_attn_hc
//!  7. **dGPU**: mHC pre-ffn → ffn_cur → ffn_input_norm
//!  8. **dGPU**: router → d_selected / d_ew
//!  9. **dGPU→iGPU**: peer push ffn_input_norm + d_selected/d_ew (on dGPU.xfer)
//! 10. **iGPU**: routed MoE → ffn_moe
//! 11. **iGPU→dGPU**: peer push ffn_moe → ffn_moe_recv
//! 12. **dGPU**: shared expert → ffn_shared (overlaps with 9-11)
//! 13. **dGPU**: ffn_moe_recv += ffn_shared then mHC post-ffn → residual_next
//!
//! `ExecMode::HetSingleStream` is a correctness oracle that runs the same
//! pipeline with `.synchronize()` between every kernel.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{Device, DeviceBuffer};

use crate::config::{
    BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, EXPERT_WEIGHT_SCALE, GROUP_DIM, HC_DIM, HC_MIX_DIM,
    INDEXER_COMP_WIDTH, INDEXER_TOP_K, N_EMBD, N_EXPERT, N_EXPERT_USED, N_FF_EXP, N_FF_SHARED,
    N_GROUPS, N_HC, N_HEAD, N_HEAD_DIM, N_INDEXER_HEAD, N_INDEXER_HEAD_DIM, N_LORA_Q, N_ROT,
    OUT_LOW, Q_FLAT, RANK, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS, SWA_WINDOW, SWIGLU_CLAMP_EXP,
};
use crate::routing::hash_router_select;
use crate::q8_k::BLOCK_Q8_K_BYTES;

/// Floor for the router-weight sum, mirroring the host topk path (f16
/// epsilon).
const ROUTER_WEIGHT_EPS: f32 = 6.103515625e-5;

use super::engine::{DeviceEngine, ExecMode, HeterogeneousEngine};
use super::scratch::{DgpuScratch, IgpuScratch};
use super::state::{CompKvStore, HetLayerState};
use crate::comp_kv_fp8::FP8_KV_HEAD_ROWS;
use crate::index_kv_e2m1::E2M1_KEY_ROW_BYTES;
use super::sync::{peer_push_f32, peer_push_i32};

/// Verify every routed pick is computed by EXACTLY ONE device.
///
/// The MoE partials from the iGPU pool, box 2, and (once wired) the dGPU hot
/// tier are summed with `vec_add` at `ffn_combine`. That sum is only correct if
/// the three claim sets partition the token's picks. Nothing in the types
/// Lever 1 — shrink the pre-`submit` critical path (`V41_DECODE_PRESUBMIT=1`).
///
/// Decode hands box 2 its work in `submit`, but everything box 1 queues on
/// `de.compute` BEFORE the `synchronize()` that reads back the router's picks is
/// time box 2 spends idle. Two things sit there for no reason:
///
///   * the shared-expert graph (~97 us/layer measured) — its result is not
///     consumed until `ffn_combine`, so it can run AFTER the submit and overlap
///     box 2 instead of preceding it;
///   * `q8k(ffn_input_norm) -> moe_xq`, which needs only `ffn_input_norm` (the
///     router's own input) and so can be enqueued before the FIRST sync. That
///     removes the SECOND full `de.compute.synchronize()` per layer outright.
///
/// Kept behind an env flag so both arms live in ONE binary: a `cargo build -p
/// v4flash-kernels` does not relink deepstrix-server, and an A/B across two
/// builds has already produced one fabricated result in this project.
///
/// MEASURED 2026-09-14, back-to-back in one binary, 3x256-token decode each:
///     off: 10.70 / 10.86 tok/s   on: 11.34 / 11.30 tok/s   (+5.0%)
///     total_us 59,857 -> 57,203      sel_sync_us 27,313 -> 24,258 (-3,055)
///     remote_rtt_us 21,567 -> 22,647 (+1,080)  pager_ensure_us unchanged
/// Output was BYTE-IDENTICAL across all six runs (sha b7f58f53b529) — the
/// reorder only moves work nothing downstream had consumed yet.
///
/// Note the shape of the win: sel_sync drops the predicted ~3 ms but rtt rises
/// ~1 ms, because box 2 starting earlier just means the hub waits for it longer.
/// Net -2.65 ms/token, ~half the -5 to -7 predicted. While box 2 is the
/// bottleneck, THAT is the ceiling for any pre-submit reordering; the rest of
/// the 23 ms expert phase needs picks taken off box 2, not scheduled earlier.
///
/// Default ON since the A/B above; `V41_DECODE_PRESUBMIT=0` rolls back.
/// `V41_INDEX_K=1`: compute the V4.1 CSA2 index keys during the compressor step.
/// Default OFF. Inert while on — nothing reads `HetCompressorState::index_k` until S1
/// flips the sparse gate — so it is safe to enable for numerical validation.
/// Mirror of `het::weights::is_index_source` (private there).
#[cfg(feature = "v41")]
pub(crate) fn is_index_source_layer(layer: i32) -> bool {
    crate::config::INDEX_SOURCE_LAYERS.contains(&layer)
}
#[cfg(not(feature = "v41"))]
pub(crate) fn is_index_source_layer(_layer: i32) -> bool {
    false
}

/// `V41_LOCAL_PICKS=N`: force box 1 to claim N of the 6 routed picks per layer
/// REGARDLESS of residency (it pages what it lacks), handing the rest to box 2.
///
/// Why this exists: the expert phase costs **max(box1, box2)**, not their sum — box 1's
/// iGPU MoE is issued between `submit` and `wait`, so the two run concurrently. With
/// box1(n) = 73n us and box2(m) = 220 + 95m us, the optimum is n~5 at 365 us/layer
/// versus 790 today, i.e. -17 ms/token. The pool sweep could not test this because box 1
/// claims only what it ALREADY HOLDS, and its LRU fills by first touch (coverage ~
/// slots/384), so extra capacity went unused. This forces n directly.
///
/// NOTE the confound: forcing n also forces PAGING on box 1's dm-crypt NVMe, which lands
/// on the critical path via `ensure`. Read `pager_ensure_us` alongside `remote_rtt_us`.
fn local_picks_override() -> Option<usize> {
    static N: std::sync::LazyLock<Option<usize>> = std::sync::LazyLock::new(|| {
        std::env::var("V41_LOCAL_PICKS").ok().and_then(|v| v.parse::<usize>().ok())
    });
    *N
}

/// `V41_LOCAL_CLAIM_MAX=N`: cap how many routed picks box 1 computes locally per layer,
/// handing the surplus to box 2 even when box 1 HOLDS them.
///
/// This is a PRECONDITION for better placement, not a refinement. The expert phase costs
/// max(box1, box2); with c1 ~= 98 us/pick and box2(m) = 80 + 95m, f(n) for n=0..6 is
/// 650/555/460/365/392/490/588 — it turns UP after n=3. Measured on the decode trace
/// (E[f(n)], warm half, 20 encoder layers): a warm-oracle 116-expert share costs
/// **11.19 ms/token uncapped — WORSE than today's 9.99** because box 1 becomes the long
/// pole at n~5.5; capped at 3 the same placement is 7.31 ms. Residency and claiming are
/// separate decisions: hold as much as you like, compute only n*.
fn local_claim_max() -> Option<usize> {
    static N: std::sync::LazyLock<Option<usize>> = std::sync::LazyLock::new(|| {
        std::env::var("V41_LOCAL_CLAIM_MAX").ok().and_then(|v| v.parse::<usize>().ok())
    });
    *N
}

fn index_k_enabled() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        matches!(std::env::var("V41_INDEX_K").as_deref(), Ok("1") | Ok("on"))
    });
    *B
}

fn decode_presubmit_reorder() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !matches!(std::env::var("V41_DECODE_PRESUBMIT").as_deref(), Ok("0") | Ok("off"))
    });
    *B
}

/// enforces it: the pager's remap, box 2's ownership bitmap and the hot remap
/// are built independently, and a per-device CAP can silently drop a pick
/// (nobody computes it) or hand it to a second device by raw id (computed
/// twice). Both failure modes are SILENT — wrong logits, no error. The M63
/// over-cap bug and this session's `hot_prefill_cap()=4` vs `N_EXPERT_USED=6`
/// bug were both of exactly this shape.
///
/// `remap` is the iGPU view: NEGATIVE = the iGPU owns it (slot `-e-1`),
/// non-negative = some other device's.
///
/// UNCONDITIONAL, deliberately. This is <= 6 integer comparisons per layer —
/// ~240 per token against a ~172 ms token, i.e. unmeasurable. An env gate would
/// only mean the check was OFF in production, which is the one place a silent
/// double-count actually costs anything.
pub fn verify_routing_exactly_once(
    layer: i32,
    sel: &[i32],
    remap: &[i32],
    owns_remote: Option<&[bool]>,
) -> eyre::Result<()> {
    // Bitmap, not `Vec::contains`: decode passes 6 picks but PREFILL passes B*6
    // (6144 at B=1024), and a linear scan per pick would be ~18M comparisons per
    // layer per chunk. O(n) either way now.
    let mut seen = [false; N_EXPERT as usize];
    for &e in sel {
        if !(0..N_EXPERT as i32).contains(&e) || seen[e as usize] {
            continue; // padding / duplicate pick: the pager dedups these too
        }
        seen[e as usize] = true;
        let igpu = remap[e as usize] < 0;
        let remote = owns_remote.is_some_and(|o| o[e as usize]);
        let claims = [("igpu", igpu), ("box2", remote)];
        let n = claims.iter().filter(|(_, c)| *c).count();
        if n != 1 {
            let who: Vec<&str> = claims.iter().filter(|(_, c)| *c).map(|(w, _)| *w).collect();
            return Err(eyre!(
                "L{layer} expert {e}: computed by {n} devices {who:?} (need exactly 1).                  remap[{e}]={} owns_remote={remote}. A pick claimed twice is DOUBLE-COUNTED                  at ffn_combine; a pick claimed zero times is silently dropped.",
                remap[e as usize]
            ));
        }
    }
    Ok(())
}

/// Run V4.1's `hc_mixes` on a side stream instead of inline.
///
/// **DEFAULT OFF — MEASURED LOSS (2026-09-13).** Back-to-back, symbol-verified,
/// identical miss counts (11.11/token both arms) and bit-identical output:
///
/// | | prefill | decode | `sel_sync_us` |
/// |---|---|---|---|
/// | off | 154 tok/s | **6.95** | **37,333** |
/// | on  | 159 tok/s | 6.22 | 67,394 |
///
/// It made the metric it targets **80% worse**. The ARCHITECTURE reasoning was
/// right — ARCH_SPEC §1.1's shift really does take `hc_mixes` off this
/// sub-block's critical path — but a second HIP stream is the wrong mechanism:
/// two graph launches plus six cross-stream record/wait pairs per layer cost
/// more than the ~58 us/layer of mixes they hide. Same lesson as `MHC_FUSED`
/// (fusion lost to a 24-WG matvec): the mHC chain resists both fusion and
/// stream-splitting, and the win is not where the stage timing suggests. Kept
/// in tree, off, as a documented negative result; `V41_MHC_SPLIT=1` re-enables.
/// A cheaper exploit of the same shift would have to avoid per-layer
/// synchronisation entirely — e.g. batching the mixes for ALL layers into one
/// launch, which the shift also permits.
fn mhc_split_enabled() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        cfg!(feature = "v41")
            && !std::env::var("MHC_FUSED").map(|v| v != "0").unwrap_or(false)
            && std::env::var("V41_MHC_SPLIT").map(|v| v != "0").unwrap_or(false)
            && std::env::var("RMS_NW_MW").map(|v| v == "fused").unwrap_or(true)
    });
    *B
}
use crate::config::{ENGRAM_IN, ENGRAM_OUT};
use super::weights::{DgpuLayerWeights, IgpuLayerWeights};
use super::expert_pager::ExpertPager;
use tracing::debug_span;


/// `V41_ATTN_COMP_CAP=<n>`: clamp the DENSE compressed-attention row count to `n`.
///
/// A timing ablation for the missing V4.1 sparse indexer (docs/v41/DECODE_M8_PLAN.md
/// "Measured (2026-09-13)"): with no indexer, V4.1 decode scores + weight-sums over
/// every compressed row of every layer >= 2, which grows linearly with context. This
/// makes the kernels do exactly the `n`-row work a working top-`n` indexer would leave
/// them, so the prize is measurable before the indexer is built. Numerically WRONG
/// (wrong rows) — never set it on a correctness run. 0 (default) = off.
fn attn_comp_cap() -> u32 {
    static N: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        let v = std::env::var("V41_ATTN_COMP_CAP").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
        if v > 0 {
            tracing::warn!(cap = v, "V41_ATTN_COMP_CAP set: dense compressed attention is CLAMPED — output is invalid, timing ablation only");
        }
        v
    });
    *N
}

impl HeterogeneousEngine {
    /// Run one layer in the het pipeline. Reads from
    /// `dgpu_scratch.residual` and writes to `dgpu_scratch.residual_next`.
    ///
    /// `next_dlw` is `Some` for all layers except the last. When present,
    /// layer N's ffn_combine is fused with layer N+1's mhc_pre_attn into
    /// one captured graph (the combined-transition graph that closes the
    /// ~115 µs/layer host-scheduling gap). Correspondingly, mhc_pre_attn
    /// is launched standalone ONLY for layer 0 — every other layer's
    /// mhc_pre_attn rides the previous layer's combined graph.
    /// V4.1 Engram: stage one token's dequantised rows (`ENGRAM_IN` f32, from
    /// `v4flash_core::EngramTable::gather_position`) for the next Engram layer
    /// this scratch runs. Hashes are token-only, so this can happen as soon as
    /// the token is known (ENGINE_PORT.md M2: off the critical path).
    pub fn stage_engram_rows(&self, dgpu_scratch: &mut DgpuScratch, rows: &[f32]) -> eyre::Result<()> {
        if rows.len() != ENGRAM_IN as usize {
            return Err(eyre!("stage_engram_rows: {} floats, want {}", rows.len(), ENGRAM_IN));
        }
        self.set_current_cached(self.dgpu.device)?;
        dgpu_scratch.engram_rows.copy_from_host(rows)?;
        dgpu_scratch.engram_rows_ready = true;
        Ok(())
    }

    pub fn forward_layer(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        next_dlw: Option<&DgpuLayerWeights>,
        ilw: &IgpuLayerWeights,
        pos: u32,
        token_id: i32,
    ) -> eyre::Result<()> {
        self.forward_layer_impl(
            dgpu_scratch,
            igpu_scratch,
            ls,
            dlw,
            next_dlw,
            ilw,
            pos,
            token_id,
            false,
            None,
        )
    }

    /// Run one layer with the cross-layer combined ffn_combine →
    /// next-mhc_pre_attn graph disabled. Each layer fires its own
    /// standalone mhc_pre_attn + standalone ffn_combine instead.
    ///
    /// Used by diagnostic tests that need per-layer control (e.g. running
    /// only layer N against the activation dump) — the combined graph
    /// would speculatively launch the NEXT layer's mhc_pre_attn before
    /// the test can read the layer-N output.
    pub fn forward_layer_standalone_graphs(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        ilw: &IgpuLayerWeights,
        pos: u32,
        token_id: i32,
    ) -> eyre::Result<()> {
        // next_dlw unused; the standalone flag forces is_first_layer +
        // is_last_layer to true regardless.
        self.forward_layer_impl(
            dgpu_scratch,
            igpu_scratch,
            ls,
            dlw,
            None,
            ilw,
            pos,
            token_id,
            true,
            None,
        )
    }

    /// M7 phase-1 expert paging: like [`forward_layer_standalone_graphs`] but the
    /// routed MoE reads experts from `pager` (paged on demand from the router's
    /// actual picks) instead of a fully-resident `ilw.routed`. `dlw.hot_experts`
    /// must be None (no dGPU hot set) so every routed expert computes on the iGPU
    /// from the pager's pool.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_layer_standalone_graphs_paged(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        ilw: &IgpuLayerWeights,
        pos: u32,
        token_id: i32,
        pager: &mut ExpertPager,
    ) -> eyre::Result<()> {
        self.forward_layer_impl(
            dgpu_scratch,
            igpu_scratch,
            ls,
            dlw,
            None,
            ilw,
            pos,
            token_id,
            true,
            Some(pager),
        )
    }

    /// Same as [`forward_layer`] but assumes the layer's iGPU MoE command
    /// sequence was already enqueued by [`issue_igpu_moe`] (M54 pre-issue
    /// mode) — the impl skips its inline iGPU section. Parallel mode only.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_layer_preissued_moe(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        next_dlw: Option<&DgpuLayerWeights>,
        ilw: &IgpuLayerWeights,
        pos: u32,
        token_id: i32,
    ) -> eyre::Result<()> {
        self.forward_layer_impl_inner(
            dgpu_scratch,
            igpu_scratch,
            ls,
            dlw,
            next_dlw,
            ilw,
            pos,
            token_id,
            false,
            true,
            None,
        )
    }

    /// M54: enqueue one layer's complete iGPU MoE sequence (wait on
    /// `selected_pushed` → routed_moe graph → record `moe_done` → peer-push
    /// ffn_moe → record `moe_arrived`). Every command is event-gated, so
    /// the whole lane for all 43 layers can be pre-issued at token start —
    /// the decode pftrace showed ~125 µs/layer of host-submission lag
    /// (`routed_moe → moe.wait` gaps) when the iGPU commands were
    /// interleaved with the ~25 dGPU submissions per layer, and that lag
    /// lands directly on the dGPU's MoE wait (the critical path).
    ///
    /// Safety relies on stream ordering: ffn_combine(L) precedes
    /// router(L+1) on the dGPU compute stream, so `selected_pushed(L+1)`
    /// implies ffn_moe_recv(L) was consumed; the activations push precedes
    /// the selected push on the dGPU xfer stream.
    pub fn issue_igpu_moe(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        ilw: &IgpuLayerWeights,
    ) -> eyre::Result<()> {
        use tracing::debug_span;
        let layer = ilw.layer_idx;
        let sev = &self.sync_events.layers[layer as usize];
        self.set_current_cached(self.igpu.device)?;
        let ie = &self.igpu;

        {
            // Value-wait, NOT event-wait: this lane is enqueued at token
            // start, before the dGPU's record/write calls for the token.
            // hipStreamWaitEvent snapshots at call time (would no-op);
            // hipStreamWaitValue32 compares at execution time.
            let _t_wait = ie.events.stage("igpu.moe.wait", &ie.compute)?;
            let seq = self
                .token_seq
                .load(std::sync::atomic::Ordering::Relaxed);
            let sig = unsafe {
                (self.moe_signal.as_slice().as_ptr() as *mut u32).add(layer as usize)
            };
            unsafe { ie.compute.wait_value32_gte(sig, seq)? };
            _t_wait.end()?;
        }

        let gbpe = ilw.routed.gate_bytes_per_expert;
        let ubpe = ilw.routed.up_bytes_per_expert;
        let dbpe = ilw.routed.down_bytes_per_expert;
        let mid_blocks_bytes = (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES;

        let _t_moe = ie.events.stage("igpu.routed_moe", &ie.compute)?;
        let _s_moe = debug_span!("routed_moe").entered();
        self.igpu_graphs.run("routed_moe", layer as u32, &ie.compute, |s| {
            ie.q8k.launch(s, &mut igpu_scratch.d_xq_q8k, &igpu_scratch.ffn_input_norm_recv, BLOCKS_Q8K_GATE_IN)?;
            super::dispatch::moe_gate_up_batch(ie, ilw.routed.gate.dtype, s, &mut igpu_scratch.d_mid_cat, &ilw.routed.gate.buffer, &ilw.routed.up.buffer, &igpu_scratch.d_xq_q8k, &igpu_scratch.d_ew, &igpu_scratch.d_selected, gbpe as u32, ubpe as u32, N_EXPERT_USED as u32, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN)?;
            ie.q8k.launch(s, &mut igpu_scratch.d_midq_cat, &igpu_scratch.d_mid_cat, BLOCKS_Q8K_DOWN_IN * (N_EXPERT_USED as u32))?;
            super::dispatch::moe_down_batched(ie, ilw.routed.down.dtype, s, &mut igpu_scratch.ffn_moe, &ilw.routed.down.buffer, &igpu_scratch.d_midq_cat, &igpu_scratch.d_selected, dbpe as u32, mid_blocks_bytes as u32, N_EXPERT_USED as u32, N_EMBD, BLOCKS_Q8K_DOWN_IN)?;
            Ok(())
        })?;
        drop(_s_moe);
        _t_moe.end()?;

        sev.moe_done.record(&ie.compute)?;
        ie.xfer.wait_event(&sev.moe_done)?;
        let _t_peer_moe = ie.events.stage("igpu.peer_push_ffn_moe", &ie.xfer)?;
        peer_push_f32(
            &igpu_scratch.ffn_moe,
            &mut dgpu_scratch.ffn_moe_recv,
            &ie.xfer,
        )?;
        sev.moe_arrived.record(&ie.xfer)?;
        _t_peer_moe.end()?;
        Ok(())
    }

    fn forward_layer_impl(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        next_dlw: Option<&DgpuLayerWeights>,
        ilw: &IgpuLayerWeights,
        pos: u32,
        token_id: i32,
        standalone_graphs: bool,
        pager: Option<&mut ExpertPager>,
    ) -> eyre::Result<()> {
        self.forward_layer_impl_inner(
            dgpu_scratch,
            igpu_scratch,
            ls,
            dlw,
            next_dlw,
            ilw,
            pos,
            token_id,
            standalone_graphs,
            false,
            pager,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_impl_inner(
        &self,
        dgpu_scratch: &mut DgpuScratch,
        igpu_scratch: &mut IgpuScratch,
        ls: &mut HetLayerState,
        dlw: &DgpuLayerWeights,
        next_dlw: Option<&DgpuLayerWeights>,
        ilw: &IgpuLayerWeights,
        pos: u32,
        token_id: i32,
        standalone_graphs: bool,
        igpu_moe_preissued: bool,
        mut pager: Option<&mut ExpertPager>,
    ) -> eyre::Result<()> {
        let layer = dlw.layer_idx;
        if ilw.layer_idx != layer {
            return Err(eyre!(
                "het forward_layer: dgpu layer {} != igpu layer {}",
                layer,
                ilw.layer_idx
            ));
        }
        let ratio = dlw.ratio;
        // `standalone_graphs` forces standalone mhc_pre_attn + standalone
        // ffn_combine each layer (i.e. bypasses the cross-layer combined
        // graph). Required when the caller needs to observe layer-N
        // output before layer N+1 starts.
        // An Engram layer must see the residual AFTER the gate+add, so it always
        // launches its own mhc_pre_attn (and the layer before it a pure combine).
        let is_first_layer = standalone_graphs || layer == 0 || dlw.engram.is_some();
        let is_last_layer = standalone_graphs || next_dlw.is_none() || next_dlw.is_some_and(|n| n.engram.is_some());

        let serial = matches!(self.mode, ExecMode::HetSingleStream);

        let _layer_span = debug_span!("het.layer", layer, pos).entered();

        // ============================================================
        // dGPU: mHC pre attn → attn_cur → attn_input_norm
        // ============================================================
        self.set_current_cached(self.dgpu.device)?;
        let de = &self.dgpu;
        if cfg!(feature = "v41") && layer == 0 {
            // Single-pass mHC: every token's layer-0 attention collapses with
            // the initial one-hot(copy 0) pre-mix (ARCH_SPEC §1.1).
            dgpu_scratch.hc_pre_carry.copy_from_host_async(&super::scratch::HC_PRE_ONEHOT, &de.compute)?;
        }
        if let Some(eg) = dlw.engram.as_ref() {
            // V4.1 Engram (ARCH_SPEC §1.7), applied to the residual copies before
            // the block: staged rows → Q8 → wkv → gate + add (engram_gate_add.hip).
            if !dgpu_scratch.engram_rows_ready {
                return Err(eyre!("layer {layer}: Engram rows not staged (stage_engram_rows before the layer)"));
            }
            let _t = de.events.stage("dgpu.engram", &de.compute)?;
            let s = &de.compute;
            de.q8.quantize_input(s, &mut dgpu_scratch.engram_xq, &mut dgpu_scratch.engram_xscale, &dgpu_scratch.engram_rows, ENGRAM_IN)?;
            de.q8.matvec(s, &mut dgpu_scratch.engram_kv, &eg.wkv.buffer, &dgpu_scratch.engram_xq, &dgpu_scratch.engram_xscale, ENGRAM_OUT, ENGRAM_IN)?;
            de.engram_gate.launch(s, &mut dgpu_scratch.residual, &dgpu_scratch.engram_kv, &eg.qk, N_HC, N_EMBD, ENGRAM_OUT, N_HC * N_EMBD, RMS_EPS, 1)?;
            dgpu_scratch.engram_rows_ready = false;
        }

        // mhc_pre_attn for layer N is launched by the PREVIOUS layer's
        // combined ffn_combine→mhc_pre_attn graph, except for layer 0
        // which has no preceding ffn_combine. The standalone mhc_pre_attn
        // graph below only fires once per token (layer 0).
        if is_first_layer {
            let _t_mhc_pre = de.events.stage("dgpu.mhc_pre_attn", &de.compute)?;
            let _s_mhc_pre = debug_span!("mhc_pre_attn").entered();
            // Single fused kernel replaces the 5-kernel chain. Wrapped in
            // the same graph-replay path so launch is amortized identically
            // to the original chain — the win (if any) comes from the
            // single in-WG pipeline instead of 5 short kernels.
            // ENV MHC_FUSED=0 rolls back to the 5-kernel chain.
            // MHC_FUSED=1 enables the fused mhc_pre_fused kernel. Default
            // OFF: the 5-kernel chain in a captured graph is FASTER than
            // the fused single-WG version because the standalone f16.matvec
            // uses 24 WGs (~24 CUs) for the HC_MIX_DIM=24 outputs while the
            // single-WG fused kernel is limited to 1 CU. Graph capture
            // already amortizes launch overhead to ~1 µs/kernel. The
            // chain's bottleneck was matvec WORK, not launch overhead —
            // our "12 ms/tok launch-overhead-bound" estimate was wrong.
            // The kernel is kept in tree for the rollback and as documented
            // negative result.
            let mhc_fused = std::env::var("MHC_FUSED")
                .map(|v| v != "0").unwrap_or(false);
            // V4.1 single-pass mHC: run `hc_mixes` on a SIDE STREAM.
            //
            // ARCH_SPEC §1.1 collapses the 4 copies with the PREVIOUS sub-block's
            // `pre` (`hc_pre_carry`), NOT with the `split` this sub-block is about
            // to compute. So `hc_weighted` -> `rms_w` -> attention does not depend
            // on the mixes at all; the mixes are consumed by `hc_post` (after
            // output_proj) and by the next sub-block's collapse. That shift is the
            // point of the V4.1 design, and running the mixes inline throws it away:
            // measured 58.3 us/layer of mhc_pre_attn + 59.6 us/layer of mhc_pre_ffn
            // = 4.7 ms/token sitting on the critical path in front of ~413 us/layer
            // of q/kv/attn/output_proj that it could hide behind.
            //
            // The mixes are ~9.5x off their byte roofline ([24,20480] fp32 = 1.97 MB
            // = 6.1 us at 640 GB/s) because `hc_split_sinkhorn` is 20 SEQUENTIAL
            // 4x4 normalisation iterations — a latency chain, not a bandwidth
            // problem. That is why MHC_FUSED lost (1 CU vs 24 WGs): fusion was the
            // wrong axis. Don't make it faster, take it off the critical path.
            //
            // Ordering (host enqueue order matters — `wait_event` snapshots the
            // event's LAST RECORD at call time and is a silent no-op if enqueued
            // ahead of the record, see Stream::wait_event):
            //   compute: record hc_src        (residual final)
            //   hc:      wait hc_src; mixes -> split
            //   compute: collapse (reads OLD carry); record hc_collapse
            //   hc:      wait hc_collapse; carry := split[0..N_HC]; record hc_mixes
            //   compute: wait hc_mixes before hc_post consumes `split`
            // V41_MHC_SPLIT=0 rolls back to the inline chain.
            if mhc_split_enabled() {
                let hsev = &self.sync_events.layers[layer as usize];
                hsev.hc_src_attn.record(&de.compute)?;
                de.hc.wait_event(&hsev.hc_src_attn)?;
                // --- Path B (side stream): hc_mixes -> split
                self.dgpu_graphs.run("mhc_mixes_attn", layer as u32, &de.hc, |s| {
                    de.rms_nw_mw.launch_inv_only(s, &mut dgpu_scratch.rms_nw_inv_scalar, &dgpu_scratch.residual, &mut dgpu_scratch.rms_nw_partials, HC_DIM, 16, RMS_EPS)?;
                    let ksplit: u32 = std::env::var("F16_KSPLIT")
                        .ok().and_then(|s| s.parse().ok()).unwrap_or(HC_DIM / 1024);
                    if ksplit > 0 {
                        de.f16.matvec_narrow_ksplit_pre_scaled(
                            s, &mut dgpu_scratch.mix, &dlw.hc_attn_fn.buffer,
                            &dgpu_scratch.residual, &dgpu_scratch.rms_nw_inv_scalar,
                            &mut dgpu_scratch.mhc_matvec_partials,
                            HC_MIX_DIM, HC_DIM, ksplit,
                        )?;
                    } else {
                        de.f16.matvec_pre_scaled(s, &mut dgpu_scratch.mix, &dlw.hc_attn_fn.buffer, &dgpu_scratch.residual, &dgpu_scratch.rms_nw_inv_scalar, HC_MIX_DIM, HC_DIM)?;
                    }
                    de.hc_sinkhorn.launch(s, &mut dgpu_scratch.split, &dgpu_scratch.mix, &dlw.hc_attn_scale, &dlw.hc_attn_base, N_HC, SINKHORN_ITERS, SINKHORN_EPS)?;
                    Ok(())
                })?;
                // --- Path A (compute): collapse with the PREVIOUS carry, then norm
                self.dgpu_graphs.run("mhc_collapse_attn", layer as u32, &de.compute, |s| {
                    de.hc_weighted.launch(s, &mut dgpu_scratch.attn_cur, &dgpu_scratch.residual, &dgpu_scratch.hc_pre_carry, N_EMBD, N_HC)?;
                    de.rms_w.launch_weighted(s, &mut dgpu_scratch.attn_input_norm, &dgpu_scratch.attn_cur, &dlw.attn_norm, N_EMBD, RMS_EPS)?;
                    Ok(())
                })?;
                hsev.hc_collapse_attn.record(&de.compute)?;
                // --- carry := this sub-block's pre, only after BOTH the old carry
                //     has been read (hc_collapse) and `split` exists (hc stream).
                de.hc.wait_event(&hsev.hc_collapse_attn)?;
                let cur_pre = dgpu_scratch.split.slice_view(0, N_HC as usize);
                let mut carry = dgpu_scratch.hc_pre_carry.slice_view_mut(0, N_HC as usize);
                carry.copy_from_buffer_async(&cur_pre, &de.hc)?;
                hsev.hc_mixes_attn.record(&de.hc)?;
            } else if mhc_fused {
                self.dgpu_graphs.run("mhc_pre_attn_fused", layer as u32, &de.compute, |s| {
                    de.mhc_pre_fused.launch(
                        s,
                        &mut dgpu_scratch.attn_input_norm,
                        &dgpu_scratch.residual,
                        &dlw.hc_attn_fn.buffer,
                        &dlw.hc_attn_scale,
                        &dlw.hc_attn_base,
                        &dlw.attn_norm,
                        RMS_EPS,
                        SINKHORN_ITERS,
                    )?;
                    Ok(())
                })?;
            } else {
            self.dgpu_graphs.run("mhc_pre_attn", layer as u32, &de.compute, |s| {
                // RMS_NW_MW values:
                //   "fused" (default): compute inv_rms only, fold scale into
                //      next f16 matvec (no apply pass, no flat[] DRAM
                //      round-trip, one fewer launch).
                //   "split": multi-WG rms_nw + standalone matvec (the
                //      previous approach).
                //   "0"/"single": original Grid(1,1,1) single-WG kernel.
                let mode = std::env::var("RMS_NW_MW").unwrap_or_else(|_| "fused".into());
                match mode.as_str() {
                    "0" | "single" => {
                        de.rms_nw.launch(s, &mut dgpu_scratch.flat, &dgpu_scratch.residual, 1, HC_DIM, RMS_EPS)?;
                        de.f16.matvec(s, &mut dgpu_scratch.mix, &dlw.hc_attn_fn.buffer, &dgpu_scratch.flat, HC_MIX_DIM, HC_DIM)?;
                    }
                    "split" => {
                        de.rms_nw_mw.launch(s, &mut dgpu_scratch.flat, &dgpu_scratch.residual, &mut dgpu_scratch.rms_nw_partials, HC_DIM, 16, RMS_EPS)?;
                        de.f16.matvec(s, &mut dgpu_scratch.mix, &dlw.hc_attn_fn.buffer, &dgpu_scratch.flat, HC_MIX_DIM, HC_DIM)?;
                    }
                    _ => {
                        de.rms_nw_mw.launch_inv_only(s, &mut dgpu_scratch.rms_nw_inv_scalar, &dgpu_scratch.residual, &mut dgpu_scratch.rms_nw_partials, HC_DIM, 16, RMS_EPS)?;
                        // F16_KSPLIT=N (default 16 = K-split + u128 vector
                        // loads). N=0 rolls back to legacy narrow.
                        let ksplit: u32 = std::env::var("F16_KSPLIT")
                            .ok().and_then(|s| s.parse().ok()).unwrap_or(HC_DIM / 1024); // k_chunk ≤ 1024 (kernel LDS): 16 @4096-wide, 20 @5120
                        if ksplit > 0 {
                            de.f16.matvec_narrow_ksplit_pre_scaled(
                                s, &mut dgpu_scratch.mix, &dlw.hc_attn_fn.buffer,
                                &dgpu_scratch.residual, &dgpu_scratch.rms_nw_inv_scalar,
                                &mut dgpu_scratch.mhc_matvec_partials,
                                HC_MIX_DIM, HC_DIM, ksplit,
                            )?;
                        } else {
                            de.f16.matvec_pre_scaled(s, &mut dgpu_scratch.mix, &dlw.hc_attn_fn.buffer, &dgpu_scratch.residual, &dgpu_scratch.rms_nw_inv_scalar, HC_MIX_DIM, HC_DIM)?;
                        }
                    }
                }
                de.hc_sinkhorn.launch(s, &mut dgpu_scratch.split, &dgpu_scratch.mix, &dlw.hc_attn_scale, &dlw.hc_attn_base, N_HC, SINKHORN_ITERS, SINKHORN_EPS)?;
                if cfg!(feature = "v41") {
                    // Single-pass mHC (ARCH_SPEC §1.1): collapse with the PREVIOUS
                    // sub-block's pre, then carry this sub-block's pre forward.
                    de.hc_weighted.launch(s, &mut dgpu_scratch.attn_cur, &dgpu_scratch.residual, &dgpu_scratch.hc_pre_carry, N_EMBD, N_HC)?;
                    let cur_pre = dgpu_scratch.split.slice_view(0, N_HC as usize);
                    let mut carry = dgpu_scratch.hc_pre_carry.slice_view_mut(0, N_HC as usize);
                    carry.copy_from_buffer_async(&cur_pre, s)?;
                } else {
                    de.hc_weighted.launch(s, &mut dgpu_scratch.attn_cur, &dgpu_scratch.residual, &dgpu_scratch.split, N_EMBD, N_HC)?;
                }
                // RMS_W_MW=1 enables multi-WG weighted RMS. Default OFF: at
                // N_EMBD=4096 the single-WG version is small enough that
                // multi-WG's extra kernel launch erases any parallelism
                // win — averaged across 256-token decode runs the variants
                // are statistical ties (~±2% within thermal noise).
                if std::env::var("RMS_W_MW").map(|v| v != "0").unwrap_or(false) {
                    de.rms_nw_mw.launch_weighted(s, &mut dgpu_scratch.attn_input_norm, &dgpu_scratch.attn_cur, &dlw.attn_norm, &mut dgpu_scratch.rms_nw_partials, N_EMBD, 16, RMS_EPS)?;
                } else {
                    de.rms_w.launch_weighted(s, &mut dgpu_scratch.attn_input_norm, &dgpu_scratch.attn_cur, &dlw.attn_norm, N_EMBD, RMS_EPS)?;
                }
                Ok(())
            })?;
            }
            drop(_s_mhc_pre);
            _t_mhc_pre.end()?;
            // Bisect within a layer (KNOWN_BUGS #0b): `attn_cur` is the mHC
            // COLLAPSE output, before attention. Dumped AFTER the graph closes --
            // the two collapse sites above sit inside `dgpu_graphs.run` capture
            // closures and synchronising in there 500s the request. Tag matches
            // the prefill side's `pf_attn_cur_p<POS>`.
            if super::engine::subtensor_dump_armed(layer as usize) {
                de.compute.synchronize()?;
                super::engine::maybe_dump_subtensor_f32(
                    layer as usize,
                    &format!("dec_attn_cur_p{pos}"),
                    &dgpu_scratch.attn_cur,
                )?;
                // Counterpart of `pf_pre_residual_p<POS>`: the layer INPUT (the
                // previous layer's FULL output -- attention AND MoE). `residual`
                // is not overwritten until `residual_next` is swapped in, so it
                // still holds the input here. Layer 1's attn_out is clean while
                // layer 2's attn_cur is not, so the MoE half needs diffing too.
                super::engine::maybe_dump_subtensor_f32(
                    layer as usize,
                    &format!("dec_pre_residual_p{pos}"),
                    &dgpu_scratch.residual,
                )?;
            }
        }

        // ============================================================
        // dGPU: Q LoRA chain → q_post_rope
        // ============================================================
        // q_chain prefix (6 kernels) captured into a graph — all params
        // are layer-constant and all I/O buffers are non-swapped scratch.
        // The trailing rope_forward takes per-token `pos` so stays a
        // direct launch.
        let _t_q = de.events.stage("dgpu.q_chain", &de.compute)?;
        let _s_q = debug_span!("q_chain").entered();
        // M59: one merged per-layer graph for the whole q+kv chain (13
        // kernels incl. device-pos ropes + device-slot append) — the
        // ungraphed segments paid ~4-6 µs of launch serialization each.
        // DECODE_QKV_GRAPH=0 rolls back to the split version.
        static QKV_GRAPH: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var("DECODE_QKV_GRAPH").map(|v| v != "0").unwrap_or(true)
        });
        // V4.1 takes the unfused chain: `kv_post_fused` bakes V4-Flash's window
        // quantisation (E4M3 over the 448 non-RoPE dims, block 64); V4.1 quantises
        // the whole post-RoPE row at block 32 (ARCH_SPEC §1.2). Fusing that is an
        // M7 item (needs a device-slot kv_append for graph capture).
        if *QKV_GRAPH && !standalone_graphs && !cfg!(feature = "v41") {
            self.dgpu_graphs.run("qkv_chain", layer as u32, &de.compute, |s| {
                de.q8.quantize_input(s, &mut dgpu_scratch.xq_n_embd, &mut dgpu_scratch.xscale_n_embd, &dgpu_scratch.attn_input_norm, N_EMBD)?;
                super::dispatch::dense_matvec(de, s, &mut dgpu_scratch.qr, &dlw.attn_q_a, &dgpu_scratch.attn_input_norm, &dgpu_scratch.xq_n_embd, &dgpu_scratch.xscale_n_embd, N_LORA_Q, N_EMBD)?;
                de.rms_w.launch_weighted_quantize_q8(s, &mut dgpu_scratch.qr_normed, &mut dgpu_scratch.qr_xq, &mut dgpu_scratch.qr_xscale, &dgpu_scratch.qr, &dlw.q_a_norm, N_LORA_Q, RMS_EPS)?;
                de.q8.matvec(s, &mut dgpu_scratch.q, &dlw.attn_q_b.buffer, &dgpu_scratch.qr_xq, &dgpu_scratch.qr_xscale, Q_FLAT, N_LORA_Q)?;
                if cfg!(feature = "v41") {
                    // V4.1 has no per-head q RMSNorm after wq_b (ARCH_SPEC §1.2: q = wq_b(qr); rope);
                    // keep the buffer flow, skip the norm.
                    dgpu_scratch.q_normed.copy_from_buffer_async(&dgpu_scratch.q, s)?;
                } else {
                    de.rms_nw.launch(s, &mut dgpu_scratch.q_normed, &dgpu_scratch.q, N_HEAD, N_HEAD_DIM, RMS_EPS)?;
                }
                de.rope.launch_forward_pdev(s, &mut dgpu_scratch.q_normed, &dgpu_scratch.pos_dev, N_HEAD, N_HEAD_DIM, N_ROT, &dlw.rope_params)?;
                de.q8.matvec(s, &mut dgpu_scratch.kv_raw, &dlw.attn_kv.buffer, &dgpu_scratch.xq_n_embd, &dgpu_scratch.xscale_n_embd, N_HEAD_DIM, N_EMBD)?;
                // M59: fused rms+rope+fp8+f16rt+append (was 5 kernels).
                de.fp8.launch_kv_post_fused(
                    s,
                    &mut dgpu_scratch.kv_normed,
                    &mut ls.kv_cache,
                    &dgpu_scratch.kv_raw,
                    &dlw.kv_a_norm,
                    &dgpu_scratch.pos_dev,
                    &dgpu_scratch.kv_slot_dev,
                    N_HEAD_DIM,
                    N_ROT,
                    RMS_EPS,
                    &dlw.rope_params,
                )?;
                Ok(())
            })?;
            debug_assert!(
                ((ls.raw_off + ls.n_raw) as usize) < super::state::KV_CACHE_ROWS,
                "kv monotonic append OOB (qkv_chain): raw_off={} n_raw={}",
                ls.raw_off,
                ls.n_raw
            );
            drop(_s_q);
            _t_q.end()?;
        } else {
        self.dgpu_graphs.run("q_chain_pre_rope", layer as u32, &de.compute, |s| {
                de.q8.quantize_input(s, &mut dgpu_scratch.xq_n_embd, &mut dgpu_scratch.xscale_n_embd, &dgpu_scratch.attn_input_norm, N_EMBD)?;
                super::dispatch::dense_matvec(de, s, &mut dgpu_scratch.qr, &dlw.attn_q_a, &dgpu_scratch.attn_input_norm, &dgpu_scratch.xq_n_embd, &dgpu_scratch.xscale_n_embd, N_LORA_Q, N_EMBD)?;
                // M55: fused rms_w + q8 quantize (was two single-WG kernels);
                // qr_normed is still written for the CSA indexer.
                de.rms_w.launch_weighted_quantize_q8(s, &mut dgpu_scratch.qr_normed, &mut dgpu_scratch.qr_xq, &mut dgpu_scratch.qr_xscale, &dgpu_scratch.qr, &dlw.q_a_norm, N_LORA_Q, RMS_EPS)?;
                de.q8.matvec(s, &mut dgpu_scratch.q, &dlw.attn_q_b.buffer, &dgpu_scratch.qr_xq, &dgpu_scratch.qr_xscale, Q_FLAT, N_LORA_Q)?;
                if cfg!(feature = "v41") {
                    // V4.1 has no per-head q RMSNorm after wq_b (ARCH_SPEC §1.2: q = wq_b(qr); rope);
                    // keep the buffer flow, skip the norm.
                    dgpu_scratch.q_normed.copy_from_buffer_async(&dgpu_scratch.q, s)?;
                } else {
                    de.rms_nw.launch(s, &mut dgpu_scratch.q_normed, &dgpu_scratch.q, N_HEAD, N_HEAD_DIM, RMS_EPS)?;
                }
                Ok(())
            })?;
            de.rope.launch_forward(
                &de.compute,
                &mut dgpu_scratch.q_normed,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                pos,
                &dlw.rope_params,
            )?;
            drop(_s_q);
            _t_q.end()?;
            // KNOWN_BUGS #0b counterpart of `pf_q_normed_p<POS>`. Dumped after
            // the q_chain graph closes (synchronising inside a capture closure
            // 500s the request -- same trap as dec_attn_cur).
            if super::engine::subtensor_dump_armed(layer as usize) {
                de.compute.synchronize()?;
                super::engine::maybe_dump_subtensor_f32(
                    layer as usize,
                    &format!("dec_q_normed_p{pos}"),
                    &dgpu_scratch.q_normed,
                )?;
            }

            // ============================================================
            // dGPU: KV chain → kv_post_rope → KV cache push
            // ============================================================
            let _t_kv = de.events.stage("dgpu.kv_chain", &de.compute)?;
            let _s_kv = debug_span!("kv_chain").entered();
            {
                let _t = de.events.stage("k.kv_chain.matvec", &de.compute)?;
                de.q8.matvec(
                    &de.compute,
                    &mut dgpu_scratch.kv_raw,
                    &dlw.attn_kv.buffer,
                    &dgpu_scratch.xq_n_embd,
                    &dgpu_scratch.xscale_n_embd,
                    N_HEAD_DIM,
                    N_EMBD,
                )?;
            }
            {
                let _t = de.events.stage("k.kv_chain.rms_w", &de.compute)?;
                de.rms_w.launch_weighted(
                    &de.compute,
                    &mut dgpu_scratch.kv_normed,
                    &dgpu_scratch.kv_raw,
                    &dlw.kv_a_norm,
                    N_HEAD_DIM,
                    RMS_EPS,
                )?;
            }
            {
                let _t = de.events.stage("k.kv_chain.rope", &de.compute)?;
                de.rope.launch_forward(
                    &de.compute,
                    &mut dgpu_scratch.kv_normed,
                    1,
                    N_HEAD_DIM,
                    N_ROT,
                    pos,
                    &dlw.rope_params,
                )?;
            }
            {
                let _t = de.events.stage("k.kv_chain.fp8", &de.compute)?;
                if cfg!(feature = "v41") {
                    // V4.1 window KV: E4M3 × 2^e per 32 over the whole row, RoPE tail included.
                    de.fp4kv.launch_fp8_window(&de.compute, &mut dgpu_scratch.kv_normed, 1, N_HEAD_DIM)?;
                } else {
                    de.fp8
                        .launch(&de.compute, &mut dgpu_scratch.kv_normed, N_HEAD_DIM - N_ROT)?;
                }
            }
            {
                let _t = de.events.stage("k.kv_chain.f16rt", &de.compute)?;
                de.f16rt
                    .launch(&de.compute, &mut dgpu_scratch.kv_normed, N_HEAD_DIM)?;
            }
            {
                // M55: MONOTONIC append at slot raw_off + n_raw — never the
                // kernel's evict-slide path (which moved 127 rows through 254
                // block-wide barriers per layer per token, ~25 µs each).
                // Passing the cache CAPACITY as the kernel's `swa_window`
                // guarantees the not-full branch. The window advances via
                // raw_off; readers below take a slice_view at raw_off.
                let _t = de.events.stage("k.kv_chain.kv_append", &de.compute)?;
                let slot = ls.raw_off + ls.n_raw;
                debug_assert!(
                    (slot as usize) < super::state::KV_CACHE_ROWS,
                    "kv monotonic append OOB: raw_off={} n_raw={}",
                    ls.raw_off,
                    ls.n_raw
                );
                de.kv_append.launch(
                    &de.compute,
                    &mut ls.kv_cache,
                    &dgpu_scratch.kv_normed,
                    slot,
                    super::state::KV_CACHE_ROWS as u32,
                    N_HEAD_DIM,
                )?;
            }
        } // !QKV_GRAPH

        if ls.n_raw < SWA_WINDOW {
            ls.n_raw += 1;
        } else {
            ls.raw_off += 1;
        }
        // Wrap (~once per B_MAX tokens per layer): eviction-down copy of
        // the live window to slots [0..W) via scratch (overlap-safe two-hop,
        // same pattern as prefill's post-chunk eviction), then reset.
        if (ls.raw_off + SWA_WINDOW) as usize >= super::state::KV_CACHE_ROWS {
            let head_dim = N_HEAD_DIM as usize;
            let win_len = (ls.n_raw as usize) * head_dim;
            let src_off = (ls.raw_off as usize) * head_dim;
            {
                let mut s = dgpu_scratch.kv_wrap_scratch.slice_view_mut(0, win_len);
                let src = ls.kv_cache.slice_view(src_off, win_len);
                s.copy_from_buffer_async(&src, &de.compute)?;
            }
            {
                let s = dgpu_scratch.kv_wrap_scratch.slice_view(0, win_len);
                let mut dst = ls.kv_cache.slice_view_mut(0, win_len);
                dst.copy_from_buffer_async(&s, &de.compute)?;
            }
            ls.raw_off = 0;
        }
        // (_t_kv/_s_kv guards, when the split path created them, end at
        // their own scope close inside the else-branch above.)

        // ============================================================
        // dGPU: Compressor (ratio>0 layers) — produces comp_kv rows on
        // boundary tokens. Reads attn_input_norm (computed above);
        // weights + state + scratch all live on dGPU so no peer push.
        // ============================================================
        let comp_fires_boundary = ratio > 0 && (pos + 1) % ratio == 0;
        let _parallel_pre = matches!(self.mode, ExecMode::HetParallel);
        let _sev_pre = &self.sync_events.layers[layer as usize];

        // V4.1 reuse layers have no compressor weights: the store they attend
        // over is the source layer's (moved in by `HetModelState::with_kv_source`).
        if ratio > 0 && dlw.compressor.is_none() && ls.compressor.is_none() {
            return Err(eyre!("L{layer}: reuse layer without its source's store (wrap the forward in HetModelState::with_kv_source)"));
        }
        if ratio > 0 && dlw.compressor.is_some() {
            let _t_comp = de.events.stage("dgpu.compressor", &de.compute)?;
            let _s_comp = debug_span!("compressor_dgpu", ratio).entered();

            let cw = dlw
                .compressor
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: missing compressor weights (dGPU)"))?;
            let comp_width = cw.width;
            let pos_mod = pos % ratio;
            let row = if ratio == 4 { 4 + pos_mod } else { pos_mod };
            if ratio == 1 {
                // V4.1 ratio 1 (layer 20): latent = norm(wkv(x)) per token — one
                // matvec straight into `pooled`, no gate/state/pool.
                let _t = de.events.stage("k.compressor_d.f16_kv", &de.compute)?;
                de.f16.matvec(&de.compute, &mut dgpu_scratch.pooled, &cw.wkv.buffer, &dgpu_scratch.attn_input_norm, comp_width, N_EMBD)?;
            } else {
                let _t = de.events.stage("k.compressor_d.f16_pair", &de.compute)?;
                de.f16.matvec_pair(
                    &de.compute,
                    &mut dgpu_scratch.kv_cur,
                    &mut dgpu_scratch.sc_cur,
                    &cw.wkv.buffer,
                    &cw.wgate.buffer,
                    &dgpu_scratch.attn_input_norm,
                    comp_width,
                    N_EMBD,
                )?;
            }
            let cs = ls
                .compressor
                .as_mut()
                .ok_or_else(|| eyre!("L{layer}: missing compressor state"))?;
            if ratio != 1 {
                let _t = de.events.stage("k.compressor_d.state_write", &de.compute)?;
                de.compressor_state_write.launch(
                    &de.compute,
                    &mut cs.state_kv,
                    &mut cs.state_score,
                    &dgpu_scratch.kv_cur,
                    &dgpu_scratch.sc_cur,
                    &cw.ape.buffer,
                    comp_width,
                    row,
                    pos_mod,
                )?;
            }

            if comp_fires_boundary {
                if ratio != 1 {
                    let _t = de.events.stage("k.compressor_d.pool", &de.compute)?;
                    de.compressor_pool.launch(
                        &de.compute,
                        &mut dgpu_scratch.pooled,
                        &cs.state_kv,
                        &cs.state_score,
                        N_HEAD_DIM,
                        ratio,
                    )?;
                }
                {
                    let _t = de.events.stage("k.compressor_d.rms_w", &de.compute)?;
                    de.rms_w.launch_weighted(
                        &de.compute,
                        &mut dgpu_scratch.comp_row,
                        &dgpu_scratch.pooled,
                        &cw.norm,
                        N_HEAD_DIM,
                        RMS_EPS,
                    )?;
                }
                let comp_pos = pos + 1 - ratio;

                // V4.1 CSA2 index-K (S1a), `V41_INDEX_K=1`. MUST sit HERE: the
                // reference reads the RoPE-free latent (`model.py:_compress_kv`,
                // "the indexer needs the latent before RoPE, so it runs before the
                // cache is written") and the block below rotates `comp_row` IN PLACE.
                //
                //   k = k_norm(wk(latent)); rope(k[-64:]) at the GROUP position
                //
                // Stored f16 for now; the reference additionally fp4-quantizes with
                // 32-block E8M0 (NOT the 16-block E4M3 the compressed KV uses, and
                // NOT `indexer_qat.hip`, which applies a Hadamard V4.1 does not).
                // NOTHING READS `index_k` yet — this is inert, like S0.
                if index_k_enabled() {
                    if let Some(iw) = dlw.indexer.as_ref() {
                        if let (Some(wk), Some(knorm)) = (iw.attn_k.as_ref(), iw.k_norm.as_ref()) {
                            let _t = de.events.stage("k.compressor_d.index_k", &de.compute)?;
                            de.f16.matvec(
                                &de.compute,
                                &mut dgpu_scratch.index_k_row,
                                &wk.buffer,
                                &dgpu_scratch.comp_row,
                                N_INDEXER_HEAD_DIM,
                                N_HEAD_DIM,
                            )?;
                            de.rms_w.launch_weighted(
                                &de.compute,
                                &mut dgpu_scratch.index_k_normed,
                                &dgpu_scratch.index_k_row,
                                knorm,
                                N_INDEXER_HEAD_DIM,
                                RMS_EPS,
                            )?;
                            de.rope.launch_forward(
                                &de.compute,
                                &mut dgpu_scratch.index_k_normed,
                                1,
                                N_INDEXER_HEAD_DIM,
                                N_ROT,
                                comp_pos,
                                &dlw.rope_params,
                            )?;
                        }
                    }
                }
                {
                    let _t = de.events.stage("k.compressor_d.rope", &de.compute)?;
                    de.rope.launch_forward(
                        &de.compute,
                        &mut dgpu_scratch.comp_row,
                        1,
                        N_HEAD_DIM,
                        N_ROT,
                        comp_pos,
                        &dlw.rope_params,
                    )?;
                }
                let packed_store = cs.comp_kv.is_fp8();
                if cfg!(feature = "v41") {
                    // V4.1: E2M1 values with one E4M3 scale per 16 over the
                    // whole post-RoPE row (ARCH_SPEC §1.3); exact in the f16 store.
                    let _t = de.events.stage("k.compressor_d.fp4kv", &de.compute)?;
                    de.fp4kv.launch(&de.compute, &mut dgpu_scratch.comp_row, 1, N_HEAD_DIM)?;
                } else if !packed_store {
                    {
                        let _t = de.events.stage("k.compressor_d.fp8", &de.compute)?;
                        de.fp8.launch(
                            &de.compute,
                            &mut dgpu_scratch.comp_row,
                            N_HEAD_DIM - N_ROT,
                        )?;
                    }
                    {
                        let _t = de.events.stage("k.compressor_d.f16rt", &de.compute)?;
                        de.f16rt.launch(
                            &de.compute,
                            &mut dgpu_scratch.comp_row,
                            N_HEAD_DIM,
                        )?;
                    }
                }
                if ratio == 4 {
                    let _t = de.events.stage("k.compressor_d.shuffle", &de.compute)?;
                    de.compressor_shuffle.launch(
                        &de.compute,
                        &mut cs.state_kv,
                        &mut cs.state_score,
                        comp_width,
                    )?;
                }

                // V4.1 index-K store (S1a step 5). The VALUE was computed above from
                // the pre-RoPE latent; it is stored HERE, beside the comp_kv append,
                // because that is where `cs` is already mutably borrowed. `index_k`
                // rows are indexed the same way comp_kv rows are, so `n_index_comp`
                // advances in lockstep with `n_comp`.
                if index_k_enabled() {
                    if let Some(ik) = cs.index_k.as_mut() {
                        let _t = de.events.stage("k.compressor_d.index_k_store", &de.compute)?;
                        // Packs E2M1 + one E8M0 per 32 and appends — the reference's
                        // `fp4_act_quant(k, 32, True)`, NOT the compressed-KV 16-block
                        // E4M3 path and NOT `indexer_qat.hip` (which Hadamards).
                        de.index_kv_e2m1.launch_append(
                            &de.compute,
                            ik,
                            &dgpu_scratch.index_k_normed,
                            cs.n_index_comp,
                        )?;
                        cs.n_index_comp += 1;
                    }
                }

                // No peer push needed — append directly into local comp_kv.
                {
                    let _t = de.events.stage("k.compressor_d.comp_kv_append", &de.compute)?;
                    match &mut cs.comp_kv {
                        CompKvStore::F16(buf) => de.comp_kv_append.launch(
                            &de.compute,
                            buf,
                            &dgpu_scratch.comp_row,
                            cs.n_comp,
                            N_HEAD_DIM,
                        )?,
                        // Packed store: quantise + pack + head-shadow write
                        // in ONE kernel (replaces fp8 -> f16rt -> append).
                        CompKvStore::Fp8 { rows, head } => de.comp_kv_fp8.launch_append(
                            &de.compute,
                            rows,
                            head,
                            &dgpu_scratch.comp_row,
                            cs.n_comp,
                            FP8_KV_HEAD_ROWS as u32,
                        )?,
                        CompKvStore::E2m1(_) => {
                            return Err(eyre!("L{layer}: main compressor store cannot be E2M1"))
                        }
                    }
                }
                cs.n_comp += 1;
            }
            drop(_s_comp);
            _t_comp.end()?;
        }

        // ============================================================
        // dGPU: CSA indexer compressor — second, parallel compressor at
        // head_dim=128, only on ratio==4 layers. Same kernel set as the
        // main compressor (matvec_pair → state_write → on boundary:
        // pool → rms_w → rope → f16rt → shuffle → comp_kv_append).
        // FP8 quantize is SKIPPED — only valid for head_dim=512.
        //
        // Scratch reuse: this block runs strictly after the main
        // compressor block on the same de.compute stream, so we can
        // reuse `kv_cur` / `sc_cur` / `pooled` / `comp_row` buffers via
        // slice views sized to the indexer's smaller dims.
        // ============================================================
        if ratio == 4 {
            let _t_icomp = de.events.stage("dgpu.indexer_compressor", &de.compute)?;
            let _s_icomp = debug_span!("indexer_compressor_dgpu").entered();

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
            let pos_mod = pos % ratio;
            let row = 4 + pos_mod; // ratio==4 always

            // matvec_pair writes [icw] into the head of kv_cur / sc_cur.
            {
                let _t = de.events.stage("k.indexer_compressor.f16_pair", &de.compute)?;
                let mut kv_view = dgpu_scratch.kv_cur.slice_view_mut(0, icw as usize);
                let mut sc_view = dgpu_scratch.sc_cur.slice_view_mut(0, icw as usize);
                de.f16.matvec_pair(
                    &de.compute,
                    &mut kv_view,
                    &mut sc_view,
                    &iw.wkv.buffer,
                    &iw.wgate.buffer,
                    &dgpu_scratch.attn_input_norm,
                    icw,
                    N_EMBD,
                )?;
            }
            {
                let _t = de.events.stage("k.indexer_compressor.state_write", &de.compute)?;
                let kv_view = dgpu_scratch.kv_cur.slice_view(0, icw as usize);
                let sc_view = dgpu_scratch.sc_cur.slice_view(0, icw as usize);
                de.compressor_state_write.launch(
                    &de.compute,
                    &mut ics.state_kv,
                    &mut ics.state_score,
                    &kv_view,
                    &sc_view,
                    &iw.ape.buffer,
                    icw,
                    row,
                    pos_mod,
                )?;
            }

            if comp_fires_boundary {
                {
                    let _t = de.events.stage("k.indexer_compressor.pool", &de.compute)?;
                    let mut pooled_view = dgpu_scratch.pooled.slice_view_mut(0, ihd as usize);
                    de.compressor_pool.launch(
                        &de.compute,
                        &mut pooled_view,
                        &ics.state_kv,
                        &ics.state_score,
                        ihd,
                        ratio,
                    )?;
                }
                {
                    let _t = de.events.stage("k.indexer_compressor.rms_w", &de.compute)?;
                    let mut row_view = dgpu_scratch.comp_row.slice_view_mut(0, ihd as usize);
                    let pooled_view = dgpu_scratch.pooled.slice_view(0, ihd as usize);
                    de.rms_w.launch_weighted(
                        &de.compute,
                        &mut row_view,
                        &pooled_view,
                        &iw.norm,
                        ihd,
                        RMS_EPS,
                    )?;
                }
                let comp_pos = pos + 1 - ratio;
                {
                    let _t = de.events.stage("k.indexer_compressor.rope", &de.compute)?;
                    let mut row_view = dgpu_scratch.comp_row.slice_view_mut(0, ihd as usize);
                    de.rope.launch_forward(
                        &de.compute,
                        &mut row_view,
                        1,
                        ihd,
                        N_ROT,
                        comp_pos,
                        &dlw.rope_params,
                    )?;
                }
                // No FP8 step — head_dim=128 ≠ 512 (ds4.c:6702 gates fp8 on head_dim==N_HEAD_DIM).
                // Instead, ds4 5bc1e6d: indexer compressor KV rows take the
                // Hadamard128 + FP4 QAT round trip (post-RoPE, pre-append).
                {
                    let _t = de.events.stage("k.indexer_compressor.qat", &de.compute)?;
                    let mut row_view = dgpu_scratch.comp_row.slice_view_mut(0, ihd as usize);
                    de.indexer_qat.launch(&de.compute, &mut row_view, 1)?;
                }
                // Packed-E2M1 key store: the append re-derives (code, e) from
                // the QAT'd f32 row itself (no f16 round trip needed).
                let packed_keys = ics.comp_kv.is_e2m1();
                if !packed_keys {
                    let _t = de.events.stage("k.indexer_compressor.f16rt", &de.compute)?;
                    let mut row_view = dgpu_scratch.comp_row.slice_view_mut(0, ihd as usize);
                    de.f16rt.launch(&de.compute, &mut row_view, ihd)?;
                }
                {
                    let _t = de.events.stage("k.indexer_compressor.shuffle", &de.compute)?;
                    de.compressor_shuffle.launch(
                        &de.compute,
                        &mut ics.state_kv,
                        &mut ics.state_score,
                        icw,
                    )?;
                }
                {
                    let _t = de.events.stage("k.indexer_compressor.comp_kv_append", &de.compute)?;
                    let row_view = dgpu_scratch.comp_row.slice_view(0, ihd as usize);
                    match &mut ics.comp_kv {
                        CompKvStore::E2m1(rows) => de.index_kv_e2m1.launch_append(
                            &de.compute,
                            rows,
                            &row_view,
                            ics.n_comp,
                        )?,
                        other => de.comp_kv_append.launch(
                            &de.compute,
                            other
                                .f16_mut()
                                .ok_or_else(|| eyre!("L{layer}: indexer compressor store must be f16 or e2m1"))?,
                            &row_view,
                            ics.n_comp,
                            ihd,
                        )?,
                    }
                }
                ics.n_comp += 1;
            }
            drop(_s_icomp);
            _t_icomp.end()?;
        }

        // ============================================================
        // dGPU: Attention compute
        // ============================================================
        // M55: the live SWA window is rows [raw_off, raw_off + n_raw) of the
        // monotonic cache; every raw-KV reader gets this view (kernels are
        // unchanged — they read rows [0, n_raw) of their base pointer).
        let kv_win = ls.kv_cache.slice_view(
            (ls.raw_off as usize) * (N_HEAD_DIM as usize),
            (ls.n_raw as usize) * (N_HEAD_DIM as usize),
        );
        if ratio == 0 {
            // KNOWN_BUGS #0b window check, SWA branch (ratio==0, e.g. layer 0).
            // `kv_win` is sliced at raw_off, so decode attends slots
            // [raw_off, raw_off + n_raw) and passes n_raw as the count.
            if std::env::var("V41_WINDOW_DBG").as_deref() == Ok("1") {
                tracing::info!(
                    layer, pos, n_raw = ls.n_raw, raw_off = ls.raw_off,
                    "window.decode swa"
                );
            }
            let _t_attn = de.events.stage("dgpu.attn_compute", &de.compute)?;
            let _s_attn = debug_span!("attn_compute").entered();
            de.attn_swa.launch(
                &de.compute,
                &mut dgpu_scratch.heads,
                &dgpu_scratch.q_normed,
                &kv_win,
                &dlw.attn_sinks,
                N_HEAD,
                N_HEAD_DIM,
                ls.n_raw,
            )?;
            drop(_s_attn);
            _t_attn.end()?;
        } else {
            // CSA indexer: at ratio==4 layers with n_index_comp > 512, run
            // matvec(attn_q_b) → RoPE → matvec(proj) → scale →
            // IndexerScore → IndexerTopk → IndexerGather. Result is a
            // dense `active_comp_kv` of ≤512 rows for the attention kernels
            // to consume instead of the full `cs.comp_kv`.
            //
            // Below the early-permit boundary (n_index_comp ≤ 512), or for
            // ratio==128 layers (no indexer), the existing dense path runs
            // unchanged.
            let cs = ls.compressor.as_ref();
            let n_comp_full = cs.map(|c| c.n_comp).unwrap_or(0);
            let ics = ls.indexer_compressor.as_ref();
            // Clamp to the scratch stride (indexer_scores [MAX_KEYS],
            // indexer_allowed_bits [ceil(MAX_KEYS/32)]). Unreachable in
            // production (the server refuses a --ctx whose
            // `attn_max_scored_keys` exceeds MAX_KEYS); FAKE_POS benches
            // decoding past the cap used to overrun the bitmap by one word.
            let n_index_comp = ics
                .map(|c| c.n_comp)
                .unwrap_or(0)
                .min(crate::attention::ATTN_MIXED_MAX_KEYS);
            // The sparse/top-512 indexer path is gated on `ratio == 4`, which ONLY
            // V4-Flash has: V4.1's ratios are 1 and 2, so every V4.1 layer >= 2
            // scores DENSELY over its whole compressed store. Widening this gate is
            // not enough to fix that — under `v41` neither the indexer weights
            // (`het::weights` loads `indexer`/`indexer_compressor` only at ratio 4)
            // nor the `indexer_compressor` state (`state.rs`) exist, and V4.1's
            // indexer is structurally different anyway (8 index-source layers
            // 2/8/14/20/24/28/32/36 sharing one top-512 with their reuse layers,
            // keys on the 4 kv-source layers only, plus the layer-20 hierarchical
            // candidate pool — ARCH_SPEC §1.4/§1.5, ENGINE_PORT M5 leftover).
            // `n_index_comp` is 0 under v41, so the condition below is false for
            // BOTH reasons; see docs/v41/DECODE_M8_PLAN.md "Measured (2026-09-13)".
            // V4.1 CSA2 (S1): index-K lives on the KV-SOURCE's compressor state
            // (`with_kv_source` has already moved it in for reuse layers), NOT in a
            // separate `indexer_compressor` — V4.1 has none. `n_index_comp` is
            // maintained by the S1a store.
            let v41_index_k = cs.and_then(|c| c.index_k.as_ref());
            let v41_n_index = cs.map(|c| c.n_index_comp).unwrap_or(0);
            let (n_index_comp, keys_v41) = if cfg!(feature = "v41") && v41_index_k.is_some() {
                (v41_n_index.min(crate::attention::ATTN_MIXED_MAX_KEYS), v41_index_k)
            } else {
                (n_index_comp, None)
            };
            // `V41_INDEXER_FORCE=1` lowers the threshold so the sparse path RUNS at
            // n_index_comp <= INDEXER_TOP_K. That regime is where top-512 selects EVERY
            // reachable row, so sparse must be BIT-IDENTICAL to dense — which makes it a
            // genuine end-to-end test of the WIRING. (The plan called the <=512 test a
            // tautology; it is one only because the normal gate stops the sparse path
            // from executing there. Forcing execution fixes that.)
            let force = std::env::var("V41_INDEXER_FORCE").as_deref() == Ok("1");
            let sparse_min = if force { 0 } else { INDEXER_TOP_K };
            let use_sparse = if keys_v41.is_some() {
                index_k_enabled() && is_index_source_layer(layer) && n_index_comp > sparse_min
            } else {
                ratio == 4 && n_index_comp > INDEXER_TOP_K
            };
            let env_disable_sparse = std::env::var("DECODE_INDEXER")
                .map(|v| v == "off" || v == "0").unwrap_or(false);
            let use_sparse = use_sparse && !env_disable_sparse;

            if use_sparse {
                let _t_ix = de.events.stage("dgpu.indexer", &de.compute)?;
                let _s_ix = debug_span!("indexer_dgpu").entered();
                let iw = dlw
                    .indexer
                    .as_ref()
                    .ok_or_else(|| eyre!("L{layer}: ratio==4 but no indexer weights"))?;
                // V4-Flash only; under v41 the keys come from `keys_v41` instead.
                let ics_ref_opt = ics;

                // 1. matvec(attn_q_b × qr_normed) → indexer_q [N_INDEXER_HEAD * N_INDEXER_HEAD_DIM]
                de.f16.matvec(
                    &de.compute,
                    &mut dgpu_scratch.indexer_q,
                    &iw.attn_q_b.buffer,
                    &dgpu_scratch.qr_normed,
                    N_INDEXER_HEAD * N_INDEXER_HEAD_DIM,
                    N_LORA_Q,
                )?;
                // 2. RoPE on indexer_q at this token's global position.
                de.rope.launch_forward(
                    &de.compute,
                    &mut dgpu_scratch.indexer_q,
                    N_INDEXER_HEAD,
                    N_INDEXER_HEAD_DIM,
                    N_ROT,
                    pos,
                    &dlw.rope_params,
                )?;
                // 2b. ds4 5bc1e6d: Hadamard128 + FP4 QAT round trip on the
                // indexer Q rows (post-RoPE, pre-scoring).
                //
                // V4.1 DOES NOT ROTATE. `model.py:535-537` applies plain
                // `fp4_act_quant(q, 32, True)` with no Hadamard, so running this here
                // would give a plausible-looking but WRONG selection, silently. This is
                // the single highest-risk line in the port.
                if keys_v41.is_none() {
                    de.indexer_qat.launch(&de.compute, &mut dgpu_scratch.indexer_q, N_INDEXER_HEAD)?;
                } else {
                    // V4.1: FP4 round trip WITHOUT the rotation. Dropping the whole
                    // kernel would leave Q in f32 against FP4 keys — a real divergence.
                    de.indexer_qat.launch_fp4(&de.compute, &mut dgpu_scratch.indexer_q, N_INDEXER_HEAD)?;
                }
                // 3. matvec(indexer.proj × attn_input_norm) → head_weights [N_INDEXER_HEAD]
                de.f16.matvec(
                    &de.compute,
                    &mut dgpu_scratch.indexer_head_weights,
                    &iw.proj.buffer,
                    &dgpu_scratch.attn_input_norm,
                    N_INDEXER_HEAD,
                    N_EMBD,
                )?;
                // 4. Scale head_weights *= 1/sqrt(head_dim * n_head)
                let scale =
                    1.0f32 / ((N_INDEXER_HEAD_DIM as f32) * (N_INDEXER_HEAD as f32)).sqrt();
                de.vec_scale.launch(
                    &de.compute,
                    &mut dgpu_scratch.indexer_head_weights,
                    scale,
                    N_INDEXER_HEAD,
                )?;
                // 5. IndexerScore over the contiguous prefix of index_comp_kv.
                // Prefer the WMMA variant when available (28× faster at
                // production decode shape); fall back to the naive kernel
                // on iGPU or any arch without WMMA support.
                // M58: multi-wave by default (INDEXER_DECODE=sw rolls
                // back to the 1-wave kernel; f16 store only).
                static MW: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
                    std::env::var("INDEXER_DECODE").map(|v| v != "sw").unwrap_or(true)
                });
                if let Some(keys) = keys_v41 {
                    let kv_slice =
                        keys.slice_view(0, (n_index_comp as usize) * E2M1_KEY_ROW_BYTES);
                    // SCALAR ONLY under V4.1. The WMMA score kernels
                    // (`launch_mw_e2m1`, `launch_batched_mw_e2m1`,
                    // `launch_batched_gemm_e2m1`) take no `n_head` — they are
                    // hard-coded to V4-Flash's **64** index heads, and V4.1 has **32**
                    // (`N_INDEXER_HEAD`). Feeding them 32-head Q makes them read twice
                    // the rows they should. `launch_e2m1` takes `(n_head, head_dim)` and
                    // is the variant the passing synthetic oracle
                    // (`tests/v41_indexer_selection_oracle.rs`) exercises at 32x128.
                    // A 32-head WMMA twin is the perf follow-up; correctness first.
                    de.indexer_score.launch_e2m1(
                        &de.compute,
                        &mut dgpu_scratch.indexer_scores,
                        &dgpu_scratch.indexer_q,
                        &dgpu_scratch.indexer_head_weights,
                        &kv_slice,
                        n_index_comp,
                        N_INDEXER_HEAD,
                        N_INDEXER_HEAD_DIM,
                    )?;
                } else {
                let ics_ref = ics_ref_opt.expect("ratio==4 must have indexer_compressor state");
                match &ics_ref.comp_kv {
                    CompKvStore::E2m1(rows) => {
                        // Packed keys: the *_e2m1 twins expand at their loads.
                        let kv_slice = rows.slice_view(0, (n_index_comp as usize) * E2M1_KEY_ROW_BYTES);
                        if let Some(wmma) = de.indexer_score_wmma.as_ref() {
                            if !*MW {
                                return Err(eyre!(
                                    "L{layer}: INDEXER_DECODE=sw is not available with the packed-E2M1 \
                                     key store (INDEXER_KEYS_E2M1=0 for the f16 store)"
                                ));
                            }
                            wmma.launch_mw_e2m1(
                                &de.compute,
                                &mut dgpu_scratch.indexer_scores,
                                &dgpu_scratch.indexer_q,
                                &dgpu_scratch.indexer_head_weights,
                                &kv_slice,
                                n_index_comp,
                            )?;
                        } else {
                            de.indexer_score.launch_e2m1(
                                &de.compute,
                                &mut dgpu_scratch.indexer_scores,
                                &dgpu_scratch.indexer_q,
                                &dgpu_scratch.indexer_head_weights,
                                &kv_slice,
                                n_index_comp,
                                N_INDEXER_HEAD,
                                N_INDEXER_HEAD_DIM,
                            )?;
                        }
                    }
                    other => {
                        let kv_slice = other
                            .f16()
                            .ok_or_else(|| eyre!("L{layer}: indexer compressor store must be f16 or e2m1"))?
                            .slice_view(0, (n_index_comp * N_INDEXER_HEAD_DIM) as usize);
                        if let Some(wmma) = de.indexer_score_wmma.as_ref() {
                            if *MW {
                                wmma.launch_mw(
                                    &de.compute,
                                    &mut dgpu_scratch.indexer_scores,
                                    &dgpu_scratch.indexer_q,
                                    &dgpu_scratch.indexer_head_weights,
                                    &kv_slice,
                                    n_index_comp,
                                )?;
                            } else {
                                wmma.launch(
                                    &de.compute,
                                    &mut dgpu_scratch.indexer_scores,
                                    &dgpu_scratch.indexer_q,
                                    &dgpu_scratch.indexer_head_weights,
                                    &kv_slice,
                                    n_index_comp,
                                )?;
                            }
                        } else {
                            de.indexer_score.launch(
                                &de.compute,
                                &mut dgpu_scratch.indexer_scores,
                                &dgpu_scratch.indexer_q,
                                &dgpu_scratch.indexer_head_weights,
                                &kv_slice,
                                n_index_comp,
                                N_INDEXER_HEAD,
                                N_INDEXER_HEAD_DIM,
                            )?;
                        }
                    }
                }
                }
                // 6. IndexerTopk → sorted indices + bitmap. The bitonic
                // variant (ported from ds4) is 72× faster than the
                // greedy fallback at n_comp=16384.
                de.indexer_topk_bitonic.launch(
                    &de.compute,
                    &mut dgpu_scratch.indexer_selected,
                    &mut dgpu_scratch.indexer_allowed_bits,
                    &mut dgpu_scratch.indexer_topk_scratch,
                    &dgpu_scratch.indexer_scores,
                    n_index_comp,
                    INDEXER_TOP_K,
                )?;
                // `V41_INDEXER_DBG=<layer>`: one-shot check that at n_index_comp <= 512
                // the selection really is ALL reachable rows (a permutation of
                // 0..n_index_comp), and that n_index_comp tracks n_comp. If either is
                // false, sparse cannot equal dense and the forced-gate comparison is
                // measuring a real defect rather than reordering.
                if std::env::var("V41_INDEXER_DBG").ok().and_then(|v| v.parse::<i32>().ok())
                    == Some(layer)
                {
                    de.compute.synchronize()?;
                    let k = (INDEXER_TOP_K.min(n_index_comp)) as usize;
                    let mut sel = vec![0i32; k];
                    dgpu_scratch.indexer_selected.slice_view(0, k).copy_to_host(&mut sel)?;
                    let mut seen = vec![false; n_index_comp as usize];
                    let (mut dup, mut oob) = (0usize, 0usize);
                    for &i in &sel {
                        if i < 0 || i as u32 >= n_index_comp {
                            oob += 1;
                        } else if seen[i as usize] {
                            dup += 1;
                        } else {
                            seen[i as usize] = true;
                        }
                    }
                    let missing = seen.iter().filter(|b| !**b).count();
                    // LIVE selection-set oracle dump (`V41_INDEXER_DUMP=<dir>`): write the
                    // exact tensors the GPU scored so a CPU recompute can check the
                    // top-512 as a SET at n_index_comp > 512 — the regime the code is
                    // designed for. Comparing logits against the dense path cannot work
                    // there (they legitimately diverge) and forcing the gate below 512
                    // leaves the designed envelope.
                    if let Ok(dir) = std::env::var("V41_INDEXER_DUMP") {
                        use std::io::Write;
                        let _ = std::fs::create_dir_all(&dir);
                        let w = |name: &str, bytes: &[u8]| {
                            if let Ok(mut f) = std::fs::File::create(format!("{dir}/{name}")) {
                                let _ = f.write_all(bytes);
                            }
                        };
                        let nq = (N_INDEXER_HEAD * N_INDEXER_HEAD_DIM) as usize;
                        let mut q = vec![0f32; nq];
                        dgpu_scratch.indexer_q.slice_view(0, nq).copy_to_host(&mut q)?;
                        let mut hw = vec![0f32; N_INDEXER_HEAD as usize];
                        dgpu_scratch.indexer_head_weights
                            .slice_view(0, N_INDEXER_HEAD as usize).copy_to_host(&mut hw)?;
                        // Expand the packed E2M1 keys so the CPU scores EXACTLY what the
                        // GPU read (quantization error cannot masquerade as a bug).
                        let nrows = n_index_comp as usize;
                        let mut exp: DeviceBuffer<u16> =
                            DeviceBuffer::new(de.device.id, nrows * N_INDEXER_HEAD_DIM as usize)?;
                        de.index_kv_e2m1.launch_expand(
                            &de.compute, &mut exp, keys_v41.unwrap(), n_index_comp)?;
                        de.compute.synchronize()?;
                        let mut kh = vec![0u16; nrows * N_INDEXER_HEAD_DIM as usize];
                        exp.copy_to_host(&mut kh)?;
                        w("q.bin", &q.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
                        w("hw.bin", &hw.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
                        w("keys_f16.bin", &kh.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
                        w("sel.bin", &sel.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
                        w("meta.txt", format!("layer={layer}\nn_index_comp={n_index_comp}\nk={k}\n").as_bytes());
                        tracing::info!(layer, n_index_comp, k, dir = %dir, "indexer.dump written");
                    }
                    tracing::info!(
                        layer, n_index_comp, n_comp_full, k,
                        oob, dup, missing,
                        is_permutation = (oob == 0 && dup == 0 && missing == 0),
                        "indexer.dbg"
                    );
                }
                // 7. Gather selected rows of cs.comp_kv into active_comp_kv.
                let cs_ref = cs.expect("ratio==4 must have main compressor state");
                match &cs_ref.comp_kv {
                    CompKvStore::F16(buf) => de.indexer_gather.launch(
                        &de.compute,
                        &mut dgpu_scratch.active_comp_kv,
                        buf,
                        &dgpu_scratch.indexer_selected,
                        INDEXER_TOP_K,
                        N_HEAD_DIM,
                    )?,
                    // Packed store: the gather expands FP8 -> f16 on the
                    // way into active_comp_kv; attention is unchanged.
                    CompKvStore::Fp8 { rows, .. } => de.comp_kv_fp8.launch_gather(
                        &de.compute,
                        &mut dgpu_scratch.active_comp_kv,
                        rows,
                        &dgpu_scratch.indexer_selected,
                        INDEXER_TOP_K,
                    )?,
                    CompKvStore::E2m1(_) => {
                        return Err(eyre!("L{layer}: main compressor store cannot be E2M1"))
                    }
                }
                // S2: publish this source's gather so the reuse layers below can use it.
                if keys_v41.is_some() {
                    let store = crate::config::kv_source_of(layer as usize)
                        .unwrap_or(layer as usize) as i32;
                    self.last_idx_gather_src
                        .store(store, std::sync::atomic::Ordering::Relaxed);
                    self.last_idx_gather_rows.store(
                        INDEXER_TOP_K.min(n_index_comp),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                drop(_s_ix);
                _t_ix.end()?;
            }

            // For sparse-attn paths we read from active_comp_kv (≤512
            // rows). Otherwise we read from cs.comp_kv directly.
            // S2 shared selection: a V4.1 layer that is NOT an index source reuses the
            // most recent source's gathered rows, provided they belong to the same
            // compressed store. This is what takes the indexer from 8 of 40 layers to
            // all 40 — the reuse layers stop scoring their whole store.
            let s2_reuse = cfg!(feature = "v41")
                && index_k_enabled()
                && !use_sparse
                && !is_index_source_layer(layer)
                && n_comp_full > 0
                && {
                    let store = crate::config::kv_source_of(layer as usize)
                        .unwrap_or(layer as usize) as i32;
                    self.last_idx_gather_src.load(std::sync::atomic::Ordering::Relaxed) == store
                };
            let (attn_comp_kv, attn_n_comp) = if s2_reuse {
                (
                    Some(&dgpu_scratch.active_comp_kv),
                    self.last_idx_gather_rows.load(std::sync::atomic::Ordering::Relaxed),
                )
            } else if use_sparse {
                // `min` NOT the bare constant. The gather produces
                // `min(INDEXER_TOP_K, n_index_comp)` valid rows; scoring a flat 512
                // makes attention read whatever stale bytes sit past the end.
                // Benign in production (the gate needs n_index_comp > 512) but WRONG
                // the moment the gate is lowered — which is exactly what
                // `V41_INDEXER_FORCE=1` does to make the sparse path testable against
                // dense. MEASURED 2026-09-14 at n_comp=293: max|logit delta| 3.012 on
                // magnitude 25.9 (11.6% relative) before this fix.
                (Some(&dgpu_scratch.active_comp_kv), INDEXER_TOP_K.min(n_index_comp))
            } else if n_comp_full > 0 {
                // Dense path: the whole f16 cache, or the FP8 store's f16
                // head shadow (errors past its 512 rows — DECODE_INDEXER=off
                // at depth is unsupported with the packed store).
                let c = cs.expect("n_comp_full > 0 implies compressor state");
                // MEASUREMENT ONLY (M8-C). `V41_ATTN_COMP_CAP=<n>` clamps the dense
                // row count so the attention kernels do the work a WORKING top-n
                // indexer would leave them, without the indexer existing. The output
                // is WRONG (it attends to the oldest n rows); this exists to price
                // the missing sparse path, never to be shipped on.
                let cap = attn_comp_cap();
                let n_used = if cap > 0 { n_comp_full.min(cap) } else { n_comp_full };
                (Some(c.comp_kv.dense_f16(n_comp_full, &format!("L{layer} decode"))?), n_used)
            } else {
                (None, 0)
            };
            {
                if std::env::var("V41_WINDOW_DBG").as_deref() == Ok("1") {
                    tracing::info!(
                        layer, pos, n_raw = ls.n_raw, raw_off = 0,
                        n_comp = attn_n_comp, "window.decode"
                    );
                }
                let _t = de.events.stage("dgpu.attn_score", &de.compute)?;
                let _s = debug_span!("attn_score").entered();
                // Default: B=1 head-tiled WMMA score (scalar-arg variant of
                // the prefill batched WMMA score). Isolated at ratio=4
                // n_comp=16384: 312 µs single-token → 58 µs = 5.4× faster.
                // Scalar args avoid the per-batch buffer reads that would
                // force ~86 copy_from_host calls/token via the batched API.
                // DECODE_SCORE=single rolls back to attention_mixed_score.
                let use_b1_wmma_score = std::env::var("DECODE_SCORE")
                    .map(|v| v != "single").unwrap_or(true);
                if use_b1_wmma_score {
                    let n_total_max = ls.n_raw + attn_n_comp;
                    de.attn_mixed.launch_score_b1_htiled_wmma(
                        &de.compute,
                        &mut dgpu_scratch.attn_scores,
                        &dgpu_scratch.q_normed,
                        &kv_win,
                        attn_comp_kv,
                        ls.n_raw, /*raw_off=*/0, attn_n_comp,
                        N_HEAD, N_HEAD_DIM, n_total_max,
                    )?;
                } else {
                    de.attn_mixed.launch_score(
                        &de.compute,
                        &mut dgpu_scratch.attn_scores,
                        &dgpu_scratch.q_normed,
                        &kv_win,
                        attn_comp_kv,
                        N_HEAD, N_HEAD_DIM, ls.n_raw, attn_n_comp,
                    )?;
                }
            }
            {
                let _t = de.events.stage("dgpu.attn_smwsum", &de.compute)?;
                let _s = debug_span!("attn_smwsum").entered();
                // Default: 3-pass K-split smwsum (head-tile=16 + k-split=16
                // + reduce). Recovers MLA V-share at B=1 by tiling 16 heads
                // per WG sharing V via LDS — the win the batched WMMA
                // kernel can't deliver at B=1 due to under-occupation.
                // Isolated bench at ratio=4 n_comp=16384:
                //   baseline      609 µs p50
                //   k-split=16    260 µs p50  (2.34× faster)
                // DECODE_SMWSUM=single rolls back to the existing kernel.
                let use_ksplit = std::env::var("DECODE_SMWSUM")
                    .map(|v| v != "single").unwrap_or(true);
                if use_ksplit {
                    const K_SPLIT: u32 = 16;
                    de.attn_mixed.launch_softmax_only(
                        &de.compute,
                        &mut dgpu_scratch.attn_scores,
                        &dlw.attn_sinks,
                        &mut dgpu_scratch.attn_inv_per_head,
                        N_HEAD, ls.n_raw, attn_n_comp,
                    )?;
                    de.attn_mixed.launch_wsum_b1_htiled_ksplit_ldsv(
                        &de.compute,
                        &mut dgpu_scratch.attn_partials,
                        &dgpu_scratch.attn_scores,
                        &kv_win,
                        attn_comp_kv,
                        N_HEAD, N_HEAD_DIM,
                        ls.n_raw, attn_n_comp,
                        K_SPLIT,
                    )?;
                    de.attn_mixed.launch_reduce_partials_apply_inv(
                        &de.compute,
                        &mut dgpu_scratch.heads,
                        &dgpu_scratch.attn_partials,
                        &dgpu_scratch.attn_inv_per_head,
                        N_HEAD, N_HEAD_DIM, K_SPLIT,
                    )?;
                } else {
                    de.attn_mixed.launch_softmax_wsum(
                        &de.compute,
                        &mut dgpu_scratch.heads,
                        &mut dgpu_scratch.attn_scores,
                        &dlw.attn_sinks,
                        &kv_win,
                        attn_comp_kv,
                        N_HEAD, N_HEAD_DIM, ls.n_raw, attn_n_comp,
                    )?;
                }
            }
        }

        // ============================================================
        // dGPU: Output projection
        // ============================================================
        // output_proj suffix (4 kernels after rope_inv) is captured into
        // a graph. rope_inv takes per-token `pos` and stays a direct launch.
        let _t_out = de.events.stage("dgpu.output_proj", &de.compute)?;
        let _s_out = debug_span!("output_proj").entered();
        de.rope.launch_inverse(
            &de.compute,
            &mut dgpu_scratch.heads,
            N_HEAD,
            N_HEAD_DIM,
            N_ROT,
            pos,
            &dlw.rope_params,
        )?;
        if super::engine::subtensor_dump_armed(layer as usize) {
            de.compute.synchronize()?;
            super::engine::maybe_dump_subtensor_f32(
                layer as usize,
                &format!("dec_heads_p{pos}"),
                &dgpu_scratch.heads,
            )?;
        }
        self.dgpu_graphs.run("output_proj_post_rope", layer as u32, &de.compute, |s| {
            de.q8.quantize_input(s, &mut dgpu_scratch.heads_xq, &mut dgpu_scratch.heads_xscale, &dgpu_scratch.heads, Q_FLAT)?;
            de.q8_grouped.matvec_grouped(s, &mut dgpu_scratch.low, &dlw.attn_output_a.buffer, &dgpu_scratch.heads_xq, &dgpu_scratch.heads_xscale, GROUP_DIM, RANK, N_GROUPS)?;
            de.q8.quantize_input(s, &mut dgpu_scratch.low_xq, &mut dgpu_scratch.low_xscale, &dgpu_scratch.low, OUT_LOW)?;
            de.q8.matvec(s, &mut dgpu_scratch.attn_out, &dlw.attn_output_b.buffer, &dgpu_scratch.low_xq, &dgpu_scratch.low_xscale, N_EMBD, OUT_LOW)?;
            Ok(())
        })?;
        drop(_s_out);
        _t_out.end()?;
        if super::engine::subtensor_dump_armed(layer as usize) {
            de.compute.synchronize()?;
            super::engine::maybe_dump_subtensor_f32(
                layer as usize,
                &format!("dec_attn_out_p{pos}"),
                &dgpu_scratch.attn_out,
            )?;
        }

        // ============================================================
        // dGPU: mHC post attn → after_attn_hc. hc_post reads post + comb
        // directly from the packed `split` buffer (no host roundtrip).
        // ============================================================
        let _t_mhc_post_attn = de.events.stage("dgpu.mhc_post_attn", &de.compute)?;
        let _s_mhc_post_attn = debug_span!("mhc_post_attn").entered();
        if mhc_split_enabled() {
            // `split` is produced on `de.hc`; this is its first consumer. By now
            // the side stream has had q/kv/attn/output_proj (~413 us) to finish
            // ~58 us of work, so this should be a no-wait in steady state.
            de.compute.wait_event(&self.sync_events.layers[layer as usize].hc_mixes_attn)?;
        }
        de.hc_post.launch_from_split(
            &de.compute,
            &mut dgpu_scratch.after_attn_hc,
            &dgpu_scratch.attn_out,
            &dgpu_scratch.residual,
            &dgpu_scratch.split,
            N_HC,   // n_w
            N_EMBD,
            N_HC,
        )?;
        drop(_s_mhc_post_attn);
        _t_mhc_post_attn.end()?;

        // ============================================================
        // dGPU: mHC pre ffn → ffn_input_norm
        // ============================================================
        // mhc_pre_ffn block (5 kernels, layer-constant params) is captured
        // into a HIP graph on first call and replayed thereafter.
        let _t_mhc_pre_ffn = de.events.stage("dgpu.mhc_pre_ffn", &de.compute)?;
        let _s_mhc_pre_ffn = debug_span!("mhc_pre_ffn").entered();
        let mhc_fused = std::env::var("MHC_FUSED")
            .map(|v| v != "0").unwrap_or(false);
        // Same side-stream split as the attention sub-block, and even more
        // lopsided: `ffn_pre/post/comb` are not consumed until `hc_post` AFTER
        // the MoE, which is ~2.5 ms/layer of paging and expert compute. 59.6
        // us of mixes hides in that completely.
        if mhc_split_enabled() {
            let hsev = &self.sync_events.layers[layer as usize];
            hsev.hc_src_ffn.record(&de.compute)?;
            de.hc.wait_event(&hsev.hc_src_ffn)?;
            self.dgpu_graphs.run("mhc_mixes_ffn", layer as u32, &de.hc, |s| {
                de.rms_nw_mw.launch_inv_only(s, &mut dgpu_scratch.rms_nw_inv_scalar, &dgpu_scratch.after_attn_hc, &mut dgpu_scratch.rms_nw_partials, HC_DIM, 16, RMS_EPS)?;
                let ksplit: u32 = std::env::var("F16_KSPLIT")
                    .ok().and_then(|s| s.parse().ok()).unwrap_or(HC_DIM / 1024);
                if ksplit > 0 {
                    de.f16.matvec_narrow_ksplit_pre_scaled(
                        s, &mut dgpu_scratch.mix, &dlw.hc_ffn_fn.buffer,
                        &dgpu_scratch.after_attn_hc, &dgpu_scratch.rms_nw_inv_scalar,
                        &mut dgpu_scratch.mhc_matvec_partials,
                        HC_MIX_DIM, HC_DIM, ksplit,
                    )?;
                } else {
                    de.f16.matvec_pre_scaled(s, &mut dgpu_scratch.mix, &dlw.hc_ffn_fn.buffer, &dgpu_scratch.after_attn_hc, &dgpu_scratch.rms_nw_inv_scalar, HC_MIX_DIM, HC_DIM)?;
                }
                de.hc_sinkhorn.launch(s, &mut dgpu_scratch.split, &dgpu_scratch.mix, &dlw.hc_ffn_scale, &dlw.hc_ffn_base, N_HC, SINKHORN_ITERS, SINKHORN_EPS)?;
                Ok(())
            })?;
            self.dgpu_graphs.run("mhc_collapse_ffn", layer as u32, &de.compute, |s| {
                de.hc_weighted.launch(s, &mut dgpu_scratch.ffn_cur, &dgpu_scratch.after_attn_hc, &dgpu_scratch.hc_pre_carry, N_EMBD, N_HC)?;
                de.rms_w.launch_weighted(s, &mut dgpu_scratch.ffn_input_norm, &dgpu_scratch.ffn_cur, &dlw.ffn_norm, N_EMBD, RMS_EPS)?;
                Ok(())
            })?;
            hsev.hc_collapse_ffn.record(&de.compute)?;
            de.hc.wait_event(&hsev.hc_collapse_ffn)?;
            let cur_pre = dgpu_scratch.split.slice_view(0, N_HC as usize);
            let mut carry = dgpu_scratch.hc_pre_carry.slice_view_mut(0, N_HC as usize);
            carry.copy_from_buffer_async(&cur_pre, &de.hc)?;
            hsev.hc_mixes_ffn.record(&de.hc)?;
        } else if mhc_fused {
            self.dgpu_graphs.run("mhc_pre_ffn_fused", layer as u32, &de.compute, |s| {
                de.mhc_pre_fused.launch(
                    s,
                    &mut dgpu_scratch.ffn_input_norm,
                    &dgpu_scratch.after_attn_hc,
                    &dlw.hc_ffn_fn.buffer,
                    &dlw.hc_ffn_scale,
                    &dlw.hc_ffn_base,
                    &dlw.ffn_norm,
                    RMS_EPS,
                    SINKHORN_ITERS,
                )?;
                Ok(())
            })?;
        } else {
        self.dgpu_graphs.run("mhc_pre_ffn", layer as u32, &de.compute, |s| {
            let mode = std::env::var("RMS_NW_MW").unwrap_or_else(|_| "fused".into());
            match mode.as_str() {
                "0" | "single" => {
                    de.rms_nw.launch(s, &mut dgpu_scratch.flat, &dgpu_scratch.after_attn_hc, 1, HC_DIM, RMS_EPS)?;
                    de.f16.matvec(s, &mut dgpu_scratch.mix, &dlw.hc_ffn_fn.buffer, &dgpu_scratch.flat, HC_MIX_DIM, HC_DIM)?;
                }
                "split" => {
                    de.rms_nw_mw.launch(s, &mut dgpu_scratch.flat, &dgpu_scratch.after_attn_hc, &mut dgpu_scratch.rms_nw_partials, HC_DIM, 16, RMS_EPS)?;
                    de.f16.matvec(s, &mut dgpu_scratch.mix, &dlw.hc_ffn_fn.buffer, &dgpu_scratch.flat, HC_MIX_DIM, HC_DIM)?;
                }
                _ => {
                    de.rms_nw_mw.launch_inv_only(s, &mut dgpu_scratch.rms_nw_inv_scalar, &dgpu_scratch.after_attn_hc, &mut dgpu_scratch.rms_nw_partials, HC_DIM, 16, RMS_EPS)?;
                    let ksplit: u32 = std::env::var("F16_KSPLIT")
                        .ok().and_then(|s| s.parse().ok()).unwrap_or(HC_DIM / 1024); // k_chunk ≤ 1024 (kernel LDS): 16 @4096-wide, 20 @5120
                    if ksplit > 0 {
                        de.f16.matvec_narrow_ksplit_pre_scaled(
                            s, &mut dgpu_scratch.mix, &dlw.hc_ffn_fn.buffer,
                            &dgpu_scratch.after_attn_hc, &dgpu_scratch.rms_nw_inv_scalar,
                            &mut dgpu_scratch.mhc_matvec_partials,
                            HC_MIX_DIM, HC_DIM, ksplit,
                        )?;
                    } else {
                        de.f16.matvec_pre_scaled(s, &mut dgpu_scratch.mix, &dlw.hc_ffn_fn.buffer, &dgpu_scratch.after_attn_hc, &dgpu_scratch.rms_nw_inv_scalar, HC_MIX_DIM, HC_DIM)?;
                    }
                }
            }
            de.hc_sinkhorn.launch(s, &mut dgpu_scratch.split, &dgpu_scratch.mix, &dlw.hc_ffn_scale, &dlw.hc_ffn_base, N_HC, SINKHORN_ITERS, SINKHORN_EPS)?;
            if cfg!(feature = "v41") {
                    // Single-pass mHC (ARCH_SPEC §1.1): collapse with the PREVIOUS
                    // sub-block's pre, then carry this sub-block's pre forward.
                    de.hc_weighted.launch(s, &mut dgpu_scratch.ffn_cur, &dgpu_scratch.after_attn_hc, &dgpu_scratch.hc_pre_carry, N_EMBD, N_HC)?;
                    let cur_pre = dgpu_scratch.split.slice_view(0, N_HC as usize);
                    let mut carry = dgpu_scratch.hc_pre_carry.slice_view_mut(0, N_HC as usize);
                    carry.copy_from_buffer_async(&cur_pre, s)?;
                } else {
                    de.hc_weighted.launch(s, &mut dgpu_scratch.ffn_cur, &dgpu_scratch.after_attn_hc, &dgpu_scratch.split, N_EMBD, N_HC)?;
                }
            if std::env::var("RMS_W_MW").map(|v| v != "0").unwrap_or(false) {
                de.rms_nw_mw.launch_weighted(s, &mut dgpu_scratch.ffn_input_norm, &dgpu_scratch.ffn_cur, &dlw.ffn_norm, &mut dgpu_scratch.rms_nw_partials, N_EMBD, 16, RMS_EPS)?;
            } else {
                de.rms_w.launch_weighted(s, &mut dgpu_scratch.ffn_input_norm, &dgpu_scratch.ffn_cur, &dlw.ffn_norm, N_EMBD, RMS_EPS)?;
            }
            Ok(())
        })?;
        }
        drop(_s_mhc_pre_ffn);
        _t_mhc_pre_ffn.end()?;

        // ============================================================
        // dGPU: router. Runs on dGPU because (a) the f16 matvec is ~1.5 ms
        // faster on dGPU's 2.6× iGPU BW, and (b) keeping router off iGPU
        // lifts it from the iGPU MoE critical path.
        //
        // Order on dGPU (all on de.compute, FIFO):
        //   1. router_logits = matvec(ffn_gate_inp, ffn_input_norm)
        //   2. learned: router_topk → d_selected, d_ew
        //      hash:    host sync, readback router_logits, hash select,
        //               write back d_selected and d_ew.
        //   3. record selected_ready event.
        //
        // dGPU.xfer pushes both ffn_input_norm (for MoE q8k_xq) and
        // d_selected/d_ew (for iq2 + q2k) to iGPU in FIFO order:
        //   xfer: wait ain_ready → push ffn_input_norm
        //         wait selected_ready → push d_selected → push d_ew
        //         record selected_pushed
        //
        // iGPU.compute waits selected_pushed (which transitively
        // covers ain_pushed since both pushes are on the same xfer
        // stream), then runs the MoE graph.
        // ============================================================
        let parallel = matches!(self.mode, ExecMode::HetParallel);
        let sev = &self.sync_events.layers[layer as usize];

        // Mark ffn_input_norm ready, queue its peer push on xfer.
        if parallel {
            sev.ain_ready.record(&de.compute)?;
            let _t_wait = de
                .events
                .stage("dgpu.peer_push_ffn_input_norm.wait", &de.xfer)?;
            de.xfer.wait_event(&sev.ain_ready)?;
            _t_wait.end()?;
        } else {
            de.compute.synchronize()?;
        }
        let _t_peer_ain = de.events.stage("dgpu.peer_push_ffn_input_norm", &de.xfer)?;
        let _s_peer_ain = debug_span!("peer_push_ffn_input_norm").entered();
        peer_push_f32(
            &dgpu_scratch.ffn_input_norm,
            &mut igpu_scratch.ffn_input_norm_recv,
            &de.xfer,
        )?;
        if parallel {
            sev.ain_pushed.record(&de.xfer)?;
        } else {
            de.xfer.synchronize()?;
        }
        drop(_s_peer_ain);
        _t_peer_ain.end()?;

        // Router runs on dGPU.compute.
        let _t_router = de.events.stage("dgpu.router", &de.compute)?;
        let _s_router = debug_span!("router_dgpu").entered();
        {
            let _t = de.events.stage("k.router.f16_matvec", &de.compute)?;
            de.f16.matvec(
                &de.compute,
                &mut dgpu_scratch.router_logits,
                &dlw.ffn_gate_inp.buffer,
                &dgpu_scratch.ffn_input_norm,
                N_EXPERT,
                N_EMBD,
            )?;
        }
        if !dlw.is_hash_router {
            let _t = de.events.stage("k.router.topk", &de.compute)?;
            de.router_topk.launch(
                &de.compute,
                &mut dgpu_scratch.d_selected,
                &mut dgpu_scratch.d_ew,
                &dgpu_scratch.router_logits,
                dlw.router_bias_dev.as_ref(),
                N_EXPERT,
                N_EXPERT_USED as u32,
                EXPERT_WEIGHT_SCALE,
                ROUTER_WEIGHT_EPS,
            )?;
            // Counterpart of `pf_sel_p<POS>` / `pf_ew_p<POS>` for #0b.
            if super::engine::subtensor_dump_armed(layer as usize) {
                de.compute.synchronize()?;
                super::engine::maybe_dump_subtensor_i32(
                    layer as usize,
                    &format!("dec_sel_p{pos}"),
                    &dgpu_scratch.d_selected,
                    N_EXPERT_USED,
                )?;
                super::engine::maybe_dump_subtensor_f32_view(
                    layer as usize,
                    &format!("dec_ew_p{pos}"),
                    &dgpu_scratch.d_ew.slice_view(0, N_EXPERT_USED),
                )?;
                super::engine::maybe_dump_subtensor_f32_view(
                    layer as usize,
                    &format!("dec_ffn_in_p{pos}"),
                    &dgpu_scratch.ffn_input_norm.slice_view(0, N_EMBD as usize),
                )?;
            }
        } else {
            // Hash router: host sync the router matvec, read 6 chosen
            // logits, write back d_selected and d_ew on dGPU.compute.
            // The shared expert that runs after this will be FIFO-
            // serialized behind the copy_from_host; the iGPU MoE
            // wait depends on selected_pushed which depends on these
            // writes via stream-FIFO.
            de.compute.synchronize()?;
            dgpu_scratch
                .router_logits
                .copy_to_host(&mut dgpu_scratch.router_logits_host)?;
            let tid2eid = dlw
                .tid2eid
                .as_ref()
                .ok_or_else(|| eyre!("L{layer}: hash router but no tid2eid"))?;
            let (sel, w) =
                hash_router_select(tid2eid, token_id, &dgpu_scratch.router_logits_host);
            dgpu_scratch.d_selected.copy_from_host(&sel)?;
            dgpu_scratch.d_ew.copy_from_host(&w)?;
        }
        drop(_s_router);
        _t_router.end()?;

        // M62: accumulate this layer's picks into the decode stats bank
        // (one tiny atomicAdd kernel on de.compute, no readback).
        self.record_sel_stats(false, &dgpu_scratch.d_selected, layer as u32, 1)?;

        // Push d_selected and d_ew to iGPU on dGPU.xfer (FIFO after the
        // ffn_input_norm push above).
        if parallel {
            sev.selected_ready.record(&de.compute)?;
            let _t_wait = de
                .events
                .stage("dgpu.peer_push_selected.wait", &de.xfer)?;
            de.xfer.wait_event(&sev.selected_ready)?;
            _t_wait.end()?;
        } else {
            de.compute.synchronize()?;
        }
        let _t_peer_sel = de.events.stage("dgpu.peer_push_selected", &de.xfer)?;
        let _s_peer_sel = debug_span!("peer_push_selected").entered();
        {
            // M60: selected + ew share one backing allocation (see
            // DgpuScratch::sel_ew_pack) — ONE 48-byte peer push instead of
            // two copies on the router→iGPU handoff path.
            let _t = de.events.stage("k.peer_push_selected.pack", &de.xfer)?;
            super::sync::peer_push_u8(
                &dgpu_scratch.sel_ew_pack,
                &mut igpu_scratch.sel_ew_pack,
                &de.xfer,
            )?;
        }
        if parallel {
            sev.selected_pushed.record(&de.xfer)?;
            // M54: value-signal companion to selected_pushed, consumed by
            // the PRE-ISSUED iGPU lane (issue_igpu_moe). An event wait
            // enqueued before its record call is a no-op (snapshot
            // semantics); the 32-bit value wait compares at execution
            // time, which makes token-start pre-issue sound. One SDMA
            // dword write per layer — negligible when pre-issue is off.
            let seq = self
                .token_seq
                .load(std::sync::atomic::Ordering::Relaxed);
            // Const-cast: the engine only ever writes this slot from
            // device streams; the host slice is allocation-stable.
            let sig = unsafe {
                (self.moe_signal.as_slice().as_ptr() as *mut u32).add(layer as usize)
            };
            unsafe { de.xfer.write_value32(sig, seq)? };
        } else {
            de.xfer.synchronize()?;
        }
        drop(_s_peer_sel);
        _t_peer_sel.end()?;

        // Decode-path two-box split. HOISTED here from just above the pager
        // block so the pre-submit reorder can see it; single binding, one lock.
        let decode_split_on = matches!(
            std::env::var("V41_REMOTE_SPLIT_DECODE").as_deref(),
            Ok("1") | Ok("on")
        ) && self
            .remote
            .as_ref()
            .and_then(|r| r.lock().ok().map(|c| c.info().owned_count(layer as u32) > 0))
            .unwrap_or(false);
        // See `decode_presubmit_reorder`. Only defer when there IS a remote to
        // wait on; with no split the shared expert is not on anyone's critical
        // path and moving it would only churn the stream order.
        let defer_shared = decode_presubmit_reorder() && parallel && decode_split_on;
        // `moe_xq` is also written by the M56 hot path's captured graph below, so
        // only pre-quantize when that path is inactive (it is, on the paged path:
        // the paged entry point requires `hot_experts == None`).
        let xq_hoisted = defer_shared && dlw.hot_experts.is_none();

        // dGPU shared expert can now run on de.compute (after router,
        // before / in parallel with iGPU MoE). It uses the same
        // ffn_input_norm input as the router.
        if parallel && !defer_shared {
            let _t_shared = de.events.stage("dgpu.shared_expert", &de.compute)?;
            let _s_shared = debug_span!("shared_expert").entered();
            self.issue_shared_expert_graph(de, dgpu_scratch, dlw, layer)?;
            drop(_s_shared);
            _t_shared.end()?;
        }
        if xq_hoisted {
            // Enqueued BEFORE the pick-readback sync, so that sync covers it and
            // the second one below disappears.
            let _t_xq = de.events.stage("dgpu.moe_xq_pre", &de.compute)?;
            de.q8k.launch(
                &de.compute,
                &mut dgpu_scratch.moe_xq,
                &dgpu_scratch.ffn_input_norm,
                crate::config::BLOCKS_Q8K_GATE_IN,
            )?;
            _t_xq.end()?;
        }

        // M56 het-split: the dGPU computes its RESIDENT hot experts during
        // its former MoE wait, in parallel with shared expert + iGPU MoE.
        // Mirrors the iGPU routed_moe pipeline on dGPU scratch; the partial
        // is added at ffn_combine. Input is the same f32 ffn_input_norm both
        // devices hold, so the q8k quantization is bit-identical.
        if parallel {
            if let Some(hot) = dlw.hot_experts.as_ref() {
        // M58.3 leg balance; see het::weights::dgpu_hot_cap (single source of
        // truth — the iGPU leg below must pass the identical value).
        let hot_cap = crate::het::weights::dgpu_hot_cap();
                let mid_blocks_bytes_d = (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES;
                let hot_gbpe = ilw.routed.gate_bytes_per_expert;
                let hot_dbpe = ilw.routed.down_bytes_per_expert;
                let _t_hot = de.events.stage("dgpu.hot_moe", &de.compute)?;
                self.dgpu_graphs.run("dgpu_hot_moe", layer as u32, &de.compute, |s| {
                    de.q8k.launch(s, &mut dgpu_scratch.moe_xq, &dgpu_scratch.ffn_input_norm, BLOCKS_Q8K_GATE_IN)?;
                    super::dispatch::moe_gate_up_batch_hetsplit(
                        de, ilw.routed.gate.dtype, s, &mut dgpu_scratch.moe_mid_cat,
                        &hot.gate, &hot.up,
                        &dgpu_scratch.moe_xq, &dgpu_scratch.d_ew, &dgpu_scratch.d_selected,
                        &hot.remap, /*mode=*/1, hot_cap,
                        hot_gbpe as u32, hot_gbpe as u32,
                        N_EXPERT_USED as u32, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
                    )?;
                    de.q8k.launch(s, &mut dgpu_scratch.moe_midq_cat, &dgpu_scratch.moe_mid_cat, BLOCKS_Q8K_DOWN_IN * (N_EXPERT_USED as u32))?;
                    super::dispatch::moe_down_batched_hetsplit(
                        de, ilw.routed.down.dtype, s, &mut dgpu_scratch.ffn_moe_dgpu,
                        &hot.down, &dgpu_scratch.moe_midq_cat, &dgpu_scratch.d_selected,
                        &hot.remap, /*mode=*/1, hot_cap,
                        hot_dbpe as u32, mid_blocks_bytes_d as u32,
                        N_EXPERT_USED as u32, N_EMBD, BLOCKS_Q8K_DOWN_IN,
                    )?;
                    Ok(())
                })?;
                _t_hot.end()?;
            }
        }

        // M54 pre-issue mode: the whole iGPU MoE lane (wait → graph →
        // push) was enqueued at token start by issue_igpu_moe; skip the
        // inline iGPU section entirely (the dGPU's moe_arrived wait below
        // synchronizes against the pre-issued lane).
        if !igpu_moe_preissued {
        // Switch to iGPU for the MoE.
        self.set_current_cached(self.igpu.device)?;
        let ie = &self.igpu;

        if parallel {
            let _t_wait = ie.events.stage("igpu.moe.wait", &ie.compute)?;
            ie.compute.wait_event(&sev.selected_pushed)?;
            _t_wait.end()?;
        }

        let gbpe = ilw.routed.gate_bytes_per_expert;
        let ubpe = ilw.routed.up_bytes_per_expert;
        let dbpe = ilw.routed.down_bytes_per_expert;

        // ============================================================
        // iGPU: routed MoE → ffn_moe
        // (Per-slot host syncs inside still block iGPU compute, but
        //  dGPU's shared-expert kernels — already queued above in
        //  parallel mode — run concurrently.)
        // ============================================================
        // Fully device-side MoE pipeline (no per-slot host syncs). Each
        // iq2 writes directly into the cat positions, swiglu_cw fires
        // once, q8k_quantize + q2k accumulate then drain the cat via the
        // d_midq_cat staging buffer.
        //
        // The 4-kernel core (q8k_xq → iq2_fused → q8k_mid → q2k_down) is
        // captured into a per-layer HIP graph on first call and replayed
        // thereafter — all params are device pointers + layer-constant
        // scalars, so the graph stays valid for every subsequent token.
        //
        // d_selected and d_ew are device-side here: peer-pushed from
        // dGPU (router runs there) and arriving via the selected_pushed
        // event already waited on above.
        let mid_blocks_bytes = (BLOCKS_Q8K_DOWN_IN as usize) * BLOCK_Q8_K_BYTES;
        let _t_moe = ie.events.stage("igpu.routed_moe", &ie.compute)?;
        let _s_moe = debug_span!("routed_moe").entered();
        // 4-kernel core (q8k_xq → iq2_fused → q8k_mid → q2k_down) captured
        // into a per-layer HIP graph on first call and replayed thereafter.
        // All kernel params are device pointers + layer-constant scalars.
        // Per-kernel event staging stays OUTSIDE the capture; the
        // event-record nodes would otherwise become part of the replayed
        // graph and corrupt the per-token harvest.
        // Decode-path two-box split. Separate env from the prefill one so the
        // two can be validated and rolled back independently — decode has a
        // different assignment (all 40 layers) and a different remap discipline
        // (LRU slots, not a dense window).
        // (`decode_split_on` is bound above, before the shared expert.)

        if let Some(pg) = pager.as_deref_mut() {
            // M7 expert paging: the engine's Q8 routing diverges from the fp8
            // dump, so page the router's ACTUAL picks (read back from d_selected)
            // and run the MoE directly from the pager's pool — no graph capture
            // (correctness-first; the resident graph path stays the default).
            // dlw.hot_experts is None here, so the all-negative remap sends every
            // routed expert to the iGPU pool and the ffn_combine adds no dGPU
            // partial.
            let mut sel_host = vec![0i32; N_EXPERT_USED];
            let t_sel = std::time::Instant::now();
            self.set_current_cached(self.dgpu.device)?;
            de.compute.synchronize()?;
            dgpu_scratch.d_selected.copy_to_host(&mut sel_host)?;
            if super::expert_pager::pick_trace_on() {
                let ids: Vec<String> = sel_host.iter().map(|v| v.to_string()).collect();
                super::expert_pager::pick_trace(&format!("D {layer} {}", ids.join(" ")));
            }
            super::trace::phase::add(
                &super::trace::phase::SEL_SYNC_NS,
                t_sel.elapsed().as_nanos() as u64,
            );
            let mut ids: Vec<u32> = Vec::with_capacity(N_EXPERT_USED);
            // Two-box split, decode path. Box 2 owns a share of EVERY layer
            // (`all:<lo>-383`), unlike the prefill-only encoder assignment, so
            // decode benefits on all 40 layers rather than half of them.
            //
            // Filtering here is what stops this box paging box 2's share — and
            // decode is ~93% miss service, so that is the whole point. It is
            // only safe together with `mark_remote_after_ensure` below: a
            // filtered-out id keeps `ensure`'s default `-(e)-1`, which reads as
            // "ours at slot e" and would compute from whatever sits in that slot.
            // Who computes each pick?
            //
            // DEFAULT (static split): box 2 owns a fixed assignment declared at
            // HELLO; the hub pages everything else from its own disk.
            //
            // T2-CATCHALL (`V41_T2_CATCHALL=1`, daemon `--paged`): box 2 is an LRU
            // over ALL experts against its OWN disk, so the hub computes only what
            // it ALREADY holds and reassigns every miss to box 2. A hub miss then
            // costs box 2's read (<=6.3 ms, overlapped with hub compute) instead of
            // the hub's own dm-crypt read (6.9 ms, blocking the layer). Weights
            // never cross the link — box 2 reads its own copy.
            let catchall = super::expert_pager::ExpertPager::t2_catchall();
            let owns_remote: Option<Vec<bool>> = if decode_split_on {
                if let Some(nloc) = local_picks_override() {
                    // Box 1 takes the first `nloc` distinct picks; box 2 takes the rest.
                    let mut taken = 0usize;
                    let mut mine = vec![false; N_EXPERT as usize];
                    for &sv in &sel_host {
                        if taken >= nloc {
                            break;
                        }
                        if (0..N_EXPERT as i32).contains(&sv) && !mine[sv as usize] {
                            mine[sv as usize] = true;
                            taken += 1;
                        }
                    }
                    Some((0..N_EXPERT).map(|e| !mine[e as usize]).collect())
                } else if catchall && super::expert_pager::ExpertPager::t2_catchall_deterministic() {
                    // `V41_T2_CATCHALL=2` — DETERMINISTIC catch-all: every routed
                    // pick goes to box 2, regardless of what box 1 happens to hold.
                    //
                    // WHY THIS EXISTS. Mode 1 decides the split with
                    // `pg.is_resident(layer, e)`, so the box1/box2 partition is a
                    // function of RESIDENCY, hence of REQUEST HISTORY. A different
                    // partition groups the per-expert partial sums differently and
                    // f32 addition is not associative, so the engine's output
                    // depends on what the server served before. MEASURED 2026-09-14,
                    // same 104K prompt, same binary:
                    //     mode 1, 100K first      -> sha 4275e8cd231d  (fluent)
                    //     mode 1, after a 37-tok  -> sha 5799afaa4959  (DEGENERATE)
                    //     mode 1, after a 32K     -> sha 6d1dbafa35e6  (degenerate)
                    //     mode 0 (static split)   -> sha a7b0e336b5b8, history-INDEPENDENT
                    // Divergence starts ~12 generated tokens in and collapses into
                    // repeated fragments. It is NOT stale KV (`V41_RESET_ZERO=1`
                    // changes nothing), NOT graphs, NOT rope, NOT context length.
                    //
                    // Mode 2 keeps the catch-all's real benefit — box 1 never blocks
                    // on its own dm-crypt disk for a miss — while making the split a
                    // constant. Box 1's decode LRU is only ~25 slots (pool minus the
                    // packed prefill windows), so it was computing ~0.5 picks/layer
                    // anyway; giving those up costs little and buys reproducibility.
                    Some(vec![true; N_EXPERT as usize])
                } else if catchall && super::expert_pager::t2_partition() {
                    // Fixed hash partition of the id space; box 1 pages its
                    // share synchronously through `ensure` (the LRU evicts), box 2
                    // pages its share on its own disk. See `t2_partition`.
                    if let Some(r) = self.remote.as_ref() {
                        if let Ok(c) = r.lock() {
                            super::expert_pager::set_partition_share(pg.decode_slots(), c.info().n_resident);
                        }
                    }
                    Some((0..N_EXPERT).map(|e| super::expert_pager::partition_box2(layer, e)).collect())
                } else if catchall && super::expert_pager::b1_prefetch() {
                    // Box 1 = L1, box 2 = victim tier. Compute what is resident
                    // here, hand the rest to box 2, and queue every miss for the
                    // BACKGROUND fill -- no synchronous admission, no freeze.
                    let o: Vec<bool> = (0..N_EXPERT).map(|e| !pg.is_resident(layer, e)).collect();
                    for &sv in &sel_host {
                        if (0..N_EXPERT as i32).contains(&sv) && o[sv as usize] {
                            pg.prefetch_hint(layer, sv as u32);
                        }
                    }
                    Some(o)
                } else if catchall {
                    // "Not resident here" == "box 2's". Pure lookup, no paging —
                    // EXCEPT while the pool is still filling, where a miss is a
                    // first touch (free slot, no eviction) and worth paying once.
                    //
                    // HISTORY-DEPENDENT — see the mode-2 note above. Prefer mode 2
                    // unless you are reproducing the old behaviour.
                    let mut budget = pg.lru_free_slots();
                    let victim = super::expert_pager::victim_cache();
                    Some(
                        (0..N_EXPERT)
                            .map(|e| {
                                if pg.is_resident(layer, e) {
                                    return false; // ours
                                }
                                // Page into a free slot — but ONLY experts box 2 told
                                // us it missed. Filling from ambient traffic makes box 1
                                // a duplicate of box 2's hot set: MEASURED, a 1396-slot
                                // LRU filled that way was SLOWER than a 25-slot one
                                // (igpu.routed_moe +89% for 8 us/pick of relief).
                                if budget > 0
                                    && sel_host.contains(&(e as i32))
                                    && (!victim || super::expert_pager::box2_missed(layer, e))
                                {
                                    budget -= 1;
                                    super::expert_pager::clear_box2_miss(layer, e);
                                    return false; // ours: page it into a free slot
                                }
                                true // box 2's
                            })
                            .collect(),
                    )
                } else {
                    self.remote.as_ref().and_then(|r| {
                        r.lock().ok().map(|c| {
                            (0..N_EXPERT).map(|e| c.owns(layer as u32, e as i32)).collect()
                        })
                    })
                }
            } else {
                None
            };
            // Claim cap: flip surplus LOCAL picks to remote, in `sel_host` order, so a
            // layer never pushes box 1 past the crossover. No-op when unset.
            let owns_remote = match (owns_remote, local_claim_max()) {
                (Some(mut o), Some(cap)) => {
                    let mut local = 0usize;
                    let mut seen: Vec<bool> = vec![false; N_EXPERT as usize];
                    for &sv in &sel_host {
                        if !(0..N_EXPERT as i32).contains(&sv) || seen[sv as usize] {
                            continue;
                        }
                        seen[sv as usize] = true;
                        if !o[sv as usize] {
                            local += 1;
                            if local > cap {
                                o[sv as usize] = true; // hand it to box 2
                            }
                        }
                    }
                    Some(o)
                }
                (o, _) => o,
            };
            for &sv in &sel_host {
                if (0..N_EXPERT as i32).contains(&sv) && !ids.contains(&(sv as u32)) {
                    if let Some(o) = owns_remote.as_ref() {
                        if o[sv as usize] {
                            continue;
                        }
                    }
                    ids.push(sv as u32);
                }
            }
            // Ship this token's activations to box 2 BEFORE the pager runs.
            //
            // This used to sit AFTER `ensure`, which meant box 2 idled through the
            // whole of box 1's paging stall — measured at 37.6-39.8 ms/token of
            // `sel_sync_us` plus the miss reads, every layer — and only then started
            // its own work (including its OWN paging). Nothing in the submit depends
            // on `ensure`: it needs `ffn_input_norm` (computed in pre-MoE), the
            // router picks (already read back into `sel_host`, that is how `ids` was
            // built) and `owns_remote`. Only `mark_remote_after_ensure` genuinely has
            // to follow the pager, and it stays below.
            if owns_remote.is_some() {
                let xq_bytes = (crate::config::BLOCKS_Q8K_GATE_IN as usize)
                    * crate::q8_k::BLOCK_Q8_K_BYTES;
                self.dgpu.device.set_current()?;
                self.current_device
                    .store(self.dgpu.device.id, std::sync::atomic::Ordering::Relaxed);
                if !xq_hoisted {
                    de.q8k.launch(
                        &de.compute,
                        &mut dgpu_scratch.moe_xq,
                        &dgpu_scratch.ffn_input_norm,
                        crate::config::BLOCKS_Q8K_GATE_IN,
                    )?;
                    // Second full stream sync per layer. When `xq_hoisted`, the
                    // pick-readback sync above already covered this quantize.
                    de.compute.synchronize()?;
                }
                let mut xq_host = vec![0u8; xq_bytes];
                dgpu_scratch
                    .moe_xq
                    .slice_view(0, xq_bytes)
                    .copy_to_host(&mut xq_host)?;
                let mut ew_host = vec![0f32; N_EXPERT_USED];
                dgpu_scratch
                    .d_ew
                    .slice_view(0, N_EXPERT_USED)
                    .copy_to_host(&mut ew_host)?;
                let (sel_remote, ew_remote): (Vec<i32>, Vec<f32>) = match owns_remote.as_ref() {
                    Some(o) if std::env::var("V41_REMOTE_NOMASK").as_deref() != Ok("1") => sel_host
                        .iter()
                        .zip(ew_host.iter())
                        .map(|(&sv, &w)| {
                            if (0..N_EXPERT as i32).contains(&sv) && o[sv as usize] {
                                (sv, w)
                            } else {
                                (super::remote_experts::NO_PICK, 0.0f32)
                            }
                        })
                        .unzip(),
                    _ => (sel_host.clone(), ew_host.clone()),
                };
                let t_sub = super::perfetto::now_ns();
                let ticket = self
                    .remote
                    .as_ref()
                    .ok_or_else(|| eyre!("decode split on but no remote client"))?
                    .lock()
                    .map_err(|_| eyre!("remote expert client mutex poisoned"))?
                    // UNMASKED under the catch-all: the hub has already decided this
                    // partition in `owns_remote`, and box 2 (`--paged`) accepts any
                    // expert. Masking here by box 2's ADVERTISED set silently dropped
                    // 77% of routed picks — see `submit_unmasked`.
                    //
                    // BUT the picks box 1 keeps ("ours", owns_remote false) MUST be
                    // blanked to NO_PICK / weight 0: box 2's `run_path` computes every
                    // pick it is handed, and `ffn_combine` adds its partial to the
                    // local one, so an unblanked local pick was computed TWICE and
                    // DOUBLE-ADDED. `verify_routing_exactly_once` cannot see it — it
                    // validates the hub's claim, not what box 2 does. Found 2026-09-17;
                    // it is what made mode-1 output depend on box 1's residency
                    // history ("DEGENERATE after a 37-tok request", KNOWN_BUGS #0).
                    // `V41_REMOTE_NOMASK=1` reproduces the old behaviour.
                    .submit_unmasked(layer as u32, 1, &xq_host, &sel_remote, &ew_remote, true)?;
                if let Some(pf) = self.perfetto.as_ref() {
                    if let Ok(pf) = pf.lock() {
                        let _ = pf.emit_host_slice(
                            pf.remote_uuid,
                            &format!("decode submit L{layer}"),
                            t_sub,
                            super::perfetto::now_ns(),
                        );
                    }
                }
                dgpu_scratch.remote_ticket = ticket;
                if dgpu_scratch.remote_ticket.is_some() {
                    dgpu_scratch.remote_sel.clear();
                    dgpu_scratch.remote_sel.extend_from_slice(&sel_host);
                }
            }
            if defer_shared {
                // Box 2 is now working. Everything from here to `remote.wait` is
                // free real estate on de.compute, so the shared expert lands in
                // it instead of delaying the submit. Still BEFORE the iGPU device
                // switch, so `de.compute` is the current device's stream.
                let _t_shared = de.events.stage("dgpu.shared_expert", &de.compute)?;
                let _s_shared = debug_span!("shared_expert").entered();
                self.issue_shared_expert_graph(de, dgpu_scratch, dlw, layer)?;
                drop(_s_shared);
                _t_shared.end()?;
            }
            self.set_current_cached(self.igpu.device)?;
            // d_selected is already on the iGPU (peer-pushed + selected_pushed waited
            // above) and equals what we read back, so do NOT overwrite it — it is a
            // view into the packed sel+ew buffer and a host copy here would clobber d_ew.
            let t_ens = std::time::Instant::now();
            let t_ens_ns = super::perfetto::now_ns();
            if super::expert_pager::pager_batch_miss() {
                pg.ensure_batched(layer, &ids)?;
            } else {
                pg.ensure(layer, &ids)?;
            }
            if let Some(pf) = self.perfetto.as_ref() {
                if let Ok(pf) = pf.lock() {
                    let _ = pf.emit_host_slice(
                        pf.pager_uuid,
                        &format!("ensure L{layer} ids={}", ids.len()),
                        t_ens_ns,
                        super::perfetto::now_ns(),
                    );
                }
            }
            if let Some(o) = owns_remote.as_ref() {
                // MUST follow `ensure` and must only touch the remote entries;
                // see the note on the method. The submit itself now runs ABOVE
                // `ensure` so box 2 works during box 1's paging stall.
                pg.mark_remote_after_ensure(|e| o[e as usize])?;
            }
            verify_routing_exactly_once(layer, &sel_host, pg.remap(), owns_remote.as_deref())?;
            super::trace::phase::add(
                &super::trace::phase::ENSURE_NS,
                t_ens.elapsed().as_nanos() as u64,
            );
            if std::env::var("V41_PAGER_DBG").is_ok() && layer == 0 {
                let mut rmp = vec![0i32; N_EXPERT as usize];
                pg.remap_dev.copy_to_host(&mut rmp)?;
                let sel_remap: Vec<i32> = ids.iter().map(|&i| rmp[i as usize]).collect();
                eprintln!("[pager-dbg] L{layer} sel={sel_host:?} ids={ids:?} remap[sel]={sel_remap:?} cap={} misses={} | pg_gbpe={} ilw_gbpe={} pg_ubpe={} ilw_ubpe={} pg_dbpe={} ilw_dbpe={} gdt={:?} ilw_gdt={:?} ddt={:?} ilw_ddt={:?} pg_nslots={} ilw_nslots={}",
                    crate::het::weights::dgpu_hot_cap(), pg.misses(),
                    pg.routed.gate_bytes_per_expert, ilw.routed.gate_bytes_per_expert,
                    pg.routed.up_bytes_per_expert, ilw.routed.up_bytes_per_expert,
                    pg.routed.down_bytes_per_expert, ilw.routed.down_bytes_per_expert,
                    pg.routed.gate.dtype, ilw.routed.gate.dtype,
                    pg.routed.down.dtype, ilw.routed.down.dtype,
                    pg.routed.n_slots, ilw.routed.n_slots);
            }
            // Under the two-box split the cap MUST be the full top-k. It is a
            // per-token RANK test over resident (other-device) picks, and the
            // kernel sends OVER-cap picks back to this device using the RAW
            // expert id — which we deliberately did not page. A cap of 4 (the
            // default) would therefore compute two of every six picks from
            // whatever sits in that pool slot, silently.
            let hot_cap_i = if decode_split_on {
                N_EXPERT_USED as u32
            } else {
                crate::het::weights::dgpu_hot_cap()
            };
            let pg_gbpe = pg.routed.gate_bytes_per_expert as u32;
            let pg_ubpe = pg.routed.up_bytes_per_expert as u32;
            let pg_dbpe = pg.routed.down_bytes_per_expert as u32;
            let gdt = pg.routed.gate.dtype;
            let ddt = pg.routed.down.dtype;
            // Diagnostic: V41_PAGER_RESIDENT drives the graph from the resident 384
            // buffer with a raw-id remap (keeping hot_experts=None) to isolate the pool
            // buffer from the hot_experts=None combine.
            let resident_dbg = std::env::var("V41_PAGER_RESIDENT").is_ok();
            if resident_dbg {
                let raw: Vec<i32> = (0..N_EXPERT as i32).map(|e| -e - 1).collect();
                pg.remap_dev.copy_from_host(&raw)?;
            }
            let (gbuf, ubuf, dbuf) = if resident_dbg {
                (&ilw.routed.gate.buffer, &ilw.routed.up.buffer, &ilw.routed.down.buffer)
            } else {
                (&pg.routed.gate.buffer, &pg.routed.up.buffer, &pg.routed.down.buffer)
            };
            // Run the paged MoE through the SAME captured-graph mechanism as the
            // resident path (a distinct key), not a bare launch: the pool + remap_dev
            // buffers are pointer-stable, only their contents change per token, so the
            // graph stays valid. A bare launch on ie.compute produced wrong output —
            // the resident path's correctness depends on the graph wrapper.
            // V41_PAGER_NOGRAPH=1: bypass graph capture and launch directly, to A/B the
            // capture wrapper against a bare launch on ie.compute.
            let mut paged_moe = |s: &v4flash_hip::Stream| -> eyre::Result<()> {
                ie.q8k.launch(s, &mut igpu_scratch.d_xq_q8k, &igpu_scratch.ffn_input_norm_recv, BLOCKS_Q8K_GATE_IN)?;
                super::dispatch::moe_gate_up_batch_hetsplit(
                    ie, gdt, s, &mut igpu_scratch.d_mid_cat,
                    gbuf, ubuf,
                    &igpu_scratch.d_xq_q8k, &igpu_scratch.d_ew, &igpu_scratch.d_selected,
                    &pg.remap_dev, /*mode=*/ 0, hot_cap_i, pg_gbpe, pg_ubpe,
                    N_EXPERT_USED as u32, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
                )?;
                ie.q8k.launch(s, &mut igpu_scratch.d_midq_cat, &igpu_scratch.d_mid_cat, BLOCKS_Q8K_DOWN_IN * (N_EXPERT_USED as u32))?;
                super::dispatch::moe_down_batched_hetsplit(
                    ie, ddt, s, &mut igpu_scratch.ffn_moe,
                    dbuf, &igpu_scratch.d_midq_cat, &igpu_scratch.d_selected,
                    &pg.remap_dev, /*mode=*/ 0, hot_cap_i, pg_dbpe, mid_blocks_bytes as u32,
                    N_EXPERT_USED as u32, N_EMBD, BLOCKS_Q8K_DOWN_IN,
                )?;
                Ok(())
            };
            if std::env::var("V41_PAGER_NOGRAPH").is_ok() {
                paged_moe(&ie.compute)?;
            } else {
                self.igpu_graphs.run("routed_moe_paged", layer as u32, &ie.compute, paged_moe)?;
            }
        } else {
        self.igpu_graphs.run("routed_moe", layer as u32, &ie.compute, |s| {
            ie.q8k.launch(s, &mut igpu_scratch.d_xq_q8k, &igpu_scratch.ffn_input_norm_recv, BLOCKS_Q8K_GATE_IN)?;
            if let Some(remap) = ilw.hot_remap.as_ref() {
                // M56: skip dGPU-resident slots (the dGPU computes those, up
                // to the cap — overflow comes back here, unless M63 de-dup
                // pinned the cap at N_EXPERT_USED, which removes overflow).
                let hot_cap_i = crate::het::weights::dgpu_hot_cap();
                super::dispatch::moe_gate_up_batch_hetsplit(ie, ilw.routed.gate.dtype, s, &mut igpu_scratch.d_mid_cat, &ilw.routed.gate.buffer, &ilw.routed.up.buffer, &igpu_scratch.d_xq_q8k, &igpu_scratch.d_ew, &igpu_scratch.d_selected, remap, /*mode=*/0, hot_cap_i, gbpe as u32, ubpe as u32, N_EXPERT_USED as u32, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN)?;
                ie.q8k.launch(s, &mut igpu_scratch.d_midq_cat, &igpu_scratch.d_mid_cat, BLOCKS_Q8K_DOWN_IN * (N_EXPERT_USED as u32))?;
                super::dispatch::moe_down_batched_hetsplit(ie, ilw.routed.down.dtype, s, &mut igpu_scratch.ffn_moe, &ilw.routed.down.buffer, &igpu_scratch.d_midq_cat, &igpu_scratch.d_selected, remap, /*mode=*/0, hot_cap_i, dbpe as u32, mid_blocks_bytes as u32, N_EXPERT_USED as u32, N_EMBD, BLOCKS_Q8K_DOWN_IN)?;
            } else {
                // The plain kernels index by RAW expert id, so they cannot
                // read a de-duplicated buffer. Load-time validation should
                // have made this unreachable; fail loudly if it didn't.
                if ilw.igpu_packed {
                    return Err(eyre!(
                        "L{layer}: iGPU experts are packed (IGPU_DEDUP_HOT) but hot_remap is \
                         missing — the plain MoE path would read the wrong experts"
                    ));
                }
                super::dispatch::moe_gate_up_batch(ie, ilw.routed.gate.dtype, s, &mut igpu_scratch.d_mid_cat, &ilw.routed.gate.buffer, &ilw.routed.up.buffer, &igpu_scratch.d_xq_q8k, &igpu_scratch.d_ew, &igpu_scratch.d_selected, gbpe as u32, ubpe as u32, N_EXPERT_USED as u32, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN)?;
                ie.q8k.launch(s, &mut igpu_scratch.d_midq_cat, &igpu_scratch.d_mid_cat, BLOCKS_Q8K_DOWN_IN * (N_EXPERT_USED as u32))?;
                super::dispatch::moe_down_batched(ie, ilw.routed.down.dtype, s, &mut igpu_scratch.ffn_moe, &ilw.routed.down.buffer, &igpu_scratch.d_midq_cat, &igpu_scratch.d_selected, dbpe as u32, mid_blocks_bytes as u32, N_EXPERT_USED as u32, N_EMBD, BLOCKS_Q8K_DOWN_IN)?;
            }
            Ok(())
        })?;
        }
        // `selected` is not materialized on host — d_selected stays
        // device-side, peer-pushed from dGPU above.
        drop(_s_moe);
        _t_moe.end()?;

        // ============================================================
        // iGPU → dGPU: peer push ffn_moe (16 KB)
        // ============================================================
        if parallel {
            sev.moe_done.record(&ie.compute)?;
            let _t_wait = ie
                .events
                .stage("igpu.peer_push_ffn_moe.wait", &ie.xfer)?;
            ie.xfer.wait_event(&sev.moe_done)?;
            _t_wait.end()?;
        } else {
            ie.compute.synchronize()?;
        }
        let _t_peer_moe = ie.events.stage("igpu.peer_push_ffn_moe", &ie.xfer)?;
        let _s_peer_moe = debug_span!("peer_push_ffn_moe").entered();
        peer_push_f32(
            &igpu_scratch.ffn_moe,
            &mut dgpu_scratch.ffn_moe_recv,
            &ie.xfer,
        )?;
        if parallel {
            sev.moe_arrived.record(&ie.xfer)?;
        } else {
            ie.xfer.synchronize()?;
        }
        drop(_s_peer_moe);
        _t_peer_moe.end()?;
        } // !igpu_moe_preissued

        // ============================================================
        // dGPU: in serial mode, issue shared expert NOW. In parallel
        // mode, it was issued earlier — just wait for ffn_moe to arrive.
        // ============================================================
        self.set_current_cached(self.dgpu.device)?;
        if !parallel {
            let _t_shared = de.events.stage("dgpu.shared_expert", &de.compute)?;
            let _s_shared = debug_span!("shared_expert").entered();
            self.issue_shared_expert_graph(de, dgpu_scratch, dlw, layer)?;
            drop(_s_shared);
            _t_shared.end()?;
        }

        // ffn_moe_recv += ffn_shared. In parallel mode wait for the
        // peer copy to land before doing the add.
        if parallel {
            let _t_wait = de.events.stage("dgpu.ffn_combine.wait", &de.compute)?;
            de.compute.wait_event(&sev.moe_arrived)?;
            _t_wait.end()?;
        }
        // KNOWN_BUGS #0b: split the MoE half into shared vs routed. Dumped
        // BEFORE `ffn_moe_recv += ffn_shared`, so ffn_moe_recv is routed-only.
        if super::engine::subtensor_dump_armed(layer as usize) {
            de.compute.synchronize()?;
            super::engine::maybe_dump_subtensor_f32_view(
                layer as usize,
                &format!("dec_ffn_shared_p{pos}"),
                &dgpu_scratch.ffn_shared.slice_view(0, N_EMBD as usize),
            )?;
            super::engine::maybe_dump_subtensor_f32_view(
                layer as usize,
                &format!("dec_ffn_routed_p{pos}"),
                &dgpu_scratch.ffn_moe_recv.slice_view(0, N_EMBD as usize),
            )?;
        }
        // Two-box split: collect box 2's partial and add it.
        //
        // MUST be after the `moe_arrived` wait above — `ffn_moe_recv` is the
        // landing buffer for the iGPU's peer push, so adding before the push
        // lands means the push overwrites the addition. (That exact mistake cost
        // a long debug on the prefill path: the buffer visibly changed and the
        // output was bit-identical anyway.)
        if let Some(t) = dgpu_scratch.remote_ticket.take() {
            let t_wait = super::perfetto::now_ns();
            let remote = self
                .remote
                .as_ref()
                .ok_or_else(|| eyre!("decode remote ticket pending but no client"))?;
            let partial = remote
                .lock()
                .map_err(|_| eyre!("remote expert client mutex poisoned"))?
                .wait(t)?;
            if let Some(pf) = self.perfetto.as_ref() {
                if let Ok(pf) = pf.lock() {
                    let _ = pf.emit_host_slice(
                        pf.remote_uuid,
                        &format!(
                            "decode wait L{layer} rtt={}us link={}us remote={}us page={}us/{}",
                            partial.rtt_us, partial.link_us(), partial.t_remote_compute_us,
                            partial.t_remote_page_us, partial.n_remote_miss,
                        ),
                        t_wait,
                        super::perfetto::now_ns(),
                    );
                    super::forward_prefill::emit_remote_page_slice(
                        &pf, layer as u32, &partial, super::perfetto::now_ns(),
                    );
                }
            }
            super::trace::phase::add(
                &super::trace::phase::REMOTE_RTT_NS,
                (super::perfetto::now_ns() - t_wait) as u64,
            );
            // Box 2's own service time for this exchange, so the summary can
            // say whether the wait was box 2 working or the link.
            super::trace::phase::add(
                &super::trace::phase::REMOTE_SRV_NS,
                (partial.t_remote_server_us as u64) * 1000,
            );
            // Box 2 reports which of these picks it had to page. Mark them so box 1's
            // decode LRU fills from box-2 MISSES instead of ambient traffic — the
            // difference between an exclusive cache (worth +6547 us/pick) and a
            // duplicate of box 2's hot set (worth -53 us/pick). See
            // `expert_pager::BOX2_MISSED`.
            if partial.miss_mask != 0 {
                for (i, &sv) in dgpu_scratch
                    .remote_sel
                    .iter()
                    .take(super::remote_experts::proto::RESP_MISS_BITS)
                    .enumerate()
                {
                    if partial.miss_mask & (1 << i) != 0 && (0..N_EXPERT as i32).contains(&sv) {
                        super::expert_pager::mark_box2_miss(layer, sv as u32);
                    }
                }
            }
            let src = partial.f32();
            if src.len() != N_EMBD as usize {
                return Err(eyre!(
                    "L{layer}: decode remote partial has {} f32, expected {N_EMBD}",
                    src.len()
                ));
            }
            self.set_current_cached(self.dgpu.device)?;
            dgpu_scratch
                .remote_ffn_moe
                .as_mut()
                .ok_or_else(|| eyre!("decode remote partial but buffer unallocated"))?
                .copy_from_host(src)?;
            let rem = dgpu_scratch.remote_ffn_moe.as_ref().unwrap();
            de.vec_add.launch(&de.compute, &mut dgpu_scratch.ffn_moe_recv, rem, N_EMBD)?;
            remote
                .lock()
                .map_err(|_| eyre!("remote expert client mutex poisoned"))?
                .recycle(partial);
        }

        // V41_PAGER_DBG: prove whether the iGPU MoE partial actually reaches the
        // dGPU combine. Dumps |ffn_moe| (iGPU side) and |ffn_moe_recv| (dGPU side)
        // right before ffn_combine consumes them.
        if std::env::var("V41_PAGER_DBG").is_ok() && layer == 0 {
            de.compute.synchronize()?;
            self.igpu.compute.synchronize()?;
            let n2 = |v: &[f32]| -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() };
            let mut moe_i = vec![0f32; N_EMBD as usize];
            self.set_current_cached(self.igpu.device)?;
            igpu_scratch.ffn_moe.slice_view(0, N_EMBD as usize).copy_to_host(&mut moe_i)?;
            self.set_current_cached(self.dgpu.device)?;
            let mut moe_r = vec![0f32; N_EMBD as usize];
            let mut shared = vec![0f32; N_EMBD as usize];
            dgpu_scratch.ffn_moe_recv.slice_view(0, N_EMBD as usize).copy_to_host(&mut moe_r)?;
            dgpu_scratch.ffn_shared.slice_view(0, N_EMBD as usize).copy_to_host(&mut shared)?;
            self.set_current_cached(self.igpu.device)?;
            let mut ew = vec![0f32; N_EXPERT_USED];
            let mut sel_i = vec![0i32; N_EXPERT_USED];
            let mut mid = vec![0f32; N_EXPERT_USED * N_FF_EXP as usize];
            igpu_scratch.d_ew.slice_view(0, N_EXPERT_USED).copy_to_host(&mut ew)?;
            igpu_scratch.d_selected.slice_view(0, N_EXPERT_USED).copy_to_host(&mut sel_i)?;
            igpu_scratch.d_mid_cat.slice_view(0, mid.len()).copy_to_host(&mut mid)?;
            let mut fin = vec![0f32; N_EMBD as usize];
            igpu_scratch.ffn_input_norm_recv.slice_view(0, N_EMBD as usize).copy_to_host(&mut fin)?;
            self.set_current_cached(self.dgpu.device)?;
            eprintln!(
                "[moe-dbg] L{layer} paged={} |ffn_moe(igpu)|={:.4e} |ffn_moe_recv(dgpu)|={:.4e} |ffn_shared|={:.4e} |d_mid_cat|={:.4e} |ffn_in_recv(igpu)|={:.4e} sel_igpu={:?}",
                pager.is_some(), n2(&moe_i), n2(&moe_r), n2(&shared), n2(&mid), n2(&fin), sel_i
            );
        }
        // Non-last layers launch the fused cross-layer graph (this
        // layer's ffn_combine + next layer's mhc_pre_attn) for a single
        // graph submit, eliminating the host-scheduling gap between
        // them. The last layer fires a pure combine.
        let combine_label = if is_last_layer {
            "dgpu.ffn_combine"
        } else {
            "dgpu.ffn_combine+next_pre_attn"
        };
        let _t_combine = de.events.stage(combine_label, &de.compute)?;
        let _s_combine = debug_span!("ffn_combine").entered();
        if is_last_layer {
            // (vec_add → hc_post writing residual_next). Stable per-layer
            // pointers thanks to the end-of-token extra swap. Used ONLY
            // for the last layer; non-last layers ride the combined graph.
            self.dgpu_graphs.run("ffn_combine", layer as u32, &de.compute, |s| {
                de.vec_add.launch(s, &mut dgpu_scratch.ffn_moe_recv, &dgpu_scratch.ffn_shared, N_EMBD)?;
                if dlw.hot_experts.is_some() {
                    // M56: + the dGPU's resident-expert MoE partial.
                    de.vec_add.launch(s, &mut dgpu_scratch.ffn_moe_recv, &dgpu_scratch.ffn_moe_dgpu, N_EMBD)?;
                }
                de.hc_post.launch_from_split(s, &mut dgpu_scratch.residual_next, &dgpu_scratch.ffn_moe_recv, &dgpu_scratch.after_attn_hc, &dgpu_scratch.split, N_HC, N_EMBD, N_HC)?;
                Ok(())
            })?;
        } else {
            // Combined ffn_combine_N + mhc_pre_attn_{N+1} graph. The
            // mhc_pre_attn block reads from layer N+1's `residual` which
            // is THIS layer's `residual_next` after the post-layer swap
            // — same physical buffer. We pass `residual_next.raw()`
            // throughout so the captured graph references one ptr.
            let next = next_dlw.expect("combined_ffn_pre_attn: next_dlw required for non-last layer");
            self.dgpu_graphs.run("combined_ffn_pre_attn", layer as u32, &de.compute, |s| {
                // ffn_combine half — writes residual_next (= layer N+1's residual).
                de.vec_add.launch(s, &mut dgpu_scratch.ffn_moe_recv, &dgpu_scratch.ffn_shared, N_EMBD)?;
                if dlw.hot_experts.is_some() {
                    // M56: + the dGPU's resident-expert MoE partial.
                    de.vec_add.launch(s, &mut dgpu_scratch.ffn_moe_recv, &dgpu_scratch.ffn_moe_dgpu, N_EMBD)?;
                }
                de.hc_post.launch_from_split(s, &mut dgpu_scratch.residual_next, &dgpu_scratch.ffn_moe_recv, &dgpu_scratch.after_attn_hc, &dgpu_scratch.split, N_HC, N_EMBD, N_HC)?;
                // mhc_pre_attn half — reads residual_next (= layer N+1's residual
                // after swap), uses layer N+1's hc/norm weights.
                let mode = std::env::var("RMS_NW_MW").unwrap_or_else(|_| "fused".into());
                match mode.as_str() {
                    "0" | "single" => {
                        de.rms_nw.launch(s, &mut dgpu_scratch.flat, &dgpu_scratch.residual_next, 1, HC_DIM, RMS_EPS)?;
                        de.f16.matvec(s, &mut dgpu_scratch.mix, &next.hc_attn_fn.buffer, &dgpu_scratch.flat, HC_MIX_DIM, HC_DIM)?;
                    }
                    "split" => {
                        de.rms_nw_mw.launch(s, &mut dgpu_scratch.flat, &dgpu_scratch.residual_next, &mut dgpu_scratch.rms_nw_partials, HC_DIM, 16, RMS_EPS)?;
                        de.f16.matvec(s, &mut dgpu_scratch.mix, &next.hc_attn_fn.buffer, &dgpu_scratch.flat, HC_MIX_DIM, HC_DIM)?;
                    }
                    _ => {
                        de.rms_nw_mw.launch_inv_only(s, &mut dgpu_scratch.rms_nw_inv_scalar, &dgpu_scratch.residual_next, &mut dgpu_scratch.rms_nw_partials, HC_DIM, 16, RMS_EPS)?;
                        let ksplit: u32 = std::env::var("F16_KSPLIT")
                            .ok().and_then(|s| s.parse().ok()).unwrap_or(HC_DIM / 1024); // k_chunk ≤ 1024 (kernel LDS): 16 @4096-wide, 20 @5120
                        if ksplit > 0 {
                            de.f16.matvec_narrow_ksplit_pre_scaled(
                                s, &mut dgpu_scratch.mix, &next.hc_attn_fn.buffer,
                                &dgpu_scratch.residual_next, &dgpu_scratch.rms_nw_inv_scalar,
                                &mut dgpu_scratch.mhc_matvec_partials,
                                HC_MIX_DIM, HC_DIM, ksplit,
                            )?;
                        } else {
                            de.f16.matvec_pre_scaled(s, &mut dgpu_scratch.mix, &next.hc_attn_fn.buffer, &dgpu_scratch.residual_next, &dgpu_scratch.rms_nw_inv_scalar, HC_MIX_DIM, HC_DIM)?;
                        }
                    }
                }
                de.hc_sinkhorn.launch(s, &mut dgpu_scratch.split, &dgpu_scratch.mix, &next.hc_attn_scale, &next.hc_attn_base, N_HC, SINKHORN_ITERS, SINKHORN_EPS)?;
                if cfg!(feature = "v41") {
                    // Single-pass mHC (ARCH_SPEC §1.1): collapse with the PREVIOUS
                    // sub-block's pre, then carry this sub-block's pre forward.
                    de.hc_weighted.launch(s, &mut dgpu_scratch.attn_cur, &dgpu_scratch.residual_next, &dgpu_scratch.hc_pre_carry, N_EMBD, N_HC)?;
                    let cur_pre = dgpu_scratch.split.slice_view(0, N_HC as usize);
                    let mut carry = dgpu_scratch.hc_pre_carry.slice_view_mut(0, N_HC as usize);
                    carry.copy_from_buffer_async(&cur_pre, s)?;
                } else {
                    de.hc_weighted.launch(s, &mut dgpu_scratch.attn_cur, &dgpu_scratch.residual_next, &dgpu_scratch.split, N_EMBD, N_HC)?;
                }
                // RMS_W_MW=1 enables multi-WG weighted RMS. Default OFF: at
                // N_EMBD=4096 the single-WG version is small enough that
                // multi-WG's extra kernel launch erases any parallelism
                // win — averaged across 256-token decode runs the variants
                // are statistical ties (~±2% within thermal noise).
                if std::env::var("RMS_W_MW").map(|v| v != "0").unwrap_or(false) {
                    de.rms_nw_mw.launch_weighted(s, &mut dgpu_scratch.attn_input_norm, &dgpu_scratch.attn_cur, &next.attn_norm, &mut dgpu_scratch.rms_nw_partials, N_EMBD, 16, RMS_EPS)?;
                } else {
                    de.rms_w.launch_weighted(s, &mut dgpu_scratch.attn_input_norm, &dgpu_scratch.attn_cur, &next.attn_norm, N_EMBD, RMS_EPS)?;
                }
                Ok(())
            })?;
        }
        drop(_s_combine);
        _t_combine.end()?;
        if serial {
            de.compute.synchronize()?;
        }
        Ok(())
    }
}

impl HeterogeneousEngine {
    /// Capture the dGPU shared-expert chain (6 kernels, all layer-constant
    /// params) into a HIP graph on first call; replay thereafter. Caller is
    /// responsible for the surrounding events stage and span.
    fn issue_shared_expert_graph(
        &self,
        de: &DeviceEngine,
        dgpu_scratch: &mut DgpuScratch,
        dlw: &DgpuLayerWeights,
        layer: i32,
    ) -> eyre::Result<()> {
        self.dgpu_graphs.run("shared_expert", layer as u32, &de.compute, |s| {
            issue_shared_expert(de, dgpu_scratch, dlw, s)
        })
    }
}

/// Issue all dGPU shared-expert kernels on `stream` (= `de.compute` in
/// practice). Reads `ffn_input_norm`, writes `ffn_shared`. Used by both
/// modes; in `HetParallel` it's invoked earlier to overlap with iGPU MoE.
fn issue_shared_expert(
    de: &DeviceEngine,
    dgpu_scratch: &mut DgpuScratch,
    dlw: &DgpuLayerWeights,
    stream: &v4flash_hip::Stream,
) -> eyre::Result<()> {
    use super::dispatch::{any_q8, dense_matvec};
    // Q8_0 consumers need the (xq, xscale) pair; K-quant weights (unsloth
    // mix: Q5_K gate/up, Q6_K down — blk.26 keeps Q8_0 down) take f32
    // directly, so each quantize runs only when something consumes it.
    if any_q8(&[&dlw.shared.gate, &dlw.shared.up]) {
        de.q8.quantize_input(stream, &mut dgpu_scratch.xq_n_embd, &mut dgpu_scratch.xscale_n_embd, &dgpu_scratch.ffn_input_norm, N_EMBD)?;
    }
    dense_matvec(de, stream, &mut dgpu_scratch.gate_sh, &dlw.shared.gate, &dgpu_scratch.ffn_input_norm, &dgpu_scratch.xq_n_embd, &dgpu_scratch.xscale_n_embd, N_FF_SHARED, N_EMBD)?;
    dense_matvec(de, stream, &mut dgpu_scratch.up_sh, &dlw.shared.up, &dgpu_scratch.ffn_input_norm, &dgpu_scratch.xq_n_embd, &dgpu_scratch.xscale_n_embd, N_FF_SHARED, N_EMBD)?;
    // ds4 5bc1e6d: shared experts use the same swiglu_limit clamp as routed
    // experts (official V4-Flash graph).
    de.swiglu.launch_clamped(stream, &mut dgpu_scratch.mid_sh, &dgpu_scratch.gate_sh, &dgpu_scratch.up_sh, N_FF_SHARED, SWIGLU_CLAMP_EXP)?;
    match dlw.shared.down.dtype {
        // dp4a GEMM@B=1: 0.019 vs 0.032 ms scalar gemv on [2048->4096]
        // (bench_kquant_dense_isolated) — ~0.55 ms/token across 43 layers.
        v4flash_core::gguf::GgufType::Q6_K => {
            de.q8k.launch(stream, &mut dgpu_scratch.mid_sh_q8k, &dgpu_scratch.mid_sh, 8)?;
            de.dense_gemm.gemm(
                stream,
                v4flash_core::gguf::GgufType::Q6_K,
                &mut dgpu_scratch.ffn_shared,
                &dlw.shared.down.buffer,
                &dgpu_scratch.mid_sh_q8k,
                1,
                N_EMBD,
                8,
            )?;
        }
        _ => {
            if any_q8(&[&dlw.shared.down]) {
                de.q8.quantize_input(stream, &mut dgpu_scratch.mid_sh_xq, &mut dgpu_scratch.mid_sh_xscale, &dgpu_scratch.mid_sh, N_FF_SHARED)?;
            }
            dense_matvec(de, stream, &mut dgpu_scratch.ffn_shared, &dlw.shared.down, &dgpu_scratch.mid_sh, &dgpu_scratch.mid_sh_xq, &dgpu_scratch.mid_sh_xscale, N_EMBD, N_FF_SHARED)?;
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn softplus_stable(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0f32 + x.exp()).ln()
    }
}

#[allow(dead_code)]
fn topk_desc(score: &[f32], k: usize) -> [i32; 6] {
    let mut idx = [-1i32; 6];
    for i in 0..score.len() {
        for j in 0..k {
            if idx[j] < 0 || score[i] > score[idx[j] as usize] {
                for m in (j + 1..k).rev() {
                    idx[m] = idx[m - 1];
                }
                idx[j] = i as i32;
                break;
            }
        }
    }
    idx
}

// Suppress dead-code warning for now (used by forward_head + tests).
#[allow(dead_code)]
fn _unused_imports_warn_suppressor(_d: &Device, _b: &DeviceBuffer<f32>, _e: &DeviceEngine) {}
