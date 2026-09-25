//! Per-layer dtype dispatch for the het/ forward pass (unsloth UD mix).
//!
//! The antirez mix is uniform (IQ2_XXS gate/up, Q2_K down, Q8_0 dense), so
//! the forward pass historically called one kernel per site. The unsloth
//! mix varies per layer (blk.26: IQ2_S gate/up + MXFP4 down + Q6_K q_a +
//! Q8_0 shexp-down; blk.42: MXFP4 down; elsewhere IQ3_XXS down, Q5_K/Q6_K
//! shexp + q_a, Q4_K head). These free functions match on the
//! `DeviceWeight::dtype` recorded at load — free functions rather than
//! methods so call sites keep their existing split borrows (the
//! `laguna.rs:997-1034` pattern).
//!
//! Graph-capture safety: dtype is fixed per layer and the graph cache is
//! keyed `(name, layer)`, so a captured graph always replays the same
//! kernel choice.

use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::GgufType;
use v4flash_hip::{DeviceBuffer, Stream};

use crate::weights::DeviceWeight;

use super::engine::DeviceEngine;

/// Decode MoE gate/up (fused SwiGLU) — batched over selected experts.
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_up_batch(
    e: &DeviceEngine,
    dt: GgufType,
    s: &Stream,
    mid: &mut DeviceBuffer<f32>,
    gate: &DeviceBuffer<u8>,
    up: &DeviceBuffer<u8>,
    xq: &DeviceBuffer<u8>,
    ew: &DeviceBuffer<f32>,
    selected: &DeviceBuffer<i32>,
    gbpe: u32,
    ubpe: u32,
    n_used: u32,
    clamp: f32,
    n_rows: u32,
    n_blocks: u32,
) -> eyre::Result<()> {
    match dt {
        GgufType::IQ2_XXS => e.iq2.launch_fused_swiglu_batch(
            s, mid, gate, up, xq, ew, selected, gbpe, ubpe, n_used, clamp, n_rows, n_blocks,
        ),
        GgufType::IQ2_S => e.iq2s.launch_fused_swiglu_batch(
            s, mid, gate, up, xq, ew, selected, gbpe, ubpe, n_used, clamp, n_rows, n_blocks,
        ),
        GgufType::IQ2_XS => e.iq2xs.launch_fused_swiglu_batch(
            s, mid, gate, up, xq, ew, selected, gbpe, ubpe, n_used, clamp, n_rows, n_blocks,
        ),
        GgufType::IQ3_XXS => e.iq3pair.launch_fused_swiglu_batch(
            s, mid, gate, up, xq, ew, selected, gbpe, ubpe, n_used, clamp, n_rows, n_blocks,
        ),
        // blk.26 of the unsloth UD-IQ3_XXS mixes.
        GgufType::IQ3_S => e.iq3s.launch_fused_swiglu_batch(
            s, mid, gate, up, xq, ew, selected, gbpe, ubpe, n_used, clamp, n_rows, n_blocks,
        ),
        // V4.1-Flash native experts.
        GgufType::MXFP4 => e.mxfp4pair.launch_fused_swiglu_batch(
            s, mid, gate, up, xq, ew, selected, gbpe, ubpe, n_used, clamp, n_rows, n_blocks,
        ),
        other => Err(eyre!("moe gate/up: no decode kernel for {other:?}")),
    }
}

/// Decode MoE gate/up — het-split variant.
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_up_batch_hetsplit(
    e: &DeviceEngine,
    dt: GgufType,
    s: &Stream,
    mid: &mut DeviceBuffer<f32>,
    gate: &DeviceBuffer<u8>,
    up: &DeviceBuffer<u8>,
    xq: &DeviceBuffer<u8>,
    ew: &DeviceBuffer<f32>,
    selected: &DeviceBuffer<i32>,
    remap: &DeviceBuffer<i32>,
    mode: u32,
    cap: u32,
    gbpe: u32,
    ubpe: u32,
    n_used: u32,
    clamp: f32,
    n_rows: u32,
    n_blocks: u32,
) -> eyre::Result<()> {
    match dt {
        GgufType::IQ2_XXS => e.iq2.launch_fused_swiglu_batch_hetsplit(
            s, mid, gate, up, xq, ew, selected, remap, mode, cap, gbpe, ubpe, n_used, clamp,
            n_rows, n_blocks,
        ),
        GgufType::IQ2_S => e.iq2s.launch_fused_swiglu_batch_hetsplit(
            s, mid, gate, up, xq, ew, selected, remap, mode, cap, gbpe, ubpe, n_used, clamp,
            n_rows, n_blocks,
        ),
        GgufType::IQ2_XS => e.iq2xs.launch_fused_swiglu_batch_hetsplit(
            s, mid, gate, up, xq, ew, selected, remap, mode, cap, gbpe, ubpe, n_used, clamp,
            n_rows, n_blocks,
        ),
        // gate/up at IQ3_XXS is blk.26 of UD-Q2_K_XL only — the PAIR family,
        // not the down-projection `iq3` one.
        GgufType::IQ3_XXS => e.iq3pair.launch_fused_swiglu_batch_hetsplit(
            s, mid, gate, up, xq, ew, selected, remap, mode, cap, gbpe, ubpe, n_used, clamp,
            n_rows, n_blocks,
        ),
        GgufType::IQ3_S => e.iq3s.launch_fused_swiglu_batch_hetsplit(
            s, mid, gate, up, xq, ew, selected, remap, mode, cap, gbpe, ubpe, n_used, clamp,
            n_rows, n_blocks,
        ),
        GgufType::MXFP4 => e.mxfp4pair.launch_fused_swiglu_batch_hetsplit(
            s, mid, gate, up, xq, ew, selected, remap, mode, cap, gbpe, ubpe, n_used, clamp,
            n_rows, n_blocks,
        ),
        other => Err(eyre!("moe gate/up hetsplit: no decode kernel for {other:?}")),
    }
}

/// Decode MoE down projection — batched over selected experts.
#[allow(clippy::too_many_arguments)]
pub fn moe_down_batched(
    e: &DeviceEngine,
    dt: GgufType,
    s: &Stream,
    out: &mut DeviceBuffer<f32>,
    down: &DeviceBuffer<u8>,
    xq: &DeviceBuffer<u8>,
    selected: &DeviceBuffer<i32>,
    dbpe: u32,
    xq_slot_stride: u32,
    n_used: u32,
    n_rows: u32,
    n_blocks_in: u32,
) -> eyre::Result<()> {
    match dt {
        GgufType::Q2_K => e.q2k.launch_batched(
            s, out, down, xq, selected, dbpe, xq_slot_stride, n_used, n_rows, n_blocks_in,
        ),
        GgufType::IQ3_XXS => e.iq3.launch_batched(
            s, out, down, xq, selected, dbpe, xq_slot_stride, n_used, n_rows, n_blocks_in,
        ),
        GgufType::MXFP4 => e.mxfp4.launch_batched(
            s, out, down, xq, selected, dbpe, xq_slot_stride, n_used, n_rows, n_blocks_in,
        ),
        other => Err(eyre!("moe down: no decode kernel for {other:?}")),
    }
}

/// Decode MoE down projection — het-split variant.
#[allow(clippy::too_many_arguments)]
pub fn moe_down_batched_hetsplit(
    e: &DeviceEngine,
    dt: GgufType,
    s: &Stream,
    out: &mut DeviceBuffer<f32>,
    down: &DeviceBuffer<u8>,
    xq: &DeviceBuffer<u8>,
    selected: &DeviceBuffer<i32>,
    remap: &DeviceBuffer<i32>,
    mode: u32,
    cap: u32,
    dbpe: u32,
    xq_slot_stride: u32,
    n_used: u32,
    n_rows: u32,
    n_blocks_in: u32,
) -> eyre::Result<()> {
    match dt {
        GgufType::Q2_K => e.q2k.launch_batched_hetsplit(
            s, out, down, xq, selected, remap, mode, cap, dbpe, xq_slot_stride, n_used, n_rows,
            n_blocks_in,
        ),
        GgufType::IQ3_XXS => e.iq3.launch_batched_hetsplit(
            s, out, down, xq, selected, remap, mode, cap, dbpe, xq_slot_stride, n_used, n_rows,
            n_blocks_in,
        ),
        GgufType::MXFP4 => e.mxfp4.launch_batched_hetsplit(
            s, out, down, xq, selected, remap, mode, cap, dbpe, xq_slot_stride, n_used, n_rows,
            n_blocks_in,
        ),
        other => Err(eyre!("moe down hetsplit: no decode kernel for {other:?}")),
    }
}

/// Dense decode matvec with per-dtype input prep:
/// - Q8_0 consumes the caller's PRE-QUANTIZED (xq, xscale) pair
/// - K-quants (Q4_K/Q5_K/Q6_K) consume the raw f32 vector directly
///
/// The caller decides whether the q8 quantize kernel runs at all (skip it
/// when NO consumer at the site is Q8_0 — see issue_shared_expert).
/// blk.26's Q8_0 `ffn_down_shexp` among 42 Q6_K siblings is the case this
/// signature exists for.
#[allow(clippy::too_many_arguments)]
pub fn dense_matvec(
    e: &DeviceEngine,
    s: &Stream,
    out: &mut DeviceBuffer<f32>,
    w: &DeviceWeight,
    x_f32: &DeviceBuffer<f32>,
    xq: &DeviceBuffer<i8>,
    xscale: &DeviceBuffer<f32>,
    n_rows: u32,
    k: u32,
) -> eyre::Result<()> {
    match w.dtype {
        GgufType::Q8_0 => e.q8.matvec(s, out, &w.buffer, xq, xscale, n_rows, k),
        GgufType::Q4_K => e.q4d.matvec(s, out, &w.buffer, x_f32, n_rows, k),
        GgufType::Q5_K => e.q5d.matvec(s, out, &w.buffer, x_f32, n_rows, k),
        GgufType::Q6_K => e.q6d.matvec(s, out, &w.buffer, x_f32, n_rows, k),
        other => Err(eyre!("dense matvec: no kernel for {other:?}")),
    }
}

/// True if any of the weights needs the Q8_0 (xq, xscale) input pair.
pub fn any_q8(ws: &[&DeviceWeight]) -> bool {
    ws.iter().any(|w| w.dtype == GgufType::Q8_0)
}

/// Prefill dense GEMM with per-dtype input prep:
/// - Q8_0 rides the f16x WMMA GEMM when f16 activations are supplied
///   (2026-09-08; 4.4x the lds_tiled kernel), else lds_tiled on (xq_i8, xscale)
/// - K-quants ride the dp4a register-tiled GEMM (Q8_K activations)
///
/// The two paths take DIFFERENT activation quantizations — the caller
/// provides both (prefill already materializes Q8_K midq for the MoE).
#[allow(clippy::too_many_arguments)]
/// Whether a dense Q8_0 GEMM at batch `b` should take decode's dp4a GEMV arm
/// instead of a WMMA GEMM. OFF by default -- MEASURED, and the measurement is
/// the point:
///
/// A DSpark verify must reproduce decode, and here it does not: decode runs
/// `dense_matvec` (dp4a over Q8 activations) where this runs `gemm_f16x` (WMMA
/// over F16 activations) -- same weight, different arithmetic. Turning the arm
/// on closes that gap and acceptance DOES improve, but only to E 2.477 ->
/// 2.500 (+0.9%), while the verify forward goes 144.9 -> 220.2 ms/step (+52%).
/// So the f16-vs-dp4a mismatch is a REAL but MINOR acceptance limiter, and the
/// WMMA GEMM beats a B-packed GEMV on these shapes even at the B=3 of a verify
/// lane. Keep WMMA; look elsewhere for the acceptance ceiling.
///
/// `V41_SMALL_B_DENSE_DP4A=1` opts in. Callers that prepare activations must
/// consult this: the dp4a arm consumes (xq_i8, xscale), the WMMA arm f16.
pub fn small_b_dense_dp4a(b: u32) -> bool {
    // DEFAULT ON up to 8 rows since 2026-09-20 (multi-stream decode steps are
    // B = 1..8): measured on the 9070 XT with the weight ring defeating the
    // infinity cache (tests/bench_small_b_dense.rs, event-timed min), the
    // B-packed dp4a GEMV vs the WMMA f16x GEMM is 13.5 vs 96 us (q_a
    // 1280x5120), 15 vs 89 (shared gate 2304x5120), 15 vs 66 (shared down),
    // 24 vs 90 / 32 vs 86 / 31 vs 67 at B = 8. The WMMA kernel has a ~70-90 us
    // floor at small B: one 128-row tile per workgroup, so a 1280-row matrix
    // runs on 10 CUs. The earlier "+52% verify forward" reading (e0ec9eb) was
    // a whole-step number with box 2 restarted per arm, i.e. the cold point.
    // `V41_SMALL_B_DENSE_DP4A=0` rolls back to WMMA at every B;
    // `V41_SMALL_B_DENSE_MAX=<n>` moves the crossover (kernel cap 16).
    static MAX: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    // Cached: this sits on a per-launch path (3 chains x 40 layers x 2 lanes).
    let max = *MAX.get_or_init(|| {
        if std::env::var("V41_SMALL_B_DENSE_DP4A").as_deref() == Ok("0") {
            return 0;
        }
        std::env::var("V41_SMALL_B_DENSE_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(8).min(16)
    });
    b <= max
}

pub fn dense_gemm_prefill(
    e: &DeviceEngine,
    s: &Stream,
    out: &mut DeviceBuffer<f32>,
    w: &DeviceWeight,
    xq_i8: &DeviceBuffer<i8>,
    xscale: &DeviceBuffer<f32>,
    xq_q8k: &DeviceBuffer<u8>,
    x16: Option<(&DeviceBuffer<u16>, u32)>,   // f16 activations + row pitch (halves): Q8_0 -> f16x GEMM
    b: u32,
    n_rows: u32,
    k: u32,
) -> eyre::Result<()> {
    match w.dtype {
        // SMALL b: take decode's own kernel, not a WMMA GEMM.
        //
        // Two reasons, and the numerics one is the important one. Decode runs
        // `dense_matvec` (dp4a over Q8 activations); `gemm_f16x` runs WMMA over
        // F16 activations. A DSpark verify must REPRODUCE decode, and this was
        // one of the places it could not -- same weight, different arithmetic.
        // The q_chain and output_proj stages already switch to the dp4a arm for
        // b <= 64 (`prefill_f32_matvec`); the shared expert never did.
        //
        // It is also faster here: a WMMA tile is 16 wide in b, so at the B=3 of
        // a verify lane most of the tile's math is thrown away, while
        // `matvec_batched` (B-packed at these sizes) reads the weight once and
        // does exactly b rows of work.
        //
        // The CALLER must have produced (xq_i8, xscale) -- see
        // `small_b_dense_dp4a` at the shared expert's two quantize stages. The
        // f16x arm writes only `x16`, so taking this arm without that prep
        // reads a buffer nothing wrote (E collapsed 2.477 -> 1.000).
        //
        // `V41_SMALL_B_DENSE_DP4A=0` rolls back to the WMMA arm.
        GgufType::Q8_0 if small_b_dense_dp4a(b) => {
            e.q8.matvec_batched(s, out, &w.buffer, xq_i8, xscale, n_rows, k, b)
        }
        GgufType::Q8_0 => match x16 {
            Some((x, pitch)) if n_rows % 128 == 0 => e.q8_wmma.gemm_f16x(s, out, &w.buffer, x, k, n_rows, 1, b, pitch),
            _ => e.q8_wmma.gemm_lds_tiled(s, out, &w.buffer, xq_i8, xscale, n_rows, k, b),
        },
        GgufType::Q4_K | GgufType::Q5_K | GgufType::Q6_K => {
            e.dense_gemm.gemm(s, w.dtype, out, &w.buffer, xq_q8k, b, n_rows, k / 256)
        }
        other => Err(eyre!("dense gemm: no prefill kernel for {other:?}")),
    }
}

/// Whether the kwide prefill kernel — weights dequantized once per lane and
/// amortized across the chunk's members — handles `dt`, given the
/// `PAIR_VARIANT=chunked` rollback flag.
///
/// Pure and public so the routing is unit-testable without a GPU: a missing
/// arm here is the whole class of bug that cost IQ2_S ~4x prefill until
/// 2026-09-04, and it is invisible to the oracles (both kernels are
/// numerically correct — only one is fast).
pub fn pair_kwide_selected(dt: GgufType, rollback_to_chunked: bool) -> bool {
    !rollback_to_chunked
        && matches!(
            dt,
            GgufType::IQ2_S | GgufType::IQ2_XS | GgufType::IQ3_XXS | GgufType::IQ3_S | GgufType::MXFP4
        )
}

/// Whether the iGPU batched-prefill MoE takes the f16 WMMA path
/// (`iq2_xs_pair_matvec_fused_swiglu_wmma_h` + `iq3_xxs_matvec_par_by_expert_wmma`,
/// 2026-09-08): gfx11 iGPU, IQ2_XS gate/up with IQ3_XXS down (42 of 43
/// layers of UD-Q2_K_XL; blk.26 = IQ3_XXS pair + MXFP4 down and blk.42 =
/// MXFP4 down stay on the kwide/Q8_K path), default IQ2_VARIANT, and
/// `IGPU_MOE_WMMA` not "0". Pure and public: the routing is unit-testable
/// and a missing arm here is invisible to the oracles (both paths are
/// numerically fine — only one is fast).
pub fn igpu_moe_wmma_selected(
    gate: GgufType,
    down: GgufType,
    igpu_is_gfx11: bool,
    iq2_variant: &str,
    env_enabled: bool,
) -> bool {
    // IQ2_S joined 2026-09-08 (the Vision-Exp UD-IQ3_XXS gate/up type on
    // 42 of 43 layers) with its own WMMA twin; blk.26 (IQ3_S / IQ3_XXS
    // pair) and blk.42 (MXFP4 down) stay on the kwide kernels.
    env_enabled
        && igpu_is_gfx11
        && (gate == GgufType::IQ2_XS || gate == GgufType::IQ2_S)
        && down == GgufType::IQ3_XXS
        && iq2_variant == "kwide"
}

/// `IGPU_MOE_WMMA=0` rolls the iGPU MoE back to the kwide/Q8_K kernels.
pub fn igpu_moe_wmma_env_enabled() -> bool {
    std::env::var("IGPU_MOE_WMMA").map(|v| v != "0").unwrap_or(true)
}

/// `PAIR_VARIANT=chunked` rolls the pair formats back to the serial
/// per-member kernel.
fn pair_variant_rollback() -> bool {
    std::env::var("PAIR_VARIANT").map(|v| v == "chunked").unwrap_or(false)
}

/// Perfetto stage name for a [`moe_gate_up_chunked`] launch, reflecting the
/// kernel that will actually run. A trace is the only way to confirm the
/// kwide path is live in production, so the label must not be a constant.
pub fn pair_prefill_stage(dt: GgufType) -> &'static str {
    if pair_kwide_selected(dt, pair_variant_rollback()) {
        "igpu.pair_kwide"
    } else {
        "igpu.pair_chunked"
    }
}

/// Prefill MoE gate/up (chunked by-expert), for the formats that have
/// exactly one prefill kernel.
///
/// Returns `Ok(false)` for IQ2_XXS, which instead has the `IQ2_VARIANT`
/// kernel zoo (staged / staged_v2 / tile8 / kwide) that the caller selects
/// between — so callers use this as: try the dispatcher, else fall through
/// to the variant chain.
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_up_chunked(
    e: &DeviceEngine,
    dt: GgufType,
    s: &Stream,
    mid: &mut DeviceBuffer<f32>,
    gate: &DeviceBuffer<u8>,
    up: &DeviceBuffer<u8>,
    xq: &DeviceBuffer<u8>,
    ew: &DeviceBuffer<f32>,
    group_count: &DeviceBuffer<i32>,
    expert_members: &DeviceBuffer<i32>,
    work_items: &DeviceBuffer<i32>,
    n_work_items: u32,
    gbpe: u32,
    ubpe: u32,
    n_used: u32,
    max_per_expert: u32,
    chunk: u32,
    clamp: f32,
    n_rows: u32,
    n_blocks: u32,
) -> eyre::Result<bool> {
    moe_gate_up_chunked_ex(
        e, dt, s, mid, gate, up, xq, ew, group_count, expert_members, work_items, n_work_items,
        gbpe, ubpe, n_used, max_per_expert, chunk, clamp, n_rows, n_blocks, None,
    )
}

/// Upper bound on the work-item builder's count, for the device-count grid.
/// `members` = rows x picks (each group entry is one (row, pick)); `groups` =
/// the builder's id bound; `chunk` = members per work item; `cap` = the
/// work-item buffer. With G non-empty groups of sizes c_g (sum = M):
/// sum ceil(c_g / C) <= G + (M - G) / C <= min(M, groups) + ceil(M / C), and
/// never more than M. Decode (M <= 48): M itself; B=512 prefill: ~480, not 3072.
pub fn moe_wi_upper_bound(members: usize, groups: u32, chunk: u32, cap: usize) -> u32 {
    let m = members;
    let g = m.min(groups as usize);
    m.min(g + m.div_ceil(chunk.max(1) as usize)).min(cap) as u32
}

/// Can the MoE work-item count stay on the device (`V41_MOE_WI_DEVCOUNT`)?
/// Only the MXFP4 kwide gate/up + MXFP4 kwide2 down pair checks a device count.
pub fn moe_wi_devcount_supported(gate: GgufType, down: GgufType) -> bool {
    gate == GgufType::MXFP4 && down == GgufType::MXFP4 && pair_kwide_selected(GgufType::MXFP4, pair_variant_rollback())
}

/// As `moe_gate_up_chunked`. With `n_wi_dev` (the builder's device-side count)
/// `n_work_items` is only an upper bound for the grid; only the MXFP4 kwide
/// kernel checks a device count, so any other kernel is an error.
#[allow(clippy::too_many_arguments)]
pub fn moe_gate_up_chunked_ex(
    e: &DeviceEngine,
    dt: GgufType,
    s: &Stream,
    mid: &mut DeviceBuffer<f32>,
    gate: &DeviceBuffer<u8>,
    up: &DeviceBuffer<u8>,
    xq: &DeviceBuffer<u8>,
    ew: &DeviceBuffer<f32>,
    group_count: &DeviceBuffer<i32>,
    expert_members: &DeviceBuffer<i32>,
    work_items: &DeviceBuffer<i32>,
    n_work_items: u32,
    gbpe: u32,
    ubpe: u32,
    n_used: u32,
    max_per_expert: u32,
    chunk: u32,
    clamp: f32,
    n_rows: u32,
    n_blocks: u32,
    n_wi_dev: Option<&DeviceBuffer<i32>>,
) -> eyre::Result<bool> {
    // IQ2_S / IQ2_XS / IQ3_XXS / IQ3_S pair kwide (2026-08-15; IQ3_S 2026-09-03;
    // IQ2_S 2026-09-04): default prefill kernel is the M51-structure kwide port
    // (weights dequantized once per lane and amortized across chunk members).
    // IQ2_XXS returns false and reaches its own kwide path in the caller.
    // PAIR_VARIANT=chunked rolls back to the serial per-member kernel.
    //
    // The per-arm `if use_kwide` guards go through `pair_kwide_selected` so
    // the format list lives in exactly one place and `pair_prefill_stage`
    // (the trace label) and the unit tests below cannot drift from it.
    let use_kwide = pair_kwide_selected(dt, pair_variant_rollback());
    if n_wi_dev.is_some() && !(dt == GgufType::MXFP4 && use_kwide) {
        return Err(eyre!("moe gate/up: a device-side work-item count needs the MXFP4 kwide kernel, not {dt:?} (kwide={use_kwide})"));
    }
    match dt {
        GgufType::IQ2_S if use_kwide => e.iq2s.launch_fused_swiglu_kwide(
            s, mid, gate, up, xq, ew, group_count, expert_members, work_items, n_work_items,
            gbpe, ubpe, n_used, max_per_expert, chunk, clamp, n_rows, n_blocks,
        )?,
        GgufType::IQ2_S => e.iq2s.launch_fused_swiglu_chunked(
            s, mid, gate, up, xq, ew, group_count, expert_members, work_items, n_work_items,
            gbpe, ubpe, n_used, max_per_expert, chunk, clamp, n_rows, n_blocks,
        )?,
        GgufType::IQ2_XS if use_kwide => e.iq2xs.launch_fused_swiglu_kwide(
            s, mid, gate, up, xq, ew, group_count, expert_members, work_items, n_work_items,
            gbpe, ubpe, n_used, max_per_expert, chunk, clamp, n_rows, n_blocks,
        )?,
        GgufType::IQ2_XS => e.iq2xs.launch_fused_swiglu_chunked(
            s, mid, gate, up, xq, ew, group_count, expert_members, work_items, n_work_items,
            gbpe, ubpe, n_used, max_per_expert, chunk, clamp, n_rows, n_blocks,
        )?,
        GgufType::IQ3_XXS if use_kwide => e.iq3pair.launch_fused_swiglu_kwide(
            s, mid, gate, up, xq, ew, group_count, expert_members, work_items, n_work_items,
            gbpe, ubpe, n_used, max_per_expert, chunk, clamp, n_rows, n_blocks,
        )?,
        GgufType::IQ3_XXS => e.iq3pair.launch_fused_swiglu_chunked(
            s, mid, gate, up, xq, ew, group_count, expert_members, work_items, n_work_items,
            gbpe, ubpe, n_used, max_per_expert, chunk, clamp, n_rows, n_blocks,
        )?,
        GgufType::IQ3_S if use_kwide => e.iq3s.launch_fused_swiglu_kwide(
            s, mid, gate, up, xq, ew, group_count, expert_members, work_items, n_work_items,
            gbpe, ubpe, n_used, max_per_expert, chunk, clamp, n_rows, n_blocks,
        )?,
        GgufType::IQ3_S => e.iq3s.launch_fused_swiglu_chunked(
            s, mid, gate, up, xq, ew, group_count, expert_members, work_items, n_work_items,
            gbpe, ubpe, n_used, max_per_expert, chunk, clamp, n_rows, n_blocks,
        )?,
        // MXFP4 (DeepSeek-V4.1-Flash's native routed experts, 2026-09-12):
        // same two-kernel family, contracts identical to iq3_s.
        GgufType::MXFP4 if use_kwide => e.mxfp4pair.launch_fused_swiglu_kwide_ex(
            s, mid, gate, up, xq, ew, group_count, expert_members, work_items, n_work_items,
            gbpe, ubpe, n_used, max_per_expert, chunk, clamp, n_rows, n_blocks, n_wi_dev,
        )?,
        GgufType::MXFP4 => e.mxfp4pair.launch_fused_swiglu_chunked(
            s, mid, gate, up, xq, ew, group_count, expert_members, work_items, n_work_items,
            gbpe, ubpe, n_used, max_per_expert, chunk, clamp, n_rows, n_blocks,
        )?,
        GgufType::IQ2_XXS => return Ok(false),
        other => return Err(eyre!("moe gate/up prefill: no kernel for {other:?}")),
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    /// `moe_wi_upper_bound` is >= the exact count for every split of M
    /// members into groups (brute force over small cases and a few big ones).
    #[test]
    fn work_item_bound_covers_every_grouping() {
        fn exact(sizes: &[usize], c: usize) -> usize {
            sizes.iter().map(|&n| n.div_ceil(c)).sum()
        }
        let mut seed = 0x5749_4443u64;
        for &(m, groups, c) in &[(1usize, 384u32, 32u32), (12, 384, 32), (24, 384, 8), (48, 384, 32), (3072, 384, 32), (3072, 1024, 8), (100, 3, 32)] {
            let bound = moe_wi_upper_bound(m, groups, c, usize::MAX) as usize;
            assert!(bound <= m);
            for trial in 0..2000 {
                // Random split of m members over at most `groups` groups.
                let k = 1 + (trial % (groups as usize).min(m));
                let mut sizes = vec![0usize; k];
                for _ in 0..m {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    sizes[(seed >> 33) as usize % k] += 1;
                }
                assert!(exact(&sizes, c as usize) <= bound, "m={m} groups={groups} c={c} sizes={sizes:?}");
            }
            // The worst case: every group but one holds a single member.
            let k = (groups as usize).min(m);
            let mut sizes = vec![1usize; k];
            sizes[0] += m - k;
            assert!(exact(&sizes, c as usize) <= bound, "worst case m={m} groups={groups} c={c}");
        }
        assert_eq!(moe_wi_upper_bound(12, 384, 32, usize::MAX), 12, "decode: M itself");
        assert_eq!(moe_wi_upper_bound(3072, 384, 32, usize::MAX), 480);
        assert_eq!(moe_wi_upper_bound(3072, 384, 32, 100), 100, "never past the buffer");
    }

    use super::*;

    /// Regression guard for the 2026-09-04 fix: IQ2_S is gate/up on 42 of 43
    /// layers of the Vision-Exp mix, and before the fix it had no kwide arm,
    /// so it silently fell to the ~4x-slower per-member re-dequant kernel
    /// with every oracle still green.
    #[test]
    fn pair_formats_default_to_kwide() {
        for dt in [
            GgufType::IQ2_S,
            GgufType::IQ2_XS,
            GgufType::IQ3_XXS,
            GgufType::IQ3_S,
            GgufType::MXFP4,
        ] {
            assert!(
                pair_kwide_selected(dt, false),
                "{dt:?} must take the kwide prefill kernel by default"
            );
            assert_eq!(
                pair_prefill_stage(dt),
                "igpu.pair_kwide",
                "{dt:?} kwide launches must be traceable as such"
            );
        }
    }

    #[test]
    fn pair_variant_chunked_rolls_back() {
        for dt in [
            GgufType::IQ2_S,
            GgufType::IQ2_XS,
            GgufType::IQ3_XXS,
            GgufType::IQ3_S,
            GgufType::MXFP4,
        ] {
            assert!(!pair_kwide_selected(dt, true), "{dt:?} rollback");
        }
    }

    #[test]
    fn wmma_path_routing() {
        use GgufType::*;
        assert!(igpu_moe_wmma_selected(IQ2_XS, IQ3_XXS, true, "kwide", true));
        assert!(igpu_moe_wmma_selected(IQ2_S, IQ3_XXS, true, "kwide", true), "UD-IQ3_XXS gate/up");
        assert!(!igpu_moe_wmma_selected(IQ2_XS, IQ3_XXS, false, "kwide", true), "dGPU/gfx12 never");
        assert!(!igpu_moe_wmma_selected(IQ2_XS, IQ3_XXS, true, "kwide", false), "env off");
        assert!(!igpu_moe_wmma_selected(IQ2_XS, MXFP4, true, "kwide", true), "blk.42");
        assert!(!igpu_moe_wmma_selected(IQ3_XXS, MXFP4, true, "kwide", true), "blk.26");
        assert!(!igpu_moe_wmma_selected(IQ2_XXS, Q2_K, true, "kwide", true), "antirez mix");
        assert!(!igpu_moe_wmma_selected(IQ2_XS, IQ3_XXS, true, "hybrid", true), "variant zoo");
    }

    /// IQ2_XXS has its own `IQ2_VARIANT` kernel zoo in the caller and must
    /// never be claimed by the pair dispatcher's kwide path.
    #[test]
    fn non_pair_formats_are_not_kwide() {
        for dt in [GgufType::IQ2_XXS, GgufType::Q2_K, GgufType::Q8_0] {
            assert!(!pair_kwide_selected(dt, false), "{dt:?}");
            assert!(!pair_kwide_selected(dt, true), "{dt:?}");
        }
    }
}
