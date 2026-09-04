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
/// - Q8_0 rides the existing WMMA GEMM ((xq_i8, xscale) per token)
/// - K-quants ride the dp4a register-tiled GEMM (Q8_K activations)
///
/// The two paths take DIFFERENT activation quantizations — the caller
/// provides both (prefill already materializes Q8_K midq for the MoE).
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_prefill(
    e: &DeviceEngine,
    s: &Stream,
    out: &mut DeviceBuffer<f32>,
    w: &DeviceWeight,
    xq_i8: &DeviceBuffer<i8>,
    xscale: &DeviceBuffer<f32>,
    xq_q8k: &DeviceBuffer<u8>,
    b: u32,
    n_rows: u32,
    k: u32,
) -> eyre::Result<()> {
    match w.dtype {
        GgufType::Q8_0 => e.q8_wmma.gemm_lds_tiled(s, out, &w.buffer, xq_i8, xscale, n_rows, k, b),
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
            GgufType::IQ2_S | GgufType::IQ2_XS | GgufType::IQ3_XXS | GgufType::IQ3_S
        )
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
        GgufType::IQ2_XXS => return Ok(false),
        other => return Err(eyre!("moe gate/up prefill: no kernel for {other:?}")),
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
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
        ] {
            assert!(!pair_kwide_selected(dt, true), "{dt:?} rollback");
        }
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
