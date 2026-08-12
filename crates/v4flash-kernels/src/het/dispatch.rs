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
    match dt {
        GgufType::IQ2_S => e.iq2s.launch_fused_swiglu_chunked(
            s, mid, gate, up, xq, ew, group_count, expert_members, work_items, n_work_items,
            gbpe, ubpe, n_used, max_per_expert, chunk, clamp, n_rows, n_blocks,
        )?,
        GgufType::IQ2_XS => e.iq2xs.launch_fused_swiglu_chunked(
            s, mid, gate, up, xq, ew, group_count, expert_members, work_items, n_work_items,
            gbpe, ubpe, n_used, max_per_expert, chunk, clamp, n_rows, n_blocks,
        )?,
        GgufType::IQ3_XXS => e.iq3pair.launch_fused_swiglu_chunked(
            s, mid, gate, up, xq, ew, group_count, expert_members, work_items, n_work_items,
            gbpe, ubpe, n_used, max_per_expert, chunk, clamp, n_rows, n_blocks,
        )?,
        GgufType::IQ2_XXS => return Ok(false),
        other => return Err(eyre!("moe gate/up prefill: no kernel for {other:?}")),
    }
    Ok(true)
}
