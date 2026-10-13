//! Decode attention FRONT in two launches (fusion group G2, 2026-10-13; `V41_DGPU_QKV_FRONT`,
//! default off). Replaces, at decode rows 1..=8 on the dGPU's dp4a arm, the eight launches
//!
//! ```text
//!   quantize(attn_input_norm) -> q_a GEMV -> rms_quant_q8_1280 -> q_b GEMV -> rope_copy
//!   kv GEMV -> kv_rms_rope_fp8 -> kv_cache_append
//! ```
//!
//! by `qkv_front_a_tB<b>` (input quantize per work-group in LDS + the q_a and kv GEMVs over one
//! grid + the q_a norm/quantize by the last work-group to finish) and `qkv_front_b_tB<b>` (the
//! q_b GEMV with the q rope in its epilogue, writing `q_normed`; work-group 0 runs the kv norm /
//! rope / fp8 tail and stores the f16 cache rows). Every output is BIT-IDENTICAL to the chain
//! (tests/qkv_front_bitexact.rs): the same per-output expressions and accumulation orders; see
//! kernels/qkv_front.hip.
//!
//! Not written any more: `q` (the un-roped q_b output) and `kv_normed` (the f32 row the append
//! read) -- hand-offs that only existed because the chain ran as separate launches.
//!
//! Measured 2026-10-13 beside the live hub (tests/bench_qkv_front_chain.rs, cold = MALL flushed,
//! 100 interleaved rounds; rocprofv3 kernel trace on the dGPU only):
//! - chain wall new - cur: -19.5 / -19.2 / -19.0 / -18.1 / -18.9 us per lane-layer at b = 1/2/3/4/8
//!   (warm -16.8..-19.2; A/A |d| p50 0.2-0.6 cold), 8 -> 2 launches.
//! - kernel time, cold: launch A 18.6 / 21.8 / 27.1 us at b = 1/4/8 (the chain's 7 non-q_b kernels:
//!   31.2 / 32.3 / 36.9 us + 6 gaps); launch B 72.3 / 76.0 / 80.5 = the q_b GEMV alone + 0..0.4.
//! - floors: 9.75 MB at the measured cold streaming rate (~575-600 GB/s: read_v4 p10, q_b at 590)
//!   = 16.3-17 us; a dependent launch 2.2 us (1.6 of pure gap). Launch A's remaining gap to its
//!   floor is the per-work-group input quantize (~1 us at b = 4, ~3 at b = 8: it disappears if the
//!   producer of attn_input_norm also emits its Q8_0) and the exposed q_a tail (~1-1.5 us at b = 4,
//!   ~3.5 at b = 8: one ticket round trip, one L2 round trip, a 256-wide tree per row).
//! - ONE launch (q_b streamed by launch A's work-groups after an in-kernel publish of the q_a tail;
//!   commit 271f512e, reverted): bit-identical but +15 / +70 / +196 us at b = 1/4/8 -- the
//!   kernel-uniform dynamic LDS caps residency, and 256 warps cannot hide the q_b warp-sum
//!   latencies the 4096-work-group launch hides. Two launches is the floor here.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

use crate::rope::{RopeParams, RopeTail};

const QKV_FRONT_GFX1201: &[u8] = include_bytes!(env!("KERNEL_QKV_FRONT_GFX1201"));
const QKV_FRONT_GFX1151: &[u8] = include_bytes!(env!("KERNEL_QKV_FRONT_GFX1151"));

/// Shapes the kernels bake in (V4.1): hidden 5120, q low-rank 1280, window KV 512 (n_rot 64),
/// q_b output 64 heads x 512.
pub const K_IN: u32 = 5120;
pub const N_QA: u32 = 1280;
pub const N_KV: u32 = 512;
pub const N_QB: u32 = 32768;
pub const HEAD_DIM: u32 = 512;
pub const N_ROT: u32 = 64;
/// Largest row count with a kernel instantiation (the dp4a arm's decode range, `SMALL_B_DENSE_MAX`).
pub const MAX_B: u32 = 8;

fn q8_bytes(rows: u32, k: u32) -> usize {
    rows as usize * (k as usize / 32) * 34
}

const A_SYMBOLS: [&str; 8] = [
    "qkv_front_a_tB1", "qkv_front_a_tB2", "qkv_front_a_tB3", "qkv_front_a_tB4",
    "qkv_front_a_tB5", "qkv_front_a_tB6", "qkv_front_a_tB7", "qkv_front_a_tB8",
];
const B_SYMBOLS: [&str; 8] = [
    "qkv_front_b_tB1", "qkv_front_b_tB2", "qkv_front_b_tB3", "qkv_front_b_tB4",
    "qkv_front_b_tB5", "qkv_front_b_tB6", "qkv_front_b_tB7", "qkv_front_b_tB8",
];

/// Launch A's work-group count at `b` rows. `V41_DGPU_QKV_FRONT_GRID=<n>` (static) overrides it
/// (benches). Each work-group re-quantizes the whole input, so fewer is cheaper as long as the
/// weight stream keeps enough loads in flight.
pub fn grid_a(b: u32) -> u32 {
    static OVR: std::sync::LazyLock<Option<u32>> = std::sync::LazyLock::new(|| {
        std::env::var("V41_DGPU_QKV_FRONT_GRID").ok().and_then(|v| v.parse().ok()).filter(|&g: &u32| g >= 1)
    });
    if let Some(g) = *OVR {
        return g;
    }
    let _ = b;
    64
}

pub struct QkvFront {
    module: Module,
}

impl QkvFront {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            QKV_FRONT_GFX1201
        } else if arch.starts_with("gfx1151") {
            QKV_FRONT_GFX1151
        } else {
            return Err(eyre!("unsupported arch for qkv_front: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
    }

    /// A module from another build of kernels/qkv_front.hip (benches: variant kernels).
    pub fn from_image(image: &[u8]) -> eyre::Result<Self> {
        Ok(Self { module: Module::load_data(image)? })
    }

    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Launch A: `qr = W_qa · Q8(x)`, `kv_raw = W_kv · Q8(x)` and, from the last work-group,
    /// `qr_normed = rms_w(qr) * q_a_norm` + its Q8_0 (`qr_xq`, `qr_xscale`). `ticket` is one u32
    /// that must be 0 before the first launch; the kernel leaves it 0. One ticket per stream
    /// (launches sharing a ticket must not overlap).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_a(
        &self,
        stream: &Stream,
        qr: &mut DeviceBuffer<f32>,
        kv_raw: &mut DeviceBuffer<f32>,
        w_qa: &DeviceBuffer<u8>,
        w_kv: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        qr_normed: &mut DeviceBuffer<f32>,
        qr_xq: &mut DeviceBuffer<i8>,
        qr_xscale: &mut DeviceBuffer<f32>,
        q_a_norm: &DeviceBuffer<f32>,
        eps: f32,
        ticket: &mut DeviceBuffer<u32>,
        b: u32,
        grid: u32,
    ) -> eyre::Result<()> {
        if !(1..=MAX_B).contains(&b) {
            return Err(eyre!("qkv_front_a: b={b} outside 1..={MAX_B}"));
        }
        let bu = b as usize;
        if w_qa.byte_len() != q8_bytes(N_QA, K_IN) || w_kv.byte_len() != q8_bytes(N_KV, K_IN) {
            return Err(eyre!(
                "qkv_front_a: weight bytes q_a {} kv {} (want {} / {})",
                w_qa.byte_len(), w_kv.byte_len(), q8_bytes(N_QA, K_IN), q8_bytes(N_KV, K_IN)
            ));
        }
        if x.len() < bu * K_IN as usize
            || qr.len() < bu * N_QA as usize
            || kv_raw.len() < bu * N_KV as usize
            || qr_normed.len() < bu * N_QA as usize
            || qr_xq.len() < bu * N_QA as usize
            || qr_xscale.len() < bu * (N_QA / 32) as usize
            || q_a_norm.len() != N_QA as usize
            || ticket.len() == 0
        {
            return Err(eyre!("qkv_front_a: buffer too small for b={b}"));
        }
        if (x.raw() as usize) % 16 != 0 {
            return Err(eyre!("qkv_front_a: x not 16-B aligned"));
        }
        let grid = grid.clamp(1, (N_QA + N_KV) / 8);
        // max(quantized input [b][5120] i8 + xscale [b][160] f32,
        //     the tail's qr [b][40 x 36] + tree [b][256] + q_a_norm [40 x 36], f32)
        let lds = (b * K_IN + b * (K_IN / 32) * 4).max((b * (40 * 36 + 256) + 40 * 36) * 4);
        let function = self.module.get_function(A_SYMBOLS[(b - 1) as usize])?;
        let cfg = LaunchConfig { grid: (grid, 1, 1), block: (256, 1, 1), shared_mem_bytes: lds };
        launch_kernel!(function, cfg, stream, [
            qr.raw(), kv_raw.raw(), w_qa.raw(), w_kv.raw(), x.raw(),
            qr_normed.raw(), qr_xq.raw(), qr_xscale.raw(), q_a_norm.raw(), eps, ticket.raw()
        ])
    }

    /// Launch B: `q_normed = rope(W_qb · (qr_xq, qr_xscale))` (forward rope on each head's last 64
    /// dims at `pos_per_b[r]`), and the window-KV tail of `kv_raw` (rms_w * kv_norm, rope, fp8
    /// window quant) stored as f16 into `kv_cache` row `slot_per[r]` (or `slot_base + r` when
    /// `slot_per` is None). The caller checks the slots against the cache.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_b(
        &self,
        stream: &Stream,
        q_normed: &mut DeviceBuffer<f32>,
        w_qb: &DeviceBuffer<u8>,
        qr_xq: &DeviceBuffer<i8>,
        qr_xscale: &DeviceBuffer<f32>,
        pos_per_b: &DeviceBuffer<i32>,
        params: &RopeParams,
        kv_cache: &mut DeviceBuffer<u16>,
        kv_raw: &DeviceBuffer<f32>,
        kv_norm: &DeviceBuffer<f32>,
        eps: f32,
        slot_per: Option<&DeviceBuffer<i32>>,
        slot_base: u32,
        b: u32,
    ) -> eyre::Result<()> {
        if !(1..=MAX_B).contains(&b) {
            return Err(eyre!("qkv_front_b: b={b} outside 1..={MAX_B}"));
        }
        let bu = b as usize;
        if w_qb.byte_len() != q8_bytes(N_QB, N_QA) {
            return Err(eyre!("qkv_front_b: q_b weight bytes {} (want {})", w_qb.byte_len(), q8_bytes(N_QB, N_QA)));
        }
        if q_normed.len() < bu * N_QB as usize
            || qr_xq.len() < bu * N_QA as usize
            || qr_xscale.len() < bu * (N_QA / 32) as usize
            || pos_per_b.len() < bu
            || kv_raw.len() < bu * N_KV as usize
            || kv_norm.len() != N_KV as usize
            || slot_per.is_some_and(|s| s.len() < bu)
        {
            return Err(eyre!("qkv_front_b: buffer too small for b={b}"));
        }
        if slot_per.is_none() && kv_cache.len() < (slot_base as usize + bu) * HEAD_DIM as usize {
            return Err(eyre!("qkv_front_b: append rows {slot_base}+{b} past the cache"));
        }
        let (theta_scale, mscale_eff, corr_low, corr_high) = RopeTail::device_args(params, N_ROT);
        let slot_ptr = slot_per.map(|s| s.raw() as *const i32).unwrap_or(std::ptr::null());
        let inverse: i32 = 0;
        let function = self.module.get_function(B_SYMBOLS[(b - 1) as usize])?;
        // work-group 0 = the kv tail, 1.. = q_b's 8-row groups
        let cfg = LaunchConfig { grid: (1 + N_QB / 8, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [
            q_normed.raw(), w_qb.raw(), qr_xq.raw(), qr_xscale.raw(), pos_per_b.raw(),
            theta_scale, params.freq_scale, params.ext_factor, mscale_eff, corr_low, corr_high, inverse,
            kv_cache.raw(), kv_raw.raw(), kv_norm.raw(), eps, slot_ptr, slot_base
        ])
    }
}
