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

    /// Launch A and launch B as ONE launch (`V41_DGPU_QKV_FRONT_ONE`): `grid` work-groups run launch
    /// A's body, the last one publishes the q_a tail through `ctl[1] = epoch` and runs the kv tail,
    /// and every work-group then streams q_b in dynamically grabbed 16-pair chunks, its first
    /// weights loaded before it waits for the publish. Same outputs, bit for bit, as `launch_a` +
    /// `launch_b`.
    ///
    /// `ctl` = 4 u32 (A ticket, published epoch, chunk counter, end ticket), zero before the first
    /// launch; the kernel leaves [0], [2], [3] at 0 and [1] at `epoch`. `epoch` must differ from
    /// the value `ctl[1]` holds (the caller counts launches per `ctl`, skipping 0).
    ///
    /// The one in-kernel wait (q_b work after the q_a tail) is deadlock-free only if all `grid`
    /// work-groups can be resident at once: every work-group runs launch A's body before it waits,
    /// so the publisher is never queued behind a waiter. `grid` is clamped to `MAX_GRID_ONE`, half
    /// the 9070 XT's capacity at the b = 8 LDS size (2 work-groups per WGP x 32 WGPs).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_one(
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
        ctl: &mut DeviceBuffer<u32>,
        epoch: u32,
        q_normed: &mut DeviceBuffer<f32>,
        w_qb: &DeviceBuffer<u8>,
        pos_per_b: &DeviceBuffer<i32>,
        params: &RopeParams,
        kv_cache: &mut DeviceBuffer<u16>,
        kv_norm: &DeviceBuffer<f32>,
        slot_per: Option<&DeviceBuffer<i32>>,
        slot_base: u32,
        b: u32,
        grid: u32,
    ) -> eyre::Result<()> {
        if !(1..=MAX_B).contains(&b) {
            return Err(eyre!("qkv_front_one: b={b} outside 1..={MAX_B}"));
        }
        let bu = b as usize;
        if w_qa.byte_len() != q8_bytes(N_QA, K_IN)
            || w_kv.byte_len() != q8_bytes(N_KV, K_IN)
            || w_qb.byte_len() != q8_bytes(N_QB, N_QA)
        {
            return Err(eyre!("qkv_front_one: weight bytes"));
        }
        if x.len() < bu * K_IN as usize
            || qr.len() < bu * N_QA as usize
            || kv_raw.len() < bu * N_KV as usize
            || qr_normed.len() < bu * N_QA as usize
            || qr_xq.len() < bu * N_QA as usize
            || qr_xscale.len() < bu * (N_QA / 32) as usize
            || q_a_norm.len() != N_QA as usize
            || ctl.len() < 4
            || q_normed.len() < bu * N_QB as usize
            || pos_per_b.len() < bu
            || kv_norm.len() != N_KV as usize
            || slot_per.is_some_and(|s| s.len() < bu)
        {
            return Err(eyre!("qkv_front_one: buffer too small for b={b}"));
        }
        if slot_per.is_none() && kv_cache.len() < (slot_base as usize + bu) * HEAD_DIM as usize {
            return Err(eyre!("qkv_front_one: append rows {slot_base}+{b} past the cache"));
        }
        if (x.raw() as usize) % 16 != 0 || (qr_xq.raw() as usize) % 16 != 0 {
            return Err(eyre!("qkv_front_one: x / qr_xq not 16-B aligned"));
        }
        if epoch == 0 {
            return Err(eyre!("qkv_front_one: epoch 0 is the initial ctl[1]"));
        }
        let grid = grid.clamp(1, MAX_GRID_ONE);
        let lds = (b * K_IN + b * (K_IN / 32) * 4).max((b * (40 * 36 + 256) + 40 * 36) * 4);
        let (theta_scale, mscale_eff, corr_low, corr_high) = RopeTail::device_args(params, N_ROT);
        let slot_ptr = slot_per.map(|s| s.raw() as *const i32).unwrap_or(std::ptr::null());
        let inverse: i32 = 0;
        let function = self.module.get_function(ONE_SYMBOLS[(b - 1) as usize])?;
        let cfg = LaunchConfig { grid: (grid, 1, 1), block: (256, 1, 1), shared_mem_bytes: lds };
        launch_kernel!(function, cfg, stream, [
            qr.raw(), kv_raw.raw(), w_qa.raw(), w_kv.raw(), x.raw(), qr_normed.raw(), qr_xq.raw(),
            qr_xscale.raw(), q_a_norm.raw(), eps, ctl.raw(), epoch, q_normed.raw(), w_qb.raw(),
            pos_per_b.raw(), theta_scale, params.freq_scale, params.ext_factor, mscale_eff, corr_low,
            corr_high, inverse, kv_cache.raw(), kv_norm.raw(), slot_ptr, slot_base
        ])
    }
}

/// See [`QkvFront::launch_one`].
pub const MAX_GRID_ONE: u32 = 32;

const ONE_SYMBOLS: [&str; 8] = [
    "qkv_front_one_tB1", "qkv_front_one_tB2", "qkv_front_one_tB3", "qkv_front_one_tB4",
    "qkv_front_one_tB5", "qkv_front_one_tB6", "qkv_front_one_tB7", "qkv_front_one_tB8",
];

/// `launch_one`'s work-group count at `b` rows (`V41_DGPU_QKV_FRONT_ONE_GRID=<n>`, static,
/// overrides; clamped to [`MAX_GRID_ONE`]).
pub fn grid_one(b: u32) -> u32 {
    static OVR: std::sync::LazyLock<Option<u32>> = std::sync::LazyLock::new(|| {
        std::env::var("V41_DGPU_QKV_FRONT_ONE_GRID").ok().and_then(|v| v.parse().ok()).filter(|&g: &u32| g >= 1)
    });
    let _ = b;
    OVR.unwrap_or(MAX_GRID_ONE).min(MAX_GRID_ONE)
}
