//! W0 microbenchmark launchers for `qe_attn_g4` (kernels/qe_attn_w0.hip;
//! docs/v41/EMBED_PHASE_DESIGN.md §19.11). gfx1201 only. The test that drives them
//! is `tests/qe_attn_g4_w0.rs` (#[ignore]: GPU window only).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const QE_ATTN_W0_GFX1201: &[u8] = include_bytes!(env!("KERNEL_QE_ATTN_W0_GFX1201"));
const QE_ATTN_W0WGP_GFX1201: &[u8] = include_bytes!(env!("KERNEL_QE_ATTN_W0WGP_GFX1201"));

/// The (W WMMA, V VALU, E exp) per-iteration mixes `qw_mix_w*_v*_e*` exists for.
pub const MIXES: [(u32, u32, u32); 8] = [(32, 0, 0), (32, 32, 0), (32, 80, 0), (32, 160, 0), (32, 0, 16), (32, 66, 16), (0, 64, 0), (0, 0, 64)];

pub struct QeAttnW0 {
    cu: Module,
    wgp: Module,
}

impl QeAttnW0 {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        if !arch.starts_with("gfx1201") {
            return Err(eyre!("qe_attn_w0 probes are gfx1201 only, not {arch}"));
        }
        Ok(Self { cu: Module::load_data(QE_ATTN_W0_GFX1201)?, wgp: Module::load_data(QE_ATTN_W0WGP_GFX1201)? })
    }

    /// (b)/(c): `n_wg` WGs of 512 threads (one per CU), `n_iters` iterations of
    /// `mix` = (W, V, E). `a_in` >= 18 f16; `out` >= n_wg * 512.
    pub fn mix(&self, st: &Stream, mix: (u32, u32, u32), out: &mut DeviceBuffer<f32>, a_in: &DeviceBuffer<u16>, n_iters: u32, n_wg: u32) -> eyre::Result<()> {
        if !MIXES.contains(&mix) {
            return Err(eyre!("qw_mix: no kernel for {mix:?}"));
        }
        if out.len() < (n_wg * 512) as usize || a_in.len() < 18 {
            return Err(eyre!("qw_mix: buffers too small"));
        }
        let f = self.cu.get_function(&format!("qw_mix_w{}_v{}_e{}", mix.0, mix.1, mix.2))?;
        let lc = LaunchConfig { grid: (n_wg, 1, 1), block: (512, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, lc, st, [out.raw(), a_in.raw(), n_iters])
    }

    /// (d): lane-linear ds_load_b128, `threads` (256 / 512) per WG, CU or WGP module.
    pub fn lds_b128(&self, st: &Stream, wgp: bool, out: &mut DeviceBuffer<u32>, n_iters: u32, n_wg: u32, threads: u32) -> eyre::Result<()> {
        if out.len() < (n_wg * threads) as usize || !(threads == 256 || threads == 512) {
            return Err(eyre!("qw_lds_b128: bad geometry"));
        }
        let f = if wgp { &self.wgp } else { &self.cu }.get_function("qw_lds_b128")?;
        let lc = LaunchConfig { grid: (n_wg, 1, 1), block: (threads, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, lc, st, [out.raw(), n_iters])
    }

    /// (e): the NW16 barrier pattern; `err` / `done` = one u32 each, zeroed by the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn barrier_rearrive(&self, st: &Stream, err: &mut DeviceBuffer<u32>, done: &mut DeviceBuffer<u32>, n_tiles: u32, sleep_wave: u32, sleep_mask: u32, n_wg: u32) -> eyre::Result<()> {
        if n_tiles == 0 || err.is_empty() || done.is_empty() {
            return Err(eyre!("qw_barrier_rearrive: bad arguments"));
        }
        let f = self.cu.get_function("qw_barrier_rearrive")?;
        let lc = LaunchConfig { grid: (n_wg, 1, 1), block: (512, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, lc, st, [err.raw(), done.raw(), n_tiles, sleep_wave, sleep_mask])
    }

    /// (f): all 2^32 f32 patterns; `mism` = 0 and `first` = u32::MAX on entry.
    pub fn fma_mix_vs_cvt(&self, st: &Stream, mism: &mut DeviceBuffer<u32>, first: &mut DeviceBuffer<u32>) -> eyre::Result<()> {
        let f = self.cu.get_function("qw_fma_mix_vs_cvt")?;
        let lc = LaunchConfig { grid: (65536, 256, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(f, lc, st, [mism.raw(), first.raw()])
    }
}
