//! Arena mHC mixes in 1 launch per sub-block, bit-identical to the chains it
//! replaces (kernels/mhc_arena.hip, tests/mhc_arena_bitexact.rs):
//! `launch_mix` = RMS scalar + narrow f16 matvec + (last WG) Sinkhorn split.

use crate::config::{HC_MIX_DIM, N_HC};
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const MHC_ARENA_GFX1201: &[u8] = include_bytes!(env!("KERNEL_MHC_ARENA_GFX1201"));
const MHC_ARENA_GFX1151: &[u8] = include_bytes!(env!("KERNEL_MHC_ARENA_GFX1151"));
const MHC_FAST_GFX1201: &[u8] = include_bytes!(env!("KERNEL_MHC_FAST_GFX1201"));
const MHC_FAST_GFX1151: &[u8] = include_bytes!(env!("KERNEL_MHC_FAST_GFX1151"));

/// `launch_fast`'s mix half: one sub-block's mixes (`launch_mix`'s arguments).
pub struct FastMix<'a> {
    pub weight: &'a DeviceBuffer<u8>,
    /// [batch, k] mix input.
    pub x: &'a DeviceBuffer<f32>,
    pub scale: &'a DeviceBuffer<f32>,
    pub base: &'a DeviceBuffer<f32>,
    /// `MIX_PRE_SCALED` or `MIX_NORMED`.
    pub mode: u32,
    pub split_out: &'a mut DeviceBuffer<f32>,
    pub mix_out: &'a mut DeviceBuffer<f32>,
    pub counters: &'a mut DeviceBuffer<u32>,
    /// [>= batch] scratch for the pre-scaled RMS scalar (written and read
    /// inside the launch).
    pub inv_rows: &'a mut DeviceBuffer<f32>,
}

/// `launch_fast`'s collapse half: `hc_weighted_sum` with the carry as weights,
/// then `rms_norm_weighted`.
pub struct FastCollapse<'a> {
    /// [batch, k] collapse input (N_HC copies of n_embd).
    pub x: &'a DeviceBuffer<f32>,
    /// [batch, n_embd] hc_weighted_sum output (what the chain left in `cur`).
    pub cur_out: &'a mut DeviceBuffer<f32>,
    /// [batch, n_embd] rms_norm_weighted output.
    pub norm_out: &'a mut DeviceBuffer<f32>,
    pub norm_w: &'a DeviceBuffer<f32>,
}

/// `launch_mix` mode: the pre-attn decode-exact path (16-slice multi-WG RMS
/// scalar, then `W @ x` scaled once) -- `rms_nw_mw.launch_inv_only` +
/// `f16.matvec_pre_scaled` per row.
pub const MIX_PRE_SCALED: u32 = 0;
/// `launch_mix` mode: normalize-then-dot -- `rms_nw.launch_batched` +
/// `f16.matvec_narrow_batched`.
pub const MIX_NORMED: u32 = 1;
/// Slices of the pre-scaled RMS scalar (`launch_inv_only`'s `n_wgs`).
pub const MIX_RMS_WGS: u32 = 16;

/// TIER 2 (`launch_mix_ksplit`, NOT bit-identical): K-chunks per mix row.
/// `k` must be a multiple of `MIX_KSPLIT * 256`.
pub const MIX_KSPLIT: u32 = 10;

pub struct MhcArena {
    module: Module,
    fast: Module,
}

impl MhcArena {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let (image, fast): (&[u8], &[u8]) = if arch.starts_with("gfx1201") {
            (MHC_ARENA_GFX1201, MHC_FAST_GFX1201)
        } else if arch.starts_with("gfx1151") {
            (MHC_ARENA_GFX1151, MHC_FAST_GFX1151)
        } else {
            return Err(eyre!("unsupported arch for mhc_arena: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)?, fast: Module::load_data(fast)? })
    }

    /// One arena mHC sub-block in ONE launch (kernels/mhc_fast.hip), every
    /// output bit-identical to the kernels it replaces:
    /// - `mix`: `launch_mix` (split + mix), and with `write_carry` also the
    ///   `carry := split` copy the chain did after its collapse;
    /// - `collapse`: `hc_weighted_sum_batched(x, carry)` into `cur_out`, then
    ///   `rms_norm_weighted_batched` into `norm_out`. It reads the carry as it
    ///   was BEFORE this launch even when `write_carry` rewrites it (the last
    ///   WG writes it after the collapse WG has finished).
    /// At least one half must be present. `k` must be `HC_DIM` (the kernel is
    /// compiled for it); `carry` is [>= batch, HC_MIX_DIM].
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fast(
        &self,
        stream: &Stream,
        mix: Option<FastMix<'_>>,
        collapse: Option<FastCollapse<'_>>,
        carry: &mut DeviceBuffer<f32>,
        write_carry: bool,
        k: u32,
        rms_eps: f32,
        sinkhorn_iters: u32,
        sinkhorn_eps: f32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let (bu, ku, m) = (batch as usize, k as usize, HC_MIX_DIM as usize);
        if k != crate::config::HC_DIM || N_HC != 4 || k % (MIX_RMS_WGS * 256) != 0 {
            return Err(eyre!("mhc_fast: k={k} (compiled for HC_DIM={}, n_hc 4)", crate::config::HC_DIM));
        }
        let ne = k / N_HC;
        if carry.len() < bu * m {
            return Err(eyre!("mhc_fast: carry too small for batch {batch}"));
        }
        if mix.is_none() && collapse.is_none() {
            return Err(eyre!("mhc_fast: neither mix nor collapse"));
        }
        if write_carry && mix.is_none() {
            return Err(eyre!("mhc_fast: write_carry needs the mix"));
        }
        let null = std::ptr::null_mut::<std::ffi::c_void>();
        let (split_p, mix_p, cnt_p, inv_p, w_p, x_p, scale_p, base_p, mode, n_mix) = match mix {
            Some(mx) => {
                if mx.mode > MIX_NORMED {
                    return Err(eyre!("mhc_fast: mode {}", mx.mode));
                }
                if mx.weight.byte_len() != m * ku * 2 {
                    return Err(eyre!("mhc_fast: weight bytes {} != {}", mx.weight.byte_len(), m * ku * 2));
                }
                if mx.x.len() < bu * ku
                    || mx.split_out.len() < bu * m
                    || mx.mix_out.len() < bu * m
                    || mx.counters.len() < bu
                    || mx.inv_rows.len() < bu
                {
                    return Err(eyre!("mhc_fast: mix buffer too small for batch {batch}"));
                }
                if mx.scale.len() < 3 || mx.base.len() < m {
                    return Err(eyre!("mhc_fast: scale/base shape"));
                }
                (
                    mx.split_out.raw(),
                    mx.mix_out.raw(),
                    mx.counters.raw(),
                    mx.inv_rows.raw(),
                    mx.weight.raw(),
                    mx.x.raw(),
                    mx.scale.raw(),
                    mx.base.raw(),
                    mx.mode,
                    HC_MIX_DIM,
                )
            }
            None => (null, null, null, null, null, null, null, null, 0u32, 0u32),
        };
        // Mode 0 with a mix: one extra WG per row computes the RMS scalar.
        let rms_wg = u32::from(n_mix > 0 && mode == MIX_PRE_SCALED);
        let (cx_p, cur_p, norm_p, nw_p, do_collapse) = match collapse {
            Some(c) => {
                if c.x.len() < bu * ku || c.cur_out.len() < bu * ne as usize || c.norm_out.len() < bu * ne as usize {
                    return Err(eyre!("mhc_fast: collapse buffer too small for batch {batch}"));
                }
                if c.norm_w.len() != ne as usize {
                    return Err(eyre!("mhc_fast: norm weight len {} != {ne}", c.norm_w.len()));
                }
                (c.x.raw(), c.cur_out.raw(), c.norm_out.raw(), c.norm_w.raw(), 1u32)
            }
            None => (null, null, null, null, 0u32),
        };
        let function = self.fast.get_function("mhc_fast_batched")?;
        let cfg = LaunchConfig { grid: (n_mix + do_collapse + rms_wg, 1, batch), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [
            split_p, mix_p, cnt_p, inv_p, w_p, x_p, scale_p, base_p, carry.raw(),
            cx_p, cur_p, norm_p, nw_p,
            k, ne, n_mix, do_collapse, write_carry as u32, mode, rms_eps,
            N_HC, sinkhorn_iters, sinkhorn_eps
        ])
    }

    /// Mix + split for `batch` rows of `x` ([batch, k]): writes `mix_out`
    /// ([batch, HC_MIX_DIM], the matvec as the old chain left it in `mix`) and
    /// `split_out` ([batch, HC_MIX_DIM], the Sinkhorn split). `counters`
    /// ([>= batch] u32) must be zero before the first launch; the kernel
    /// leaves it zero.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_mix(
        &self,
        stream: &Stream,
        split_out: &mut DeviceBuffer<f32>,
        mix_out: &mut DeviceBuffer<f32>,
        counters: &mut DeviceBuffer<u32>,
        weight: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        scale: &DeviceBuffer<f32>,
        base: &DeviceBuffer<f32>,
        k: u32,
        mode: u32,
        rms_eps: f32,
        sinkhorn_iters: u32,
        sinkhorn_eps: f32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let (bu, ku, m) = (batch as usize, k as usize, HC_MIX_DIM as usize);
        if mode > MIX_NORMED {
            return Err(eyre!("mhc_arena mix: mode {mode}"));
        }
        if mode == MIX_PRE_SCALED && k % (MIX_RMS_WGS * 256) != 0 {
            return Err(eyre!("mhc_arena mix: k={k} not a multiple of {} (the slice/thread mapping)", MIX_RMS_WGS * 256));
        }
        if weight.byte_len() != m * ku * 2 {
            return Err(eyre!("mhc_arena mix: weight bytes {} != {}", weight.byte_len(), m * ku * 2));
        }
        if x.len() < bu * ku || split_out.len() < bu * m || mix_out.len() < bu * m || counters.len() < bu {
            return Err(eyre!("mhc_arena mix: buffer too small for batch {batch}"));
        }
        if scale.len() < 3 || base.len() < m || N_HC != 4 {
            return Err(eyre!("mhc_arena mix: scale/base shape (n_hc must be 4)"));
        }
        let function = self.module.get_function("mhc_mix_batched")?;
        let cfg = LaunchConfig { grid: (HC_MIX_DIM, 1, batch), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [
            split_out.raw(), mix_out.raw(), counters.raw(), weight.raw(), x.raw(),
            scale.raw(), base.raw(), k, HC_MIX_DIM, mode, MIX_RMS_WGS, rms_eps,
            N_HC, sinkhorn_iters, sinkhorn_eps
        ])
    }

    /// TIER 2, NOT bit-identical: mix + split with each row's dot split over
    /// `MIX_KSPLIT` K-chunks (grid (HC_MIX_DIM * S, 1, B)), dot-then-scale for
    /// both sub-blocks. `dotp` >= [B, HC_MIX_DIM, S], `sqp` >= [B, S] scratch;
    /// `counters` as in `launch_mix`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_mix_ksplit(
        &self,
        stream: &Stream,
        split_out: &mut DeviceBuffer<f32>,
        mix_out: &mut DeviceBuffer<f32>,
        dotp: &mut DeviceBuffer<f32>,
        sqp: &mut DeviceBuffer<f32>,
        counters: &mut DeviceBuffer<u32>,
        weight: &DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        scale: &DeviceBuffer<f32>,
        base: &DeviceBuffer<f32>,
        k: u32,
        rms_eps: f32,
        sinkhorn_iters: u32,
        sinkhorn_eps: f32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let (bu, ku, m, sp) = (batch as usize, k as usize, HC_MIX_DIM as usize, MIX_KSPLIT as usize);
        if k % (MIX_KSPLIT * 256) != 0 {
            return Err(eyre!("mhc_arena ksplit: k={k} not a multiple of {}", MIX_KSPLIT * 256));
        }
        if weight.byte_len() != m * ku * 2 || x.len() < bu * ku || split_out.len() < bu * m || mix_out.len() < bu * m
            || dotp.len() < bu * m * sp || sqp.len() < bu * sp || counters.len() < bu
        {
            return Err(eyre!("mhc_arena ksplit: buffer too small for batch {batch}"));
        }
        if scale.len() < 3 || base.len() < m || N_HC != 4 {
            return Err(eyre!("mhc_arena ksplit: scale/base shape (n_hc must be 4)"));
        }
        let function = self.module.get_function("mhc_mix_ksplit_batched")?;
        let cfg = LaunchConfig { grid: (HC_MIX_DIM * MIX_KSPLIT, 1, batch), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [
            split_out.raw(), mix_out.raw(), dotp.raw(), sqp.raw(), counters.raw(), weight.raw(), x.raw(),
            scale.raw(), base.raw(), k, HC_MIX_DIM, MIX_KSPLIT, rms_eps, N_HC, sinkhorn_iters, sinkhorn_eps
        ])
    }
}
