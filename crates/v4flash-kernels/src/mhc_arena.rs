//! Arena mHC mixes in 1 launch per sub-block, bit-identical to the chains it
//! replaces (kernels/mhc_arena.hip, tests/mhc_arena_bitexact.rs):
//! `launch_mix` = RMS scalar + narrow f16 matvec + (last WG) Sinkhorn split.

use crate::config::{HC_MIX_DIM, N_HC};
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const MHC_ARENA_GFX1201: &[u8] = include_bytes!(env!("KERNEL_MHC_ARENA_GFX1201"));
const MHC_ARENA_GFX1151: &[u8] = include_bytes!(env!("KERNEL_MHC_ARENA_GFX1151"));

/// `launch_mix` mode: the pre-attn decode-exact path (16-slice multi-WG RMS
/// scalar, then `W @ x` scaled once) -- `rms_nw_mw.launch_inv_only` +
/// `f16.matvec_pre_scaled` per row.
pub const MIX_PRE_SCALED: u32 = 0;
/// `launch_mix` mode: normalize-then-dot -- `rms_nw.launch_batched` +
/// `f16.matvec_narrow_batched`.
pub const MIX_NORMED: u32 = 1;
/// Slices of the pre-scaled RMS scalar (`launch_inv_only`'s `n_wgs`).
pub const MIX_RMS_WGS: u32 = 16;

pub struct MhcArena {
    module: Module,
}

impl MhcArena {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            MHC_ARENA_GFX1201
        } else if arch.starts_with("gfx1151") {
            MHC_ARENA_GFX1151
        } else {
            return Err(eyre!("unsupported arch for mhc_arena: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
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

}
