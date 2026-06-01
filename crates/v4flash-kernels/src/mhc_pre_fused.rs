//! Fused mhc_pre kernel: replaces the 5-kernel
//! `rms_nw → f16.matvec → hc_sinkhorn → hc_weighted → rms_w`
//! chain with a single WG-internal pipeline. Targets decode-path
//! launch-overhead-bound stages (per [[decode-long-ctx-analysis]]).

use crate::config::SINKHORN_EPS;
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const MHC_PRE_FUSED_GFX1201: &[u8] = include_bytes!(env!("KERNEL_MHC_PRE_FUSED_GFX1201"));
const MHC_PRE_FUSED_GFX1151: &[u8] = include_bytes!(env!("KERNEL_MHC_PRE_FUSED_GFX1151"));

pub struct MhcPreFused {
    module: Module,
}

impl MhcPreFused {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            MHC_PRE_FUSED_GFX1201
        } else if arch.starts_with("gfx1151") {
            MHC_PRE_FUSED_GFX1151
        } else {
            return Err(eyre!("unsupported arch for mhc_pre_fused: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Fused mhc_pre. Replaces the rms_nw + f16.matvec + hc_sinkhorn +
    /// hc_weighted + rms_w chain in one WG. Caller swaps weights for the
    /// pre_attn vs pre_ffn variant.
    ///
    /// Layout: see kernel comment. Requires HC_DIM==16384, N_EMBD==4096,
    /// HC_MIX_DIM==24, N_HC==4 (the V4-Flash configuration). Single WG
    /// (`grid=(1,1,1)`, `block=(512,1,1)`); intermediate state is in LDS.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,            // [N_EMBD=4096]
        residual: &DeviceBuffer<f32>,           // [HC_DIM=16384]
        hc_fn_w: &DeviceBuffer<u8>,             // [HC_MIX_DIM=24, HC_DIM] f16 (raw bytes)
        sk_scale: &DeviceBuffer<f32>,           // [3]
        sk_base: &DeviceBuffer<f32>,            // [HC_MIX_DIM=24]
        norm_w: &DeviceBuffer<f32>,             // [N_EMBD=4096]
        rms_eps: f32,
        sinkhorn_iters: u32,
    ) -> eyre::Result<()> {
        let function = self.module.get_function("mhc_pre_fused")?;
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (512, 1, 1),
            shared_mem_bytes: 0, // static __shared__
        };
        let sinkhorn_eps: f32 = SINKHORN_EPS;
        launch_kernel!(function, cfg, stream, [
            out.raw(), residual.raw(), hc_fn_w.raw(),
            sk_scale.raw(), sk_base.raw(), norm_w.raw(),
            rms_eps, sinkhorn_eps, sinkhorn_iters
        ])
    }
}
