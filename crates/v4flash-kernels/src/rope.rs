//! RoPE — applies `rope_tail_ext_inplace` (ds4.c:4728) on the GPU.
//!
//! Mirrors ds4's per-layer-varying RoPE: rotates only the tail `n_rot` dims
//! of each head; uses YaRN ramp interpolation when the layer is "compressed"
//! (ext_factor != 0). Forward and inverse share one kernel via a flag arg;
//! inverse is used by the attention output projection in M5+.
//!
//! Per-layer YaRN parameters live in the activation dump under
//! `weight:rope_params` as 6 floats: `[freq_base, freq_scale, ext_factor,
//! attn_factor, beta_fast, beta_slow]`. `n_ctx_orig` is layer-invariant
//! (`DS4_ROPE_ORIG_CTX = 65536` when compressed, 0 otherwise) and is set by
//! the caller; in practice it's only consulted when `ext_factor != 0`.

use std::f32::consts::PI;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const ROPE_TAIL_GFX1201: &[u8] = include_bytes!(env!("KERNEL_ROPE_TAIL_GFX1201"));
const ROPE_TAIL_GFX1151: &[u8] = include_bytes!(env!("KERNEL_ROPE_TAIL_GFX1151"));

/// Per-layer RoPE parameters; matches the 6-float `weight:rope_params` blob
/// emitted by the ds4-dump 0004 patch, plus the layer-invariant `n_ctx_orig`.
#[derive(Debug, Clone, Copy)]
pub struct RopeParams {
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    /// `DS4_ROPE_ORIG_CTX` for compressed layers, 0 otherwise. Only consulted
    /// when `ext_factor != 0`, so non-compressed layers can pass 0.
    pub n_ctx_orig: u64,
}

impl RopeParams {
    /// Parse from the 6-float `weight:rope_params` blob captured by the dump.
    /// `n_ctx_orig` is supplied separately (it's the same for every compressed
    /// layer, and ignored when ext_factor == 0).
    pub fn from_dump_blob(params: &[f32], n_ctx_orig: u64) -> eyre::Result<Self> {
        if params.len() != 6 {
            return Err(eyre!(
                "rope_params blob has {} floats, expected 6",
                params.len()
            ));
        }
        Ok(Self {
            freq_base: params[0],
            freq_scale: params[1],
            ext_factor: params[2],
            attn_factor: params[3],
            beta_fast: params[4],
            beta_slow: params[5],
            n_ctx_orig,
        })
    }
}

pub struct RopeTail {
    module: Module,
}

impl RopeTail {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            ROPE_TAIL_GFX1201
        } else if arch.starts_with("gfx1151") {
            ROPE_TAIL_GFX1151
        } else {
            return Err(eyre!("unsupported arch for rope_tail kernel: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    pub fn launch_forward(
        &self,
        stream: &Stream,
        x: &mut DeviceBuffer<f32>,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        pos: u32,
        params: &RopeParams,
    ) -> eyre::Result<()> {
        self.launch(stream, x, n_head, head_dim, n_rot, pos, params, false)
    }

    pub fn launch_inverse(
        &self,
        stream: &Stream,
        x: &mut DeviceBuffer<f32>,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        pos: u32,
        params: &RopeParams,
    ) -> eyre::Result<()> {
        self.launch(stream, x, n_head, head_dim, n_rot, pos, params, true)
    }

    /// Batched variant: process B independent (n_head × head_dim) rows in
    /// one launch. `pos_per_b` is a device buffer of B i32 positions.
    /// Forward = `inverse=false`, mirror to `launch_inverse_batched`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_forward_batched(
        &self,
        stream: &Stream,
        x: &mut DeviceBuffer<f32>,
        pos_per_b: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        b: u32,
        params: &RopeParams,
    ) -> eyre::Result<()> {
        self.launch_batched(stream, x, pos_per_b, n_head, head_dim, n_rot, b, params, false)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_inverse_batched(
        &self,
        stream: &Stream,
        x: &mut DeviceBuffer<f32>,
        pos_per_b: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        b: u32,
        params: &RopeParams,
    ) -> eyre::Result<()> {
        self.launch_batched(stream, x, pos_per_b, n_head, head_dim, n_rot, b, params, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_batched(
        &self,
        stream: &Stream,
        x: &mut DeviceBuffer<f32>,
        pos_per_b: &DeviceBuffer<i32>,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        b: u32,
        params: &RopeParams,
        inverse: bool,
    ) -> eyre::Result<()> {
        if n_rot % 2 != 0 {
            return Err(eyre!("rope_tail batched: n_rot={n_rot} must be even"));
        }
        if n_rot > head_dim {
            return Err(eyre!("rope_tail batched: n_rot={n_rot} > head_dim={head_dim}"));
        }
        if b == 0 {
            return Ok(());
        }
        let theta_scale = params.freq_base.powf(-2.0 / n_rot as f32);
        let mscale_eff = if params.ext_factor != 0.0 && params.freq_scale > 0.0 {
            params.attn_factor * (1.0 + 0.1 * (1.0 / params.freq_scale).ln())
        } else {
            params.attn_factor
        };
        let (corr_low, corr_high) = if params.ext_factor != 0.0 {
            corr_dims(n_rot, params.n_ctx_orig, params.freq_base, params.beta_fast, params.beta_slow)
        } else {
            (0.0, 0.0)
        };
        let function = self.module.get_function("rope_tail_batched")?;
        let inverse_i: i32 = if inverse { 1 } else { 0 };
        let cfg = LaunchConfig {
            grid: (n_head, 1, b),
            block: (n_rot / 2, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            x.raw(), pos_per_b.raw(), n_head, head_dim, n_rot,
            theta_scale, params.freq_scale, params.ext_factor,
            mscale_eff, corr_low, corr_high, inverse_i
        ])
    }

    fn launch(
        &self,
        stream: &Stream,
        x: &mut DeviceBuffer<f32>,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        pos: u32,
        params: &RopeParams,
        inverse: bool,
    ) -> eyre::Result<()> {
        if n_rot % 2 != 0 {
            return Err(eyre!("rope_tail: n_rot={n_rot} must be even"));
        }
        if n_rot > head_dim {
            return Err(eyre!(
                "rope_tail: n_rot={n_rot} must be <= head_dim={head_dim}"
            ));
        }
        let needed = (n_head as usize) * (head_dim as usize);
        if x.len() < needed {
            return Err(eyre!(
                "rope_tail: x has {} elements, need n_head*head_dim={}",
                x.len(),
                needed
            ));
        }

        // theta_scale = freq_base^(-2/n_rot)
        let theta_scale = params.freq_base.powf(-2.0 / n_rot as f32);

        // mscale_eff = attn_factor * (1 + 0.1*log(1/freq_scale)) if ext_factor != 0,
        //              else attn_factor.
        // Mirrors the per-iteration mscale update in rope_tail_ext_inplace.
        let mscale_eff =
            if params.ext_factor != 0.0 && params.freq_scale > 0.0 {
                params.attn_factor * (1.0 + 0.1 * (1.0 / params.freq_scale).ln())
            } else {
                params.attn_factor
            };

        // corr_low / corr_high — only meaningful when ext_factor != 0. Compute
        // host-side so the kernel doesn't redo the work per workgroup.
        let (corr_low, corr_high) = if params.ext_factor != 0.0 {
            corr_dims(
                n_rot,
                params.n_ctx_orig,
                params.freq_base,
                params.beta_fast,
                params.beta_slow,
            )
        } else {
            (0.0, 0.0)
        };

        let function = self.module.get_function("rope_tail")?;

        let inverse_i: i32 = if inverse { 1 } else { 0 };
        let cfg = LaunchConfig {
            grid: (n_head, 1, 1),
            block: (n_rot / 2, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            x.raw(), n_head, head_dim, n_rot, pos,
            theta_scale, params.freq_scale, params.ext_factor,
            mscale_eff, corr_low, corr_high, inverse_i
        ])
    }
}

/// `rope_yarn_corr_dims` — port of ds4.c:4718. Returns `(low, high)` clamped
/// per the CPU function:
///   low  = max(0, floor(corr_dim(beta_fast)))
///   high = min(n_rot - 1, ceil(corr_dim(beta_slow)))
fn corr_dims(n_rot: u32, n_ctx_orig: u64, freq_base: f32, beta_fast: f32, beta_slow: f32) -> (f32, f32) {
    let start = corr_dim(n_rot as f32, n_ctx_orig, beta_fast, freq_base).floor();
    let end = corr_dim(n_rot as f32, n_ctx_orig, beta_slow, freq_base).ceil();
    let low = start.max(0.0);
    let high = end.min((n_rot - 1) as f32);
    (low, high)
}

fn corr_dim(n_dims: f32, n_ctx_orig: u64, n_rot_arg: f32, base: f32) -> f32 {
    n_dims * ((n_ctx_orig as f32) / (n_rot_arg * 2.0 * PI)).ln() / (2.0 * base.ln())
}
