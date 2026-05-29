//! V4 Flash head ops: HC sigmoid+bias + HC weighted-sum.
//!
//! Mirrors `output_hc_head_one` / `output_logits_one_decode_scratch`
//! (ds4.c:8139, 8195):
//!   flat   = rms_norm_no_weight(inp_hc, hc_dim)
//!   pre    = matvec_f16(output_hc_fn, flat)       [n_hc]
//!   w      = sigmoid_stable(pre*scale + base) + DS4_HC_EPS
//!   embd   = sum_h inp_hc[h*n_embd + d] * w[h]   [n_embd]
//!
//! `RmsNormNoWeight` (M3) and `F16Matvec` (M7) cover the first two steps;
//! this module adds the last two as `HcSigmoidBias` and `HcWeightedSum`.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const HC_SIGMOID_BIAS_GFX1201: &[u8] = include_bytes!(env!("KERNEL_HC_SIGMOID_BIAS_GFX1201"));
const HC_SIGMOID_BIAS_GFX1151: &[u8] = include_bytes!(env!("KERNEL_HC_SIGMOID_BIAS_GFX1151"));
const HC_WEIGHTED_SUM_GFX1201: &[u8] = include_bytes!(env!("KERNEL_HC_WEIGHTED_SUM_GFX1201"));
const HC_WEIGHTED_SUM_GFX1151: &[u8] = include_bytes!(env!("KERNEL_HC_WEIGHTED_SUM_GFX1151"));
const HC_SINKHORN_GFX1201: &[u8] = include_bytes!(env!("KERNEL_HC_SINKHORN_GFX1201"));
const HC_SINKHORN_GFX1151: &[u8] = include_bytes!(env!("KERNEL_HC_SINKHORN_GFX1151"));
const HC_POST_GFX1201: &[u8] = include_bytes!(env!("KERNEL_HC_POST_GFX1201"));
const HC_POST_GFX1151: &[u8] = include_bytes!(env!("KERNEL_HC_POST_GFX1151"));
const HC_SINKHORN_PAR_GFX1201: &[u8] = include_bytes!(env!("KERNEL_HC_SINKHORN_PAR_GFX1201"));
const HC_SINKHORN_PAR_GFX1151: &[u8] = include_bytes!(env!("KERNEL_HC_SINKHORN_PAR_GFX1151"));

/// `w[i] = sigmoid_stable(pre[i] * scale[0] + base[i]) + DS4_HC_EPS`.
pub struct HcSigmoidBias {
    module: Module,
}

impl HcSigmoidBias {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            HC_SIGMOID_BIAS_GFX1201
        } else if arch.starts_with("gfx1151") {
            HC_SIGMOID_BIAS_GFX1151
        } else {
            return Err(eyre!("unsupported arch for hc_sigmoid_bias: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        pre: &DeviceBuffer<f32>,
        scale: &DeviceBuffer<f32>,
        base: &DeviceBuffer<f32>,
        n: u32,
    ) -> eyre::Result<()> {
        let function = self.module.get_function("hc_sigmoid_bias")?;
        let block_x = 32u32;
        let grid_x = n.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), pre.raw(), scale.raw(), base.raw(), n])
    }

    /// M50 Phase 2: batched. pre[B, n], out[B, n]. scale, base shared.
    pub fn launch_batched(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        pre: &DeviceBuffer<f32>,
        scale: &DeviceBuffer<f32>,
        base: &DeviceBuffer<f32>,
        n: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let function = self.module.get_function("hc_sigmoid_bias_batched")?;
        let block_x = 32u32;
        let grid_x = n.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, 1, batch),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), pre.raw(), scale.raw(), base.raw(), n])
    }
}

/// `out[d] = sum_h x[h*n_embd + d] * weights[h]`. Used by the head's HC collapse
/// and (in future) by `hc_pre_from_state_one` per-layer.
pub struct HcWeightedSum {
    module: Module,
}

impl HcWeightedSum {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            HC_WEIGHTED_SUM_GFX1201
        } else if arch.starts_with("gfx1151") {
            HC_WEIGHTED_SUM_GFX1151
        } else {
            return Err(eyre!("unsupported arch for hc_weighted_sum: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<f32>,
        n_embd: u32,
        n_hc: u32,
    ) -> eyre::Result<()> {
        let function = self.module.get_function("hc_weighted_sum")?;
        let block_x = 256u32;
        let grid_x = n_embd.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), x.raw(), weights.raw(), n_embd, n_hc])
    }

    /// M50 Phase 2: batched. x[B, n_hc, n_embd], weights[B, n_hc],
    /// out[B, n_embd]. Grid (n_embd/256, 1, B).
    /// M50 Phase 2: batched hc_weighted_sum.
    ///
    /// `w_stride` is the per-batch stride of `weights`. When `weights`
    /// is the packed `split[B, HC_MIX_DIM]` from sinkhorn, pass
    /// `w_stride = HC_MIX_DIM` (= 2*n_hc + n_hc*n_hc); the kernel reads
    /// the first `n_hc` elements (= pre-sigmoid "w" portion) per batch.
    /// For a tightly-packed `weights[B, n_hc]`, pass `w_stride = n_hc`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        weights: &DeviceBuffer<f32>,
        n_embd: u32,
        n_hc: u32,
        w_stride: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let function = self.module.get_function("hc_weighted_sum_batched")?;
        let block_x = 256u32;
        let grid_x = n_embd.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, 1, batch),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), x.raw(), weights.raw(), n_embd, n_hc, w_stride
        ])
    }
}

/// Per-layer mHC Sinkhorn split. Mirrors `hc_split_sinkhorn_one`
/// (ds4.c:4220). Output is 2*n_hc + n_hc*n_hc = 24 floats for n_hc=4.
///
/// Holds both kernel images: the original single-thread variant (used
/// for general `n_hc`) and the M14a 16-thread shared-mem variant
/// specialized for `n_hc=4` (V4-Flash's only configuration). The
/// `launch()` entry point auto-picks the parallel version when
/// applicable.
pub struct HcSinkhorn {
    serial: Module,
    par: Module,
}

impl HcSinkhorn {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let (serial_img, par_img): (&[u8], &[u8]) = if arch.starts_with("gfx1201") {
            (HC_SINKHORN_GFX1201, HC_SINKHORN_PAR_GFX1201)
        } else if arch.starts_with("gfx1151") {
            (HC_SINKHORN_GFX1151, HC_SINKHORN_PAR_GFX1151)
        } else {
            return Err(eyre!("unsupported arch for hc_sinkhorn: {arch}"));
        };
        let serial = Module::load_data(serial_img)?;
        let par = Module::load_data(par_img)?;
        Ok(Self { serial, par })
    }

    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        mix: &DeviceBuffer<f32>,
        scale: &DeviceBuffer<f32>,
        base: &DeviceBuffer<f32>,
        n_hc: u32,
        iters: u32,
        eps: f32,
    ) -> eyre::Result<()> {
        // V4-Flash always uses n_hc=4. Take the M14a parallel path.
        // Other n_hc values still work via the original single-thread
        // kernel (only exercised by older oracle tests, if any).
        let (function, cfg) = if n_hc == 4 {
            (
                self.par.get_function("hc_sinkhorn_par")?,
                LaunchConfig {
                    grid: (1, 1, 1),
                    block: (16, 1, 1),
                    shared_mem_bytes: 0,
                },
            )
        } else {
            (
                self.serial.get_function("hc_sinkhorn")?,
                LaunchConfig {
                    grid: (1, 1, 1),
                    block: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
            )
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), mix.raw(), scale.raw(), base.raw(), n_hc, iters, eps
        ])
    }

    /// M50 Phase 2: batched sinkhorn (n_hc=4 only). Grid (B, 1, 1),
    /// block (16). Each WG independently runs the 20-iteration
    /// Sinkhorn-Knopp on its batch element's mix.
    /// mix[B, 2*n_hc + n_hc*n_hc], out same shape. scale, base shared.
    pub fn launch_batched(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        mix: &DeviceBuffer<f32>,
        scale: &DeviceBuffer<f32>,
        base: &DeviceBuffer<f32>,
        n_hc: u32,
        iters: u32,
        eps: f32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if n_hc != 4 {
            return Err(eyre!("sinkhorn batched only supports n_hc=4"));
        }
        let function = self.par.get_function("hc_sinkhorn_par_batched")?;
        let cfg = LaunchConfig {
            grid: (batch, 1, 1),
            block: (16, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), mix.raw(), scale.raw(), base.raw(), n_hc, iters, eps
        ])
    }
}

/// HC post: blend sublayer output with HC residual. Mirrors `hc_post_one`
/// (ds4.c:4400). Output `[n_hc, n_embd]` row-major.
pub struct HcPost {
    module: Module,
}

impl HcPost {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            HC_POST_GFX1201
        } else if arch.starts_with("gfx1151") {
            HC_POST_GFX1151
        } else {
            return Err(eyre!("unsupported arch for hc_post: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    pub fn launch(
        &self,
        stream: &Stream,
        out_hc: &mut DeviceBuffer<f32>,
        block_out: &DeviceBuffer<f32>,
        residual_hc: &DeviceBuffer<f32>,
        post: &DeviceBuffer<f32>,
        comb: &DeviceBuffer<f32>,
        n_embd: u32,
        n_hc: u32,
    ) -> eyre::Result<()> {
        let function = self.module.get_function("hc_post")?;
        let block_x = 256u32;
        let grid_x = n_embd.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, n_hc, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out_hc.raw(), block_out.raw(), residual_hc.raw(), post.raw(), comb.raw(), n_embd, n_hc
        ])
    }

    /// M50 Phase 2: batched. out_hc[B, n_hc, n_embd], block_out[B, n_embd],
    /// residual_hc[B, n_hc, n_embd], post[B, n_hc], comb[B, n_hc, n_hc].
    /// Grid (n_embd/256, n_hc, B).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched(
        &self,
        stream: &Stream,
        out_hc: &mut DeviceBuffer<f32>,
        block_out: &DeviceBuffer<f32>,
        residual_hc: &DeviceBuffer<f32>,
        post: &DeviceBuffer<f32>,
        comb: &DeviceBuffer<f32>,
        n_embd: u32,
        n_hc: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let function = self.module.get_function("hc_post_batched")?;
        let block_x = 256u32;
        let grid_x = n_embd.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, n_hc, batch),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out_hc.raw(), block_out.raw(), residual_hc.raw(), post.raw(), comb.raw(), n_embd, n_hc
        ])
    }

    /// M50 Phase 2: batched from_split. Per-batch split layout:
    /// `[B][n_w + n_hc + n_hc*n_hc]`. The kernel reads `post` at
    /// offset `n_w` and `comb` at offset `n_w + n_hc` within each
    /// per-batch row.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_from_split_batched(
        &self,
        stream: &Stream,
        out_hc: &mut DeviceBuffer<f32>,
        block_out: &DeviceBuffer<f32>,
        residual_hc: &DeviceBuffer<f32>,
        split: &DeviceBuffer<f32>,
        n_w: u32,
        n_embd: u32,
        n_hc: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let function = self.module.get_function("hc_post_from_split_batched")?;
        let block_x = 256u32;
        let grid_x = n_embd.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, n_hc, batch),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out_hc.raw(), block_out.raw(), residual_hc.raw(), split.raw(), n_w, n_embd, n_hc
        ])
    }

    /// Launch reading `post` and `comb` directly from a packed `split`
    /// buffer with layout `[w(n_w), post(n_hc), comb(n_hc*n_hc)]`. This
    /// is exactly the layout produced by `HcSinkhorn`, so callers can
    /// skip the device→host→device extraction round-trip entirely
    /// (M13.3).
    pub fn launch_from_split(
        &self,
        stream: &Stream,
        out_hc: &mut DeviceBuffer<f32>,
        block_out: &DeviceBuffer<f32>,
        residual_hc: &DeviceBuffer<f32>,
        split: &DeviceBuffer<f32>,
        n_w: u32,
        n_embd: u32,
        n_hc: u32,
    ) -> eyre::Result<()> {
        let function = self.module.get_function("hc_post")?;
        // SAFETY: caller must ensure split has length >= n_w + n_hc + n_hc*n_hc.
        let split_base = split.raw() as *mut u8;
        let post_offset_bytes = (n_w as usize) * 4;
        let comb_offset_bytes = ((n_w + n_hc) as usize) * 4;
        let p_ptr =
            unsafe { split_base.add(post_offset_bytes) } as v4flash_hip::sys::hipDeviceptr_t;
        let c_ptr =
            unsafe { split_base.add(comb_offset_bytes) } as v4flash_hip::sys::hipDeviceptr_t;
        let block_x = 256u32;
        let grid_x = n_embd.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, n_hc, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out_hc.raw(), block_out.raw(), residual_hc.raw(), p_ptr, c_ptr, n_embd, n_hc
        ])
    }
}
