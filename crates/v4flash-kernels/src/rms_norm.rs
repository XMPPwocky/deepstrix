//! RMSNorm HIP kernel — first port. Mirrors ds4.c:2709 `rms_norm_weight`:
//!
//!     out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
//!
//! V4 Flash uses this at `attn_cur` → `attn_input_norm` and at
//! `ffn_cur` → `ffn_input_norm`, per layer per token. Per-layer scale
//! vectors are `layer->attn_norm` and `layer->ffn_norm` (also captured
//! in the activation dump under `L<LL>/weight/`).
//!
//! Threshold from `docs/PHASE2_KERNEL_VALIDATION.md`: tests assert
//! `max_abs_diff < 1e-4` against the canonical ds4 CPU output.

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, LaunchConfig, Module, Stream};

const RMS_NORM_GFX1201: &[u8] = include_bytes!(env!("KERNEL_RMS_NORM_GFX1201"));
const RMS_NORM_GFX1151: &[u8] = include_bytes!(env!("KERNEL_RMS_NORM_GFX1151"));

const RMS_NORM_NO_WEIGHT_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_RMS_NORM_NO_WEIGHT_GFX1201"));
const RMS_NORM_NO_WEIGHT_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_RMS_NORM_NO_WEIGHT_GFX1151"));

/// Loaded RMSNorm kernel for one device. Bind the current HIP device
/// before calling [`RmsNorm::for_arch`], then re-use the resulting
/// handle across launches.
pub struct RmsNorm {
    module: Module,
}

impl RmsNorm {
    /// Load the kernel blob for the given gfx arch. `gcn_arch_name`
    /// comes from `v4flash_hip::Device::properties().gcn_arch_name`.
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        // gcn_arch_name on RDNA reports e.g. "gfx1151:sramecc-:xnack-".
        // Match on prefix.
        let image: &[u8] = if arch.starts_with("gfx1201") {
            RMS_NORM_GFX1201
        } else if arch.starts_with("gfx1151") {
            RMS_NORM_GFX1151
        } else {
            return Err(eyre!("unsupported arch for rms_norm kernel: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Launch the `rms_norm_weighted` kernel asynchronously on `stream`.
    /// `n` must equal `out.len() == x.len() == weight.len()` and is also
    /// capped by the kernel's reduction layout (currently n ≤ 4096).
    pub fn launch_weighted(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        n: u32,
        eps: f32,
    ) -> eyre::Result<()> {
        if out.len() != n as usize || x.len() != n as usize || weight.len() != n as usize {
            return Err(eyre!(
                "rms_norm_weighted len mismatch: n={}, out={}, x={}, w={}",
                n,
                out.len(),
                x.len(),
                weight.len()
            ));
        }
        if n > 4096 {
            return Err(eyre!("rms_norm_weighted n={n} exceeds kernel cap of 4096"));
        }

        let function = self.module.get_function("rms_norm_weighted")?;

        // Kernel signature: (float *out, const float *x, const float *weight,
        //                   unsigned int n, float eps)
        // HIP kernelParams ABI: array of pointers to each argument's storage.
        let mut out_ptr = out.raw();
        let mut x_ptr = x.raw();
        let mut w_ptr = weight.raw();
        let mut n_val = n;
        let mut eps_val = eps;
        let mut args: [*mut c_void; 5] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut eps_val as *mut _ as *mut c_void,
        ];

        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }

    /// M50 Phase 2: batched rms_norm_weighted. `x[B, n]`, `out[B, n]`,
    /// `weight[n]` shared across batch. Grid (B, 1, 1), block (256).
    pub fn launch_weighted_batched(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        n: u32,
        eps: f32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let needed = (batch as usize) * (n as usize);
        if out.len() < needed || x.len() < needed {
            return Err(eyre!(
                "rms_norm_weighted_batched: buffer too small (need {needed})"
            ));
        }
        if weight.len() != n as usize {
            return Err(eyre!("rms_norm_weighted_batched: weight len != n"));
        }
        if n > 4096 {
            return Err(eyre!("rms_norm_weighted_batched: n={n} > 4096"));
        }
        let function = self
            .module
            .get_function("rms_norm_weighted_batched")?;
        let mut out_ptr = out.raw();
        let mut x_ptr = x.raw();
        let mut w_ptr = weight.raw();
        let mut n_val = n;
        let mut eps_val = eps;
        let mut args: [*mut c_void; 5] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut eps_val as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (batch, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}

/// No-weight RMSNorm — mirrors ds4.c `rms_norm_no_weight`. Operates on
/// `n_rows` independent rows of length `n` (stride n); one workgroup per
/// row. Used by V4 Flash for `head_rms_norm_inplace` (n_rows=64, n=512)
/// and by the head's `output_flat` normalisation (M10; n_rows=1, n=16384).
pub struct RmsNormNoWeight {
    module: Module,
}

impl RmsNormNoWeight {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            RMS_NORM_NO_WEIGHT_GFX1201
        } else if arch.starts_with("gfx1151") {
            RMS_NORM_NO_WEIGHT_GFX1151
        } else {
            return Err(eyre!(
                "unsupported arch for rms_norm_no_weight kernel: {arch}"
            ));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Launch `rms_norm_no_weight` over `n_rows` rows of length `n` in
    /// `x` (stride n, row-major), writing to `out`. Both buffers must
    /// hold at least `n_rows * n` elements.
    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        n_rows: u32,
        n: u32,
        eps: f32,
    ) -> eyre::Result<()> {
        let needed = (n_rows as usize) * (n as usize);
        if out.len() < needed || x.len() < needed {
            return Err(eyre!(
                "rms_norm_no_weight len mismatch: n_rows={n_rows}, n={n}, need={needed}, out={}, x={}",
                out.len(),
                x.len()
            ));
        }

        let function = self.module.get_function("rms_norm_no_weight")?;

        let mut out_ptr = out.raw();
        let mut x_ptr = x.raw();
        let mut n_val = n;
        let mut eps_val = eps;
        let mut args: [*mut c_void; 4] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut eps_val as *mut _ as *mut c_void,
        ];

        let cfg = LaunchConfig {
            grid: (n_rows, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }

    /// M50 Phase 2: batched rms_norm_no_weight. `x[B, n_rows, n]`,
    /// `out[B, n_rows, n]`. Grid (B, n_rows, 1), block (256). Each WG
    /// processes one (batch, row) pair.
    pub fn launch_batched(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        n_rows: u32,
        n: u32,
        eps: f32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let needed = (batch as usize) * (n_rows as usize) * (n as usize);
        if out.len() < needed || x.len() < needed {
            return Err(eyre!(
                "rms_norm_no_weight_batched: buffer too small (need {needed})"
            ));
        }
        let function = self.module.get_function("rms_norm_no_weight_batched")?;
        let mut out_ptr = out.raw();
        let mut x_ptr = x.raw();
        let mut nr_val = n_rows;
        let mut n_val = n;
        let mut eps_val = eps;
        let mut args: [*mut c_void; 5] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut nr_val as *mut _ as *mut c_void,
            &mut n_val as *mut _ as *mut c_void,
            &mut eps_val as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (batch, n_rows, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}
