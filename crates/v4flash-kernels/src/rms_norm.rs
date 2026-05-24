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
}
