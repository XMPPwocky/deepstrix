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

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, LaunchConfig, Module, Stream};

const HC_SIGMOID_BIAS_GFX1201: &[u8] = include_bytes!(env!("KERNEL_HC_SIGMOID_BIAS_GFX1201"));
const HC_SIGMOID_BIAS_GFX1151: &[u8] = include_bytes!(env!("KERNEL_HC_SIGMOID_BIAS_GFX1151"));
const HC_WEIGHTED_SUM_GFX1201: &[u8] = include_bytes!(env!("KERNEL_HC_WEIGHTED_SUM_GFX1201"));
const HC_WEIGHTED_SUM_GFX1151: &[u8] = include_bytes!(env!("KERNEL_HC_WEIGHTED_SUM_GFX1151"));

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
        let mut out_ptr = out.raw();
        let mut pre_ptr = pre.raw();
        let mut scale_ptr = scale.raw();
        let mut base_ptr = base.raw();
        let mut n_v = n;
        let mut args: [*mut c_void; 5] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut pre_ptr as *mut _ as *mut c_void,
            &mut scale_ptr as *mut _ as *mut c_void,
            &mut base_ptr as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];
        let block_x = 32u32;
        let grid_x = n.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
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
        let mut out_ptr = out.raw();
        let mut x_ptr = x.raw();
        let mut w_ptr = weights.raw();
        let mut ne = n_embd;
        let mut nh = n_hc;
        let mut args: [*mut c_void; 5] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
        ];
        let block_x = 256u32;
        let grid_x = n_embd.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}
