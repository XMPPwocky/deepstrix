//! V4 Flash FFN building blocks: SwiGLU (and future expert MLP kernels).

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, LaunchConfig, Module, Stream};

const SWIGLU_GFX1201: &[u8] = include_bytes!(env!("KERNEL_SWIGLU_GFX1201"));
const SWIGLU_GFX1151: &[u8] = include_bytes!(env!("KERNEL_SWIGLU_GFX1151"));
const SWIGLU_CW_GFX1201: &[u8] = include_bytes!(env!("KERNEL_SWIGLU_CLAMP_WEIGHTED_GFX1201"));
const SWIGLU_CW_GFX1151: &[u8] = include_bytes!(env!("KERNEL_SWIGLU_CLAMP_WEIGHTED_GFX1151"));

/// SwiGLU activation: `out[i] = silu(gate[i]) * up[i]`. Mirrors ds4
/// `swiglu()` (ds4.c:5083). Used by the shared expert.
pub struct Swiglu {
    module: Module,
}

impl Swiglu {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            SWIGLU_GFX1201
        } else if arch.starts_with("gfx1151") {
            SWIGLU_GFX1151
        } else {
            return Err(eyre!("unsupported arch for swiglu: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        n: u32,
    ) -> eyre::Result<()> {
        let function = self.module.get_function("swiglu")?;
        let mut out_ptr = out.raw();
        let mut g_ptr = gate.raw();
        let mut u_ptr = up.raw();
        let mut n_v = n;
        let mut args: [*mut c_void; 4] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut g_ptr as *mut _ as *mut c_void,
            &mut u_ptr as *mut _ as *mut c_void,
            &mut n_v as *mut _ as *mut c_void,
        ];
        let block_x = 256u32;
        let grid_x = n.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}

/// Routed-MoE SwiGLU: clamp gate top + up (two-sided), multiply by per-expert
/// weight. Mirrors `matvec_iq2_xxs_mid_worker` body (ds4.c:3868-3873). Output
/// layout is `mid[expert * ff_exp + r]` for r in [0, ff_exp), e in [0, n_experts).
pub struct SwigluClampWeighted {
    module: Module,
}

impl SwigluClampWeighted {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            SWIGLU_CW_GFX1201
        } else if arch.starts_with("gfx1151") {
            SWIGLU_CW_GFX1151
        } else {
            return Err(eyre!("unsupported arch for swiglu_clamp_weighted: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    pub fn launch(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        expert_w: &DeviceBuffer<f32>,
        clamp: f32,
        ff_exp: u32,
        n_experts: u32,
    ) -> eyre::Result<()> {
        let function = self.module.get_function("swiglu_clamp_weighted")?;
        let mut mid_ptr = mid.raw();
        let mut g_ptr = gate.raw();
        let mut u_ptr = up.raw();
        let mut ew_ptr = expert_w.raw();
        let mut clamp_v = clamp;
        let mut ff = ff_exp;
        let mut args: [*mut c_void; 6] = [
            &mut mid_ptr as *mut _ as *mut c_void,
            &mut g_ptr as *mut _ as *mut c_void,
            &mut u_ptr as *mut _ as *mut c_void,
            &mut ew_ptr as *mut _ as *mut c_void,
            &mut clamp_v as *mut _ as *mut c_void,
            &mut ff as *mut _ as *mut c_void,
        ];
        let block_x = 256u32;
        let grid_x = ff_exp.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, n_experts, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}
