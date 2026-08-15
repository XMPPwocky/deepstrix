//! V4 Flash FFN building blocks: SwiGLU (and future expert MLP kernels).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const SWIGLU_GFX1201: &[u8] = include_bytes!(env!("KERNEL_SWIGLU_GFX1201"));
const SWIGLU_GFX1151: &[u8] = include_bytes!(env!("KERNEL_SWIGLU_GFX1151"));
const SWIGLU_CW_GFX1201: &[u8] = include_bytes!(env!("KERNEL_SWIGLU_CLAMP_WEIGHTED_GFX1201"));
const SWIGLU_CW_GFX1151: &[u8] = include_bytes!(env!("KERNEL_SWIGLU_CLAMP_WEIGHTED_GFX1151"));
const VEC_ADD_GFX1201: &[u8] = include_bytes!(env!("KERNEL_VEC_ADD_GFX1201"));
const VEC_ADD_GFX1151: &[u8] = include_bytes!(env!("KERNEL_VEC_ADD_GFX1151"));

/// SwiGLU activation: `out[i] = silu(gate[i]) * up[i]`, optionally with
/// ds4's swiglu_limit clamp (one-sided on gate, two-sided on up). Mirrors
/// ds4 `swiglu()` post-5bc1e6d. The V4-Flash shared expert uses
/// `launch_clamped` with `SWIGLU_CLAMP_EXP` (same limit as routed
/// experts); Laguna uses the unclamped `launch`.
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

    /// Unclamped swiglu (`clamp = 0` disables the limit in-kernel).
    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        n: u32,
    ) -> eyre::Result<()> {
        self.launch_clamped(stream, out, gate, up, n, 0.0)
    }

    /// swiglu with ds4's swiglu_limit clamp: `gate = min(gate, clamp)`,
    /// `up = min(max(up, -clamp), clamp)` before `silu(gate) * up`.
    /// `clamp <= 1e-6` disables clamping.
    pub fn launch_clamped(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        n: u32,
        clamp: f32,
    ) -> eyre::Result<()> {
        let function = self.module.get_function("swiglu")?;
        let block_x = 256u32;
        let grid_x = n.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), gate.raw(), up.raw(), n, clamp])
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
        let block_x = 256u32;
        let grid_x = ff_exp.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, n_experts, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            mid.raw(), gate.raw(), up.raw(), expert_w.raw(), clamp, ff_exp
        ])
    }
}

/// In-place vector add: `out[i] += rhs[i]`. Used by the forward orchestrator
/// to combine `ffn_moe + ffn_shared`.
pub struct VecAddInplace {
    module: Module,
}

impl VecAddInplace {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            VEC_ADD_GFX1201
        } else if arch.starts_with("gfx1151") {
            VEC_ADD_GFX1151
        } else {
            return Err(eyre!("unsupported arch for vec_add: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        rhs: &DeviceBuffer<f32>,
        n: u32,
    ) -> eyre::Result<()> {
        let function = self.module.get_function("vec_add_inplace")?;
        let block_x = 256u32;
        let grid_x = n.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), rhs.raw(), n])
    }
}
