//! Device-side router top-K (M13.3).
//!
//! Replaces the `synchronize + copy_to_host + topk_desc + ...` host
//! roundtrip with a single launch. Used by the het orchestrator's iGPU
//! router stage for learned routers (L≥3); the hash-router L0-L2 path
//! still uses host code because it indexes into `tid2eid` with the
//! per-token id, which is host-side anyway.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, sys, DeviceBuffer, LaunchConfig, Module, Stream};

const ROUTER_TOPK_GFX1201: &[u8] = include_bytes!(env!("KERNEL_ROUTER_TOPK_GFX1201"));
const ROUTER_TOPK_GFX1151: &[u8] = include_bytes!(env!("KERNEL_ROUTER_TOPK_GFX1151"));
const ROUTER_TOPK_PAR_GFX1201: &[u8] = include_bytes!(env!("KERNEL_ROUTER_TOPK_PAR_GFX1201"));
const ROUTER_TOPK_PAR_GFX1151: &[u8] = include_bytes!(env!("KERNEL_ROUTER_TOPK_PAR_GFX1151"));

/// Hard caps mirroring the kernel `#define`s. If the architecture ever
/// changes these, both have to move in lock-step.
pub const ROUTER_MAX_EXPERTS: u32 = 256;
pub const ROUTER_MAX_USED: u32 = 8;

pub struct RouterTopk {
    module: Module,
}

impl RouterTopk {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        // Use the M14b parallel variant by default — same numerics for
        // tie-free inputs as the serial reference, ~10× faster.
        let image: &[u8] = if arch.starts_with("gfx1201") {
            ROUTER_TOPK_PAR_GFX1201
        } else if arch.starts_with("gfx1151") {
            ROUTER_TOPK_PAR_GFX1151
        } else {
            return Err(eyre!("unsupported arch for router_topk: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Construct using the original serial kernel — used by the
    /// regression test that compares serial vs parallel outputs.
    pub fn for_arch_serial(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            ROUTER_TOPK_GFX1201
        } else if arch.starts_with("gfx1151") {
            ROUTER_TOPK_GFX1151
        } else {
            return Err(eyre!("unsupported arch for router_topk: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Selected: `[n_used]` i32. Weights: `[n_used]` f32. Logits:
    /// `[n_expert]` f32. Bias: optional `[n_expert]` f32.
    pub fn launch(
        &self,
        stream: &Stream,
        selected: &mut DeviceBuffer<i32>,
        weights: &mut DeviceBuffer<f32>,
        logits: &DeviceBuffer<f32>,
        bias: Option<&DeviceBuffer<f32>>,
        n_expert: u32,
        n_used: u32,
        expert_weight_scale: f32,
        weight_eps: f32,
    ) -> eyre::Result<()> {
        if n_expert == 0 || n_expert > ROUTER_MAX_EXPERTS {
            return Err(eyre!(
                "router_topk: n_expert {n_expert} must be in [1, {ROUTER_MAX_EXPERTS}]"
            ));
        }
        if n_used == 0 || n_used > ROUTER_MAX_USED {
            return Err(eyre!(
                "router_topk: n_used {n_used} must be in [1, {ROUTER_MAX_USED}]"
            ));
        }
        if selected.len() < n_used as usize {
            return Err(eyre!(
                "router_topk: selected len {} < n_used {n_used}",
                selected.len()
            ));
        }
        if weights.len() < n_used as usize {
            return Err(eyre!(
                "router_topk: weights len {} < n_used {n_used}",
                weights.len()
            ));
        }
        if logits.len() < n_expert as usize {
            return Err(eyre!(
                "router_topk: logits len {} < n_expert {n_expert}",
                logits.len()
            ));
        }
        if let Some(b) = bias {
            if b.len() < n_expert as usize {
                return Err(eyre!(
                    "router_topk: bias len {} < n_expert {n_expert}",
                    b.len()
                ));
            }
        }

        // The parallel variant exports `router_topk_par`; the serial
        // variant exports `router_topk`. Try the par name first.
        let function = self
            .module
            .get_function("router_topk_par")
            .or_else(|_| self.module.get_function("router_topk"))?;
        let b_ptr: sys::hipDeviceptr_t = match bias {
            Some(b) => b.raw(),
            None => std::ptr::null_mut(),
        };
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (n_expert, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            selected.raw(), weights.raw(), logits.raw(), b_ptr,
            n_expert, n_used, expert_weight_scale, weight_eps
        ])
    }

    /// Batched top-k: one block per token (grid.x = B). `logits` is
    /// `[B, n_expert]`, `selected`/`weights` are `[B, n_used]`; bias is shared
    /// across tokens. Each block's result is identical to a single `launch`
    /// for that token. Requires the parallel kernel (`router_topk_par`).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched(
        &self,
        stream: &Stream,
        selected: &mut DeviceBuffer<i32>,
        weights: &mut DeviceBuffer<f32>,
        logits: &DeviceBuffer<f32>,
        bias: Option<&DeviceBuffer<f32>>,
        n_expert: u32,
        n_used: u32,
        expert_weight_scale: f32,
        weight_eps: f32,
        b: u32,
    ) -> eyre::Result<()> {
        if b == 0 {
            return Ok(());
        }
        if n_expert == 0 || n_expert > ROUTER_MAX_EXPERTS {
            return Err(eyre!(
                "router_topk: n_expert {n_expert} must be in [1, {ROUTER_MAX_EXPERTS}]"
            ));
        }
        if n_used == 0 || n_used > ROUTER_MAX_USED {
            return Err(eyre!(
                "router_topk: n_used {n_used} must be in [1, {ROUTER_MAX_USED}]"
            ));
        }
        let bn = b as usize;
        if selected.len() < bn * n_used as usize {
            return Err(eyre!(
                "router_topk batched: selected len {} < b*n_used {}",
                selected.len(),
                bn * n_used as usize
            ));
        }
        if weights.len() < bn * n_used as usize {
            return Err(eyre!(
                "router_topk batched: weights len {} < b*n_used {}",
                weights.len(),
                bn * n_used as usize
            ));
        }
        if logits.len() < bn * n_expert as usize {
            return Err(eyre!(
                "router_topk batched: logits len {} < b*n_expert {}",
                logits.len(),
                bn * n_expert as usize
            ));
        }
        if let Some(bs) = bias {
            if bs.len() < n_expert as usize {
                return Err(eyre!(
                    "router_topk batched: bias len {} < n_expert {n_expert}",
                    bs.len()
                ));
            }
        }
        // Batched path needs the par kernel's blockIdx.x offsetting.
        let function = self.module.get_function("router_topk_par")?;
        let b_ptr: sys::hipDeviceptr_t = match bias {
            Some(bs) => bs.raw(),
            None => std::ptr::null_mut(),
        };
        let cfg = LaunchConfig {
            grid: (b, 1, 1),
            block: (n_expert, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            selected.raw(), weights.raw(), logits.raw(), b_ptr,
            n_expert, n_used, expert_weight_scale, weight_eps
        ])
    }
}
