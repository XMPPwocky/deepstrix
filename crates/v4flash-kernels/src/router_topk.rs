//! Device-side router top-K (M13.3).
//!
//! Replaces the `synchronize + copy_to_host + topk_desc + ...` host
//! roundtrip with a single launch. Used by the het orchestrator's iGPU
//! router stage for learned routers (L≥3); the hash-router L0-L2 path
//! still uses host code because it indexes into `tid2eid` with the
//! per-token id, which is host-side anyway.

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{sys, DeviceBuffer, LaunchConfig, Module, Stream};

const ROUTER_TOPK_GFX1201: &[u8] = include_bytes!(env!("KERNEL_ROUTER_TOPK_GFX1201"));
const ROUTER_TOPK_GFX1151: &[u8] = include_bytes!(env!("KERNEL_ROUTER_TOPK_GFX1151"));

/// Hard caps mirroring the kernel `#define`s. If the architecture ever
/// changes these, both have to move in lock-step.
pub const ROUTER_MAX_EXPERTS: u32 = 256;
pub const ROUTER_MAX_USED: u32 = 8;

pub struct RouterTopk {
    module: Module,
}

impl RouterTopk {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
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

        let function = self.module.get_function("router_topk")?;
        let mut sel_ptr = selected.raw();
        let mut w_ptr = weights.raw();
        let mut l_ptr = logits.raw();
        let mut b_ptr: sys::hipDeviceptr_t = match bias {
            Some(b) => b.raw(),
            None => std::ptr::null_mut(),
        };
        let mut ne = n_expert;
        let mut nu = n_used;
        let mut scale = expert_weight_scale;
        let mut eps = weight_eps;
        let mut args: [*mut c_void; 8] = [
            &mut sel_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut l_ptr as *mut _ as *mut c_void,
            &mut b_ptr as *mut _ as *mut c_void,
            &mut ne as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void,
            &mut scale as *mut _ as *mut c_void,
            &mut eps as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (n_expert, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}
