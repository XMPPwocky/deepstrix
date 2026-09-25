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
/// Padded expert count = kernel block size (power of two; ids in
/// [n_expert, MAX) are -INF-padded). Must match `-DROUTER_MAX_EXPERTS` in build.rs.
#[cfg(not(feature = "v41"))]
pub const ROUTER_MAX_EXPERTS: u32 = 256;
#[cfg(feature = "v41")]
pub const ROUTER_MAX_EXPERTS: u32 = 512;
pub const ROUTER_MAX_USED: u32 = 8;
/// Max alternatives (ranks `n_used+1..=n_used+n_alt`) per token. Matches
/// `ROUTER_MAX_ALT` in both kernels.
pub const ROUTER_MAX_ALT: u32 = 4;

/// Optional extras for [`RouterTopk::launch_batched_ex`]. `Default` = the
/// plain router.
#[derive(Default)]
pub struct RouterEx<'a> {
    /// Ranks n_used+1..=n_used+n_alt per token (`[B, n_alt]`).
    pub alts: Option<&'a mut DeviceBuffer<i32>>,
    pub n_alt: u32,
    /// The alternatives' weights on `weights`' scale (`[B, n_alt]`, needs `alts`).
    pub alt_w: Option<&'a mut DeviceBuffer<f32>>,
    /// CACHE-PRIOR (`[n_expert]`, shared by all tokens): added to each expert's
    /// selection score; the original top-`n_protect` are always kept; weights
    /// from the unbiased probs of the final set. `None` = off (bit-identical).
    pub prior: Option<&'a DeviceBuffer<f32>>,
    pub n_protect: u32,
    /// Dry run: emit the PLAIN picks/weights and put the prior's picks in
    /// `orig_sel` instead (roles swapped). Numerics unchanged.
    pub prior_dry: bool,
    /// The ORIGINAL top-n_used per token (`[B, n_used]`), without the prior.
    pub orig_sel: Option<&'a mut DeviceBuffer<i32>>,
    /// max - min of each token's selection scores (`[B]`).
    pub range_out: Option<&'a mut DeviceBuffer<f32>>,
}

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
            // The padded lanes [n_expert, MAX) must exist to write their -INF
            // sentinels for the tree reduce (== n_expert when the model fills MAX).
            block: (ROUTER_MAX_EXPERTS, 1, 1),
            shared_mem_bytes: 0,
        };
        let null: sys::hipDeviceptr_t = std::ptr::null_mut();
        launch_kernel!(function, cfg, stream, [
            selected.raw(), weights.raw(), logits.raw(), b_ptr,
            n_expert, n_used, expert_weight_scale, weight_eps, null, 0u32, null,
            null, 0u32, 0u32, null, null
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
        self.launch_batched_ex(
            stream, selected, weights, logits, bias, n_expert, n_used, expert_weight_scale, weight_eps, b, RouterEx::default(),
        )
    }

    /// [`Self::launch_batched`] that also writes each token's next `n_alt`
    /// ranks to `alts` (`[B, n_alt]`, rank order). `selected` / `weights` are
    /// bit-identical to `n_alt = 0`: the alternatives are extra argmax passes
    /// after the first `n_used`. `alts = None` or `n_alt = 0` is exactly
    /// `launch_batched`.
    ///
    /// `alt_w` (optional, needs `alts`): each alternative's prob divided by the
    /// top-`n_used` prob sum, times the scale. That's the same scale as
    /// `weights`, so swapping pick j for alternative k renormalizes exactly
    /// (ref.Gate) by dividing every weight in the row, and `alt_w[k]`, by
    /// `1 - weights[j]/scale + alt_w[k]/scale`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched_alts(
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
        alts: Option<&mut DeviceBuffer<i32>>,
        n_alt: u32,
        alt_w: Option<&mut DeviceBuffer<f32>>,
    ) -> eyre::Result<()> {
        self.launch_batched_ex(
            stream, selected, weights, logits, bias, n_expert, n_used, expert_weight_scale, weight_eps, b,
            RouterEx { alts, n_alt, alt_w, ..Default::default() },
        )
    }

    /// The batched router with every option (`RouterEx`): alternatives, their
    /// weights, the cache-prior, the original picks and the score range.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched_ex(
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
        ex: RouterEx<'_>,
    ) -> eyre::Result<()> {
        let RouterEx { alts, n_alt, alt_w, prior, n_protect, prior_dry, orig_sel, range_out } = ex;
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
        let n_alt = if alts.is_some() { n_alt } else { 0 };
        if n_alt > ROUTER_MAX_ALT || n_used + n_alt > n_expert {
            return Err(eyre!("router_topk batched: n_alt {n_alt} must be <= {ROUTER_MAX_ALT} and n_used+n_alt <= n_expert"));
        }
        if let Some(a) = alts.as_ref() {
            if a.len() < bn * n_alt as usize {
                return Err(eyre!("router_topk batched: alts len {} < b*n_alt {}", a.len(), bn * n_alt as usize));
            }
        }
        if let Some(w) = alt_w.as_ref() {
            if w.len() < bn * n_alt as usize {
                return Err(eyre!("router_topk batched: alt_w len {} < b*n_alt {}", w.len(), bn * n_alt as usize));
            }
        }
        let a_ptr: sys::hipDeviceptr_t = match alts {
            Some(a) if n_alt > 0 => a.raw(),
            _ => std::ptr::null_mut(),
        };
        let aw_ptr: sys::hipDeviceptr_t = match alt_w {
            Some(w) if n_alt > 0 => w.raw(),
            _ => std::ptr::null_mut(),
        };
        if let Some(p) = prior {
            if p.len() < n_expert as usize {
                return Err(eyre!("router_topk batched: prior len {} < n_expert {n_expert}", p.len()));
            }
        }
        if let Some(o) = orig_sel.as_ref() {
            if o.len() < bn * n_used as usize {
                return Err(eyre!("router_topk batched: orig_sel len {} < b*n_used {}", o.len(), bn * n_used as usize));
            }
        }
        if let Some(r) = range_out.as_ref() {
            if r.len() < bn {
                return Err(eyre!("router_topk batched: range_out len {} < b {bn}", r.len()));
            }
        }
        let pr_ptr: sys::hipDeviceptr_t = prior.map_or(std::ptr::null_mut(), |p| p.raw());
        let os_ptr: sys::hipDeviceptr_t = orig_sel.map_or(std::ptr::null_mut(), |o| o.raw());
        let ro_ptr: sys::hipDeviceptr_t = range_out.map_or(std::ptr::null_mut(), |r| r.raw());
        let n_protect = n_protect.min(n_used);
        // Batched path needs the par kernel's blockIdx.x offsetting.
        let function = self.module.get_function("router_topk_par")?;
        let b_ptr: sys::hipDeviceptr_t = match bias {
            Some(bs) => bs.raw(),
            None => std::ptr::null_mut(),
        };
        let cfg = LaunchConfig {
            grid: (b, 1, 1),
            // The padded lanes [n_expert, MAX) must exist to write their -INF
            // sentinels for the tree reduce (== n_expert when the model fills MAX).
            block: (ROUTER_MAX_EXPERTS, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            selected.raw(), weights.raw(), logits.raw(), b_ptr,
            n_expert, n_used, expert_weight_scale, weight_eps, a_ptr, n_alt, aw_ptr,
            pr_ptr, n_protect, prior_dry as u32, os_ptr, ro_ptr
        ])
    }
}
