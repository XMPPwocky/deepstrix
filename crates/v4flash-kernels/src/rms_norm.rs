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

use crate::config::HC_DIM;
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const RMS_NORM_GFX1201: &[u8] = include_bytes!(env!("KERNEL_RMS_NORM_GFX1201"));
const RMS_NORM_GFX1151: &[u8] = include_bytes!(env!("KERNEL_RMS_NORM_GFX1151"));

const RMS_NORM_NO_WEIGHT_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_RMS_NORM_NO_WEIGHT_GFX1201"));
const RMS_NORM_NO_WEIGHT_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_RMS_NORM_NO_WEIGHT_GFX1151"));

/// `V41_RMS_FAST` (default ON; `0` = `rms_norm_weighted_batched` /
/// `rms_norm_weighted` at every n): for n in {512, 1280, 5120} both weighted
/// launches run `rms_norm_weighted_batched_fast` (same signature and grid; the
/// per-row `launch_weighted` is its grid-(1) case), which issues all x and
/// weight loads up front instead of one dependent load per loop iteration.
/// BIT-IDENTICAL (same double accumulation order, same LDS tree, output from
/// the same f32 values; tests/mhc_glue_bitexact.rs). 2026-09-26 sweep
/// (F_mhc_glue/rms_fast, dGPU graph): q_a n=1280 4.91 -> 3.38 us, kv/comp
/// n=512 3.82 -> 3.25, head prep n=5120 10.46 -> 4.07; prefill B=512 n=5120
/// 31.9 -> 25.5. Other n keep the old kernels.
fn rms_fast_for(n: u32) -> bool {
    static D: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_RMS_FAST").as_deref() != Ok("0"));
    *D && matches!(n, 512 | 1280 | 5120)
}

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

    /// The loaded module (tests: explicit-symbol launches).
    pub fn module(&self) -> &Module { &self.module }

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
        if n > HC_DIM as u32 {
            return Err(eyre!("rms_norm_weighted n={n} exceeds the wrapper cap HC_DIM"));
        }

        // `V41_RMS_FAST`: the loads-up-front kernel at grid (1), bit-identical.
        let function = self.module.get_function(if rms_fast_for(n) {
            "rms_norm_weighted_batched_fast"
        } else {
            "rms_norm_weighted"
        })?;

        // Kernel signature: (float *out, const float *x, const float *weight,
        //                   unsigned int n, float eps)
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), x.raw(), weight.raw(), n, eps])
    }

    /// M55: fused rms_norm_weighted + q8_0 input quantization — replaces a
    /// (rms_w, q8_0_quantize_f32) kernel pair in the decode q_chain.
    /// Writes the normed vector AND its q8 blocks in one pass; numerics
    /// identical to the separate kernels. `n` must be a multiple of 32.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_weighted_quantize_q8(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        xq: &mut DeviceBuffer<i8>,
        xscale: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        n: u32,
        eps: f32,
    ) -> eyre::Result<()> {
        if out.len() != n as usize || x.len() != n as usize || weight.len() != n as usize {
            return Err(eyre!(
                "rms_norm_weighted_quantize_q8 len mismatch: n={}, out={}, x={}, w={}",
                n,
                out.len(),
                x.len(),
                weight.len()
            ));
        }
        if n > HC_DIM as u32 || n % 32 != 0 {
            return Err(eyre!(
                "rms_norm_weighted_quantize_q8: n={n} must be ≤4096 and %32"
            ));
        }
        if xq.len() < n as usize || xscale.len() < (n / 32) as usize {
            return Err(eyre!(
                "rms_norm_weighted_quantize_q8: xq={} xscale={} too small for n={n}",
                xq.len(),
                xscale.len()
            ));
        }

        let function = self.module.get_function("rms_norm_weighted_quantize_q8")?;
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), xq.raw(), xscale.raw(), x.raw(), weight.raw(), n, eps
        ])
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
        if n > HC_DIM as u32 {
            return Err(eyre!("rms_norm_weighted_batched: n={n} > HC_DIM"));
        }
        // `V41_RMS_FAST`: loads-up-front twin for n in {512, 1280, 5120}, bit-identical.
        let function = self.module.get_function(if rms_fast_for(n) {
            "rms_norm_weighted_batched_fast"
        } else {
            "rms_norm_weighted_batched"
        })?;
        let cfg = LaunchConfig {
            grid: (batch, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), x.raw(), weight.raw(), n, eps])
    }

    /// `rms_quant_q8_1280_batched` (2026-09-27 round 2, `V41_DEC_FUSE`): the decode q_a
    /// norm `launch_weighted_batched(n = 1280)` + the Q8_0 quantize of its output
    /// (`Q8_0Matvec::quantize_input_batched`, wave kernel) in one launch; `out`, `xq`,
    /// `xscale` BIT-IDENTICAL to the two launches (tests/decode_fusion_bitexact.rs).
    /// Chain (incl. the dead f16 cast it also drops) 10.1 -> 5.6 us at b = 4 (graph,
    /// warm, 5 runs; 0.55-0.75 at b = 1..8).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_weighted_quant_q8_1280(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        xq: &mut DeviceBuffer<i8>,
        xscale: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        eps: f32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let needed = (batch as usize) * 1280;
        if out.len() < needed || x.len() < needed || xq.len() < needed || xscale.len() < needed / 32 {
            return Err(eyre!("rms_quant_q8_1280_batched: buffer too small (need {needed})"));
        }
        if weight.len() != 1280 {
            return Err(eyre!("rms_quant_q8_1280_batched: weight len != 1280"));
        }
        let function = self.module.get_function("rms_quant_q8_1280_batched")?;
        let cfg = LaunchConfig { grid: (batch, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [out.raw(), xq.raw(), xscale.raw(), x.raw(), weight.raw(), eps])
    }

    /// `launch_weighted_quant_q8_1280` through its `_ind` twin (docs/v41/GRAPH_KEYS_DESIGN.md
    /// 2.3): operands 0..4 (out, xq, xscale, x, weight) marked in `ind` come from the arena
    /// context; the buffers passed are the real ones (checked). `_canary` when `ind.canary` is set.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_weighted_quant_q8_1280_ind(
        &self,
        stream: &Stream,
        ind: crate::het::arena_ctx::Ind,
        out: &mut DeviceBuffer<f32>,
        xq: &mut DeviceBuffer<i8>,
        xscale: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        eps: f32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let needed = (batch as usize) * 1280;
        if out.len() < needed || x.len() < needed || xq.len() < needed || xscale.len() < needed / 32 || weight.len() != 1280 {
            return Err(eyre!("rms_quant_q8_1280_batched_ind: buffer sizes (need {needed})"));
        }
        let function = self.module.get_function(&ind.symbol("rms_quant_q8_1280_batched"))?;
        let cfg = LaunchConfig { grid: (batch, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        let p: [u64; 5] = [out.raw() as u64, xq.raw() as u64, xscale.raw() as u64, x.raw() as u64, weight.raw() as u64];
        let q: [u64; 5] = std::array::from_fn(|i| ind.ptr(i, p[i]));
        crate::het::arena_ctx::vet_ind(&ind, &p);
        launch_kernel!(function, cfg, stream, [ind.mask(), ind.canary, ind.tag, q[0], q[1], q[2], q[3], q[4], eps])
    }
}

/// No-weight RMSNorm — mirrors ds4.c `rms_norm_no_weight`. Operates on
/// `n_rows` independent rows of length `n` (stride n); one workgroup per
/// row. Used by V4 Flash for `head_rms_norm_inplace` (n_rows=64, n=512)
/// and by the head's `output_flat` normalisation (M10; n_rows=1, n=16384).
// Multi-WG variant of rms_norm_no_weight. Single-WG version is 150× off
// BW roofline because 63/64 CUs sit idle for a 128 KB memcpy-style op.
pub struct RmsNormNoWeightMultiWG {
    module: Module,
}

const RMS_NORM_NW_MULTIWG_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_RMS_NORM_NO_WEIGHT_MULTIWG_GFX1201"));
const RMS_NORM_NW_MULTIWG_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_RMS_NORM_NO_WEIGHT_MULTIWG_GFX1151"));

impl RmsNormNoWeightMultiWG {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            RMS_NORM_NW_MULTIWG_GFX1201
        } else if arch.starts_with("gfx1151") {
            RMS_NORM_NW_MULTIWG_GFX1151
        } else {
            return Err(eyre!("unsupported arch for rms_norm_no_weight_multiwg: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Two-kernel multi-WG RMS norm without weight. `partial_buf` is sized
    /// `n_wgs` f32 elements — each WG writes one slot, no zeroing required.
    /// Grid uses `n_wgs` WGs of 256 threads each; `n` must be a multiple
    /// of `n_wgs`.
    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        partial_buf: &mut DeviceBuffer<f32>,    // [n_wgs] — per-WG partials (no zero needed)
        n: u32,
        n_wgs: u32,
        eps: f32,
    ) -> eyre::Result<()> {
        if n % n_wgs != 0 {
            return Err(eyre!(
                "rms_norm_no_weight_multiwg: n={n} not divisible by n_wgs={n_wgs}"
            ));
        }
        if (partial_buf.len() as u32) < n_wgs {
            return Err(eyre!(
                "rms_norm_no_weight_multiwg: partial_buf len={} < n_wgs={n_wgs}",
                partial_buf.len()
            ));
        }
        let f_part = self
            .module
            .get_function("rms_norm_partial_sum_sq")?;
        let f_apply = self.module.get_function("rms_norm_apply_scale")?;
        let cfg = LaunchConfig {
            grid: (n_wgs, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(f_part, cfg, stream, [partial_buf.raw(), x.raw(), n])?;
        launch_kernel!(f_apply, cfg, stream, [out.raw(), x.raw(), partial_buf.raw(), n, n_wgs, eps])
    }

    /// Multi-WG weighted RMS: same partial-sum pass as the unweighted
    /// variant, then a multi-WG apply that does `out[i] = x[i] * inv_rms
    /// * weight[i]`. Pairs with `rms_norm_partial_sum_sq` so the partial
    /// kernel is shared between weighted and unweighted use. `n` must be
    /// divisible by `n_wgs`; `partial_buf` sized for `n_wgs` f32.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_weighted(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        x: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        partial_buf: &mut DeviceBuffer<f32>,
        n: u32,
        n_wgs: u32,
        eps: f32,
    ) -> eyre::Result<()> {
        if n % n_wgs != 0 {
            return Err(eyre!(
                "rms_norm_weighted_multiwg: n={n} not divisible by n_wgs={n_wgs}"
            ));
        }
        let f_part  = self.module.get_function("rms_norm_partial_sum_sq")?;
        let f_apply = self.module.get_function("rms_norm_weighted_apply")?;
        let cfg = LaunchConfig {
            grid: (n_wgs, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(f_part, cfg, stream, [partial_buf.raw(), x.raw(), n])?;
        launch_kernel!(f_apply, cfg, stream, [
            out.raw(), x.raw(), weight.raw(), partial_buf.raw(), n, n_wgs, eps
        ])
    }

    /// Compute inv_rms scalar only (no apply pass). Pairs with
    /// `F16Matvec::matvec_pre_scaled` to fold the per-element scale into
    /// the next kernel — saves one N-sized DRAM round-trip + one launch.
    /// Two-kernel: multi-WG partial sum, then a 1-thread finalize that
    /// reduces the partials and computes `1/sqrt(mean_sq + eps)`.
    pub fn launch_inv_only(
        &self,
        stream: &Stream,
        inv_out: &mut DeviceBuffer<f32>,        // [1]
        x: &DeviceBuffer<f32>,
        partial_buf: &mut DeviceBuffer<f32>,    // [n_wgs]
        n: u32,
        n_wgs: u32,
        eps: f32,
    ) -> eyre::Result<()> {
        if n % n_wgs != 0 {
            return Err(eyre!("inv_only: n={n} not divisible by n_wgs={n_wgs}"));
        }
        let f_part = self.module.get_function("rms_norm_partial_sum_sq")?;
        let f_fin  = self.module.get_function("rms_norm_finalize_inv")?;
        let cfg_part = LaunchConfig {
            grid: (n_wgs, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let cfg_fin = LaunchConfig {
            grid: (1, 1, 1),
            block: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(f_part, cfg_part, stream, [partial_buf.raw(), x.raw(), n])?;
        launch_kernel!(f_fin, cfg_fin, stream, [inv_out.raw(), partial_buf.raw(), n, n_wgs, eps])
    }
}

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

        let cfg = LaunchConfig {
            grid: (n_rows, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), x.raw(), n, eps])
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
        let cfg = LaunchConfig {
            grid: (batch, n_rows, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out.raw(), x.raw(), n_rows, n, eps])
    }
}
