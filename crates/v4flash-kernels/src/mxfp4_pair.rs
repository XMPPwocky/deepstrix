//! Launchers for `kernels/mxfp4_pair_matvec.hip` — the fused gate+up SwiGLU
//! pair kernels for MXFP4 routed experts (DeepSeek-V4.1-Flash's native
//! expert format). Same contracts as [`crate::iq3_s::Iq3SPairMatvec`], so
//! `het::dispatch` routes `GgufType::MXFP4` gate/up here.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const MXFP4_PAIR_GFX1201: &[u8] = include_bytes!(env!("KERNEL_MXFP4_PAIR_MATVEC_GFX1201"));
const MXFP4_PAIR_GFX1151: &[u8] = include_bytes!(env!("KERNEL_MXFP4_PAIR_MATVEC_GFX1151"));
/// `kernels/mxfp4_moe_wmma.hip` (the int8-WMMA gate+up / down arm, 2026-09-26
/// sweep B_moe_prefill). gfx1151 only: the RDNA3 WMMA intrinsic; the gfx12 build
/// of that file traps, so it is never loaded there.
pub(crate) const MXFP4_MOE_WMMA_GFX1151: &[u8] = include_bytes!(env!("KERNEL_MXFP4_MOE_WMMA_GFX1151"));

/// Members per work item the WMMA kernels stage (`WM_MAX_CHUNK`; the kernel
/// silently clamps a larger chunk, which would DROP members, so the launchers
/// refuse it).
pub const MXFP4_WMMA_MAX_CHUNK: u32 = 32;
/// Weight rows per work-group of the WMMA kernels (`WM_ROWS_PER_WG`).
pub const MXFP4_WMMA_ROWS_PER_WG: u32 = 128;

/// Q8_K superblocks the kernels stage in LDS (`MXFP4_PAIR_MAX_BLOCKS`).
pub const MXFP4_PAIR_MAX_BLOCKS: u32 = 32;
/// Upper bound on the prefill kwide kernel's `chunk_size` (`MXFP4_KW_MAX_CHUNK`
/// in the kernel: sizes its LDS staging and per-lane accumulators).
pub const MXFP4_KW_MAX_CHUNK: u32 = 32;

pub struct Mxfp4PairMatvec {
    module: Module,
    /// The int8-WMMA module (gfx1151 only), see [`Self::launch_fused_swiglu_wmma_ex`].
    wmma: Option<Module>,
}

/// Warps per workgroup (= rows per workgroup, one row each) for the decode
/// pair kernels. `V41_MXFP4_PAIR_WARPS` sweeps it; default 8 = the historical
/// hardcoded geometry.
fn pair_warps() -> u32 {
    static W: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        std::env::var("V41_MXFP4_PAIR_WARPS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|w| (1..=32).contains(w))
            .unwrap_or(8)
    });
    *W
}

impl Mxfp4PairMatvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            MXFP4_PAIR_GFX1201
        } else if arch.starts_with("gfx1151") {
            MXFP4_PAIR_GFX1151
        } else {
            return Err(eyre!("unsupported arch for mxfp4_pair_matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        let wmma = if arch.starts_with("gfx1151") {
            Some(Module::load_data(MXFP4_MOE_WMMA_GFX1151)?)
        } else {
            None
        };
        Ok(Self { module, wmma })
    }

    /// Whether the int8-WMMA gate+up kernel exists on this device (gfx1151).
    pub fn has_wmma(&self) -> bool { self.wmma.is_some() }

    /// The loaded modules (tests: explicit-symbol launches).
    pub fn module(&self) -> &Module { &self.module }
    pub fn wmma_module(&self) -> Option<&Module> { self.wmma.as_ref() }

    fn check(mid: &DeviceBuffer<f32>, n_used: u32, n_rows: u32, n_blocks: u32) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("mxfp4 pair: n_rows={n_rows} not %8"));
        }
        if n_blocks == 0 || n_blocks > MXFP4_PAIR_MAX_BLOCKS {
            return Err(eyre!(
                "mxfp4 pair: n_blocks={n_blocks} outside [1, {MXFP4_PAIR_MAX_BLOCKS}] (LDS staging)"
            ));
        }
        if mid.len() < (n_used as usize) * (n_rows as usize) {
            return Err(eyre!("mxfp4 pair mid: len {} < n_used*n_rows", mid.len()));
        }
        Ok(())
    }

    /// Decode: `mid[slot, row] = swiglu(gate·x, up·x) * expert_w[slot]` for
    /// the `n_used` selected experts. `n_blocks` counts Q8_K superblocks.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_batch(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        selected: &DeviceBuffer<i32>,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        Self::check(mid, n_used, n_rows, n_blocks)?;
        let function = self.module.get_function("mxfp4_pair_matvec_fused_swiglu_batch")?;
        let cfg = LaunchConfig {
            grid: (n_rows.div_ceil(pair_warps()), n_used, 1),
            block: (pair_warps() * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(
            function,
            cfg,
            stream,
            [
                mid.raw(),
                gate_w_base.raw(),
                up_w_base.raw(),
                xq.raw(),
                expert_w.raw(),
                selected.raw(),
                gate_bpe,
                up_bpe,
                clamp,
                n_rows,
                n_blocks
            ]
        )
    }

    /// Decode het-split (see the kernel comment for the `remap` / `mode` /
    /// `dgpu_cap` contract).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_batch_hetsplit(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        selected: &DeviceBuffer<i32>,
        remap: &DeviceBuffer<i32>,
        mode: u32,
        dgpu_cap: u32,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        Self::check(mid, n_used, n_rows, n_blocks)?;
        let function = self
            .module
            .get_function("mxfp4_pair_matvec_fused_swiglu_batch_hetsplit")?;
        let cfg = LaunchConfig {
            grid: (n_rows.div_ceil(pair_warps()), n_used, 1),
            block: (pair_warps() * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(
            function,
            cfg,
            stream,
            [
                mid.raw(),
                gate_w_base.raw(),
                up_w_base.raw(),
                xq.raw(),
                expert_w.raw(),
                selected.raw(),
                remap.raw(),
                mode,
                dgpu_cap,
                gate_bpe,
                up_bpe,
                clamp,
                n_rows,
                n_blocks
            ]
        )
    }

    /// Prefill chunked by-expert (work-items contract of
    /// [`crate::iq3_s::Iq3SPairMatvec::launch_fused_swiglu_chunked`]): the
    /// per-member re-dequant fallback (`PAIR_VARIANT=chunked`).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_chunked(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("mxfp4 pair chunked: n_rows={n_rows} not %8"));
        }
        if n_blocks == 0 || n_blocks > MXFP4_PAIR_MAX_BLOCKS {
            return Err(eyre!("mxfp4 pair chunked: n_blocks={n_blocks} outside [1, {MXFP4_PAIR_MAX_BLOCKS}]"));
        }
        if n_work_items == 0 {
            return Ok(());
        }
        let function = self.module.get_function("mxfp4_pair_matvec_fused_swiglu_chunked")?;
        let cfg = LaunchConfig { grid: (n_rows / 8, n_work_items, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(
            function,
            cfg,
            stream,
            [
                mid.raw(), gate_w_base.raw(), up_w_base.raw(), xq.raw(), expert_w.raw(),
                group_count.raw(), expert_members.raw(), work_items.raw(),
                gate_bpe, up_bpe, n_used, max_per_expert, chunk_size, clamp, n_rows, n_blocks
            ]
        )
    }

    /// Prefill kwide (M51 structure: each lane's 16 gate+up weights dequantised
    /// ONCE per super-block pair and amortised across the chunk's members).
    /// Same work-items contract as `launch_fused_swiglu_chunked`; additionally
    /// requires `n_blocks % 2 == 0` and `chunk_size <= MXFP4_KW_MAX_CHUNK`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_kwide(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        self.launch_fused_swiglu_kwide_ex(
            stream, mid, gate_w_base, up_w_base, xq, expert_w, group_count, expert_members,
            work_items, n_work_items, gate_bpe, up_bpe, n_used, max_per_expert, chunk_size,
            clamp, n_rows, n_blocks, None,
        )
    }

    /// As `launch_fused_swiglu_kwide`. With `n_work_items_dev` (the builder's
    /// device-side count) `n_work_items` is only an UPPER BOUND for grid.y
    /// (<= `work_items.len()`): work-groups past the device count exit at once,
    /// so the host never reads the count back (`V41_MOE_WI_DEVCOUNT`).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_kwide_ex(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
        n_work_items_dev: Option<&DeviceBuffer<i32>>,
    ) -> eyre::Result<()> {
        if n_work_items_dev.is_some() && n_work_items as usize > work_items.len() {
            return Err(eyre!("mxfp4 pair kwide: grid bound {n_work_items} > work_items {}", work_items.len()));
        }
        let n_wi_dev = n_work_items_dev.map_or(std::ptr::null_mut(), |c| c.raw());
        if n_rows % 8 != 0 {
            return Err(eyre!("mxfp4 pair kwide: n_rows={n_rows} not %8"));
        }
        if n_blocks == 0 || n_blocks % 2 != 0 {
            return Err(eyre!("mxfp4 pair kwide: n_blocks={n_blocks} must be even and > 0"));
        }
        if chunk_size > MXFP4_KW_MAX_CHUNK {
            return Err(eyre!("mxfp4 pair kwide: chunk_size={chunk_size} exceeds MXFP4_KW_MAX_CHUNK={MXFP4_KW_MAX_CHUNK}"));
        }
        if n_work_items == 0 {
            return Ok(());
        }
        let function = self.module.get_function("mxfp4_pair_matvec_fused_swiglu_kwide")?;
        let cfg = LaunchConfig { grid: (n_rows / 8, n_work_items, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(
            function,
            cfg,
            stream,
            [
                mid.raw(), gate_w_base.raw(), up_w_base.raw(), xq.raw(), expert_w.raw(),
                group_count.raw(), expert_members.raw(), work_items.raw(),
                gate_bpe, up_bpe, n_used, max_per_expert, chunk_size, clamp, n_rows, n_blocks,
                n_wi_dev
            ]
        )
    }

    /// int8-WMMA twin of [`Self::launch_fused_swiglu_kwide_ex`]
    /// (`mxfp4_pair_matvec_fused_swiglu_wmma`, kernels/mxfp4_moe_wmma.hip): the
    /// SAME argument list and work-item / device-count contract, grid
    /// (ceil(n_rows/128), n_work_items) x 256. NOT bit-exact vs kwide (f32
    /// re-association of identical int32 dots, rel_rmse ~5e-7); needs Q8_K `xq`
    /// with bsums (production q8_k_quantize writes them). gfx1151 only.
    /// Selected by `dispatch::moe_gate_up_chunked_rows` (`V41_MOE_WMMA_GATEUP`).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_wmma_ex(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
        n_work_items_dev: Option<&DeviceBuffer<i32>>,
    ) -> eyre::Result<()> {
        let module = self.wmma.as_ref().ok_or_else(|| eyre!("mxfp4 pair wmma: no WMMA module on this arch (gfx1151 only)"))?;
        if n_work_items_dev.is_some() && n_work_items as usize > work_items.len() {
            return Err(eyre!("mxfp4 pair wmma: grid bound {n_work_items} > work_items {}", work_items.len()));
        }
        let n_wi_dev = n_work_items_dev.map_or(std::ptr::null_mut(), |c| c.raw());
        if n_blocks == 0 || n_blocks > MXFP4_PAIR_MAX_BLOCKS {
            return Err(eyre!("mxfp4 pair wmma: n_blocks={n_blocks} outside [1, {MXFP4_PAIR_MAX_BLOCKS}]"));
        }
        if chunk_size == 0 || chunk_size > MXFP4_WMMA_MAX_CHUNK {
            return Err(eyre!("mxfp4 pair wmma: chunk_size={chunk_size} not in 1..={MXFP4_WMMA_MAX_CHUNK} (the kernel would drop members)"));
        }
        if n_work_items == 0 {
            return Ok(());
        }
        let function = module.get_function("mxfp4_pair_matvec_fused_swiglu_wmma")?;
        let cfg = LaunchConfig {
            grid: (n_rows.div_ceil(MXFP4_WMMA_ROWS_PER_WG), n_work_items, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(
            function,
            cfg,
            stream,
            [
                mid.raw(), gate_w_base.raw(), up_w_base.raw(), xq.raw(), expert_w.raw(),
                group_count.raw(), expert_members.raw(), work_items.raw(),
                gate_bpe, up_bpe, n_used, max_per_expert, chunk_size, clamp, n_rows, n_blocks,
                n_wi_dev
            ]
        )
    }
}
