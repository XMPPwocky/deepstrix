//! Launchers for `kernels/mxfp4_pair_matvec.hip` — the fused gate+up SwiGLU
//! pair kernels for MXFP4 routed experts (DeepSeek-V4.1-Flash's native
//! expert format). Same contracts as [`crate::iq3_s::Iq3SPairMatvec`], so
//! `het::dispatch` routes `GgufType::MXFP4` gate/up here.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const MXFP4_PAIR_GFX1201: &[u8] = include_bytes!(env!("KERNEL_MXFP4_PAIR_MATVEC_GFX1201"));
const MXFP4_PAIR_GFX1151: &[u8] = include_bytes!(env!("KERNEL_MXFP4_PAIR_MATVEC_GFX1151"));

/// Q8_K superblocks the kernels stage in LDS (`MXFP4_PAIR_MAX_BLOCKS`).
pub const MXFP4_PAIR_MAX_BLOCKS: u32 = 32;
/// Upper bound on the prefill kwide kernel's `chunk_size` (`MXFP4_KW_MAX_CHUNK`
/// in the kernel: sizes its LDS staging and per-lane accumulators).
pub const MXFP4_KW_MAX_CHUNK: u32 = 32;

pub struct Mxfp4PairMatvec {
    module: Module,
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
        Ok(Self { module })
    }

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
                gate_bpe, up_bpe, n_used, max_per_expert, chunk_size, clamp, n_rows, n_blocks
            ]
        )
    }
}
