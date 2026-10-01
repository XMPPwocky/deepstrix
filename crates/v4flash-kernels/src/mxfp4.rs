//! MXFP4 × Q8_K matvec family — routed-MoE down projection for blk.26 +
//! blk.42 of the unsloth UD mix (the official checkpoint's native expert
//! format — plausibly lossless weights on those layers).
//!
//! Same contract as the q2_k/iq3_xxs down family (Q8_K activations,
//! slot-major midq, q2_k_reduce_partials-compatible partials). NOTE:
//! `n_blocks_in` counts Q8_K superblocks (256 elems = 136 MXFP4 bytes).
//! CPU reference: [`crate::mxfp4_tables::cpu_dot_mxfp4_q8_k`].

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const MXFP4_GFX1201: &[u8] = include_bytes!(env!("KERNEL_MXFP4_MATVEC_GFX1201"));
const MXFP4_GFX1151: &[u8] = include_bytes!(env!("KERNEL_MXFP4_MATVEC_GFX1151"));

pub const SUPER_MXFP4_BYTES: usize = 136;

/// `n_blocks_in` the small-b down twin is compiled for (`DN2_NB` = N_FF_EXP / 256).
pub const SMALLB_DOWN_N_BLOCKS_IN: u32 = 9;

pub struct Mxfp4Matvec {
    module: Module,
    /// The int8-WMMA module (gfx1151 only), see [`Self::launch_by_expert_wmma_ex`].
    wmma: Option<Module>,
}

impl Mxfp4Matvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            MXFP4_GFX1201
        } else if arch.starts_with("gfx1151") {
            MXFP4_GFX1151
        } else {
            return Err(eyre!("unsupported arch for mxfp4_matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        let wmma = if arch.starts_with("gfx1151") {
            Some(Module::load_data(crate::mxfp4_pair::MXFP4_MOE_WMMA_GFX1151)?)
        } else {
            None
        };
        Ok(Self { module, wmma })
    }

    /// Whether the int8-WMMA down kernel exists on this device (gfx1151).
    pub fn has_wmma(&self) -> bool { self.wmma.is_some() }

    /// The loaded modules (tests: explicit-symbol launches).
    pub fn module(&self) -> &Module { &self.module }
    pub fn wmma_module(&self) -> Option<&Module> { self.wmma.as_ref() }

    /// Small-b twin of [`Self::launch_by_expert_kwide2_ex`]
    /// (`mxfp4_matvec_par_by_expert_smallb`, member-outer, 2 rows per warp):
    /// the SAME arguments, grid and work-item / device-count contract, and
    /// BIT-IDENTICAL partials. Compiled for `n_blocks_in` = 9 only (it would
    /// return without writing otherwise), so any other value is refused here.
    /// Selected by `dispatch::moe_down_mxfp4` for rows <= 8 (`V41_MOE_DOWN_DN2`).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_by_expert_smallb_ex(
        &self,
        stream: &Stream,
        partials: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        n_rows: u32,
        n_blocks_in: u32,
        n_work_items_dev: Option<&DeviceBuffer<i32>>,
    ) -> eyre::Result<()> {
        if n_blocks_in != SMALLB_DOWN_N_BLOCKS_IN {
            return Err(eyre!("mxfp4 smallb down: n_blocks_in={n_blocks_in} but the kernel is compiled for {SMALLB_DOWN_N_BLOCKS_IN}"));
        }
        self.launch_by_expert_sym(
            "mxfp4_matvec_par_by_expert_smallb", stream, partials, w_base, xq_base, group_count,
            expert_members, work_items, n_work_items, dbpe, xq_slot_stride, n_used, max_per_expert,
            chunk_size, n_rows, n_blocks_in, n_work_items_dev,
        )
    }

    /// int8-WMMA twin of [`Self::launch_by_expert_kwide2_ex`]
    /// (`mxfp4_matvec_par_by_expert_wmma`, kernels/mxfp4_moe_wmma.hip): the SAME
    /// arguments and work-item / device-count contract and the same (b, slot, row)
    /// partials set, grid (ceil(n_rows/128), n_work_items) x 256. NOT bit-exact
    /// vs kwide2 (f32 re-association of identical int32 dots, rel_rmse ~1.7e-7);
    /// needs Q8_K midq with bsums. gfx1151 only. Selected by
    /// `dispatch::moe_down_mxfp4` for rows >= 128 (`V41_MOE_WMMA_DOWN`).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_by_expert_wmma_ex(
        &self,
        stream: &Stream,
        partials: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        n_rows: u32,
        n_blocks_in: u32,
        n_work_items_dev: Option<&DeviceBuffer<i32>>,
    ) -> eyre::Result<()> {
        let module = self.wmma.as_ref().ok_or_else(|| eyre!("mxfp4 wmma down: no WMMA module on this arch (gfx1151 only)"))?;
        if n_work_items_dev.is_some() && n_work_items as usize > work_items.len() {
            return Err(eyre!("mxfp4 wmma down: grid bound {n_work_items} > work_items {}", work_items.len()));
        }
        let n_wi_dev = n_work_items_dev.map_or(std::ptr::null_mut(), |c| c.raw());
        if chunk_size == 0 || chunk_size > crate::mxfp4_pair::MXFP4_WMMA_MAX_CHUNK {
            return Err(eyre!("mxfp4 wmma down: chunk_size={chunk_size} not in 1..=32 (the kernel would drop members)"));
        }
        if n_blocks_in == 0 {
            return Err(eyre!("mxfp4 wmma down: n_blocks_in must be > 0"));
        }
        if n_work_items == 0 {
            return Ok(());
        }
        let function = module.get_function("mxfp4_matvec_par_by_expert_wmma")?;
        let cfg = LaunchConfig {
            grid: (n_rows.div_ceil(crate::mxfp4_pair::MXFP4_WMMA_ROWS_PER_WG), n_work_items, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            partials.raw(), w_base.raw(), xq_base.raw(),
            group_count.raw(), expert_members.raw(), work_items.raw(),
            dbpe, xq_slot_stride, n_used, max_per_expert, chunk_size,
            n_rows, n_blocks_in, n_wi_dev
        ])
    }

    /// Decode batched over `n_used` selected experts (graph-captured core).
    /// Contract identical to `Q2KAccumulateMatvec::launch_batched`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        selected: &DeviceBuffer<i32>,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        n_rows: u32,
        n_blocks_in: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("mxfp4_matvec_par_batched: n_rows={n_rows} not %8"));
        }
        if out.len() < n_rows as usize {
            return Err(eyre!("mxfp4 batched out: len {} < n_rows {n_rows}", out.len()));
        }
        if (selected.len() as u32) < n_used {
            return Err(eyre!("selected len {} < n_used {n_used}", selected.len()));
        }
        let function = self.module.get_function("mxfp4_matvec_par_batched")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), w_base.raw(), xq_base.raw(), selected.raw(),
            dbpe, xq_slot_stride, n_used, n_rows, n_blocks_in
        ])
    }

    /// Decode het-split; contract identical to
    /// `Q2KAccumulateMatvec::launch_batched_hetsplit`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched_hetsplit(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        selected: &DeviceBuffer<i32>,
        remap: &DeviceBuffer<i32>,
        mode: u32,
        dgpu_cap: u32,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        n_rows: u32,
        n_blocks_in: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("mxfp4 hetsplit: n_rows={n_rows} not %8"));
        }
        if remap.len() < 256 {
            return Err(eyre!("mxfp4 hetsplit: remap len {} < 256", remap.len()));
        }
        let function = self
            .module
            .get_function("mxfp4_matvec_par_batched_hetsplit")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), w_base.raw(), xq_base.raw(), selected.raw(), remap.raw(), mode, dgpu_cap,
            dbpe, xq_slot_stride, n_used, n_rows, n_blocks_in
        ])
    }

    /// Drafter block down projection GROUPED BY EXPERT (`mxfp4_matvec_par_grouped`,
    /// `V41_MTP_MOE_GROUPED`; gate+up twin
    /// [`crate::mxfp4_pair::Mxfp4PairMatvec::launch_fused_swiglu_grouped`]): all
    /// `n_tok` tokens in ONE launch, each DISTINCT resident expert's rows read
    /// once. `xq_base` is pick-major midq (pick `p = token * n_used + slot` at
    /// `p * xq_slot_stride`), `out` is `[n_tok, n_rows]`.
    ///
    /// BIT-IDENTICAL per (token, row) to `n_tok` launches of
    /// [`Self::launch_batched_hetsplit`] with mode 0 and `dgpu_cap >= n_used`
    /// (ours iff `remap[sel] < 0`): the kernel keeps each lane's per-pick dot
    /// and replays the per-row kernel's in-lane slot-order sum before the warp
    /// reduction, which a by-expert partials + reduce cannot. Graph-capturable.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_grouped(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        selected: &DeviceBuffer<i32>,
        remap: &DeviceBuffer<i32>,
        dbpe: u32,
        xq_slot_stride: u32,
        n_rows: u32,
        n_blocks_in: u32,
        n_tok: u32,
        n_used: u32,
    ) -> eyre::Result<()> {
        let n_picks = n_tok * n_used;
        let max_picks = crate::mxfp4_pair::MXFP4_GROUPED_MAX_PICKS;
        if n_tok == 0 || n_used == 0 || n_picks > max_picks {
            return Err(eyre!("mxfp4 grouped down: n_tok={n_tok} x n_used={n_used} outside 1..={max_picks} picks"));
        }
        // Two super-block iterations per lane (block_lane < 8), as compiled.
        if n_blocks_in == 0 || n_blocks_in > 16 {
            return Err(eyre!("mxfp4 grouped down: n_blocks_in={n_blocks_in} outside 1..=16"));
        }
        if n_rows % 8 != 0 {
            return Err(eyre!("mxfp4 grouped down: n_rows={n_rows} not %8"));
        }
        if out.len() < (n_tok * n_rows) as usize {
            return Err(eyre!("mxfp4 grouped down out: len {} < n_tok * n_rows", out.len()));
        }
        if (xq_slot_stride as usize) < n_blocks_in as usize * crate::q8_k::BLOCK_Q8_K_BYTES
            || xq_base.len() < n_picks as usize * xq_slot_stride as usize
        {
            return Err(eyre!(
                "mxfp4 grouped down xq: {} bytes, stride {xq_slot_stride}, for {n_picks} picks of {n_blocks_in} blocks",
                xq_base.len()
            ));
        }
        if (selected.len() as u32) < n_picks {
            return Err(eyre!("mxfp4 grouped down: selected len {} < n_picks {n_picks}", selected.len()));
        }
        if remap.len() < 256 {
            return Err(eyre!("mxfp4 grouped down: remap len {} < 256", remap.len()));
        }
        let function = self.module.get_function("mxfp4_matvec_par_grouped")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), w_base.raw(), xq_base.raw(), selected.raw(), remap.raw(),
            dbpe, xq_slot_stride, n_used, n_picks, n_rows, n_blocks_in
        ])
    }

    /// Prefill by-expert kwide2 (production analog of
    /// `Q2KAccumulateMatvec::launch_by_expert_kwide2`): grid
    /// `(n_rows/16, n_work_items)`, 2 rows per warp, members in halves of
    /// 16, weights unpacked once per (row, block). Caller zeroes
    /// `partials` and pairs with `q2_k_reduce_partials`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_by_expert_kwide2(
        &self,
        stream: &Stream,
        partials: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        n_rows: u32,
        n_blocks_in: u32,
    ) -> eyre::Result<()> {
        self.launch_by_expert_kwide2_ex(
            stream, partials, w_base, xq_base, group_count, expert_members, work_items,
            n_work_items, dbpe, xq_slot_stride, n_used, max_per_expert, chunk_size, n_rows,
            n_blocks_in, None,
        )
    }

    /// As `launch_by_expert_kwide2`. With `n_work_items_dev` (the builder's
    /// device-side count) `n_work_items` is only an UPPER BOUND for grid.y
    /// (<= `work_items.len()`): work-groups past the device count exit at once
    /// (`V41_MOE_WI_DEVCOUNT`).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_by_expert_kwide2_ex(
        &self,
        stream: &Stream,
        partials: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        n_rows: u32,
        n_blocks_in: u32,
        n_work_items_dev: Option<&DeviceBuffer<i32>>,
    ) -> eyre::Result<()> {
        self.launch_by_expert_sym(
            "mxfp4_matvec_par_by_expert_kwide2", stream, partials, w_base, xq_base, group_count,
            expert_members, work_items, n_work_items, dbpe, xq_slot_stride, n_used, max_per_expert,
            chunk_size, n_rows, n_blocks_in, n_work_items_dev,
        )
    }

    /// The kwide2-geometry launch (grid (n_rows/16, n_work_items) x 256) of
    /// `sym`: `mxfp4_matvec_par_by_expert_kwide2` or its small-b twin.
    #[allow(clippy::too_many_arguments)]
    fn launch_by_expert_sym(
        &self,
        sym: &str,
        stream: &Stream,
        partials: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        n_rows: u32,
        n_blocks_in: u32,
        n_work_items_dev: Option<&DeviceBuffer<i32>>,
    ) -> eyre::Result<()> {
        if n_work_items_dev.is_some() && n_work_items as usize > work_items.len() {
            return Err(eyre!("mxfp4 kwide2: grid bound {n_work_items} > work_items {}", work_items.len()));
        }
        let n_wi_dev = n_work_items_dev.map_or(std::ptr::null_mut(), |c| c.raw());
        if n_rows % 16 != 0 {
            return Err(eyre!("mxfp4 kwide2: n_rows={n_rows} not %16"));
        }
        if chunk_size == 0 || chunk_size > 32 {
            return Err(eyre!("mxfp4 kwide2: chunk_size={chunk_size} not in 1..=32"));
        }
        if n_work_items == 0 {
            return Ok(());
        }
        let function = self.module.get_function(sym)?;
        let cfg = LaunchConfig {
            grid: (n_rows / 16, n_work_items, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            partials.raw(), w_base.raw(), xq_base.raw(),
            group_count.raw(), expert_members.raw(), work_items.raw(),
            dbpe, xq_slot_stride, n_used, max_per_expert, chunk_size,
            n_rows, n_blocks_in, n_wi_dev
        ])
    }
}
