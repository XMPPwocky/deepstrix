//! By-expert, bandwidth-amortized MoE GEMM for Laguna prefill.
//!
//! The BxN kernels ([`crate::Q4KMatvec::launch_pair_swiglu_bxn`] etc.) re-read
//! each expert's weight tile once per (token, slot) that routed to it, so
//! weight bandwidth scales with `B * top_k`. These by-expert kernels launch
//! `grid.y = n_expert`: each workgroup streams expert `e`'s weight row-tile
//! ONCE (DRAM→L2) and dots it against ALL tokens that routed to `e` (its group
//! members, produced by [`crate::MoeGroupBuilder`]). Per-chunk weight traffic
//! collapses to the active-expert bytes — the iGPU BW roofline for prefill.
//!
//! Pipeline (drop-in for the BxN path; identical layouts):
//!   1. [`Self::gate_up_swiglu`] — fused gate×up×swiglu×ew → `mid[B,n_used,n_rows]`.
//!   2. Q8_K-quantize `mid`.
//!   3. [`Self::down`] — down projection, **atomicAdd-accumulated** into
//!      `out[B, n_rows]` (each token sums top_k contributions from top_k
//!      distinct expert WGs). Caller MUST zero `out` before the launch.
//!
//! `group_count`/`expert_members` come from `MoeGroupBuilder::launch`
//! (`expert_members[e*max_per_expert + i] = (b<<16 | slot)`).

use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::GgufType;
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const LAGUNA_MOE_TILED_GFX1201: &[u8] = include_bytes!(env!("KERNEL_LAGUNA_MOE_TILED_GFX1201"));
const LAGUNA_MOE_TILED_GFX1151: &[u8] = include_bytes!(env!("KERNEL_LAGUNA_MOE_TILED_GFX1151"));

pub struct LagunaMoeTiled {
    module: Module,
}

impl LagunaMoeTiled {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            LAGUNA_MOE_TILED_GFX1201
        } else if arch.starts_with("gfx1151") {
            LAGUNA_MOE_TILED_GFX1151
        } else {
            return Err(eyre!("unsupported arch for laguna_moe_tiled: {arch}"));
        };
        Ok(Self { module: Module::load_data(image)? })
    }

    /// By-expert fused gate×up×swiglu×ew. Writes `mid[b, slot, r]` for every
    /// group member (b, slot) of every selected expert. `n_blocks_in =
    /// HIDDEN/256`. `mid` sized `[B, n_used, n_rows]`. `n_rows` (= FF_EXP) %8.
    #[allow(clippy::too_many_arguments)]
    pub fn gate_up_swiglu(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks_in: u32,
        n_expert: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("gate_up_swiglu: n_rows={n_rows} must be %8"));
        }
        let function = self.module.get_function("q4_k_gate_up_swiglu_tiled")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_expert, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            mid.raw(), gate_w_base.raw(), up_w_base.raw(), xq.raw(), expert_w.raw(),
            group_count.raw(), expert_members.raw(),
            gate_bpe, up_bpe, n_used, max_per_expert, clamp, n_rows, n_blocks_in
        ])
    }

    /// REGISTER-TILED gate×up×swiglu×ew: decodes each lane's owned weight
    /// quarter ONCE into registers, then streams all members through a bare
    /// dp4a (weight bytes touched once per chunk). Same layouts/args as
    /// [`Self::gate_up_swiglu`].
    #[allow(clippy::too_many_arguments)]
    pub fn gate_up_swiglu_reg(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks_in: u32,
        n_expert: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("gate_up_swiglu_reg: n_rows={n_rows} must be %8"));
        }
        let function = self.module.get_function("q4_k_gate_up_swiglu_tiled_reg")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_expert, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            mid.raw(), gate_w_base.raw(), up_w_base.raw(), xq.raw(), expert_w.raw(),
            group_count.raw(), expert_members.raw(),
            gate_bpe, up_bpe, n_used, max_per_expert, clamp, n_rows, n_blocks_in
        ])
    }

    /// REGISTER-TILED down projection (Q6_K only for now). Decodes each lane's
    /// weight quarter once, streams members through dp4a. `out` MUST be zeroed.
    #[allow(clippy::too_many_arguments)]
    pub fn down_reg_q6k(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        max_per_expert: u32,
        n_rows: u32,
        n_blocks_in: u32,
        n_expert: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("down_reg_q6k: n_rows={n_rows} must be %8"));
        }
        let function = self.module.get_function("q6_k_down_tiled_reg")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_expert, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), w_base.raw(), xq_base.raw(),
            group_count.raw(), expert_members.raw(),
            dbpe, xq_slot_stride, n_used, max_per_expert, n_rows, n_blocks_in
        ])
    }

    /// WIN #4 — COLUMN-TILED gate×up×swiglu×ew: register-held weight, but stages
    /// NT_COL members' activations in LDS per barrier (barrier count drops
    /// n_members → ceil(n_members/NT_COL)). Same layouts/args as
    /// [`Self::gate_up_swiglu`].
    #[allow(clippy::too_many_arguments)]
    pub fn gate_up_swiglu_reg_col(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks_in: u32,
        n_expert: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("gate_up_swiglu_reg_col: n_rows={n_rows} must be %8"));
        }
        let function = self.module.get_function("q4_k_gate_up_swiglu_tiled_reg_col")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_expert, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            mid.raw(), gate_w_base.raw(), up_w_base.raw(), xq.raw(), expert_w.raw(),
            group_count.raw(), expert_members.raw(),
            gate_bpe, up_bpe, n_used, max_per_expert, clamp, n_rows, n_blocks_in
        ])
    }

    /// WIN #2 — WIDE-ROW column-tiled gate×up×swiglu×ew: 32 rows/WG (block=1024)
    /// so the LDS-staged member activation amortizes over 4× more weight rows
    /// (activation DRAM traffic /4). Math identical to [`Self::gate_up_swiglu_reg_col`];
    /// `n_rows` (=FF_EXP) must be %32.
    #[allow(clippy::too_many_arguments)]
    pub fn gate_up_swiglu_reg_col_r32(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks_in: u32,
        n_expert: u32,
    ) -> eyre::Result<()> {
        if n_rows % 32 != 0 {
            return Err(eyre!("gate_up_swiglu_reg_col_r32: n_rows={n_rows} must be %32"));
        }
        // ABLATION hook (timing only, parity broken): LAGUNA_GU_ABL=nostage|noload|
        // nodot|noreduce|noscale|floor|full swaps in a component-stripped kernel.
        let kname = match std::env::var("LAGUNA_GU_ABL").ok().as_deref() {
            Some("nostage") => "q4_k_gate_up_r32_abl_nostage",
            Some("noload") => "q4_k_gate_up_r32_abl_noload",
            Some("nodot") => "q4_k_gate_up_r32_abl_nodot",
            Some("noreduce") => "q4_k_gate_up_r32_abl_noreduce",
            Some("noscale") => "q4_k_gate_up_r32_abl_noscale",
            Some("floor") => "q4_k_gate_up_r32_abl_floor",
            Some("full") => "q4_k_gate_up_r32_abl_full",
            Some("nolds") => "q4_k_gate_up_swiglu_tiled_reg_col_r32_nolds",
            Some("lds") => "q4_k_gate_up_swiglu_tiled_reg_col_r32",
            // default: no-LDS (reads activation from global; L2 serves intra-WG
            // reuse) — beats the LDS staging round-trip on gfx1151.
            _ => "q4_k_gate_up_swiglu_tiled_reg_col_r32_nolds",
        };
        let function = self.module.get_function(kname)?;
        let rpw: u32 = std::env::var("LAGUNA_GU_RPW").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
        let cfg = LaunchConfig {
            grid: (n_rows / rpw, n_expert, 1),
            block: (rpw * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            mid.raw(), gate_w_base.raw(), up_w_base.raw(), xq.raw(), expert_w.raw(),
            group_count.raw(), expert_members.raw(),
            gate_bpe, up_bpe, n_used, max_per_expert, clamp, n_rows, n_blocks_in
        ])
    }

    /// WIN #4 — COLUMN-TILED full-warp Q6_K down: 2 rows/wave + NT_COL members
    /// staged per barrier. `out` MUST be zeroed. `n_rows` (=HIDDEN) %16.
    #[allow(clippy::too_many_arguments)]
    pub fn down_reg_q6k_w32_col(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        max_per_expert: u32,
        n_rows: u32,
        n_blocks_in: u32,
        n_expert: u32,
    ) -> eyre::Result<()> {
        if n_rows % 16 != 0 {
            return Err(eyre!("down_reg_q6k_w32_col: n_rows={n_rows} must be %16"));
        }
        let lds = std::env::var("LAGUNA_DOWN_LDS").is_ok();
        // LAGUNA_DOWN_ROWS=<16|32|64...>: rows/WG for the wide no-LDS kernel
        // (block = rows/2*32). Default 16 = the shipped w32_col_nolds geometry.
        let rows: u32 = std::env::var("LAGUNA_DOWN_ROWS").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(16);
        let abl = std::env::var("LAGUNA_DOWN_ABL").ok();
        let (kname, block, grid_x) = if let Some(a) = abl.as_deref() {
            let k = match a {
                "nodot" => "q6_k_down_nolds_abl_nodot",
                "noreduce" => "q6_k_down_nolds_abl_noreduce",
                "noatomic" => "q6_k_down_nolds_abl_noatomic",
                _ => "q6_k_down_nolds_abl_full",
            };
            (k, 256u32, n_rows / 16)
        } else if lds {
            ("q6_k_down_tiled_reg_w32_col", 256u32, n_rows / 16)
        } else if rows == 16 {
            ("q6_k_down_tiled_reg_w32_col_nolds", 256u32, n_rows / 16)
        } else {
            ("q6_k_down_tiled_reg_w32_col_nolds_wide", rows / 2 * 32, n_rows / rows)
        };
        let function = self.module.get_function(kname)?;
        let cfg = LaunchConfig {
            grid: (grid_x, n_expert, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), w_base.raw(), xq_base.raw(),
            group_count.raw(), expert_members.raw(),
            dbpe, xq_slot_stride, n_used, max_per_expert, n_rows, n_blocks_in
        ])
    }

    /// WIN #2 — full-warp Q6_K reg down: packs 2 output rows per 32-lane wave
    /// (16 rows/WG) so no lane idles. `out` MUST be zeroed. `n_rows` (=HIDDEN) %16.
    #[allow(clippy::too_many_arguments)]
    pub fn down_reg_q6k_w32(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        max_per_expert: u32,
        n_rows: u32,
        n_blocks_in: u32,
        n_expert: u32,
    ) -> eyre::Result<()> {
        if n_rows % 16 != 0 {
            return Err(eyre!("down_reg_q6k_w32: n_rows={n_rows} must be %16"));
        }
        let function = self.module.get_function("q6_k_down_tiled_reg_w32")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 16, n_expert, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), w_base.raw(), xq_base.raw(),
            group_count.raw(), expert_members.raw(),
            dbpe, xq_slot_stride, n_used, max_per_expert, n_rows, n_blocks_in
        ])
    }

    /// WIN #2 — full-warp Q4_K down (re-decode per member), 2 rows/wave.
    #[allow(clippy::too_many_arguments)]
    pub fn down_q4k_w32(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        max_per_expert: u32,
        n_rows: u32,
        n_blocks_in: u32,
        n_expert: u32,
    ) -> eyre::Result<()> {
        if n_rows % 16 != 0 {
            return Err(eyre!("down_q4k_w32: n_rows={n_rows} must be %16"));
        }
        let function = self.module.get_function("q4_k_down_tiled_w32")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 16, n_expert, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), w_base.raw(), xq_base.raw(),
            group_count.raw(), expert_members.raw(),
            dbpe, xq_slot_stride, n_used, max_per_expert, n_rows, n_blocks_in
        ])
    }

    /// COLUMN-TILED full-warp Q4_K down (Q4_K analog of
    /// [`Self::down_reg_q6k_w32_col`]): register-held weight quarter + NT_COL
    /// members staged per barrier. Replaces the per-member re-decode +
    /// per-member barrier `q4_k_down_tiled_w32`. `out` MUST be zeroed.
    /// `n_rows` (=HIDDEN) %16.
    #[allow(clippy::too_many_arguments)]
    pub fn down_reg_q4k_w32_col(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        max_per_expert: u32,
        n_rows: u32,
        n_blocks_in: u32,
        n_expert: u32,
    ) -> eyre::Result<()> {
        if n_rows % 16 != 0 {
            return Err(eyre!("down_reg_q4k_w32_col: n_rows={n_rows} must be %16"));
        }
        let lds = std::env::var("LAGUNA_DOWN_LDS").is_ok();
        let rows: u32 = std::env::var("LAGUNA_DOWN_ROWS").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(16);
        let abl = std::env::var("LAGUNA_DOWN_ABL").ok();
        let (kname, block, grid_x) = if let Some(a) = abl.as_deref() {
            let k = match a {
                "nodot" => "q4_k_down_nolds_abl_nodot",
                "noreduce" => "q4_k_down_nolds_abl_noreduce",
                "noatomic" => "q4_k_down_nolds_abl_noatomic",
                _ => "q4_k_down_nolds_abl_full",
            };
            (k, 256u32, n_rows / 16)
        } else if lds {
            ("q4_k_down_tiled_reg_w32_col", 256u32, n_rows / 16)
        } else if rows == 16 {
            ("q4_k_down_tiled_reg_w32_col_nolds", 256u32, n_rows / 16)
        } else {
            ("q4_k_down_tiled_reg_w32_col_nolds_wide", rows / 2 * 32, n_rows / rows)
        };
        let function = self.module.get_function(kname)?;
        let cfg = LaunchConfig {
            grid: (grid_x, n_expert, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), w_base.raw(), xq_base.raw(),
            group_count.raw(), expert_members.raw(),
            dbpe, xq_slot_stride, n_used, max_per_expert, n_rows, n_blocks_in
        ])
    }

    /// WIN #3 diagnostic — Q6_K reg down with the dp4a MAC chain stubbed out
    /// (LDS stage + barriers + reduce + atomic kept). Measures the barrier/LDS
    /// floor. BREAKS PARITY — timing only. Same layout as [`Self::down_reg_q6k`].
    #[allow(clippy::too_many_arguments)]
    pub fn down_reg_q6k_nodot(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        max_per_expert: u32,
        n_rows: u32,
        n_blocks_in: u32,
        n_expert: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("down_reg_q6k_nodot: n_rows={n_rows} must be %8"));
        }
        let function = self.module.get_function("q6_k_down_tiled_reg_nodot")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_expert, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), w_base.raw(), xq_base.raw(),
            group_count.raw(), expert_members.raw(),
            dbpe, xq_slot_stride, n_used, max_per_expert, n_rows, n_blocks_in
        ])
    }

    /// WIN #1 — DENSE weight-read-once dp4a GEMM. Applies ONE weight matrix
    /// `w[n_rows, K]` to ALL `b` token-columns (Q8_K activations `xq[b, K]`),
    /// register-tiling the weight so per-chunk weight traffic is 1× not b×.
    /// Replaces `q4_k/q6_k_dense_gemv_batched` for the dense-L0 FFN + shared
    /// expert. `n_rows` %8; `n_blk` (=K/256) must fit the kernel's LDS stride:
    /// Q4_K path caps at 12 (HIDDEN), Q6_K path at 4 (FF/256 mid). `out[b,i]`.
    #[allow(clippy::too_many_arguments)]
    pub fn dense_gemm_dp4a(
        &self,
        stream: &Stream,
        down_dt: GgufType,
        out: &mut DeviceBuffer<f32>,
        w: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        b: u32,
        n_rows: u32,
        n_blk: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("dense_gemm_dp4a: n_rows={n_rows} must be %8"));
        }
        // Wide-row (32/WG) + no-LDS variant (default): applies WIN #2 wide-row +
        // the nolds read-from-global lever to the shared-expert / dense-L0 GEMM.
        // Requires n_rows%32; falls back to the 8-row LDS kernel otherwise or via
        // LAGUNA_DENSE_WIDE=0. Math bit-identical (same per-row accumulation).
        let wide = std::env::var("LAGUNA_DENSE_WIDE").map(|v| v != "0").unwrap_or(true)
            && n_rows % 32 == 0;
        let (fname, max_blk) = match (down_dt, wide) {
            (GgufType::Q4_K, true) => ("q4_k_dense_gemm_dp4a_r32_nolds", 12u32),
            (GgufType::Q6_K, true) => ("q6_k_dense_gemm_dp4a_r32_nolds", 4u32),
            (GgufType::Q4_K, false) => ("q4_k_dense_gemm_dp4a", 12u32),
            (GgufType::Q6_K, false) => ("q6_k_dense_gemm_dp4a", 4u32),
            (other, _) => return Err(eyre!("dense_gemm_dp4a: unsupported dtype {other:?}")),
        };
        if n_blk > max_blk {
            return Err(eyre!(
                "dense_gemm_dp4a: n_blk={n_blk} exceeds {fname} cap {max_blk}"
            ));
        }
        let function = self.module.get_function(fname)?;
        let cfg = if wide {
            LaunchConfig { grid: (n_rows / 32, 1, 1), block: (1024, 1, 1), shared_mem_bytes: 0 }
        } else {
            LaunchConfig { grid: (n_rows / 8, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 }
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), w.raw(), xq.raw(), b, n_rows, n_blk
        ])
    }

    /// ATOMIC-FREE down: PLAIN-STORE each member's (b,slot) contribution into
    /// its unique slice `part[(b*n_used+slot)*n_rows + r]` (no cross-expert
    /// atomic contention). Follow with [`Self::down_reduce_slots`] to sum the
    /// n_used=TOPK slots into `acc[b, r]`. `part` need not be zeroed (n_rows%16).
    /// dtype-dispatched (Q6_K/Q4_K). Same args as [`Self::down`] but `out`=part.
    #[allow(clippy::too_many_arguments)]
    pub fn down_part(
        &self,
        stream: &Stream,
        down_dt: GgufType,
        part: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        max_per_expert: u32,
        n_rows: u32,
        n_blocks_in: u32,
        n_expert: u32,
    ) -> eyre::Result<()> {
        if n_rows % 16 != 0 {
            return Err(eyre!("down_part: n_rows={n_rows} must be %16"));
        }
        let rows: u32 = std::env::var("LAGUNA_DOWN_ROWS").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(16);
        let block = if rows == 16 { 256 } else { rows / 2 * 32 };
        let grid_x = if rows == 16 { n_rows / 16 } else { n_rows / rows };
        let fname = match down_dt {
            GgufType::Q6_K => "q6_k_down_tiled_reg_w32_col_nolds_part",
            GgufType::Q4_K => "q4_k_down_tiled_reg_w32_col_nolds_part",
            other => return Err(eyre!("down_part: unsupported dtype {other:?}")),
        };
        let function = self.module.get_function(fname)?;
        let cfg = LaunchConfig {
            grid: (grid_x, n_expert, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            part.raw(), w_base.raw(), xq_base.raw(),
            group_count.raw(), expert_members.raw(),
            dbpe, xq_slot_stride, n_used, max_per_expert, n_rows, n_blocks_in
        ])
    }

    /// Sum the `n_used` slot-partials written by [`Self::down_part`] into
    /// `acc[b*n_rows + r]`. `total = B * n_rows`.
    pub fn down_reduce_slots(
        &self,
        stream: &Stream,
        acc: &mut DeviceBuffer<f32>,
        part: &DeviceBuffer<f32>,
        n_rows: u32,
        n_used: u32,
        total: u32,
    ) -> eyre::Result<()> {
        let function = self.module.get_function("moe_down_reduce_slots")?;
        let cfg = LaunchConfig {
            grid: ((total + 255) / 256, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            acc.raw(), part.raw(), n_rows, n_used, total
        ])
    }

    /// By-expert down projection, dtype-dispatched (Q6_K or Q4_K). Accumulates
    /// (atomicAdd) each expert's contribution into `out[b, r]`. `out` MUST be
    /// zeroed before this launch. `xq_base` = Q8_K of `mid`, `[B, n_used,
    /// xq_slot_stride]`. `n_blocks_in = FF_EXP/256`, `n_rows = HIDDEN` (%8).
    #[allow(clippy::too_many_arguments)]
    pub fn down(
        &self,
        stream: &Stream,
        down_dt: GgufType,
        out: &mut DeviceBuffer<f32>,
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        max_per_expert: u32,
        n_rows: u32,
        n_blocks_in: u32,
        n_expert: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("down: n_rows={n_rows} must be %8"));
        }
        let fname = match down_dt {
            GgufType::Q6_K => "q6_k_down_tiled",
            GgufType::Q4_K => "q4_k_down_tiled",
            other => return Err(eyre!("down: unsupported dtype {other:?}")),
        };
        let function = self.module.get_function(fname)?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_expert, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), w_base.raw(), xq_base.raw(),
            group_count.raw(), expert_members.raw(),
            dbpe, xq_slot_stride, n_used, max_per_expert, n_rows, n_blocks_in
        ])
    }
}
