//! Q6_K × Q8_K matvec — batched routed-MoE **down** projection.
//!
//! Companion to [`crate::q4_k::Q4KMatvec`] (gate/up) and
//! [`crate::q2_k::Q2KAccumulateMatvec`] (V4-Flash down). For the Laguna port
//! the down-projection experts are Q6_K on most layers, so this provides the
//! batched Q6_K down-accumulate that no existing kernel covered.
//!
//! `q6_k_matvec_par_batched` sums `y = Σ_s down_{selected[s]}(midq[s])` over
//! the top-k selected experts in a single launch (the routing scale is baked
//! into `mid` upstream in the gate/up swiglu step). Layers whose `ffn_down`
//! tensor is Q4_K instead reuse [`crate::q4_k::Q4KMatvec::launch_batched`].
//!
//! Q6_K block layout (210 bytes / 256 elements): ql[128] | qh[64] |
//! scales[16] (int8) | d (ggml_half, 2 B). The 6-bit weight's -32 offset is
//! baked into the signed int8 decode, so the Q8_K dot is a plain signed dp4a.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const Q6_K_ACC_PAR_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_Q6_K_ACCUMULATE_MATVEC_PAR_GFX1201"));
const Q6_K_ACC_PAR_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_Q6_K_ACCUMULATE_MATVEC_PAR_GFX1151"));

pub const BLOCK_Q6_K_BYTES: usize = 210;

pub struct Q6KMatvec {
    module: Module,
}

impl Q6KMatvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            Q6_K_ACC_PAR_GFX1201
        } else if arch.starts_with("gfx1151") {
            Q6_K_ACC_PAR_GFX1151
        } else {
            return Err(eyre!("unsupported arch for q6_k_matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Batched MoE down-accumulate. Single launch loops over `n_used`
    /// selected experts and writes the summed `out[n_rows]` directly.
    /// Per-slot activation blocks come from `xq_base + s*xq_slot_stride`;
    /// per-expert weight from `w_base + selected[s]*dbpe`.
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
            return Err(eyre!("q6_k_matvec_par_batched: n_rows={n_rows} not %8"));
        }
        if out.len() < n_rows as usize {
            return Err(eyre!("q6_k batched out: len {} < n_rows {n_rows}", out.len()));
        }
        if (selected.len() as u32) < n_used {
            return Err(eyre!("selected len {} < n_used {n_used}", selected.len()));
        }

        let function = self.module.get_function("q6_k_matvec_par_batched")?;
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

    /// BATCHED-over-tokens down projection: sum over `n_used` experts for each
    /// of `batch` tokens in ONE launch (`grid.z = batch`). `selected` is
    /// `[batch, n_used]`, `xq_base` is `[batch, n_used*xq_slot_stride]`, and
    /// `out` is `[batch, n_rows]` (token-major). Q6_K companion to
    /// [`crate::q4_k::Q4KMatvec::launch_batched_bxn`].
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched_bxn(
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
        batch: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("q6_k_matvec_par_batched_bxn: n_rows={n_rows} not %8"));
        }
        if out.len() < (batch as usize) * (n_rows as usize) {
            return Err(eyre!("q6_k bxn out: len {} < batch*n_rows", out.len()));
        }
        if (selected.len() as u32) < batch * n_used {
            return Err(eyre!("selected len {} < batch*n_used", selected.len()));
        }
        let function = self.module.get_function("q6_k_matvec_par_batched_bxn")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, 1, batch),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), w_base.raw(), xq_base.raw(), selected.raw(),
            dbpe, xq_slot_stride, n_used, n_rows, n_blocks_in
        ])
    }

    /// Single-expert accumulate: `out[row] {=,+=} Σ_b dot(W[row,b], xq[b])`.
    /// `zero_init` writes on the first expert, then accumulate.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_accumulate(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        w_expert: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        n_rows: u32,
        n_blocks_in: u32,
        zero_init: bool,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("q6_k_accumulate_matvec_par: n_rows={n_rows} not %8"));
        }
        if out.len() < n_rows as usize {
            return Err(eyre!("q6_k acc out: len {} < n_rows {n_rows}", out.len()));
        }
        let function = self.module.get_function("q6_k_accumulate_matvec_par")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            out.raw(), w_expert.raw(), xq.raw(), n_rows, n_blocks_in,
            if zero_init { 1u32 } else { 0u32 }
        ])
    }
}
