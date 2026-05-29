//! Q4_K × Q8_K matvec, batched MoE down/gate/up projection.
//!
//! Mirrors the Q2_K batched-par kernel but for Q4_K-quantized weights.
//! Required for MTP support: antirez's V4-Flash MTP GGUF stores routed
//! experts as Q4_K (the main-model file uses iq2_xxs gate/up + q2_k down).
//!
//! Block layout (144 bytes per 256 elements):
//!   d (ggml_half, 2 B) | dmin (ggml_half, 2 B) | scales[12] | qs[128]
//!
//! Per element: w = d_super * d_sub * q4 - dmin_super * m_sub
//! (8 sub-blocks of 32 elements each, 6-bit packed (d_sub, m_sub) pairs).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const Q4_K_MATVEC_PAR_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_Q4_K_MATVEC_PAR_GFX1201"));
const Q4_K_MATVEC_PAR_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_Q4_K_MATVEC_PAR_GFX1151"));

pub const BLOCK_Q4_K_BYTES: usize = 144;

pub struct Q4KMatvec {
    module: Module,
}

impl Q4KMatvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            Q4_K_MATVEC_PAR_GFX1201
        } else if arch.starts_with("gfx1151") {
            Q4_K_MATVEC_PAR_GFX1151
        } else {
            return Err(eyre!("unsupported arch for q4_k_matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Fused gate × up × swiglu × expert_w for MoE gate+up step.
    /// Single launch handles all `n_used` slots via grid.y. Writes
    /// `mid[slot * n_rows + r]` for each (slot, r). Mirrors iq2's
    /// `iq2_xxs_pair_matvec_fused_swiglu_batch` but for Q4_K weights.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_pair_swiglu_batched(
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
        n_blocks_in: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!(
                "q4_k_pair_swiglu: n_rows={n_rows} must be %8"
            ));
        }
        if mid.len() < (n_used as usize) * (n_rows as usize) {
            return Err(eyre!(
                "mid len {} < n_used {} * n_rows {} = {}",
                mid.len(),
                n_used,
                n_rows,
                (n_used as usize) * (n_rows as usize)
            ));
        }
        if (selected.len() as u32) < n_used {
            return Err(eyre!("selected len {} < n_used {n_used}", selected.len()));
        }
        if (expert_w.len() as u32) < n_used {
            return Err(eyre!("expert_w len {} < n_used {n_used}", expert_w.len()));
        }

        let function = self
            .module
            .get_function("q4_k_pair_matvec_fused_swiglu_batch")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_used, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [
            mid.raw(), gate_w_base.raw(), up_w_base.raw(), xq.raw(), expert_w.raw(),
            selected.raw(), gate_bpe, up_bpe, clamp, n_rows, n_blocks_in
        ])
    }

    /// Batched MoE matvec. Single launch loops over `n_used` selected
    /// experts internally per workgroup. Writes the summed result
    /// directly to `out[n_rows]` (no zero-init or accumulate dance).
    ///
    /// Used for the down projection in the routed MoE pipeline (where
    /// `n_rows = n_embd` and `xq_slot_stride` covers `n_ff_exp_in_blocks
    /// * BLOCK_Q8_K_BYTES`). Also usable for gate or up by passing
    /// `n_rows = n_ff_exp` and a single-slot `xq`.
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
            return Err(eyre!("q4_k_matvec_par_batched: n_rows={n_rows} not %8"));
        }
        if out.len() < n_rows as usize {
            return Err(eyre!(
                "q4_k batched out: len {} < n_rows {n_rows}",
                out.len()
            ));
        }
        if (selected.len() as u32) < n_used {
            return Err(eyre!("selected len {} < n_used {n_used}", selected.len()));
        }

        let function = self.module.get_function("q4_k_matvec_par_batched")?;
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
}

/// CPU reference port of ds4's `dev_dot_q4_K_q8_K_block` (ds4_cuda.cu:7279).
/// Used by the oracle test to validate the kernel's output.
pub fn cpu_dot_q4_k_q8_k(n_blocks: usize, w_bytes: &[u8], y_bytes: &[u8]) -> f32 {
    assert_eq!(w_bytes.len(), n_blocks * BLOCK_Q4_K_BYTES);
    assert_eq!(y_bytes.len(), n_blocks * 292);
    let mut sumf = 0.0f32;
    for b in 0..n_blocks {
        let w_off = b * BLOCK_Q4_K_BYTES;
        let y_off = b * 292;
        let xd_bits = u16::from_le_bytes([w_bytes[w_off], w_bytes[w_off + 1]]);
        let xdm_bits = u16::from_le_bytes([w_bytes[w_off + 2], w_bytes[w_off + 3]]);
        let scales = &w_bytes[w_off + 4..w_off + 16];
        let qs = &w_bytes[w_off + 16..w_off + 144];
        let yd = f32::from_le_bytes([
            y_bytes[y_off],
            y_bytes[y_off + 1],
            y_bytes[y_off + 2],
            y_bytes[y_off + 3],
        ]);
        let q8 = &y_bytes[y_off + 4..y_off + 260];
        let bsums_bytes = &y_bytes[y_off + 260..y_off + 292];

        let xd = crate::iq2_xxs_tables::f16_to_f32(xd_bits);
        let xmin = crate::iq2_xxs_tables::f16_to_f32(xdm_bits);

        let mut isum: i32 = 0;
        let mut summs: i32 = 0;
        for j in 0..8u32 {
            // Per sub-block decode of (d_sub, m_sub) — matches ggml's get_scale_min_k4.
            let (sc, m): (u8, u8) = if j < 4 {
                (scales[j as usize] & 0x3F, scales[(j + 4) as usize] & 0x3F)
            } else {
                let d = (scales[(j + 4) as usize] & 0x0F)
                    | ((scales[(j - 4) as usize] >> 6) << 4);
                let m = (scales[(j + 4) as usize] >> 4)
                    | ((scales[j as usize] >> 6) << 4);
                (d, m)
            };

            let bs0 = i16::from_le_bytes([
                bsums_bytes[(2 * j) as usize * 2],
                bsums_bytes[(2 * j) as usize * 2 + 1],
            ]) as i32;
            let bs1 = i16::from_le_bytes([
                bsums_bytes[(2 * j + 1) as usize * 2],
                bsums_bytes[(2 * j + 1) as usize * 2 + 1],
            ]) as i32;
            summs += (m as i32) * (bs0 + bs1);

            let byte_off = (j >> 1) as usize * 32;
            let shift = if j & 1 == 1 { 4 } else { 0 };
            let mut dot32: i32 = 0;
            for i in (0..32).step_by(4) {
                let dw = i32::from_le_bytes([
                    qs[byte_off + i],
                    qs[byte_off + i + 1],
                    qs[byte_off + i + 2],
                    qs[byte_off + i + 3],
                ]);
                let v = (dw >> shift) & 0x0f0f0f0f;
                let q8_off = j as usize * 32 + i;
                let q8_dw = i32::from_le_bytes([
                    q8[q8_off] as u8 as u8,
                    q8[q8_off + 1] as u8 as u8,
                    q8[q8_off + 2] as u8 as u8,
                    q8[q8_off + 3] as u8 as u8,
                ]);
                // Per-byte signed × signed dot.
                for byte_i in 0..4 {
                    let a = ((v >> (byte_i * 8)) & 0xFF) as i8 as i32;
                    let b = ((q8_dw >> (byte_i * 8)) & 0xFF) as i8 as i32;
                    dot32 += a * b;
                }
            }
            isum += (sc as i32) * dot32;
        }
        sumf += yd * xd * isum as f32 - yd * xmin * summs as f32;
    }
    sumf
}
