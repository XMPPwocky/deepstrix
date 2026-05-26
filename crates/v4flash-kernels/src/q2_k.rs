//! Q2_K × Q8_K matvec, accumulating across the routed-MoE down projection.
//! Mirrors ds4's CPU expert-down accumulator (and CUDA's `dev_dot_q2_K_q8_K`-based
//! down matvec). One launch per selected expert; first launch zero-initialises,
//! subsequent launches add into `out`.

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, LaunchConfig, Module, Stream};

const Q2_K_ACC_GFX1201: &[u8] = include_bytes!(env!("KERNEL_Q2_K_ACCUMULATE_MATVEC_GFX1201"));
const Q2_K_ACC_GFX1151: &[u8] = include_bytes!(env!("KERNEL_Q2_K_ACCUMULATE_MATVEC_GFX1151"));
const Q2_K_ACC_PAR_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_Q2_K_ACCUMULATE_MATVEC_PAR_GFX1201"));
const Q2_K_ACC_PAR_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_Q2_K_ACCUMULATE_MATVEC_PAR_GFX1151"));

pub const BLOCK_Q2_K_BYTES: usize = 84;

pub struct Q2KAccumulateMatvec {
    module: Module,
}

impl Q2KAccumulateMatvec {
    /// M14g parallel variant by default — 4 lanes per super-block, all 32
    /// warp lanes active (vs 8 in the serial kernel for n_blocks_in=8).
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            Q2_K_ACC_PAR_GFX1201
        } else if arch.starts_with("gfx1151") {
            Q2_K_ACC_PAR_GFX1151
        } else {
            return Err(eyre!("unsupported arch for q2_k_accumulate_matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Original serial kernel — kept for regression testing.
    pub fn for_arch_serial(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            Q2_K_ACC_GFX1201
        } else if arch.starts_with("gfx1151") {
            Q2_K_ACC_GFX1151
        } else {
            return Err(eyre!("unsupported arch for q2_k_accumulate_matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        w_expert: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        n_rows: u32,
        n_blocks_in: u32,
        zero_init: bool,
    ) -> eyre::Result<()> {
        self.launch_with_offset(stream, out, w_expert, 0, xq, n_rows, n_blocks_in, zero_init)
    }

    /// Same as [`launch`] but reads `w_expert` starting at a byte offset.
    pub fn launch_with_offset(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        w_expert: &DeviceBuffer<u8>,
        w_expert_offset: usize,
        xq: &DeviceBuffer<u8>,
        n_rows: u32,
        n_blocks_in: u32,
        zero_init: bool,
    ) -> eyre::Result<()> {
        self.launch_with_full_offsets(
            stream,
            out,
            w_expert,
            w_expert_offset,
            xq,
            0,
            n_rows,
            n_blocks_in,
            zero_init,
        )
    }

    /// Full-offset variant: also offsets the activation buffer `xq`.
    /// Lets the MoE pipeline read per-slot q8k blocks from a single
    /// concatenated mid-quant buffer (paired with
    /// [`crate::q8_k::Q8KQuantize::launch_with_offsets`]) — no per-slot
    /// host syncs.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_with_full_offsets(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,
        w_expert: &DeviceBuffer<u8>,
        w_expert_offset: usize,
        xq: &DeviceBuffer<u8>,
        xq_offset_bytes: usize,
        n_rows: u32,
        n_blocks_in: u32,
        zero_init: bool,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!(
                "q2_k_accumulate_matvec: n_rows={n_rows} must be multiple of 8"
            ));
        }
        let row_bytes = (n_blocks_in as usize) * BLOCK_Q2_K_BYTES;
        let need = (n_rows as usize) * row_bytes;
        if w_expert.byte_len() < w_expert_offset + need {
            return Err(eyre!(
                "w_expert bytes: have {}, need offset {} + {} = {}",
                w_expert.byte_len(),
                w_expert_offset,
                need,
                w_expert_offset + need
            ));
        }
        // xq blocks are 292 bytes per (n_blocks_in matches q8k blocks).
        let xq_need = (n_blocks_in as usize) * 292;
        if xq.byte_len() < xq_offset_bytes + xq_need {
            return Err(eyre!(
                "xq bytes: have {}, need offset {} + {} = {}",
                xq.byte_len(),
                xq_offset_bytes,
                xq_need,
                xq_offset_bytes + xq_need
            ));
        }

        let function = self
            .module
            .get_function("q2_k_accumulate_matvec_par")
            .or_else(|_| self.module.get_function("q2_k_accumulate_matvec"))?;
        let mut out_ptr = out.raw();
        // SAFETY: bounds-checked above.
        let mut w_ptr = unsafe { (w_expert.raw() as *mut u8).add(w_expert_offset) }
            as v4flash_hip::sys::hipDeviceptr_t;
        let mut x_ptr = unsafe { (xq.raw() as *mut u8).add(xq_offset_bytes) }
            as v4flash_hip::sys::hipDeviceptr_t;
        let mut nr = n_rows;
        let mut nb = n_blocks_in;
        let mut zi: u32 = if zero_init { 1 } else { 0 };
        let mut args: [*mut c_void; 6] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut x_ptr as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
            &mut zi as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_rows / 8, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}

impl Q2KAccumulateMatvec {
    /// Batched variant: single launch loops over all `n_used` slots
    /// internally per workgroup. Writes summed result directly (no
    /// accumulate). Eliminates 5 launches + 5 boundary syncs per layer.
    /// (M14k.5)
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
            return Err(eyre!("q2_k_matvec_par_batched: n_rows={n_rows} not %8"));
        }
        if out.len() < n_rows as usize {
            return Err(eyre!(
                "q2_k batched out: len {} < n_rows {n_rows}",
                out.len()
            ));
        }
        if (selected.len() as u32) < n_used {
            return Err(eyre!("selected len {} < n_used {n_used}", selected.len()));
        }

        let function = self.module.get_function("q2_k_matvec_par_batched")?;
        let mut out_ptr = out.raw();
        let mut w_ptr = w_base.raw();
        let mut xq_ptr = xq_base.raw();
        let mut sel_ptr = selected.raw();
        let mut dbpe_v = dbpe;
        let mut xq_stride_v = xq_slot_stride;
        let mut nu = n_used;
        let mut nr = n_rows;
        let mut nb = n_blocks_in;
        let mut args: [*mut c_void; 9] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut sel_ptr as *mut _ as *mut c_void,
            &mut dbpe_v as *mut _ as *mut c_void,
            &mut xq_stride_v as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_rows / 8, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }

    /// M50 Phase 3 v0: B-batched variant — grid.z = B. Per-batch xq, selected,
    /// out. Weight shared across batch.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_batched_bxn(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<f32>,         // [B, n_rows]
        w_base: &DeviceBuffer<u8>,
        xq_base: &DeviceBuffer<u8>,          // [B, n_used * xq_slot_stride]
        selected: &DeviceBuffer<i32>,        // [B, n_used]
        dbpe: u32,
        xq_slot_stride: u32,
        n_used: u32,
        n_rows: u32,
        n_blocks_in: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if n_rows % 8 != 0 {
            return Err(eyre!(
                "q2_k_matvec_par_batched_bxn: n_rows={n_rows} not %8"
            ));
        }
        let needed = (batch as usize) * (n_rows as usize);
        if out.len() < needed {
            return Err(eyre!(
                "q2_k batched_bxn out: len {} < {needed}",
                out.len()
            ));
        }
        let function = self
            .module
            .get_function("q2_k_matvec_par_batched_BxN")?;
        let mut out_ptr = out.raw();
        let mut w_ptr = w_base.raw();
        let mut xq_ptr = xq_base.raw();
        let mut sel_ptr = selected.raw();
        let mut dbpe_v = dbpe;
        let mut xq_stride_v = xq_slot_stride;
        let mut nu = n_used;
        let mut nr = n_rows;
        let mut nb = n_blocks_in;
        let mut args: [*mut c_void; 9] = [
            &mut out_ptr as *mut _ as *mut c_void,
            &mut w_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut sel_ptr as *mut _ as *mut c_void,
            &mut dbpe_v as *mut _ as *mut c_void,
            &mut xq_stride_v as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_rows / 8, 1, batch),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}

/// CPU port of `dev_dot_q2_K_q8_K_block` (ds4_cuda.cu:7296). Used by oracle.
pub fn cpu_dot_q2_k_q8_k(n_blocks: usize, w_bytes: &[u8], y_bytes: &[u8]) -> f32 {
    assert_eq!(w_bytes.len(), n_blocks * BLOCK_Q2_K_BYTES);
    assert_eq!(y_bytes.len(), n_blocks * 292);
    let mut sumf = 0.0f32;
    for b in 0..n_blocks {
        let w_off = b * BLOCK_Q2_K_BYTES;
        let y_off = b * 292;
        let sc = &w_bytes[w_off..w_off + 16];
        let q2_base = &w_bytes[w_off + 16..w_off + 80];
        let xd_bits = u16::from_le_bytes([w_bytes[w_off + 80], w_bytes[w_off + 81]]);
        let xdm_bits = u16::from_le_bytes([w_bytes[w_off + 82], w_bytes[w_off + 83]]);
        let yd = f32::from_le_bytes([
            y_bytes[y_off],
            y_bytes[y_off + 1],
            y_bytes[y_off + 2],
            y_bytes[y_off + 3],
        ]);
        let q8 = &y_bytes[y_off + 4..y_off + 260];
        let bsums_bytes = &y_bytes[y_off + 260..y_off + 292];

        let mut summs: i32 = 0;
        for j in 0..16 {
            let bs = i16::from_le_bytes([bsums_bytes[j * 2], bsums_bytes[j * 2 + 1]]) as i32;
            summs += bs * ((sc[j] >> 4) as i32);
        }
        let dall = yd * crate::iq2_xxs_tables::f16_to_f32(xd_bits);
        let dmin = yd * crate::iq2_xxs_tables::f16_to_f32(xdm_bits);

        let mut isum: i32 = 0;
        let mut is = 0usize;
        let mut q2_off = 0usize;
        let mut q8_off = 0usize;
        for _k in 0..(256 / 128) {
            let mut shift = 0;
            for _j in 0..4 {
                let d = (sc[is] & 0x0f) as i32;
                is += 1;
                let mut inner: i32 = 0;
                for i in 0..16 {
                    let v = ((q2_base[q2_off + i] >> shift) & 0x03) as i32;
                    inner += v * (q8[q8_off + i] as i8 as i32);
                }
                isum += d * inner;
                let d_b = (sc[is] & 0x0f) as i32;
                is += 1;
                let mut inner2: i32 = 0;
                for i in 0..16 {
                    let v = ((q2_base[q2_off + 16 + i] >> shift) & 0x03) as i32;
                    inner2 += v * (q8[q8_off + 16 + i] as i8 as i32);
                }
                isum += d_b * inner2;
                shift += 2;
                q8_off += 32;
            }
            q2_off += 32;
        }
        sumf += dall * (isum as f32) - dmin * (summs as f32);
    }
    sumf
}
