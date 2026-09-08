//! Q8_K activation quantization — packs an F32 vector into 256-element
//! Q8_K blocks (292 bytes each) consumed by routed-expert IQ2_XXS/Q2_K
//! matvecs. Mirrors `ds4_quantize_row_q8_K` (ds4.c:1655).

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const Q8_K_QUANTIZE_GFX1201: &[u8] = include_bytes!(env!("KERNEL_Q8_K_QUANTIZE_GFX1201"));
const Q8_K_QUANTIZE_GFX1151: &[u8] = include_bytes!(env!("KERNEL_Q8_K_QUANTIZE_GFX1151"));

pub const QK_K: u32 = 256;
pub const BLOCK_Q8_K_BYTES: usize = 292;

pub struct Q8KQuantize {
    module: Module,
}

impl Q8KQuantize {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            Q8_K_QUANTIZE_GFX1201
        } else if arch.starts_with("gfx1151") {
            Q8_K_QUANTIZE_GFX1151
        } else {
            return Err(eyre!("unsupported arch for q8_k_quantize: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    pub fn launch(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<u8>,
        x: &DeviceBuffer<f32>,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        let needed_x = (n_blocks as usize) * (QK_K as usize);
        let needed_out = (n_blocks as usize) * BLOCK_Q8_K_BYTES;
        if x.len() < needed_x {
            return Err(eyre!(
                "q8_k_quantize x len: have {}, need {}",
                x.len(),
                needed_x
            ));
        }
        if out.byte_len() < needed_out {
            return Err(eyre!(
                "q8_k_quantize out bytes: have {}, need {}",
                out.byte_len(),
                needed_out
            ));
        }

        self.launch_with_offsets(stream, out, 0, x, 0, n_blocks)
    }

    /// Same as [`launch`] but with byte offset into `out` and element
    /// offset into `x`. Lets the MoE pipeline write per-slot q8k blocks
    /// into a single concatenated buffer without per-slot host copies.
    pub fn launch_with_offsets(
        &self,
        stream: &Stream,
        out: &mut DeviceBuffer<u8>,
        out_offset_bytes: usize,
        x: &DeviceBuffer<f32>,
        x_offset_elems: usize,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        let needed_x = (n_blocks as usize) * (QK_K as usize);
        let needed_out = (n_blocks as usize) * BLOCK_Q8_K_BYTES;
        if x.len() < x_offset_elems + needed_x {
            return Err(eyre!(
                "q8_k_quantize x: len {} < offset {} + need {}",
                x.len(),
                x_offset_elems,
                needed_x
            ));
        }
        if out.byte_len() < out_offset_bytes + needed_out {
            return Err(eyre!(
                "q8_k_quantize out: bytes {} < offset {} + need {}",
                out.byte_len(),
                out_offset_bytes,
                needed_out
            ));
        }

        let function = self.module.get_function("q8_k_quantize")?;
        // SAFETY: bounds-checked above.
        let out_ptr = unsafe { (out.raw() as *mut u8).add(out_offset_bytes) }
            as v4flash_hip::sys::hipDeviceptr_t;
        let x_ptr = unsafe { (x.raw() as *mut f32).add(x_offset_elems) }
            as v4flash_hip::sys::hipDeviceptr_t;
        let cfg = LaunchConfig {
            grid: (n_blocks, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(function, cfg, stream, [out_ptr, x_ptr, n_blocks])
    }

    /// f32 -> f16 cast of `n` elements (WMMA MoE activation prep).
    pub fn launch_cast_f16(&self, stream: &Stream, out: &mut DeviceBuffer<u16>, x: &DeviceBuffer<f32>, n: u32) -> eyre::Result<()> {
        if out.len() < n as usize || x.len() < n as usize {
            return Err(eyre!("f32_to_f16_cast: buffers too small for n={n}"));
        }
        if n == 0 { return Ok(()); }
        let function = self.module.get_function("f32_to_f16_cast")?;
        let threads = (n as usize).div_ceil(8);
        let cfg = LaunchConfig { grid: (threads.div_ceil(256) as u32, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [out.raw(), x.raw(), n])
    }

    /// f32 [rows][cols] -> f16 [rows][out_pitch]; cols and out_pitch % 8 == 0.
    pub fn launch_cast_f16_2d(&self, stream: &Stream, out: &mut DeviceBuffer<u16>, x: &DeviceBuffer<f32>, rows: u32, cols: u32, out_pitch: u32) -> eyre::Result<()> {
        if cols % 8 != 0 || out_pitch % 8 != 0 || out_pitch < cols {
            return Err(eyre!("f32_to_f16_cast_2d: cols={cols} out_pitch={out_pitch} invalid"));
        }
        if out.len() < (rows as usize) * (out_pitch as usize) || x.len() < (rows as usize) * (cols as usize) {
            return Err(eyre!("f32_to_f16_cast_2d: buffers too small (rows={rows})"));
        }
        if rows == 0 { return Ok(()); }
        let function = self.module.get_function("f32_to_f16_cast_2d")?;
        let threads = (rows as usize) * (cols as usize / 8);
        let cfg = LaunchConfig { grid: (threads.div_ceil(256) as u32, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(function, cfg, stream, [out.raw(), x.raw(), rows, cols, out_pitch])
    }
}
